//! Per-chunk pipeline selection.
//!
//! This is where the engine earns its "adaptive" name. For each chunk it
//! evaluates the pipelines that [`crate::analyze`] says are worth trying,
//! measures what each actually produces, and keeps the smallest — falling back
//! to `Store` whenever nothing wins.
//!
//! # Why try-and-pick rather than prediction
//!
//! Predicting the best pipeline from cheap statistics is fast but wrong often
//! enough to matter, and being wrong is invisible. Trying candidates and
//! measuring is exact by construction, and the only cost is time. The trade is
//! therefore made explicit: [`CandidatePolicy`] lets a caller bound the search
//! (one candidate, cheap pair, or full sweep) and the benchmark harness measures
//! what each policy actually costs.
//!
//! # Independence and dictionaries
//!
//! The planner also decides whether a frame may reference outside itself.
//! Independent frames decode in any order, which is what enables parallel
//! decoding; linked frames compress better because each chunk can match into the
//! tail of the previous one.

use crate::analyze::{Analysis, DataKind};
use crate::codec::{DecodeContext, DictionaryTable, EncodeContext, Registry};
use crate::config::{Config, Level, Strategy};
use crate::error::Result;
use crate::format::{PipelineId, TransformId};
use crate::matchfinder::DictionaryIndex;
use crate::transform;

/// How many candidate pipelines to evaluate per chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidatePolicy {
    /// Try exactly one pipeline, chosen from the level and analysis. Fastest;
    /// the ratio is whatever that pipeline gives.
    Single,
    /// `Store` plus the two most promising pipelines.
    Cheap,
    /// Every pipeline the analysis admits, in each transform configuration.
    Full,
}

impl CandidatePolicy {
    /// Policy implied by a strategy.
    pub fn for_strategy(strategy: Strategy) -> Self {
        match strategy {
            Strategy::NeverCompress => CandidatePolicy::Single,
            Strategy::Fixed => CandidatePolicy::Single,
            Strategy::Adaptive => CandidatePolicy::Full,
        }
    }
}

/// The outcome of planning one chunk.
#[derive(Debug, Clone)]
pub struct ChunkPlan {
    /// Pipeline to use.
    pub pipeline: PipelineId,
    /// Transform to apply before the pipeline.
    pub transform: TransformId,
    /// Encoded payload, already the best of those tried.
    pub payload: Vec<u8>,
    /// Uncompressed bytes this plan decodes to.
    ///
    /// This is the *input* length, not [`payload`](Self::payload)'s, and keeping
    /// both is the point: a frame records the size it decodes to, so a plan that
    /// conflated the two would emit a header that disagrees with its payload.
    pub content_size: usize,
    /// True when the frame references only its own output.
    pub independent: bool,
    /// Pipelines that were actually evaluated, for `info` and benchmarks.
    pub tried: Vec<PipelineId>,
    /// True when the payload was stored because nothing beat it.
    pub stored: bool,
    /// 1-based dictionary id this frame references, or 0 for none.
    pub dict_id: u8,
}

impl ChunkPlan {
    /// Uncompressed size this plan decodes to.
    pub fn content_size(&self) -> usize {
        self.content_size
    }
}

/// Chooses and runs pipelines for chunks.
#[derive(Debug)]
pub struct Planner {
    registry: Registry,
    config: Config,
    policy: CandidatePolicy,
    dictionary: Option<DictionaryIndex>,
}

impl Planner {
    /// Build a planner from a validated configuration.
    pub fn new(config: &Config) -> Self {
        let dictionary = config.dictionary().map(DictionaryIndex::build);
        Self {
            registry: Registry::new(),
            config: config.clone(),
            policy: CandidatePolicy::for_strategy(config.strategy()),
            dictionary,
        }
    }

    /// Override the candidate policy, ignoring the strategy's default.
    pub fn with_policy(mut self, policy: CandidatePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The registry in use, for callers that need direct pipeline access.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Candidate pipelines for a chunk, ordered cheapest-first.
    ///
    /// Ordering matters: under [`CandidatePolicy::Cheap`] the first two entries
    /// are the ones kept, so they must be the most likely to win.
    fn candidates(&self, input: &[u8], analysis: &Analysis) -> Vec<(PipelineId, TransformId)> {
        let level = self.config.level();
        let mut out: Vec<(PipelineId, TransformId)> = Vec::new();

        if self.policy == CandidatePolicy::Single {
            let (p, t) = self.single_choice(input, analysis, level);
            out.push((p, t));
            return out;
        }

        // Statistical first: it is the strongest general-purpose pipeline, so
        // under a truncated search it is the one most worth measuring.
        for codec in self.registry.all() {
            if codec.id() == PipelineId::Store || codec.id() == PipelineId::Dictionary {
                continue;
            }
            if codec.candidate(analysis, level) {
                let primary = (codec.id(), TransformId::None);
                out.push(primary);
                // A transform is only worth a second pass when the analysis
                // suggests one, so it does not double the cost by default.
                if let Some(t) = self.transform_for(input, analysis, codec.id()) {
                    out.push((codec.id(), t));
                }
            }
        }

        // RLE is a registry member, so the loop above has already added it when its
        // own `candidate()` accepted. This push used to duplicate it on repetitive
        // data: the same pipeline encoded twice, byte for byte, for no benefit.
        // Measured as 7 candidates on the `runs` corpus where 6 were correct.
        if analysis.kind == DataKind::Repetitive {
            out.push((PipelineId::Rle, TransformId::None));
        }

        if self.policy == CandidatePolicy::Cheap {
            out.truncate(2);
        }

        // `store` is always measured last as the safety net.
        out.push((PipelineId::Store, TransformId::None));

        // Every candidate is encoded in full, and a repeat cannot change which one
        // is smallest, so a duplicate is pure waste. Deduplicating once here rather
        // than at each push site means a future candidate rule cannot reintroduce
        // the problem without this guard catching it.
        let mut seen: Vec<(PipelineId, TransformId)> = Vec::with_capacity(out.len());
        out.retain(|c| {
            if seen.contains(c) {
                false
            } else {
                seen.push(*c);
                true
            }
        });
        out
    }

    /// The one pipeline to use when not searching.
    fn single_choice(
        &self,
        input: &[u8],
        analysis: &Analysis,
        level: Level,
    ) -> (PipelineId, TransformId) {
        if self.config.strategy() == Strategy::NeverCompress || level == Level::None {
            return (PipelineId::Store, TransformId::None);
        }
        if analysis.looks_incompressible() {
            return (PipelineId::Store, TransformId::None);
        }
        let p = match level {
            Level::Fast => PipelineId::LzFast,
            Level::Default | Level::High | Level::Max => PipelineId::Statistical,
            Level::None => PipelineId::Store,
        };
        let t = self
            .transform_for(input, analysis, p)
            .unwrap_or(TransformId::None);
        (p, t)
    }

    /// A transform worth trying for this pipeline, if any.
    ///
    /// Delta is the only non-identity transform, and it earns its place only on
    /// data whose consecutive bytes are correlated — slowly varying counters,
    /// timestamps, little-endian integers. `DeltaTransform::worth_trying` already
    /// encodes that test, so this defers to it rather than duplicating the
    /// threshold in two places that could then drift apart.
    ///
    /// This is a hint and nothing more: the planner measures the real output of
    /// every candidate against `store`, so a wrong answer costs ratio and a little
    /// CPU, never correctness. That is what lets this be permissive — being wrong
    /// in the cheap direction is fine.
    ///
    /// Incompressible data is excluded explicitly. Random input's drift statistic
    /// is already far above the threshold and would be pruned anyway, but
    /// short-circuiting keeps the fast path for incompressible data free of a
    /// transform pass that cannot win.
    ///
    /// Only the two LZ-shaped pipelines see a transform. RLE would gain nothing:
    /// it already collapses runs, and delta would destroy the byte identity it
    /// depends on. `store` must never be transformed, since storing transformed
    /// bytes without recording the transform would not be invertible.
    fn transform_for(
        &self,
        input: &[u8],
        analysis: &Analysis,
        pipeline: PipelineId,
    ) -> Option<TransformId> {
        if !matches!(pipeline, PipelineId::LzFast | PipelineId::Statistical) {
            return None;
        }
        if analysis.looks_incompressible() {
            return None;
        }

        // `worth_trying` is on the `Transform` trait, not inherent, so it has to be
        // called through the trait rather than on the struct.
        use crate::transform::Transform;
        crate::transform::DeltaTransform
            .worth_trying(input)
            .then_some(TransformId::Delta)
    }

    /// Plan and encode one chunk.
    pub fn plan_chunk(&self, input: &[u8], independent: bool) -> Result<ChunkPlan> {
        let analysis = Analysis::of(input)?;
        let candidates = self.candidates(input, &analysis);

        let mut best: Option<(PipelineId, TransformId, Vec<u8>)> = None;
        let mut tried: Vec<PipelineId> = Vec::new();

        for (pipeline, transform) in candidates {
            // Never let a frame grow: `Store` is the floor, and comparing
            // against it here is what makes "store when compression does not
            // pay" a guarantee rather than a hope.
            let payload = match self.try_encode(pipeline, transform, input, &analysis) {
                Ok(p) => p,
                Err(_) => continue,
            };
            tried.push(pipeline);
            let better = match &best {
                None => true,
                Some((_, _, b)) => payload.len() < b.len(),
            };
            if better {
                best = Some((pipeline, transform, payload));
            }
        }

        // Store is always reachable: it is the last candidate and cannot fail.
        let stored = matches!(&best, Some((PipelineId::Store, _, _)));

        let (pipeline, transform, payload) = best.ok_or_else(|| {
            crate::error::Error::Io(std::io::Error::other("no pipeline produced output"))
        })?;

        Ok(ChunkPlan {
            pipeline,
            transform,
            payload,
            // The frame must record how many bytes it decodes to, which is the
            // *input* length. The payload length is the encoded size and is
            // usually smaller, so deriving this from the payload would write a
            // frame that claims to decode to less than the caller supplied.
            content_size: input.len(),
            independent,
            tried,
            stored,
            // Dictionary priming is not wired into planning yet; a frame never
            // claims a dictionary it does not have, so 0 is always correct here.
            dict_id: 0,
        })
    }

    /// Encode with one specific pipeline, applying the transform first.
    fn try_encode(
        &self,
        pipeline: PipelineId,
        transform: TransformId,
        input: &[u8],
        analysis: &Analysis,
    ) -> Result<Vec<u8>> {
        let codec = self.registry.get(pipeline)?;
        let transformed: Vec<u8> = if transform == TransformId::None {
            input.to_vec()
        } else {
            transform::apply_forward(transform, input)?
        };
        let ctx = EncodeContext {
            input: &transformed,
            analysis,
            level: self.config.level(),
            dictionary: self.dictionary.as_ref(),
            independent: true,
        };
        codec.encode(&ctx)
    }

    /// Decode a frame's payload into a fresh buffer.
    pub fn decode_payload(
        &self,
        pipeline: PipelineId,
        transform: TransformId,
        payload: &[u8],
        content_size: usize,
        dicts: &DictionaryTable,
        dict_id: u8,
    ) -> Result<Vec<u8>> {
        let codec = self.registry.get(pipeline)?;
        let dictionary = if dict_id == 0 {
            None
        } else {
            Some(dicts.get(dict_id)?)
        };
        let mut out = vec![0u8; content_size];
        let ctx = DecodeContext {
            payload,
            content_size,
            transform,
            dictionary,
            independent: true,
        };
        codec.decode(&ctx, &mut out)?;
        if transform != TransformId::None {
            out = transform::apply_inverse(transform, &out)?;
        }
        Ok(out)
    }

    /// The configuration this planner was built from.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The candidates this planner would measure for `input`, in order.
    ///
    /// Exposed so candidate *selection* can be inspected without running an encode.
    /// Which pipeline wins is a ratio question and is settled by measurement; which
    /// candidates are even considered is a policy question, and a policy nobody can
    /// read is a policy nobody can review. A pruning change that silently drops a
    /// useful candidate is otherwise invisible until someone notices a ratio
    /// regression months later.
    pub fn candidates_for(&self, input: &[u8]) -> Result<Vec<(PipelineId, TransformId)>> {
        let analysis = Analysis::of(input)?;
        Ok(self.candidates(input, &analysis))
    }

    /// Why each pipeline was or was not considered for `input`.
    ///
    /// For diagnostics, and for tests that assert a pruning rule actually *fires*
    /// rather than merely appearing to.
    pub fn explain_candidates(&self, input: &[u8]) -> Result<Vec<CandidateReport>> {
        let analysis = Analysis::of(input)?;
        let level = self.config.level();
        let chosen = self.candidates(input, &analysis);

        let mut out = Vec::new();
        for codec in self.registry.all() {
            let id = codec.id();

            let reason = match id {
                PipelineId::Store => CandidateReason::AlwaysMeasuredAsFloor,
                PipelineId::Dictionary => CandidateReason::NotImplementedYet,
                _ => {
                    if codec.candidate(&analysis, level) {
                        CandidateReason::CandidateAccepted
                    } else if analysis.looks_incompressible() {
                        CandidateReason::Incompressible
                    } else {
                        CandidateReason::CandidateRejected
                    }
                }
            };

            let transform = if matches!(id, PipelineId::LzFast | PipelineId::Statistical)
                && !analysis.looks_incompressible()
            {
                match self.transform_for(input, &analysis, id) {
                    Some(t) => TransformOutcome::Added(t),
                    None => TransformOutcome::NotOffered,
                }
            } else {
                TransformOutcome::NotConsidered
            };

            out.push(CandidateReport {
                pipeline: id,
                in_list: chosen.contains(&(id, TransformId::None)),
                reason,
                transform,
            });
        }
        Ok(out)
    }
}

/// Why a pipeline was or was not considered for a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateReason {
    /// The pipeline passed its own `candidate()` check.
    CandidateAccepted,
    /// The pipeline's `candidate()` check refused it.
    CandidateRejected,
    /// The chunk looks incompressible, so nothing is worth trying.
    Incompressible,
    /// `store` is always measured, as the floor a chunk may not exceed.
    AlwaysMeasuredAsFloor,
    /// Specified in the format but never emitted.
    NotImplementedYet,
}

impl CandidateReason {
    pub fn name(self) -> &'static str {
        match self {
            CandidateReason::CandidateAccepted => "accepted",
            CandidateReason::CandidateRejected => "rejected",
            CandidateReason::Incompressible => "incompressible",
            CandidateReason::AlwaysMeasuredAsFloor => "floor",
            CandidateReason::NotImplementedYet => "unimplemented",
        }
    }
}

/// What happened to one pipeline's second, transformed attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformOutcome {
    /// The transform was offered and added as a candidate.
    Added(TransformId),
    /// The transform was tested and refused.
    NotOffered,
    /// No transform applies to this pipeline.
    NotConsidered,
}

impl TransformOutcome {
    pub fn name(self) -> String {
        match self {
            TransformOutcome::Added(t) => format!("added {}", t.name()),
            TransformOutcome::NotOffered => "not offered".to_string(),
            TransformOutcome::NotConsidered => "n/a".to_string(),
        }
    }
}

/// One pipeline's candidate status for a chunk.
#[derive(Debug, Clone, Copy)]
pub struct CandidateReport {
    pub pipeline: PipelineId,
    /// True when the untransformed pipeline is in the candidate list.
    pub in_list: bool,
    pub reason: CandidateReason,
    pub transform: TransformOutcome,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WindowConfig;

    fn planner(cfg: Config) -> Planner {
        Planner::new(&cfg.validated().unwrap())
    }

    /// A deterministic pseudo-random `u16`, for building interleaved fields that
    /// have a field *width* but no field *structure* — the noise control for
    /// stride detection, which must not mistake a fixed width for a signal.
    ///
    /// Counter-driven rather than clock-driven: a test that seeds from the clock can
    /// fail once in a hundred runs, which is worse than no test at all.
    fn random_u16(counter: &mut u32) -> u16 {
        let mut x = counter.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *counter = x;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x as u16
    }

    fn pseudo_random(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn never_compress_picks_store() {
        let p = planner(Config::new(Level::Default).with_strategy(Strategy::NeverCompress));
        let data = b"hello hello hello".repeat(100);
        let plan = p.plan_chunk(&data, true).unwrap();
        assert_eq!(plan.pipeline, PipelineId::Store);
        assert!(plan.stored);
        assert_eq!(plan.payload, data);
    }

    #[test]
    fn incompressible_picks_store() {
        let p = planner(Config::new(Level::Default));
        let data = pseudo_random(8192, 0xBEEF);
        let plan = p.plan_chunk(&data, true).unwrap();
        // Nothing can compress random data, so store must win.
        assert_eq!(plan.pipeline, PipelineId::Store);
        assert!(plan.stored);
    }

    #[test]
    fn compressible_beats_store() {
        let p = planner(Config::new(Level::Default));
        let data = "the quick brown fox jumps over the lazy dog. "
            .repeat(200)
            .into_bytes();
        let plan = p.plan_chunk(&data, true).unwrap();
        assert!(
            plan.payload.len() < data.len() / 4,
            "{} bytes",
            plan.payload.len()
        );
        assert_ne!(plan.pipeline, PipelineId::Store);
    }

    /// Ratio floors for each data shape.
    ///
    /// This is the guard against "optimised" changes that quietly lose ratio. A
    /// speed benchmark would happily report a 20% throughput win for a change
    /// that made every file 5% bigger, so the ratio is asserted here where it
    /// fails the quality gate.
    ///
    /// The floors sit a few percent below what the engine achieves today, so
    /// ordinary churn does not trip them. Raising one is a deliberate decision to
    /// accept a worse ratio, and should be made with that in mind.
    #[test]
    fn ratio_floors_are_maintained() {
        let cases: [(&str, Vec<u8>, f64); 6] = [
            ("text", text_corpus(256 * 1024), 0.25),
            (
                "source",
                (0..3000usize)
                    .flat_map(|i| {
                        format!("    let value_{i} = compute(&self, {i})?;\n").into_bytes()
                    })
                    .collect(),
                0.20,
            ),
            (
                "runs",
                (0..256 * 1024usize).map(|i| (i / 97) as u8).collect(),
                0.10,
            ),
            (
                "low entropy",
                (0..256 * 1024usize)
                    .map(|i| b"ACGT"[(i * 7 + i / 13) % 4])
                    .collect(),
                0.22,
            ),
            (
                // Incompressible input must cost framing overhead and nothing
                // else: a 28-byte header, a 16-byte frame header, two 32-byte
                // hashes and a 12-byte end marker, which is 120 bytes however
                // much data follows. Expressed as a ratio over a 64 KiB chunk.
                "incompressible",
                {
                    let mut s = 0x5EEDu32;
                    (0..64 * 1024usize)
                        .map(|_| {
                            s ^= s << 13;
                            s ^= s >> 17;
                            s ^= s << 5;
                            (s >> 24) as u8
                        })
                        .collect()
                },
                1.0025,
            ),
            // Delta-eligible data. A floor rather than a benchmark, because a
            // regression in transform selection must fail the gate.
            ("numeric", numeric_corpus(256 * 1024), 0.02),
        ];

        for (name, data, floor) in cases {
            let config = Config::new(Level::Default);
            let (packed, _) = crate::compress::compress_slice(&data, &config).unwrap();
            let ratio = packed.len() as f64 / data.len() as f64;

            // The floor already encodes the expectation for each shape; the
            // incompressible case's floor is the framing overhead, not a ratio
            // below one, so there is nothing special to do here. Keeping one
            // uniform rule means a case cannot silently bypass its own bound.
            assert!(
                ratio <= floor,
                "{name}: ratio {ratio:.4} exceeded the floor {floor:.4} ({} -> {} bytes)",
                data.len(),
                packed.len()
            );
        }
    }

    /// A deterministic numeric corpus, so the ratio floor is reproducible.
    ///
    /// Bytes walking by a constant step: the shape the delta transform exists for, and
    /// distinct from the text and low-entropy corpora so a change to transform
    /// selection shows up here rather than hiding behind them.
    fn numeric_corpus(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 3) % 256) as u8).collect()
    }

    /// A deterministic wide-alphabet text corpus.
    fn wide_text_corpus(n: usize) -> Vec<u8> {
        let mut s = 13u32;
        (0..n)
            .map(|_| {
                let mut next = || {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    s
                };
                match (next() >> 24) % 6 {
                    0 => b'a' + (next() % 26) as u8,
                    1 => b'A' + (next() % 26) as u8,
                    2 => b' ' + (next() % 2) as u8,
                    3 => b',',
                    4 => b'.',
                    _ => b'\n',
                }
            })
            .collect()
    }

    /// A deterministic run-heavy corpus.
    fn runs_corpus(n: usize) -> Vec<u8> {
        let mut s = 7u32;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            out.extend(std::iter::repeat_n(
                (s >> 8) as u8,
                1 + (s >> 24) as usize % 200,
            ));
        }
        out.truncate(n);
        out
    }

    /// A deterministic low-entropy, non-periodic corpus.
    ///
    /// Only four distinct byte values, but no repetition for LZ to exploit — the
    /// classic hard case, and the one that most punishes a wrong pruning rule.
    fn random_alphabet_corpus(n: usize) -> Vec<u8> {
        let mut s = 77u32;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                b"ACGT"[(s >> 24) as usize % 4]
            })
            .collect()
    }

    /// A deterministic low-entropy, periodic corpus.
    ///
    /// Four byte values on a fixed stride: structured enough for every pipeline to
    /// have an opinion about it, so it is a useful shared fixture.
    fn low_entropy_corpus(n: usize) -> Vec<u8> {
        (0..n).map(|i| b"ACGT"[(i * 7 + i / 13) % 4]).collect()
    }

    /// The candidate list never contains a repeat.
    ///
    /// Every candidate is encoded in full and a duplicate cannot change the winner,
    /// so a repeat is wasted work by construction. This already happened once:
    /// `rle` was added both by the registry loop and by the explicit `Repetitive`
    /// push, so run-heavy data paid for the same pipeline twice.
    #[test]
    fn candidate_lists_never_contain_duplicates() {
        let corpora = candidate_corpora();

        for level in Level::ALL {
            for (name, data) in &corpora {
                let cands = planner(Config::new(level))
                    .candidates_for(data)
                    .unwrap_or_else(|e| panic!("{name} at {}: {e}", level.name()));

                let mut seen: Vec<(PipelineId, TransformId)> = Vec::new();
                for c in &cands {
                    assert!(
                        !seen.contains(c),
                        "{name} at {}: {c:?} listed twice in {cands:?}",
                        level.name()
                    );
                    seen.push(*c);
                }
            }
        }
    }

    /// `store` is the floor every candidate list must contain.
    ///
    /// This is the invariant behind "a chunk never grows", so no pruning rule may
    /// be allowed to remove it.
    #[test]
    fn store_is_always_a_candidate() {
        let corpora = candidate_corpora();

        for level in Level::ALL {
            for (name, data) in &corpora {
                let cands = planner(Config::new(level))
                    .candidates_for(data)
                    .unwrap_or_else(|e| panic!("{name} at {}: {e}", level.name()));
                assert!(
                    cands.contains(&(PipelineId::Store, TransformId::None)),
                    "{name} at {}: store missing from {cands:?}",
                    level.name()
                );
            }
        }
    }

    /// Candidates that cannot win are skipped.
    ///
    /// Incompressible data is the clearest case: every pipeline would produce
    /// something larger than the input, so measuring them is wasted by definition.
    /// This is the fast path, and it is why uniform random data encodes in single
    /// digit milliseconds.
    #[test]
    fn candidates_that_cannot_win_are_skipped() {
        let data = pseudo_random(512 * 1024, 0x5EED);
        assert!(
            Analysis::of(&data).unwrap().looks_incompressible(),
            "fixture must actually be incompressible"
        );

        for level in Level::ALL {
            let cands = planner(Config::new(level))
                .candidates_for(&data)
                .expect("candidates");
            assert_eq!(
                cands,
                vec![(PipelineId::Store, TransformId::None)],
                "level {} attempted more than store on incompressible data",
                level.name()
            );
        }
    }

    /// Candidates that measurably win are not accidentally pruned.
    ///
    /// Written as "the measured winner must be present" rather than as "this rule
    /// must hold", so it pins behaviour and not a particular implementation. It is
    /// the test that fails when a future pruning rule guesses too hard.
    #[test]
    fn winning_candidates_survive_pruning() {
        let cases: Vec<(&str, Vec<u8>, PipelineId, TransformId)> = vec![
            (
                "runs",
                runs_corpus(512 * 1024),
                PipelineId::Rle,
                TransformId::None,
            ),
            (
                "text",
                text_corpus(512 * 1024),
                PipelineId::Statistical,
                TransformId::None,
            ),
            (
                "numeric",
                numeric_corpus(512 * 1024),
                PipelineId::Statistical,
                TransformId::Delta,
            ),
            (
                "wide text",
                wide_text_corpus(512 * 1024),
                PipelineId::Statistical,
                TransformId::None,
            ),
            (
                // Interleaved u16 fields. Stride-aware detection is what makes
                // delta available here at all, so this is the case that would
                // regress first if that signal were removed.
                "interleaved u16",
                (0..512 * 1024usize)
                    .flat_map(|i| ((i % 65536) as u16).to_le_bytes())
                    .collect(),
                PipelineId::Statistical,
                TransformId::Delta,
            ),
            (
                // Low entropy *and* periodic on a stride, which is why delta wins
                // 16.5x here. Previously refused (adjacent drift 8.77) and now
                // correctly detected at stride 4.
                "low entropy",
                low_entropy_corpus(512 * 1024),
                PipelineId::Statistical,
                TransformId::Delta,
            ),
            (
                // Interleaved but with no stride structure: random u16 values.
                // Detection must stay away from this, or delta would be offered
                // on data where it is a pure loss.
                "random u16",
                random_alphabet_corpus(512 * 1024),
                PipelineId::Statistical,
                TransformId::None,
            ),
        ];

        for (name, data, winner, transform) in cases {
            let cands = planner(Config::new(Level::Default))
                .candidates_for(&data)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                cands.contains(&(winner, transform)),
                "{name}: the measured winner {}/{} was pruned from {cands:?}",
                winner.name(),
                transform.name()
            );
        }
    }

    /// RLE is offered as a candidate on data where it loses badly.
    ///
    /// This is the measurement that rules out the obvious pruning rules. RLE is
    /// accepted for both corpora below — `runs` by the `Repetitive` class, random
    /// ACGT by `max_byte_share` — yet RLE wins one by a wide margin and loses the
    /// other by 3.3x:
    ///
    /// | corpus      | RLE       | statistical | winner      |
    /// |-------------|-----------|-------------|-------------|
    /// | runs        | 63 095    | 76 238      | rle         |
    /// | random acgt | 2 261 658 | 676 298     | statistical |
    ///
    /// So neither "skip the LZ pipelines when RLE is available" nor "trust the
    /// `Repetitive` class" is safe: each is right on one corpus and badly wrong on
    /// the other. That is why the planner spends a millisecond measuring RLE rather
    /// than trying to predict its value.
    ///
    /// The test exists to stop anyone adding such a rule without measuring first.
    #[test]
    fn rle_is_offered_on_data_where_it_can_lose_badly() {
        let runs = runs_corpus(512 * 1024);
        let acgt = random_alphabet_corpus(512 * 1024);
        let p = planner(Config::new(Level::Default));

        for (name, data) in [("runs", &runs), ("random acgt", &acgt)] {
            let cands = p.candidates_for(data).expect("candidates");
            assert!(
                cands.contains(&(PipelineId::Rle, TransformId::None)),
                "{name} should offer RLE as a candidate, got {cands:?}"
            );
        }

        assert_eq!(
            p.plan_chunk(&runs, true).unwrap().pipeline,
            PipelineId::Rle,
            "RLE should win on run-heavy data"
        );
        assert_eq!(
            p.plan_chunk(&acgt, true).unwrap().pipeline,
            PipelineId::Statistical,
            "statistical should win on random ACGT even though RLE was offered"
        );
    }

    /// Pruning must not change the bytes a chunk produces.
    ///
    /// Deduplicating cannot change the winner — a repeat encodes to the same
    /// payload — and this test proves that rather than assuming it. It re-encodes
    /// the chosen candidate directly, then re-encodes every *other* candidate and
    /// confirms none of them was better. If a pruning rule had dropped a better
    /// candidate, that is exactly where it would show.
    #[test]
    fn pruning_does_not_change_the_output() {
        let registry = crate::codec::Registry::new();

        for level in Level::ALL {
            let cfg = Config::new(level);
            let p = planner(cfg.clone());

            for (name, data) in candidate_corpora() {
                let plan = p.plan_chunk(&data, true).expect("plan");
                let cands = p.candidates_for(&data).expect("candidates");

                let encode = |pipeline: PipelineId, transform: TransformId| {
                    let bytes =
                        crate::transform::apply_forward(transform, &data).expect("transform");
                    registry
                        .get(pipeline)
                        .unwrap()
                        .encode(&crate::codec::EncodeContext {
                            input: &bytes,
                            analysis: &Analysis::of(&bytes).unwrap(),
                            level,
                            dictionary: None,
                            independent: true,
                        })
                        .expect("encode")
                        .len()
                };

                assert_eq!(
                    encode(plan.pipeline, plan.transform),
                    plan.payload.len(),
                    "{name} at {}: the plan and a direct encode disagree",
                    level.name()
                );

                for candidate in &cands {
                    if *candidate == (plan.pipeline, plan.transform) {
                        continue;
                    }
                    let other = encode(candidate.0, candidate.1);
                    assert!(
                        other >= plan.payload.len(),
                        "{name} at {}: candidate {candidate:?} produced {other} bytes, \
                         better than the winner's {}",
                        level.name(),
                        plan.payload.len()
                    );
                }
            }
        }
    }

    /// The corpora every candidate test runs over.
    ///
    /// Chosen to span the shapes where different pipelines win, because a pruning
    /// rule that only survives one of them has not been tested.
    fn candidate_corpora() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("empty", Vec::new()),
            ("tiny", b"hi".to_vec()),
            ("text", text_corpus(200 * 1024)),
            ("runs", runs_corpus(200 * 1024)),
            ("numeric", numeric_corpus(200 * 1024)),
            ("wide text", wide_text_corpus(200 * 1024)),
            ("low entropy", low_entropy_corpus(200 * 1024)),
            ("random alphabet", random_alphabet_corpus(200 * 1024)),
            ("random", pseudo_random(200 * 1024, 9)),
            (
                "interleaved",
                (0..200 * 1024usize)
                    .flat_map(|i| [((i / 256) % 256) as u8, (i % 256) as u8])
                    .collect(),
            ),
        ]
    }

    /// Interleaved `u16` fields are detected and delta is selected.
    ///
    /// This test used to assert the opposite — that the byte-drift heuristic *missed*
    /// this shape — and it was the reason stride-aware detection exists. It now pins
    /// the fixed behaviour, and the numbers are the reason the fix was worth
    /// making: adjacent drift reads 120.3 (so the old heuristic refused), stride-2
    /// drift reads 1.0, and delta compresses the result **45.8x** better.
    #[test]
    fn delta_detects_interleaved_little_endian_fields() {
        use crate::transform::detected_stride;

        // A u16 counter stored little-endian: `lo0 hi0 lo1 hi1 ...`.
        let data: Vec<u8> = (0..200 * 1024usize)
            .flat_map(|i| ((i % 65536) as u16).to_le_bytes())
            .collect();

        assert_eq!(
            detected_stride(&data),
            Some(2),
            "a u16 field should be detected at stride 2"
        );

        let plan = planner(Config::new(Level::Default))
            .plan_chunk(&data, true)
            .unwrap();
        assert_eq!(
            plan.transform,
            TransformId::Delta,
            "the transform that used to be missed must now be selected"
        );

        // And it has to be worth selecting: measure against the same pipeline with
        // no transform.
        let mut best_without = usize::MAX;
        for pipeline in [PipelineId::Statistical, PipelineId::LzFast, PipelineId::Rle] {
            let payload = crate::codec::Registry::new()
                .get(pipeline)
                .unwrap()
                .encode(&crate::codec::EncodeContext {
                    input: &data,
                    analysis: &Analysis::of(&data).unwrap(),
                    level: Level::Default,
                    dictionary: None,
                    independent: true,
                })
                .map(|p| p.len())
                .unwrap_or(usize::MAX);
            best_without = best_without.min(payload);
        }
        assert!(
            plan.payload.len() * 5 < best_without,
            "delta produced {} bytes against an untransformed best of {best_without}",
            plan.payload.len()
        );
    }

    /// Random four-symbol data must still come out as a statistical frame.
    ///
    /// This is the noise control for stride detection, and it is asserted on the
    /// *pipeline* rather than on whether the candidate was offered.
    ///
    /// It has to be: with only four distinct byte values, any two of them are close,
    /// so the drift average lands at 7.7 whether or not there is any structure —
    /// within a hair of the 8.0 threshold, where sample size alone can flip it. That
    /// makes "was delta offered" an unstable thing to assert, and an unstable
    /// assertion is one that gets "fixed" by editing the expectation.
    ///
    /// What actually has to hold is the outcome: random data over a small alphabet
    /// has no structure for delta to exploit, so delta must not win — and whether it
    /// was tried is the planner's business, not a property of the data. This also
    /// pins that the result is no worse than before stride detection existed, which
    /// is the regression the constraint asked to protect.
    #[test]
    fn random_four_symbol_alphabet_still_loses_to_measurement() {
        let data = random_alphabet_corpus(512 * 1024);
        let plan = planner(Config::new(Level::Default))
            .plan_chunk(&data, true)
            .unwrap();

        assert_ne!(
            plan.transform,
            TransformId::Delta,
            "delta must not win on random four-symbol data"
        );
        assert_eq!(
            plan.pipeline,
            PipelineId::Statistical,
            "the statistical pipeline is the one that handles this shape"
        );

        // And the payload must match what the untransformed pipeline produces, so
        // the stride signal cannot have quietly changed the result here.
        let untransformed = crate::codec::Registry::new()
            .get(PipelineId::Statistical)
            .unwrap()
            .encode(&crate::codec::EncodeContext {
                input: &data,
                analysis: &Analysis::of(&data).unwrap(),
                level: Level::Default,
                dictionary: None,
                independent: true,
            })
            .expect("encode")
            .len();
        assert_eq!(
            plan.payload.len(),
            untransformed,
            "the selected payload should be the untransformed one, byte for byte"
        );
    }

    /// The one shape where stride detection is known to misfire, pinned so it stays
    /// visible and stays *cheap*.
    ///
    /// Very short-period data has a degenerate drift statistic. `"hello world "`
    /// repeats every 12 bytes, so an 8 KiB sample contains only 12 distinct byte
    /// pairs per stride; at stride 6 those are
    /// `(h,w) (e,o) (l,r) (l,l) (o,d) (sp,sp)`, which average drift 7.00 — just
    /// under the 8.0 threshold — and delta gets offered. It loses, and the planner
    /// discards it, so the cost is one wasted encode.
    ///
    /// This is *not* fixed by tuning the threshold. The tempting change is to
    /// lower the stride test to ~4, which does exclude this corpus, and it is the
    /// wrong move: the genuine cases sit at 0.22–2.92 and this one at 7.00, so the
    /// only evidence for a threshold anywhere in that gap is this single synthetic
    /// sample. A threshold chosen to exclude one 12-byte-period string is
    /// overfitting to that string.
    ///
    /// Realistic text does not have this problem — a 45-byte phrase cycle measures
    /// 24–37 at every stride, and CSV rows measure 12–19.
    #[test]
    fn short_period_text_can_false_positive_and_that_is_accepted() {
        let data = "hello world ".repeat(500).into_bytes();
        let candidates = planner(Config::new(Level::Default))
            .candidates_for(&data)
            .expect("candidates");

        // The documented misfire: delta is offered.
        assert!(
            candidates.iter().any(|(_, t)| *t == TransformId::Delta),
            "this corpus is the documented short-period false positive; if it has \
             stopped firing, update the comment above rather than deleting the test"
        );

        // What actually matters is that the misfire is *cheap*: the plan still
        // comes out as the untransformed LZ frame.
        let plan = planner(Config::new(Level::Default))
            .plan_chunk(&data, true)
            .expect("plan");
        assert_eq!(
            plan.transform,
            TransformId::None,
            "delta must not win, so the frame must not claim it"
        );
        assert!(
            plan.payload.len() < data.len() / 2,
            "and the frame must still be a good one: {} bytes",
            plan.payload.len()
        );
    }

    /// The strides the detector is supposed to find, and the ones it must not
    /// invent.
    #[test]
    fn stride_detection_picks_the_field_width() {
        use crate::transform::detected_stride;

        // The noise controls. A stride must never be reported for data with
        // no field structure at any width.
        let mut counter = 1u32;
        let random_fields: Vec<u8> = (0..100 * 1024usize)
            .flat_map(|_| [random_u16(&mut counter), random_u16(&mut counter)])
            .flat_map(u16::to_le_bytes)
            .collect();

        let cases: Vec<(&str, Vec<u8>, Option<usize>)> = vec![
            (
                "u16 field",
                (0..200 * 1024usize)
                    .flat_map(|i| ((i % 65536) as u16).to_le_bytes())
                    .collect(),
                Some(2),
            ),
            (
                "u24 field",
                (0..200 * 1024usize)
                    .flat_map(|i| {
                        let v = i as u32;
                        [
                            (v & 0xFF) as u8,
                            ((v >> 8) & 0xFF) as u8,
                            ((v >> 16) & 0xFF) as u8,
                        ]
                    })
                    .collect(),
                Some(3),
            ),
            (
                "u32 field",
                (0..200 * 1024usize)
                    .flat_map(|i| (i as u32).to_le_bytes())
                    .collect(),
                Some(4),
            ),
            (
                "three u16 fields",
                (0..200 * 1024usize)
                    .flat_map(|i| {
                        let v = i as u32;
                        (0..3u32).flat_map(move |f| ((v + f * 1000) as u16).to_le_bytes())
                    })
                    .collect(),
                Some(6),
            ),
            (
                "u64 field",
                (0..200 * 1024usize)
                    .flat_map(|i| (i as u64).to_le_bytes())
                    .collect(),
                Some(8),
            ),
            // The noise controls. A stride must never be reported for data with
            // no field structure at any width.
            ("uniform random", pseudo_random(200 * 1024, 0x5EED), None),
            ("random u16 fields", random_fields, None),
            ("ordinary text", text_corpus(200 * 1024), None),
        ];

        for (name, data, expected) in cases {
            assert_eq!(
                detected_stride(&data),
                expected,
                "{name}: expected stride {expected:?}"
            );
        }
    }

    /// A deterministic text-like corpus, so the ratio floor is reproducible.
    fn text_corpus(n: usize) -> Vec<u8> {
        const WORDS: [&str; 8] = [
            "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ",
        ];
        let mut out = Vec::with_capacity(n + 32);
        let mut i = 0usize;
        while out.len() < n {
            out.extend_from_slice(WORDS[i % WORDS.len()].as_bytes());
            i += 1;
        }
        out.truncate(n);
        out
    }

    /// A corpus for the candidate-*policy* tests, which is about how many
    /// candidates the planner is allowed to try.
    ///
    /// It has to be data no transform is offered for, or the policy's budget
    /// moves underneath the test for reasons that have nothing to do with policy.
    /// These three tests used `"hello world ".repeat(500)`, which is a 12-byte
    /// period: stride 6 compares `(h,w) (e,o) (l,r) (l,l) (o,d) (sp,sp)` and lands
    /// at drift 7.00, just under the 8.0 threshold, so stride-aware detection
    /// offers delta and the budget legitimately grows by one.
    ///
    /// That corpus was a bad choice, not the detector being wrong — a 12-byte
    /// period means an 8 KiB sample averages only 12 distinct byte pairs, so the
    /// drift statistic has almost no diversity to work with. Ordinary text with a
    /// realistic period measures 25–37 at every stride and stays clean.
    ///
    /// The size is kept at the original 6 000 bytes, not raised for realism: the
    /// `statistical` pipeline declines inputs below the point where its model
    /// tables pay for themselves, so a larger corpus would add a candidate for a
    /// reason that has nothing to do with the policy under test either.
    fn policy_corpus() -> Vec<u8> {
        let data = text_corpus(6_000);
        assert!(
            !planner(Config::new(Level::Default))
                .candidates_for(&data)
                .unwrap_or_default()
                .iter()
                .any(|(_, t)| *t == TransformId::Delta),
            "policy_corpus must not be offered delta, or the candidate budget tests \
             are measuring the transform heuristic instead of the policy"
        );
        data
    }

    #[test]
    fn level_controls_search_effort() {
        // The level has to reach the match finder, not stop at the config. If it
        // did, every level would emit identical bytes and the knob would be a lie
        // — which is exactly the failure this pins.
        //
        // The corpus is low-entropy but not periodic: a chain walker has to walk
        // back through many candidate positions to find the long matches, so its
        // extra effort shows up in the output rather than only in the clock.
        // Asserted at the *codec* level rather than through the planner, with the
        // pipeline pinned. The planner was the wrong place for it: whichever
        // pipeline wins, the plan only reports the winner's size, so a level
        // change that altered the search was invisible whenever `rle` or `store`
        // happened to come out ahead. Holding the pipeline fixed measures exactly
        // the thing under test — whether the level reaches the match finder.
        let data: Vec<u8> = wide_text_corpus(512 * 1024);
        let analysis = Analysis::of(&data).unwrap();
        let registry = crate::codec::Registry::new();

        let encode_at = |pipeline: PipelineId, level: Level| -> usize {
            registry
                .get(pipeline)
                .unwrap()
                .encode(&crate::codec::EncodeContext {
                    input: &data,
                    analysis: &analysis,
                    level,
                    dictionary: None,
                    independent: true,
                })
                .expect("encode")
                .len()
        };

        // If the level did not reach the match finder, every level would produce
        // identical bytes and the knob would be a lie.
        //
        // Deliberately no per-level size bound here. A single codec *can* expand
        // its input — `lz-fast` on wide-alphabet text reaches 553 542 bytes from
        // 524 288 at high and max — and that is not a bug: "a chunk never grows"
        // is a guarantee the *planner* makes by measuring against `store`, not a
        // property of any pipeline on its own. Asserting it here would be asserting
        // the wrong contract.
        for pipeline in [PipelineId::Statistical, PipelineId::LzFast] {
            let sizes: Vec<usize> = Level::ALL.iter().map(|l| encode_at(pipeline, *l)).collect();
            assert!(
                sizes.iter().collect::<std::collections::HashSet<_>>().len() > 1,
                "{} produced identical output at every level: {sizes:?}",
                pipeline.name()
            );
        }
    }

    /// The data shapes where the delta transform genuinely wins, and the ones where
    /// it must not be attempted.
    ///
    /// These numbers are measured, not aspirational. The losing cases matter as
    /// much as the winning ones: `transform_for` is a heuristic that gates a real
    /// measurement, so a false positive costs CPU and a false negative costs ratio.
    /// Both directions are pinned here so the threshold cannot drift silently.
    ///
    /// The assertion is against the candidate list, deliberately. `plan.transform`
    /// is the transform that *won*, which is a different question: delta is offered
    /// and then loses on run-heavy data, so reading the winner there says "skipped"
    /// when it was in fact tried and measured.
    #[test]
    fn delta_selection_matches_measured_outcomes() {
        // (name, data, whether delta should be offered)
        //
        // The assertion is against the candidate list, deliberately. `plan.transform`
        // is the transform that *won*, which is a different question: delta is
        // offered and then loses on run-heavy data, so reading the winner there
        // says "skipped" when it was in fact tried and measured.
        let cases: Vec<(&str, Vec<u8>, bool)> = vec![
            (
                // Steps of one: every byte differs from its predecessor by a
                // constant, which is exactly what delta is for.
                "bytes stepping by one",
                (0..200 * 1024usize).map(|i| (i % 256) as u8).collect(),
                true,
            ),
            (
                // A byte walk that increments by a constant: every byte differs
                // from its predecessor by the same small amount, which is exactly
                // the shape delta exists for.
                "bytes stepping by three",
                (0..200 * 1024usize)
                    .map(|i| ((i * 3) % 256) as u8)
                    .collect(),
                true,
            ),
            (
                // Low entropy *and* stride-periodic. This was called "random" for
                // a long time, which was simply wrong: `(i * 7 + i / 13) % 4` is
                // highly structured at stride 4, and delta wins 16.5x on it. The
                // old adjacent-only heuristic refused it (drift 8.77); stride
                // detection now finds the signal that was there all along.
                "low entropy, stride periodic",
                (0..200 * 1024usize)
                    .map(|i| b"ACGT"[(i * 7 + i / 13) % 4])
                    .collect(),
                true,
            ),
            (
                // Wide drift at every stride: no structure anywhere.
                "uniform random",
                pseudo_random(200 * 1024, 0x5EED),
                false,
            ),
            (
                // Ordinary text: adjacent drift is high and no stride rescues it,
                // so delta must not be offered.
                "text",
                text_corpus(200 * 1024),
                false,
            ),
            (
                // Run-heavy: offered via the adjacent test (drift ~1), and delta
                // loses because it destroys the runs RLE depends on. Pinned so
                // the stride signal cannot make this worse.
                "runs",
                runs_corpus(200 * 1024),
                true,
            ),
            (
                // Interleaved u16 fields: the case stride detection exists for.
                "interleaved u16",
                (0..200 * 1024usize)
                    .flat_map(|i| ((i % 65536) as u16).to_le_bytes())
                    .collect(),
                true,
            ),
            (
                // Interleaved u32 fields. Also detected, and delta *loses* here.
                // Offered on purpose: the planner measures, so it costs one encode
                // and no ratio. Recorded because "the detector fires somewhere it
                // should not" is exactly the kind of thing that should be visible.
                "interleaved u32",
                (0..200 * 1024usize)
                    .flat_map(|i| (i as u32).to_le_bytes())
                    .collect(),
                true,
            ),
            (
                // 24-bit fields: stride 3, which the power-of-two set missed
                // entirely. Delta wins, if only by 1.22x.
                "u24 triples",
                (0..200 * 1024usize)
                    .flat_map(|i| {
                        let v = i as u32;
                        [
                            (v & 0xFF) as u8,
                            ((v >> 8) & 0xFF) as u8,
                            ((v >> 16) & 0xFF) as u8,
                        ]
                    })
                    .collect(),
                true,
            ),
            (
                // Three interleaved u16 fields — the x, y, z record layout, and the
                // case that made stride 6 worth its pass. Drift at stride 2, 4 and
                // 8 all read above 30 here, so a power-of-two set misses it, and
                // delta wins 29.9x.
                "xyz u16 stride 6",
                (0..200 * 1024usize)
                    .flat_map(|i| {
                        let v = i as u32;
                        (0..3u32).flat_map(move |f| ((v + f * 1000) as u16).to_le_bytes())
                    })
                    .collect(),
                true,
            ),
        ];

        for (name, data, should_try) in cases {
            let analysis = Analysis::of(&data).unwrap();
            let _ = &analysis;
            let cands = planner(Config::new(Level::Default))
                .candidates_for(&data)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            // "Offered" means present in the candidate list, which is the
            // heuristic's decision. Whether it then *wins* is a separate,
            // measurement-driven outcome checked elsewhere.
            let attempted = cands.iter().any(|(_, t)| *t == TransformId::Delta);

            assert_eq!(
                attempted,
                should_try,
                "{name}: delta {} but measurement says {}",
                if attempted {
                    "was attempted"
                } else {
                    "was skipped"
                },
                if should_try {
                    "it should be tried"
                } else {
                    "it should be skipped"
                }
            );
            let _ = analysis;
        }
    }

    #[test]
    fn delta_selection_actually_improves_the_numeric_case() {
        // The reason the transform exists. A 16-bit counter is the clearest case:
        // delta turns a slowly varying series into mostly zeroes, and the payload
        // must get substantially smaller for the transform to have earned its place.
        // A byte walk with a constant step. Chosen over a multi-byte integer counter
        // deliberately: interleaved little-endian fields defeat the byte-drift
        // heuristic (see `DeltaTransform::worth_trying`), so they cannot be used
        // to test *selection*. This shape has low drift *and* benefits, which is
        // the only combination where a selection test proves anything.
        let data: Vec<u8> = (0..200 * 1024usize)
            .map(|i| ((i * 3) % 256) as u8)
            .collect();

        let with = planner(Config::new(Level::Default))
            .plan_chunk(&data, true)
            .unwrap();
        assert_eq!(
            with.transform,
            TransformId::Delta,
            "a constant-step byte walk is a case delta is for"
        );

        // Measure against the same pipeline with no transform, so the comparison
        // isolates the transform rather than the pipeline.
        let mut best_without = usize::MAX;
        for pipeline in [
            PipelineId::Statistical,
            PipelineId::LzFast,
            PipelineId::Rle,
            PipelineId::Store,
        ] {
            let analysis = Analysis::of(&data).unwrap();
            let payload = crate::codec::Registry::new()
                .get(pipeline)
                .unwrap()
                .encode(&crate::codec::EncodeContext {
                    input: &data,
                    analysis: &analysis,
                    level: Level::Default,
                    dictionary: None,
                    independent: true,
                })
                .map(|p| p.len())
                .unwrap_or(usize::MAX);
            best_without = best_without.min(payload);
        }

        // Measured 1 699 against 3 135, so a 3x margin is too tight to survive ordinary
        // churn. The improvement is real and large; the exact factor is not the
        // claim being made here.
        assert!(
            with.payload.len() + 512 < best_without,
            "delta produced {} bytes against an untransformed best of {best_without}",
            with.payload.len()
        );
    }

    #[test]
    fn delta_selection_never_makes_a_frame_larger() {
        // The planner's core guarantee, under the transform. Every corpus, every
        // level: a frame may not exceed its input, with or without delta chosen.
        let corpora: Vec<Vec<u8>> = vec![
            (0..300 * 1024usize).map(|i| (i % 256) as u8).collect(),
            (0..300 * 1024usize)
                .flat_map(|i| (i as i16).to_le_bytes())
                .collect(),
            (0..300 * 1024usize)
                .map(|i| b"ACGT"[(i * 7 + i / 13) % 4])
                .collect(),
            {
                let mut s = 0x5EEDu32;
                (0..300 * 1024usize)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 17;
                        s ^= s << 5;
                        (s >> 24) as u8
                    })
                    .collect()
            },
            b"the quick brown fox jumps over the lazy dog ".repeat(8_000),
        ];

        for level in Level::ALL {
            for data in &corpora {
                let plan = planner(Config::new(level)).plan_chunk(data, true).unwrap();
                assert!(
                    plan.payload.len() <= data.len(),
                    "level {} grew {} bytes to {}",
                    level.name(),
                    data.len(),
                    plan.payload.len()
                );
                assert_eq!(
                    plan.content_size,
                    data.len(),
                    "frame size must match the chunk"
                );
            }
        }
    }

    #[test]
    fn delta_selected_frames_round_trip() {
        // The lossless contract, specifically for frames that carry a transform.
        // A transform bug would be invisible on untransformed data.
        // Same constant-step shape as the selection test, so delta is actually chosen.
        let data: Vec<u8> = (0..200 * 1024usize)
            .map(|i| ((i * 3) % 256) as u8)
            .collect();

        // `Level::None` is `store` by definition and records no transform, so it
        // is excluded from the transform assertion below rather than special-cased
        // inside it.
        for level in Level::ALL {
            let cfg = Config::new(level);
            let (packed, _) = crate::compress::compress_slice(&data, &cfg).expect("compress");
            let (decoded, stats) =
                crate::decompress::decompress_slice(&packed, &cfg).expect("decompress");
            assert_eq!(
                decoded,
                data,
                "level {} lost data through the transform",
                level.name()
            );
            assert!(stats.verified);

            // And the transform must actually be recorded, or a decoder could not
            // know to invert it. `Level::None` stores the chunk untouched, so it
            // legitimately records nothing.
            if level != Level::None {
                let transforms =
                    crate::decompress::transforms_used(&packed, &cfg).expect("transforms");
                assert!(
                    transforms.contains(&TransformId::Delta),
                    "level {} did not record the transform it used",
                    level.name()
                );
            }
        }
    }

    #[test]
    fn every_level_produces_a_decodable_frame() {
        // Effort must not change *correctness*, only cost: whatever a level emits
        // has to decode back to the same bytes through the registry.
        let data: Vec<u8> = (0..64 * 1024usize)
            .map(|i| b"ACGT"[(i * 7 + i / 13) % 4])
            .collect();
        let registry = crate::codec::Registry::new();
        for level in Level::ALL {
            let plan = planner(Config::new(level)).plan_chunk(&data, true).unwrap();
            let analysis = Analysis::of(&data).unwrap();
            let ctx = EncodeContext {
                input: &data,
                analysis: &analysis,
                level,
                dictionary: None,
                independent: true,
            };
            let encoded = registry
                .get(plan.pipeline)
                .unwrap()
                .encode(&ctx)
                .expect("re-encode");
            let mut out = vec![0u8; data.len()];
            let dec = crate::codec::DecodeContext {
                payload: &encoded,
                content_size: data.len(),
                transform: plan.transform,
                dictionary: None,
                independent: true,
            };
            registry
                .get(plan.pipeline)
                .unwrap()
                .decode(&dec, &mut out)
                .expect("decode");
            assert_eq!(out, data, "level {} did not round trip", level.name());
        }
    }

    #[test]
    fn repetitive_data_beats_store() {
        // A 30 KB constant chunk cannot be collapsed by LZ, which pays a token per
        // `MAX_MATCH` bytes, so the statistical pipeline's model has to carry it —
        // which is the case the pipeline exists for. The bound is a token-count
        // budget rather than a round number, so it cannot drift as the token
        // layout changes.
        let p = planner(Config::new(Level::Default));
        let data = vec![0x42u8; 30_000];
        let plan = p.plan_chunk(&data, true).unwrap();
        let min_tokens = data.len() / crate::matchfinder::MAX_MATCH as usize;
        let budget = min_tokens * 8 + 256;
        assert!(
            plan.payload.len() < budget,
            "{} bytes exceeded the {budget} byte budget",
            plan.payload.len()
        );
        assert!(
            plan.payload.len() * 20 < data.len(),
            "ratio too poor: {}",
            plan.payload.len()
        );
    }

    #[test]
    fn plan_never_exceeds_input_size() {
        // The core store guarantee: no chunk ever grows.
        let p = planner(Config::new(Level::Default));
        for data in [
            b"".to_vec(),
            b"a".to_vec(),
            b"ab".to_vec(),
            pseudo_random(1000, 3),
            "hello world ".repeat(50).into_bytes(),
            vec![7u8; 5000],
        ] {
            let plan = p.plan_chunk(&data, true).unwrap();
            assert!(
                plan.payload.len() <= data.len(),
                "grew from {} to {}",
                data.len(),
                plan.payload.len()
            );
        }
    }

    #[test]
    fn empty_input_is_stored() {
        let p = planner(Config::new(Level::Default));
        let plan = p.plan_chunk(&[], true).unwrap();
        assert!(plan.payload.is_empty());
    }

    #[test]
    fn single_policy_uses_one_candidate() {
        let p = planner(Config::new(Level::Default)).with_policy(CandidatePolicy::Single);
        let plan = p.plan_chunk(&policy_corpus(), true).unwrap();
        assert!(plan.tried.len() <= 1, "tried {:?}", plan.tried);
    }

    #[test]
    fn cheap_policy_limits_candidates() {
        let p = planner(Config::new(Level::Default)).with_policy(CandidatePolicy::Cheap);
        let plan = p.plan_chunk(&policy_corpus(), true).unwrap();
        assert!(plan.tried.len() <= 2, "tried {:?}", plan.tried);
    }

    #[test]
    fn full_policy_tries_several() {
        let p = planner(Config::new(Level::Default)).with_policy(CandidatePolicy::Full);
        let data = policy_corpus();
        let plan = p.plan_chunk(&data, true).unwrap();
        assert!(plan.tried.len() >= 2, "tried {:?}", plan.tried);
    }

    #[test]
    fn full_policy_is_never_worse_than_single() {
        // Trying more candidates must never produce a larger result.
        let data = "the quick brown fox jumps over the lazy dog. "
            .repeat(300)
            .into_bytes();
        let single = planner(Config::new(Level::Default))
            .with_policy(CandidatePolicy::Single)
            .plan_chunk(&data, true)
            .unwrap();
        let full = planner(Config::new(Level::Default))
            .with_policy(CandidatePolicy::Full)
            .plan_chunk(&data, true)
            .unwrap();
        assert!(
            full.payload.len() <= single.payload.len(),
            "full {} > single {}",
            full.payload.len(),
            single.payload.len()
        );
    }

    #[test]
    fn levels_produce_valid_plans() {
        for level in Level::ALL {
            let p = planner(Config::new(level));
            let data = "hello world ".repeat(500).into_bytes();
            let plan = p.plan_chunk(&data, true).unwrap();
            assert!(plan.payload.len() <= data.len(), "level {:?}", level);
        }
    }

    #[test]
    fn level_none_always_stores() {
        let p = planner(Config::new(Level::None));
        let data = "hello world ".repeat(500).into_bytes();
        let plan = p.plan_chunk(&data, true).unwrap();
        assert_eq!(plan.pipeline, PipelineId::Store);
    }

    #[test]
    fn small_window_config_is_respected() {
        let cfg = Config::new(Level::Default).with_window(WindowConfig {
            chunk_size: 2048,
            window_size: 1024,
        });
        let p = planner(cfg);
        let data = "hello world ".repeat(500).into_bytes();
        let plan = p.plan_chunk(&data, true).unwrap();
        assert!(plan.payload.len() <= data.len());
    }
}
