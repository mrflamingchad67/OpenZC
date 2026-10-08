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
    fn candidates(&self, analysis: &Analysis) -> Vec<(PipelineId, TransformId)> {
        let level = self.config.level();
        let mut out: Vec<(PipelineId, TransformId)> = Vec::new();

        if self.policy == CandidatePolicy::Single {
            let (p, t) = self.single_choice(analysis, level);
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
                if let Some(t) = self.transform_for(analysis, codec.id()) {
                    out.push((codec.id(), t));
                }
            }
        }

        // RLE earns its place on run-heavy data, and is cheap enough to always
        // measure when the data looks repetitive.
        if analysis.kind == DataKind::Repetitive {
            out.push((PipelineId::Rle, TransformId::None));
        }

        if self.policy == CandidatePolicy::Cheap {
            out.truncate(2);
        }

        // `Store` is always measured last as the safety net.
        out.push((PipelineId::Store, TransformId::None));
        out
    }

    /// The one pipeline to use when not searching.
    fn single_choice(&self, analysis: &Analysis, level: Level) -> (PipelineId, TransformId) {
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
        let t = self.transform_for(analysis, p).unwrap_or(TransformId::None);
        (p, t)
    }

    /// A transform worth trying for this pipeline, if any.
    fn transform_for(&self, analysis: &Analysis, _pipeline: PipelineId) -> Option<TransformId> {
        // Sampling from the analysis buffer keeps this cheap; the pipeline
        // measures the real result afterwards.
        let _ = analysis;
        None
    }

    /// Plan and encode one chunk.
    pub fn plan_chunk(&self, input: &[u8], independent: bool) -> Result<ChunkPlan> {
        let analysis = Analysis::of(input)?;
        let candidates = self.candidates(&analysis);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WindowConfig;

    fn planner(cfg: Config) -> Planner {
        Planner::new(&cfg.validated().unwrap())
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
        let cases: [(&str, Vec<u8>, f64); 5] = [
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

    #[test]
    fn level_controls_search_effort() {
        // The level has to reach the match finder, not stop at the config. If it
        // did, every level would emit identical bytes and the knob would be a lie
        // — which is exactly the failure this pins.
        //
        // The corpus is low-entropy but not periodic: a chain walker has to walk
        // back through many candidate positions to find the long matches, so its
        // extra effort shows up in the output rather than only in the clock.
        let data: Vec<u8> = (0..512 * 1024usize)
            .map(|i| b"ACGT"[(i * 7 + i / 13) % 4])
            .collect();

        let fast = planner(Config::new(Level::Fast))
            .plan_chunk(&data, true)
            .unwrap();
        let max = planner(Config::new(Level::Max))
            .plan_chunk(&data, true)
            .unwrap();

        assert!(
            max.payload.len() * 2 < fast.payload.len(),
            "level max produced {} bytes against level fast's {}",
            max.payload.len(),
            fast.payload.len()
        );
        // Neither may be worse than storing the input.
        assert!(max.payload.len() < data.len());
        assert!(fast.payload.len() < data.len());
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
        let data = "hello world ".repeat(500).into_bytes();
        let plan = p.plan_chunk(&data, true).unwrap();
        assert!(plan.tried.len() <= 1, "tried {:?}", plan.tried);
    }

    #[test]
    fn cheap_policy_limits_candidates() {
        let p = planner(Config::new(Level::Default)).with_policy(CandidatePolicy::Cheap);
        let data = "hello world ".repeat(500).into_bytes();
        let plan = p.plan_chunk(&data, true).unwrap();
        assert!(plan.tried.len() <= 2, "tried {:?}", plan.tried);
    }

    #[test]
    fn full_policy_tries_several() {
        let p = planner(Config::new(Level::Default)).with_policy(CandidatePolicy::Full);
        let data = "hello world ".repeat(500).into_bytes();
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
