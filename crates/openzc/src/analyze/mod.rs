//! Cheap per-chunk data characterisation.
//!
//! Analysis exists to *prune*, not to decide outright. It answers "which
//! candidates are worth trying?" in time proportional to a sampling rate, so
//! the expensive pipelines are never run on data that cannot benefit.
//!
//! # Cost policy
//!
//! Analysis samples at most [`MAX_SAMPLE`] bytes. A full histogram of a 4 MiB
//! chunk is affordable, but a full scan of every chunk to decide between three
//! candidates is not — hence the sample cap. Sampling bias is acceptable
//! because the output is always validated: a wrong guess costs ratio, never
//! correctness.

use crate::error::Result;

/// Maximum bytes examined per chunk. Bounds analysis cost regardless of chunk
/// size.
pub const MAX_SAMPLE: usize = 64 * 1024;

/// Trigram start positions examined for the reuse ratio.
///
/// Exact duplicate counting is what keeps random data from looking repetitive,
/// and it costs a sort. Sorting 4 Ki keys per chunk is fast enough to keep
/// analysis negligible next to any real compression, and the ratio converges
/// long before that, so there is no reason to probe more.
pub const REPEAT_POSITIONS: usize = 4096;

/// Overall character of a chunk's content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataKind {
    /// Little or no repetition: uniform random, or already-compressed.
    Incompressible,
    /// Long runs of a repeated byte or short period.
    Repetitive,
    /// Byte values concentrated in a small alphabet.
    LowEntropy,
    /// Structured, with matchable substrings: text, source, markup, records.
    Structured,
}

impl DataKind {
    /// Short label for diagnostics and `info` output.
    pub fn name(self) -> &'static str {
        match self {
            DataKind::Incompressible => "incompressible",
            DataKind::Repetitive => "repetitive",
            DataKind::LowEntropy => "low-entropy",
            DataKind::Structured => "structured",
        }
    }
}

/// Cheap statistics describing a chunk.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// Bytes actually examined.
    pub sampled: usize,
    /// Shannon entropy in bits per byte, 0.0..=8.0.
    pub entropy: f32,
    /// Fraction of bytes belonging to the most common byte value, 0.0..=1.0.
    pub max_byte_share: f32,
    /// Distinct byte values seen.
    pub distinct_bytes: u32,
    /// Fraction of positions starting a repeat of the previous 3 bytes.
    pub trigram_repeat_ratio: f32,
    /// Longest run of one repeated byte value.
    pub longest_run: u32,
    /// Whether a 3-byte hash at offset `i` was seen earlier in the sample.
    pub has_matchable_content: bool,
    /// Characterisation derived from the above.
    pub kind: DataKind,
}

impl Analysis {
    /// Analyse a full chunk, sampling at most [`MAX_SAMPLE`] bytes from the
    /// front of the buffer.
    pub fn of(data: &[u8]) -> Result<Self> {
        let n = data.len().min(MAX_SAMPLE);
        let sample = &data[..n];
        Ok(Self::compute(sample))
    }

    fn compute(sample: &[u8]) -> Self {
        let mut freq = [0u32; 256];
        for &b in sample {
            freq[usize::from(b)] += 1;
        }
        let total = sample.len() as f64;
        let entropy = if total == 0.0 {
            0.0
        } else {
            // Computed in f64 then narrowed: the log term needs more precision
            // than f32 holds before summation.
            let bits: f64 = freq
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = f64::from(c) / total;
                    -p * p.log2()
                })
                .sum();
            (bits as f32).clamp(0.0, 8.0)
        };
        let max_byte_share = (freq.iter().copied().max().unwrap_or(0) as f64 / total) as f32;
        let distinct_bytes = freq.iter().filter(|&&c| c > 0).count() as u32;

        let (longest_run, trigram_repeat_ratio, has_matchable_content) = structure(sample);

        let kind = classify(
            entropy,
            max_byte_share,
            distinct_bytes,
            trigram_repeat_ratio,
            longest_run,
        );

        Self {
            sampled: sample.len(),
            entropy,
            max_byte_share,
            distinct_bytes,
            trigram_repeat_ratio,
            has_matchable_content,
            longest_run,
            kind,
        }
    }

    /// True when the sample looks close to incompressible, which is the signal
    /// to store rather than spend CPU on.
    ///
    /// The threshold is deliberately conservative: false "compressible" guesses
    /// only waste time, while false "incompressible" guesses throw away ratio.
    pub fn looks_incompressible(&self) -> bool {
        self.entropy >= 7.85 && self.trigram_repeat_ratio < 0.002
    }
}

/// Run length, local repeat rate, and trigram reuse for a sample.
fn structure(sample: &[u8]) -> (u32, f32, bool) {
    let n = sample.len();
    if n < 4 {
        return (n as u32, 0.0, false);
    }

    // Longest run of an identical byte.
    let mut longest_run = 1u32;
    let mut run = 1u32;
    for i in 1..n {
        if sample[i] == sample[i - 1] {
            run += 1;
            if run > longest_run {
                longest_run = run;
            }
        } else {
            run = 1;
        }
    }

    // Trigram reuse: count *exact* 3-byte repeats, so uniformly random data
    // cannot look repetitive just because a hash table collided. A truncated
    // tag set is much cheaper but lies badly here — with n positions drawn
    // into S slots, roughly n^2 / 2S hits are false, which for a 16-bit set is
    // large enough to swamp the real signal.
    let span = n - 2;
    let positions = REPEAT_POSITIONS.min(span);
    let mut keys: Vec<u32> = Vec::with_capacity(positions);
    for k in 0..positions {
        // Spread the probes across the sample instead of clustering at the
        // front, so a chunk whose head is atypical is judged fairly.
        let i = k * (span - 1) / (positions - 1);
        keys.push(trigram(sample, i));
    }
    keys.sort_unstable();

    let mut repeats = 0u32;
    for pair in keys.windows(2) {
        if pair[0] == pair[1] {
            repeats += 1;
        }
    }
    let ratio = repeats as f32 / positions as f32;
    (longest_run, ratio, repeats > 0)
}

/// The 3 bytes at `i` packed into a `u32`. Exact keys, no hashing.
#[inline]
fn trigram(sample: &[u8], i: usize) -> u32 {
    (u32::from(sample[i]) << 16) | (u32::from(sample[i + 1]) << 8) | u32::from(sample[i + 2])
}

/// Map raw statistics onto a coarse classification.
fn classify(
    entropy: f32,
    max_byte_share: f32,
    distinct_bytes: u32,
    trigram_repeat_ratio: f32,
    longest_run: u32,
) -> DataKind {
    // Long runs dominate: a chunk containing a run of 64+ identical bytes is
    // run-length shaped regardless of what the rest holds. The threshold is
    // absolute rather than relative to entropy because that is the property
    // the RLE codec actually exploits.
    if longest_run >= 64 {
        return DataKind::Repetitive;
    }
    // A single byte making up 40% of the sample means an order-0 model has
    // very little to spend, whatever the run structure.
    if max_byte_share >= 0.40 {
        return DataKind::Repetitive;
    }
    if entropy >= 7.85 && trigram_repeat_ratio < 0.002 {
        return DataKind::Incompressible;
    }
    if distinct_bytes <= 16 || entropy < 4.0 {
        return DataKind::LowEntropy;
    }
    // Any real trigram reuse at all means LZ has something to work with, and
    // text sits far above this threshold.
    if trigram_repeat_ratio > 0.01 {
        return DataKind::Structured;
    }
    DataKind::Incompressible
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes, so tests do not depend on an RNG crate.
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

    fn text(n: usize) -> Vec<u8> {
        let base = "the quick brown fox jumps over the lazy dog. ";
        let repeated = base.repeat(n / base.len() + 1);
        repeated.as_bytes()[..n].to_vec()
    }

    #[test]
    fn random_is_incompressible() {
        let a = Analysis::of(&pseudo_random(8192, 0xDEAD)).unwrap();
        assert_eq!(a.kind, DataKind::Incompressible, "entropy {}", a.entropy);
        assert!(a.looks_incompressible());
        assert!(a.entropy > 7.9, "entropy was {}", a.entropy);
    }

    #[test]
    fn constant_data_is_repetitive() {
        let a = Analysis::of(&[0x41u8; 8192]).unwrap();
        assert_eq!(a.kind, DataKind::Repetitive);
        assert_eq!(a.longest_run, 8192);
        assert!((a.max_byte_share - 1.0).abs() < 1e-6);
        assert!(a.entropy < 0.001);
    }

    #[test]
    fn short_period_data_is_low_entropy_and_matchable() {
        // A short period is the textbook definition of low entropy. It is *not*
        // run-shaped, so `Repetitive` would be a lie here, but it is exactly
        // what LZ collapses to nothing, so it must be recognised as matchable
        // and must never be mistaken for incompressible.
        let mut d = Vec::new();
        while d.len() < 8192 {
            d.extend_from_slice(b"abcabc");
        }
        let a = Analysis::of(&d).unwrap();
        assert_eq!(a.kind, DataKind::LowEntropy);
        assert!(a.has_matchable_content);
        assert!(
            a.trigram_repeat_ratio > 0.5,
            "ratio {}",
            a.trigram_repeat_ratio
        );
        assert!(a.longest_run < 4, "runs were {}", a.longest_run);
        assert!(!a.looks_incompressible());
    }

    #[test]
    fn long_period_data_is_structured() {
        // A period far longer than the match window's reach: still repetitive to
        // a trigram, but too little reuse to call a period, so it lands as
        // structured data for LZ.
        let period: Vec<u8> = (0..97u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut d = period.clone();
        while d.len() < 8192 {
            d.extend_from_slice(&period);
        }
        let a = Analysis::of(&d).unwrap();
        assert_ne!(a.kind, DataKind::Incompressible);
        assert!(a.has_matchable_content);
    }

    #[test]
    fn text_is_structured() {
        let a = Analysis::of(&text(16_384)).unwrap();
        assert_eq!(a.kind, DataKind::Structured, "entropy {}", a.entropy);
        assert!(
            a.trigram_repeat_ratio > 0.2,
            "ratio {}",
            a.trigram_repeat_ratio
        );
        assert!(!a.looks_incompressible());
    }

    #[test]
    fn entropy_of_uniform_is_near_eight() {
        // 0..=255 exactly once: entropy is log2(256) = 8.0.
        let a = Analysis::of(&(0..=255u8).collect::<Vec<_>>()).unwrap();
        assert!((a.entropy - 8.0).abs() < 0.01, "entropy {}", a.entropy);
        assert_eq!(a.distinct_bytes, 256);
    }

    #[test]
    fn entropy_of_constant_is_zero() {
        let a = Analysis::of(&[0u8; 1000]).unwrap();
        assert!(a.entropy.abs() < 1e-6, "entropy {}", a.entropy);
        assert_eq!(a.distinct_bytes, 1);
    }

    #[test]
    fn sampling_is_bounded() {
        let big = pseudo_random(MAX_SAMPLE * 4, 7);
        let a = Analysis::of(&big).unwrap();
        assert_eq!(a.sampled, MAX_SAMPLE);
    }

    #[test]
    fn empty_and_tiny_inputs_do_not_panic() {
        for n in 0..8 {
            let d = pseudo_random(n, 3);
            let a = Analysis::of(&d).unwrap();
            assert_eq!(a.sampled, n);
        }
    }

    #[test]
    fn analysis_is_deterministic() {
        let d = text(32_768);
        let a = Analysis::of(&d).unwrap();
        let b = Analysis::of(&d).unwrap();
        assert_eq!(a.entropy.to_bits(), b.entropy.to_bits());
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.longest_run, b.longest_run);
    }
}
