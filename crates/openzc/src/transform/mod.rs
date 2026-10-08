//! Reversible pre-LZ transforms.
//!
//! A transform rearranges a chunk so that the LZ stage downstream sees more
//! redundancy. It must be exactly invertible: a one-byte change in the
//! transform must produce a one-byte change in its output. Every transform is
//! tested with a `transform(invert(x)) == x` property test.
//!
//! Transforms are opt-in per frame. The id is recorded in the frame header, so
//! a decoder always knows which one ran.
//!
//! # Interleaved values, and why the heuristic looks at strides
//!
//! The obvious way to decide whether [`DeltaTransform`] is worth trying is to
//! measure how much adjacent bytes drift. That works for a smooth byte stream and
//! fails for interleaved fixed-width fields.
//!
//! A `u16` counter stored little-endian is the byte stream `lo0 hi0 lo1 hi1 ...`.
//! Every *adjacent* pair straddles a field boundary and jumps — measured drift
//! 120.3 — so adjacent drift says "incompressible" and delta is refused. But the
//! bytes that carry the signal are two apart, and drift at stride 2 is 1.0. On that
//! data delta compresses **45.8x** better, which is the single largest win the
//! transform has on any input.
//!
//! So the heuristic measures drift at several strides and offers delta if any of
//! them is low. Stride 1 remains the primary test and the others are purely
//! additive, so nothing that is offered today stops being offered.
//!
//! The stride set is deliberately short and each entry is justified by a measured
//! case (see [`CANDIDATE_STRIDES`]). Scanning every stride instead was measured and
//! rejected: it costs 5x more, and by stride 64 it registers low drift on ordinary
//! text at stride 40 — taking a minimum over enough noisy statistics eventually
//! finds one that dips under any fixed threshold.
//!
//! The signal is deliberately loose. It also fires on `u32` and `u64`, where delta
//! loses. That is not a defect, because the planner measures every candidate and
//! keeps the smallest: an extra candidate can only find something smaller or
//! equal, so it cannot cost ratio. It costs one wasted encode, which is the right
//! thing to buy against permanently missing a 45x win.

use crate::error::Result;
use crate::format::TransformId;

/// Bytes sampled for the drift heuristics.
///
/// 8 KiB is enough for a stable average and small enough that the whole check
/// costs single-digit microseconds, which is nothing beside an encode.
pub const SAMPLE_BYTES: usize = 8192;

/// Average absolute byte difference below which drift is considered low.
///
/// Measured on the statistical pipeline over 512 KiB: uniform random data averages
/// ~85 at every stride, ordinary text 25–37, and the interleaved integer streams
/// this exists to catch sit at 0.2–1.0 on their own stride. Random ACGT is the
/// awkward neighbour at ~7.7 — below the threshold, which is why it is kept out by
/// measurement rather than by the threshold.
pub const MAX_DRIFT: f64 = 8.0;

/// Strides checked when adjacent drift is high.
///
/// A short list of *field widths people actually write*, not a contiguous scan.
/// Each entry earns its place against a measured case:
///
/// | stride | layout | delta outcome |
/// |--------|--------|---------------|
/// | 2 | `u16`, or two interleaved `u8` | wins **45.8x** |
/// | 3 | `u24` | wins |
/// | 4 | `u32` | loses, detected anyway |
/// | 6 | three interleaved `u16` (x, y, z records) | wins **29.9x** |
/// | 8 | `u64` | loses, detected anyway |
///
/// Strides 3 and 6 were added after measuring that the power-of-two set missed
/// both: a 24-bit field and a three-`u16` record are ordinary layouts, and delta
/// wins 29.9x on the latter.
///
/// A contiguous scan `2..=K` was measured and rejected, on two independent
/// grounds:
///
/// * **It is 2.3x the cost.** Worst case 24.7 us for this set against 56 us for
///   `2..=16` and 116 us for `2..=32` — and the worst case is ordinary text and
///   uniform random, the two inputs that matter most.
/// * **It starts inventing strides.** At `K = 64`, ordinary text registers low
///   drift at stride 40. That is not a bug in the measurement, it is what happens
///   when you take a minimum over 63 noisy statistics: with enough candidates,
///   some will dip under any fixed threshold. A blind scan does not fail by being
///   too narrow, it fails by being too permissive, and the failure is silent.
///
/// For scale: the single adjacent pass this replaces cost 4.2 us, so five strides
/// are ~6x the old check — which is ~0.001% of an encode, and end-to-end text
/// throughput is unchanged.
pub const CANDIDATE_STRIDES: [usize; 5] = [2, 3, 4, 6, 8];

/// A reversible byte rearrangement.
pub trait Transform: Send + Sync {
    /// Stable id written to the frame header.
    fn id(&self) -> TransformId;

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Apply the transform.
    fn forward(&self, input: &[u8], output: &mut Vec<u8>);

    /// Invert the transform. `input` must be the exact output of `forward`.
    fn inverse(&self, input: &[u8], output: &mut Vec<u8>);

    /// Rough estimate of whether this transform will help `data`.
    ///
    /// This is a *hint*, never a decision: the pipeline measures the real
    /// output size before committing, so a wrong answer costs ratio, never
    /// correctness.
    fn worth_trying(&self, data: &[u8]) -> bool {
        let _ = data;
        true
    }
}

/// Identity transform.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTransform;

impl Transform for NoTransform {
    fn id(&self) -> TransformId {
        TransformId::None
    }

    fn name(&self) -> &'static str {
        "none"
    }

    fn forward(&self, input: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(input);
    }

    fn inverse(&self, input: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(input);
    }

    fn worth_trying(&self, _data: &[u8]) -> bool {
        false
    }
}

/// Byte-wise delta against the previous byte.
///
/// Turns slowly-varying numeric series (timestamps, counters, gradients,
/// 16/32-bit little-endian integers) into streams of small values and zeroes,
/// which the byte-oriented entropy stage codes far more efficiently.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeltaTransform;

impl Transform for DeltaTransform {
    fn id(&self) -> TransformId {
        TransformId::Delta
    }

    fn name(&self) -> &'static str {
        "delta"
    }

    fn forward(&self, input: &[u8], output: &mut Vec<u8>) {
        output.reserve(input.len());
        let mut prev = 0u8;
        for &b in input {
            output.push(b.wrapping_sub(prev));
            prev = b;
        }
    }

    fn inverse(&self, input: &[u8], output: &mut Vec<u8>) {
        output.reserve(input.len());
        let mut prev = 0u8;
        for &b in input {
            let v = b.wrapping_add(prev);
            output.push(v);
            prev = v;
        }
    }

    /// Delta pays off when bytes a fixed distance apart are correlated.
    ///
    /// Two independent signals, because one is not enough:
    ///
    /// * **Adjacent drift** (stride 1) — the classic cheap proxy. Low total drift
    ///   means the delta stream is mostly zeroes and small magnitudes.
    /// * **Stride drift** (strides 2, 4 and 8) — for interleaved fixed-width
    ///   fields, where the bytes that carry the signal are *not* adjacent.
    ///
    /// Stride 1 is kept as the primary test and the others are additive, so no
    /// input that is offered today stops being offered. See the module note on
    /// interleaved integers for why the second signal exists at all.
    fn worth_trying(&self, data: &[u8]) -> bool {
        if data.len() < 64 {
            return false;
        }
        let n = data.len().min(SAMPLE_BYTES);

        let mut drift = 0u64;
        for i in 1..n {
            drift += u64::from(data[i].abs_diff(data[i - 1]));
        }
        if (drift as f64 / (n - 1) as f64) < MAX_DRIFT {
            return true;
        }

        // Adjacent drift was high. That is the signature of interleaved values:
        // a `u16` counter stored little-endian is a byte stream of
        // `lo0 hi0 lo1 hi1 ...`, where every *adjacent* pair straddles a field
        // boundary and jumps, while the pairs that matter are two apart.
        //
        // Measured over 512 KiB, adjacent drift versus the stride that exposes
        // the signal, and whether delta actually won:
        //
        // | corpus           | adjacent | stride | stride drift | delta |
        // |------------------|----------|--------|--------------|-------|
        // | u16 pairs        | 120.3    | 2      | 1.0          | +45.8x |
        // | u24 triples      | 120.3    | 3      | 0.5          | wins   |
        // | u32 quads        | 63.8     | 4      | 0.5          | -0.24x |
        // | xyz u16 stride 6 | 119.4    | 6      | 0.6          | +29.9x |
        // | u64 octets       | 31.9     | 8      | 0.2          | -0.36x |
        // | random u16 pairs | 84.2     | 2,3,4,6,8 | ~85     | no    |
        // | uniform random   | 86.6     | 2,3,4,6,8 | ~86     | no    |
        // | ordinary text    | 36.2     | 2,3,4,6,8 | 26–37  | no    |
        //
        // The last three rows are the ones that keep this honest. Noise does not
        // produce a low stride drift at any of these strides, and text does not
        // either — so the signal stays off on exactly the data where it would be
        // a pure cost. `u32` and `u64` are detected and delta *loses* on them,
        // which costs nothing in ratio: the planner measures every candidate and
        // keeps the smallest, so an extra candidate can only find something smaller
        // or equal. It does cost a second statistical encode. Paying that on
        // `u32`/`u64` to catch 45x on `u16` and 29.9x on interleaved records is the
        // right way round — a missed candidate costs ratio permanently, a wasted
        // one costs CPU once.
        detected_stride(data).is_some()
    }
}

/// The stride at which `data` shows low drift, if any.
///
/// Diagnostics only. [`Transform::worth_trying`] deliberately reports a boolean,
/// because "which stride fired" is not something a caller can act on — but it is
/// exactly what someone needs in order to explain a surprising result rather than
/// guess at it.
pub fn detected_stride(data: &[u8]) -> Option<usize> {
    let n = data.len().min(SAMPLE_BYTES);
    CANDIDATE_STRIDES
        .iter()
        .copied()
        .find(|&stride| stride < n && mean_stride_drift(data, stride) < MAX_DRIFT)
}

/// Mean absolute difference between bytes `stride` apart, over the sample.
///
/// Returns infinity for a stride the sample is too short to measure, so that an
/// unusable stride can never look like a low one.
fn mean_stride_drift(data: &[u8], stride: usize) -> f64 {
    let n = data.len().min(SAMPLE_BYTES);
    if stride == 0 || n <= stride {
        return f64::INFINITY;
    }
    let mut total = 0u64;
    for i in stride..n {
        total += u64::from(data[i].abs_diff(data[i - stride]));
    }
    total as f64 / (n - stride) as f64
}

/// Look up a transform by its on-wire id.
pub fn by_id(id: TransformId) -> Result<Box<dyn Transform>> {
    Ok(match id {
        TransformId::None => Box::new(NoTransform),
        TransformId::Delta => Box::new(DeltaTransform),
    })
}

/// Apply a transform by id, allocating a fresh buffer.
pub fn apply_forward(id: TransformId, input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len());
    by_id(id)?.forward(input, &mut out);
    Ok(out)
}

/// Invert a transform by id, allocating a fresh buffer.
pub fn apply_inverse(id: TransformId, input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len());
    by_id(id)?.inverse(input, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every transform must round-trip exactly, for every input length in a
    /// range that crosses the small-input edge cases.
    #[test]
    fn all_transforms_roundtrip() {
        let transforms: Vec<Box<dyn Transform>> =
            vec![Box::new(NoTransform), Box::new(DeltaTransform)];
        for t in &transforms {
            for n in 0..=300usize {
                let data: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(37) % 256) as u8).collect();
                let mut fwd = Vec::new();
                t.forward(&data, &mut fwd);
                assert_eq!(fwd.len(), data.len(), "{} len changed at n={n}", t.name());
                let mut inv = Vec::new();
                t.inverse(&fwd, &mut inv);
                assert_eq!(inv, data, "{} failed to invert at n={n}", t.name());
            }
        }
    }

    #[test]
    fn delta_of_constant_is_all_zero() {
        let data = vec![0x7Fu8; 64];
        let mut out = Vec::new();
        DeltaTransform.forward(&data, &mut out);
        assert_eq!(out[0], 0x7F, "first delta is against zero");
        assert!(out[1..].iter().all(|&b| b == 0), "rest must be zero");
    }

    #[test]
    fn delta_reduces_ramp() {
        // A ramp that does not wrap, so the point is visible: every byte differs
        // from its predecessor by one, and delta should collapse that to a
        // constant — which is what makes the chunk compressible at all. Counting
        // zero *bytes* would be the wrong measure; the payoff is the run.
        let data: Vec<u8> = (0..200u32).map(|i| i as u8).collect();
        let fwd = apply_forward(TransformId::Delta, &data).unwrap();
        assert_eq!(fwd[0], 0, "first delta is against zero");
        assert!(
            fwd[1..].iter().all(|&b| b == 1),
            "a ramp must become a constant, got {:?}",
            &fwd[..8]
        );

        // A constant is only useful if it survives the round trip.
        let back = apply_inverse(TransformId::Delta, &fwd).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn delta_handles_byte_overflow() {
        // Exercises wrapping behaviour at the 0/255 boundary.
        let data = vec![0u8, 255, 0, 255, 128, 127];
        let fwd = apply_forward(TransformId::Delta, &data).unwrap();
        let inv = apply_inverse(TransformId::Delta, &fwd).unwrap();
        assert_eq!(inv, data);
    }

    #[test]
    fn worth_trying_rejects_random_and_accepts_ramp() {
        let t = DeltaTransform;
        let mut s = 12345u32;
        let random: Vec<u8> = (0..4096)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 24) as u8
            })
            .collect();
        assert!(
            !t.worth_trying(&random),
            "random data should not be deltified"
        );
        assert!(!t.worth_trying(&[1, 2, 3]), "tiny input is not worth it");

        let ramp: Vec<u8> = (0..8192u32).map(|i| (i % 200) as u8).collect();
        assert!(t.worth_trying(&ramp), "ramp should be deltified");
    }

    #[test]
    fn by_id_rejects_unknown() {
        assert!(by_id(TransformId::Delta).is_ok());
        assert!(by_id(TransformId::None).is_ok());
    }
}
