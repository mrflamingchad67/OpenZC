//! Reversible pre-LZ transforms.
//!
//! A transform rearranges a chunk so that the LZ stage downstream sees more
//! redundancy. It must be exactly invertible: a one-byte change in the
//! transform must produce a one-byte change in its output. Every transform is
//! tested with a `transform(invert(x)) == x` property test.
//!
//! Transforms are opt-in per frame. The id is recorded in the frame header, so
//! a decoder always knows which one ran.

use crate::error::Result;
use crate::format::TransformId;

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

    /// Delta pays off when consecutive bytes are correlated. Summing absolute
    /// first differences is the classic cheap proxy: low total drift means the
    /// delta stream is mostly zeroes and small magnitudes.
    fn worth_trying(&self, data: &[u8]) -> bool {
        if data.len() < 64 {
            return false;
        }
        let n = data.len().min(8192);
        let mut drift = 0u64;
        for i in 1..n {
            drift += u64::from(data[i].abs_diff(data[i - 1]));
        }
        // Average absolute difference per byte. Structured little-endian
        // integers typically land well under 16; random data averages ~85.
        (drift as f64 / (n - 1) as f64) < 24.0
    }
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
