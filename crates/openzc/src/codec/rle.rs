//! Run-length pipeline.
//!
//! RLE earns its place for run-heavy data — a run of `n` identical bytes costs a
//! two-byte control pair plus the value, so it only beats `Store` once runs are
//! long. It also needs no match space at all, which makes a chunk it encodes
//! decodable with no history and therefore trivially parallel.
//!
//! # Stream layout
//!
//! A sequence of two-byte control groups:
//!
//! ```text
//! <control> <value>   control bit 7 set: a run of (control & 0x7F) + 1 copies
//! <control> <n bytes> control bit 7 clear: n = (control & 0x7F) + 1 literal bytes
//! ```
//!
//! The decoder knows when to stop from the frame's `content_size`, so there is no
//! terminator. Literal groups mean a chunk with no runs costs `n + ceil(n / 128)`
//! bytes — slightly more than `Store`, which is precisely why the planner
//! measures both and keeps the smaller.

use crate::analyze::Analysis;
use crate::bytes::{ByteReader, ByteWriter};
use crate::codec::{Codec, DecodeContext, EncodeContext};
use crate::config::Level;
use crate::error::{Error, Result};
use crate::format::PipelineId;

/// Flag marking a run group.
const RUN_FLAG: u8 = 0x80;
/// Maximum group payload, encoded as `(control & 0x7F) + 1`.
const MAX_GROUP: usize = 128;

/// Run-length codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct RleCodec;

impl RleCodec {
    pub fn new() -> Self {
        Self
    }
}

impl Codec for RleCodec {
    fn id(&self) -> PipelineId {
        PipelineId::Rle
    }

    fn name(&self) -> &'static str {
        "rle"
    }

    /// Only worth trying when runs are long enough for the two-byte control to
    /// pay for itself. A run must exceed roughly 3 bytes to beat `Store`.
    fn candidate(&self, analysis: &Analysis, _level: Level) -> bool {
        match analysis.kind {
            crate::analyze::DataKind::Repetitive => true,
            crate::analyze::DataKind::LowEntropy => analysis.max_byte_share > 0.25,
            _ => false,
        }
    }

    fn encode(&self, ctx: &EncodeContext<'_>) -> Result<Vec<u8>> {
        let data = ctx.input;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let mut w = ByteWriter::with_capacity(data.len() / 2 + 16);

        let mut i = 0usize;
        while i < data.len() {
            let b = data[i];
            // Measure the run, capped at a full group.
            let mut run = 1usize;
            while i + run < data.len() && data[i + run] == b && run < MAX_GROUP {
                run += 1;
            }

            if run >= 2 {
                w.u8(RUN_FLAG | (run as u8 - 1));
                w.u8(b);
                i += run;
            } else {
                // Collect a literal group up to the next run of two or more.
                let start = i;
                while i < data.len() && i - start < MAX_GROUP {
                    let b = data[i];
                    let mut look = 1usize;
                    while i + look < data.len() && data[i + look] == b {
                        look += 1;
                    }
                    if look >= 2 {
                        break;
                    }
                    i += 1;
                }
                let n = i - start;
                w.u8((n as u8) - 1);
                w.bytes(&data[start..i]);
            }
        }
        Ok(w.into_vec())
    }

    fn decode(&self, ctx: &DecodeContext<'_>, out: &mut [u8]) -> Result<()> {
        let target = out.len();
        let mut r = ByteReader::new(ctx.payload);
        let mut pos = 0usize;

        while pos < target {
            let control = r.u8()?;
            let n = usize::from(control & 0x7F) + 1;
            if n > target - pos {
                return Err(Error::CorruptSequence("run-length group overruns chunk"));
            }
            if control & RUN_FLAG != 0 {
                let b = r.u8()?;
                out[pos..pos + n].fill(b);
            } else {
                let lits = r.take(n)?;
                out[pos..pos + n].copy_from_slice(lits);
            }
            pos += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::TransformId;

    fn roundtrip(data: &[u8]) {
        let c = RleCodec::new();
        let a = Analysis::of(data).unwrap();
        let enc_ctx = EncodeContext {
            input: data,
            analysis: &a,
            level: Level::Default,
            dictionary: None,
            independent: true,
        };
        let payload = c.encode(&enc_ctx).unwrap();
        let mut out = vec![0u8; data.len()];
        let dec = DecodeContext {
            payload: &payload,
            content_size: data.len(),
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        c.decode(&dec, &mut out)
            .unwrap_or_else(|e| panic!("decode failed for {} bytes: {e}", data.len()));
        assert_eq!(out, data, "roundtrip mismatch for {} bytes", data.len());
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(b"");
    }

    #[test]
    fn roundtrip_single_byte() {
        roundtrip(b"a");
    }

    #[test]
    fn roundtrip_all_lengths_uniform() {
        for n in 0..300usize {
            roundtrip(&vec![0x5Au8; n]);
        }
    }

    #[test]
    fn roundtrip_mixed_runs() {
        roundtrip(b"aaabbbbccd");
        roundtrip(b"a");
        roundtrip(b"ab");
        roundtrip(b"aaaaaaaaaaaaaaaaaaaaaab");
        roundtrip(b"baaaaaaaaaaaaaaaaaaaaaaa");
    }

    #[test]
    fn roundtrip_group_boundaries() {
        // Lengths either side of the 128-byte group cap.
        for n in [126usize, 127, 128, 129, 255, 256, 257, 384, 385] {
            roundtrip(&vec![7u8; n]);
            roundtrip(&vec![9u8; n]);
        }
    }

    #[test]
    fn long_runs_roundtrip() {
        for n in [16_383usize, 16_384, 16_385, 200_000] {
            roundtrip(&vec![0xEEu8; n]);
        }
    }

    #[test]
    fn literal_only_data_stays_near_store_size() {
        // A run group costs two bytes, so a chunk with no runs must not blow up
        // by more than the literal-group overhead.
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 2) as u8).collect();
        roundtrip(&data);
        let a = Analysis::of(&data).unwrap();
        let enc = RleCodec::new()
            .encode(&EncodeContext {
                input: &data,
                analysis: &a,
                level: Level::Default,
                dictionary: None,
                independent: true,
            })
            .unwrap();
        let overhead = enc.len() - data.len();
        assert!(
            overhead <= data.len() / MAX_GROUP + 1,
            "grew by {overhead} bytes"
        );
    }

    #[test]
    fn run_data_shrinks() {
        let data = vec![7u8; 100_000];
        let a = Analysis::of(&data).unwrap();
        let payload = RleCodec::new()
            .encode(&EncodeContext {
                input: &data,
                analysis: &a,
                level: Level::Default,
                dictionary: None,
                independent: true,
            })
            .unwrap();
        assert!(payload.len() < 2000, "payload was {} bytes", payload.len());
    }

    #[test]
    fn many_alternating_groups_roundtrip() {
        let mut data = Vec::new();
        for i in 0..2000u32 {
            data.push((i % 7) as u8);
        }
        roundtrip(&data);
    }

    #[test]
    fn decode_rejects_group_overrunning_chunk() {
        let c = RleCodec::new();
        let mut w = ByteWriter::new();
        w.u8(RUN_FLAG | 127); // a run of 128 into a 10-byte chunk
        w.u8(0x41);
        let payload = w.into_vec();
        let mut out = vec![0u8; 10];
        let dec = DecodeContext {
            payload: &payload,
            content_size: 10,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        assert!(c.decode(&dec, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_truncated_literal_group() {
        let c = RleCodec::new();
        let mut w = ByteWriter::new();
        w.u8(9); // 10 literal bytes
        w.bytes(b"only fou");
        let payload = w.into_vec();
        let mut out = vec![0u8; 10];
        let dec = DecodeContext {
            payload: &payload,
            content_size: 10,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        assert!(c.decode(&dec, &mut out).is_err());
    }

    #[test]
    fn decode_of_empty_payload_into_empty_output_succeeds() {
        let c = RleCodec::new();
        let mut out: Vec<u8> = Vec::new();
        let dec = DecodeContext {
            payload: &[],
            content_size: 0,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        c.decode(&dec, &mut out).unwrap();
        assert!(out.is_empty());
    }
}
