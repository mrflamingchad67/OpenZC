//! Store and Dictionary pipelines: identity transforms.
//!
//! These are the escape hatches that make the format total. When nothing
//! compresses, `Store` is not a failure — it is the correct answer, and it is
//! what makes `compressed_size >= original_size` a decision the engine can make
//! rather than a bug it has to avoid.
//!
//! `Dictionary` is the same payload format with a different meaning: the bytes
//! it decodes to are registered for later frames to use as a match source.

use crate::analyze::Analysis;
use crate::codec::{Codec, DecodeContext, EncodeContext};
use crate::config::Level;
use crate::error::Result;
use crate::format::PipelineId;

/// Copies the chunk through unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub struct StoreCodec {
    dictionary: bool,
}

impl StoreCodec {
    pub fn new() -> Self {
        Self { dictionary: false }
    }

    /// A store codec whose decoded content defines a dictionary.
    pub fn dictionary() -> Self {
        Self { dictionary: true }
    }
}

impl Codec for StoreCodec {
    fn id(&self) -> PipelineId {
        if self.dictionary {
            PipelineId::Dictionary
        } else {
            PipelineId::Store
        }
    }

    fn name(&self) -> &'static str {
        if self.dictionary {
            "dictionary"
        } else {
            "store"
        }
    }

    /// Always a candidate. Storing is correct for any input, which is precisely
    /// why it is the fallback when every other option is worse.
    fn candidate(&self, _analysis: &Analysis, _level: Level) -> bool {
        true
    }

    fn encode(&self, ctx: &EncodeContext<'_>) -> Result<Vec<u8>> {
        Ok(ctx.input.to_vec())
    }

    fn decode(&self, ctx: &DecodeContext<'_>, out: &mut [u8]) -> Result<()> {
        // `out.len() == ctx.content_size` is guaranteed by the stream layer,
        // and the frame header already verified payload_size == content_size.
        if ctx.payload.len() != out.len() {
            return Err(crate::error::Error::CorruptChunk(
                "store frame payload does not match content size",
            ));
        }
        out.copy_from_slice(ctx.payload);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::Analysis;

    fn ctx<'a>(input: &'a [u8], analysis: &'a Analysis) -> EncodeContext<'a> {
        EncodeContext {
            input,
            analysis,
            level: crate::config::Level::Default,
            dictionary: None,
            independent: true,
        }
    }

    #[test]
    fn store_roundtrips_every_length() {
        let c = StoreCodec::new();
        for n in 0..200usize {
            let data: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
            let a = Analysis::of(&data).unwrap();
            let enc = ctx(&data, &a);
            let payload = c.encode(&enc).unwrap();
            assert_eq!(payload, data, "n={n}");

            let mut out = vec![0u8; n];
            let d = DecodeContext {
                payload: &payload,
                content_size: n,
                transform: crate::format::TransformId::None,
                dictionary: None,
                independent: true,
            };
            c.decode(&d, &mut out).unwrap();
            assert_eq!(out, data, "n={n}");
        }
    }

    #[test]
    fn store_rejects_size_mismatch() {
        let c = StoreCodec::new();
        let d = DecodeContext {
            payload: b"abc",
            content_size: 4,
            transform: crate::format::TransformId::None,
            dictionary: None,
            independent: true,
        };
        let mut out = vec![0u8; 4];
        assert!(matches!(
            c.decode(&d, &mut out),
            Err(crate::error::Error::CorruptChunk(_))
        ));
    }

    #[test]
    fn dictionary_codec_has_its_own_id() {
        assert_eq!(StoreCodec::new().id(), PipelineId::Store);
        assert_eq!(StoreCodec::dictionary().id(), PipelineId::Dictionary);
        assert_eq!(StoreCodec::dictionary().name(), "dictionary");
    }

    #[test]
    fn store_is_always_a_candidate() {
        let c = StoreCodec::new();
        for level in Level::ALL {
            let a = Analysis::of(&[0u8; 100]).unwrap();
            assert!(c.candidate(&a, level));
        }
    }
}
