//! Frame header: one independently decodable unit of the stream.

use crate::bytes::{ByteReader, ByteWriter};
use crate::error::{Error, Result};
use crate::format::{PipelineId, TransformId, HASH_LEN};

/// Frame magic: `FRM 1A`.
pub const FRAME_MAGIC: [u8; 4] = *b"FRM\x1a";
pub const FRAME_MAGIC_LEN: usize = 4;

/// Fixed frame header length, excluding payload and the optional content hash.
///
/// ```text
/// magic(4) pipeline(1) transform(1) flags(1) dict_id(1)
/// content_size(4) payload_size(4)
/// ```
pub const FRAME_HEADER_LEN: usize = FRAME_MAGIC_LEN + 4 + 8;

/// How a frame may reference data outside itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameFlags(u8);

impl FrameFlags {
    /// The frame may only reference bytes it has already produced itself, so
    /// frames of this kind can be decoded in any order and in parallel.
    pub const INDEPENDENT: u8 = 1 << 0;

    pub const KNOWN: u8 = Self::INDEPENDENT;

    #[inline]
    pub fn bits(self) -> u8 {
        self.0
    }

    #[inline]
    pub fn contains(self, f: u8) -> bool {
        self.0 & f == f
    }

    pub fn from_bits_strict(bits: u8) -> Result<Self> {
        if bits & !Self::KNOWN != 0 {
            return Err(Error::InvalidHeader("unknown frame flag bits set"));
        }
        Ok(Self(bits))
    }

    /// Build from raw bits, panicking on undefined bits.
    ///
    /// Only for internal construction where the bits are compile-time known;
    /// parsing must use [`FrameFlags::from_bits_strict`].
    pub(crate) fn from_bits_unchecked(bits: u8) -> Self {
        debug_assert_eq!(bits & !Self::KNOWN, 0, "undefined frame flag bit set");
        Self(bits)
    }
}

/// Parsed frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub pipeline: PipelineId,
    pub transform: TransformId,
    pub flags: FrameFlags,
    /// 0 = no dictionary. Otherwise the 1-based ordinal of a preceding
    /// [`PipelineId::Dictionary`] frame.
    pub dict_id: u8,
    /// Number of bytes this frame decodes to.
    pub content_size: u32,
    /// Number of payload bytes that follow the header.
    pub payload_size: u32,
}

impl FrameHeader {
    #[inline]
    pub fn is_independent(&self) -> bool {
        self.flags.contains(FrameFlags::INDEPENDENT)
    }

    /// Total bytes this frame occupies, including the content hash if the
    /// container header enabled per-frame checksums.
    pub fn frame_len(&self, with_hash: bool) -> usize {
        FRAME_HEADER_LEN + self.payload_size as usize + if with_hash { HASH_LEN } else { 0 }
    }

    /// Encode into a fresh buffer. The caller must be able to state
    /// `content_size` and `payload_size` up front, which is why frames are
    /// compressed in bounded memory before being written.
    pub fn encode(&self, w: &mut ByteWriter) {
        debug_assert!(w.is_empty());
        w.bytes(&FRAME_MAGIC);
        w.u8(self.pipeline as u8);
        w.u8(self.transform as u8);
        w.u8(self.flags.bits());
        w.u8(self.dict_id);
        w.u32le(self.content_size);
        w.u32le(self.payload_size);
    }

    /// Parse a fixed-size frame header from exactly `FRAME_HEADER_LEN` bytes.
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < FRAME_HEADER_LEN {
            return Err(Error::UnexpectedEof {
                needed: FRAME_HEADER_LEN,
                available: data.len(),
            });
        }
        let mut r = ByteReader::new(data);
        let m = r.take(FRAME_MAGIC_LEN)?;
        if m != FRAME_MAGIC {
            return Err(Error::InvalidHeader("frame magic mismatch"));
        }
        let pipeline = PipelineId::from_u8(r.u8()?)?;
        let transform = TransformId::from_u8(r.u8()?)?;
        let flags = FrameFlags::from_bits_strict(r.u8()?)?;
        let dict_id = r.u8()?;
        let content_size = r.u32le()?;
        let payload_size = r.u32le()?;

        if pipeline == PipelineId::Reserved0 {
            return Err(Error::UnknownPipeline(0));
        }
        // A dictionary frame is by definition a standalone artifact; letting it
        // also depend on a dictionary would be a cycle.
        if pipeline == PipelineId::Dictionary && dict_id != 0 {
            return Err(Error::InvalidHeader(
                "dictionary frame must not reference a dictionary",
            ));
        }
        if pipeline == PipelineId::Store && payload_size != content_size {
            return Err(Error::CorruptChunk(
                "store frame payload size differs from content size",
            ));
        }
        if pipeline == PipelineId::Rle && content_size == 0 {
            return Err(Error::CorruptChunk("empty run-length frame"));
        }
        Ok(Self {
            pipeline,
            transform,
            flags,
            dict_id,
            content_size,
            payload_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::header::ContainerFlags;

    fn sample() -> FrameHeader {
        FrameHeader {
            pipeline: PipelineId::Statistical,
            transform: TransformId::Delta,
            flags: FrameFlags(FrameFlags::INDEPENDENT),
            dict_id: 1,
            content_size: 1 << 20,
            payload_size: 4096,
        }
    }

    #[test]
    fn frame_header_roundtrip() {
        let h = sample();
        let mut w = ByteWriter::new();
        h.encode(&mut w);
        assert_eq!(w.len(), FRAME_HEADER_LEN);
        assert_eq!(FrameHeader::decode(w.as_slice()).unwrap(), h);
    }

    #[test]
    fn frame_len_accounts_for_hash() {
        let h = sample();
        assert_eq!(h.frame_len(false), FRAME_HEADER_LEN + 4096);
        assert_eq!(h.frame_len(true), FRAME_HEADER_LEN + 4096 + HASH_LEN);
        let _ = ContainerFlags::CHUNK_CHECKSUM;
    }

    #[test]
    fn store_frame_size_mismatch_rejected() {
        let h = FrameHeader {
            pipeline: PipelineId::Store,
            transform: TransformId::None,
            flags: FrameFlags(0),
            dict_id: 0,
            content_size: 10,
            payload_size: 11,
        };
        let mut w = ByteWriter::new();
        h.encode(&mut w);
        assert!(matches!(
            FrameHeader::decode(w.as_slice()),
            Err(Error::CorruptChunk(_))
        ));
    }

    #[test]
    fn unknown_flags_rejected() {
        let mut w = ByteWriter::new();
        sample().encode(&mut w);
        let mut v = w.into_vec();
        v[FRAME_MAGIC_LEN + 2] = 0x40;
        assert!(FrameHeader::decode(&v).is_err());
    }

    #[test]
    fn dictionary_frame_cannot_reference_dictionary() {
        let h = FrameHeader {
            pipeline: PipelineId::Dictionary,
            transform: TransformId::None,
            flags: FrameFlags(0),
            dict_id: 1,
            content_size: 10,
            payload_size: 5,
        };
        let mut w = ByteWriter::new();
        h.encode(&mut w);
        assert!(FrameHeader::decode(w.as_slice()).is_err());
    }
}
