//! The OpenZC container and frame format, version 1.
//!
//! The format is deliberately small and fully specified so that OpenZE — or any
//! third party — can implement a compatible decoder without linking this crate.
//! The normative description lives in `docs/FORMAT.md`; this module is the
//! reference implementation of it.
//!
//! # Grammar
//!
//! ```text
//! Stream     := ContainerHeader Frame* EndMarker
//! Frame      := FrameMagic Pipeline Transform Flags DictId ContentSize
//!               PayloadSize Payload [ContentHash]
//! EndMarker  := EndMagic ContentSize [ContentHash]
//! ```
//!
//! # Extensibility rules
//!
//! * `version_major` must be understood before anything else is read. A major
//!   version that this build does not know is a hard error.
//! * `header_len` lets a future minor version append fields that older readers
//!   skip without understanding.
//! * Flag bits are *not* forward compatible. An encoder must only set bits
//!   defined by the major version it writes, so a reader can reject an
//!   unknown bit and know the stream is not one it can safely decode.
//! * `pipeline`, `transform`, and `dict_id` are small integer ids. New values
//!   are additive; old decoders reject unknown ids instead of guessing.

pub mod frame;
pub mod header;

pub use frame::{FrameFlags, FrameHeader, FRAME_HEADER_LEN, FRAME_MAGIC, FRAME_MAGIC_LEN};
pub use header::{
    ContainerFlags, ContainerHeader, EndMarker, MAGIC, MAGIC_LEN, VERSION_MAJOR, VERSION_MINOR,
};

/// Size of the fixed container header preamble:
/// `magic(4) + version_major(1) + version_minor(1) + header_len(2)`.
pub const CONTAINER_PREAMBLE_LEN: usize = 8;

/// Hard ceiling on the container header, so a corrupt `header_len` cannot make
/// a decoder allocate or skip an absurd amount.
pub const MAX_CONTAINER_HEADER_LEN: usize = 4096;

/// Total size of the OpenZC end marker excluding the optional content hash.
pub const END_MARKER_FIXED_LEN: usize = 4 + 8;

/// Size of a BLAKE3 digest in bytes.
pub const HASH_LEN: usize = 32;

/// Identifier for the compression pipeline applied to a frame.
///
/// Ids are stable: a decoder dispatches on this value, so ids are never reused
/// and new ids are only appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PipelineId {
    /// Reserved; the end marker uses a distinct magic instead of a pipeline id.
    Reserved0 = 0,
    /// No compression. Payload bytes *are* the frame content.
    Store = 1,
    /// Byte-oriented LZ77 with no entropy coding. Very fast to decode.
    LzFast = 2,
    /// LZ77 + adaptive context model + range coder.
    Statistical = 3,
    /// Run-length only. No LZ window, so decoding is trivially parallel.
    Rle = 4,
    /// A frame whose decoded content defines a dictionary for later frames.
    Dictionary = 5,
}

impl PipelineId {
    /// Decode from the on-wire byte. Unknown ids are rejected.
    pub fn from_u8(v: u8) -> crate::error::Result<Self> {
        Ok(match v {
            0 => PipelineId::Reserved0,
            1 => PipelineId::Store,
            2 => PipelineId::LzFast,
            3 => PipelineId::Statistical,
            4 => PipelineId::Rle,
            5 => PipelineId::Dictionary,
            other => return Err(crate::error::Error::UnknownPipeline(u32::from(other))),
        })
    }

    /// Short human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            PipelineId::Reserved0 => "reserved",
            PipelineId::Store => "store",
            PipelineId::LzFast => "lz-fast",
            PipelineId::Statistical => "statistical",
            PipelineId::Rle => "rle",
            PipelineId::Dictionary => "dictionary",
        }
    }

    /// Every pipeline id this build implements, in id order.
    pub fn all() -> &'static [PipelineId] {
        &[
            PipelineId::Store,
            PipelineId::LzFast,
            PipelineId::Statistical,
            PipelineId::Rle,
            PipelineId::Dictionary,
        ]
    }
}

/// Identifier for a reversible pre-LZ transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TransformId {
    /// Identity.
    None = 0,
    /// Byte-wise delta against the previous byte in the frame.
    Delta = 1,
}

impl TransformId {
    pub fn from_u8(v: u8) -> crate::error::Result<Self> {
        Ok(match v {
            0 => TransformId::None,
            1 => TransformId::Delta,
            other => return Err(crate::error::Error::UnknownTransform(other)),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            TransformId::None => "none",
            TransformId::Delta => "delta",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_ids_roundtrip() {
        for &p in PipelineId::all() {
            assert_eq!(PipelineId::from_u8(p as u8).unwrap(), p);
        }
    }

    #[test]
    fn unknown_pipeline_rejected() {
        assert!(PipelineId::from_u8(200).is_err());
    }

    #[test]
    fn pipeline_ids_are_unique() {
        let mut ids: Vec<u8> = PipelineId::all().iter().map(|p| *p as u8).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before);
    }
}
