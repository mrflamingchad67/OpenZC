//! Container header and end marker.

use crate::bytes::{ByteReader, ByteWriter};
use crate::error::{Error, Result};

/// Container magic: `1A 'C' 'Z' 'O'`.
///
/// The leading `0x1A` (DOS EOF) makes accidental truncation or text-mode
/// damage obvious, and the remaining bytes spell the format name.
pub const MAGIC: [u8; 4] = [0x1A, b'C', b'Z', b'O'];
pub const MAGIC_LEN: usize = 4;

/// Format major version. A decoder must implement the major version to read a
/// stream at all.
///
/// 2 because the `statistical` payload changed: the escaped literal-run length
/// went from a fixed two bytes to a varint. Streams written by 1.x are *not*
/// decodable here, and not only for the frames that were already broken — a 1.x
/// frame with a short literal run is perfectly valid and would now mis-decode, so
/// this is the major-version bump the format requires rather than a silent change.
pub const VERSION_MAJOR: u8 = 2;
/// Format minor version. Bumped for backwards-compatible additions.
pub const VERSION_MINOR: u8 = 0;

/// Highest major version this build can decode.
pub const MAX_SUPPORTED_MAJOR: u16 = VERSION_MAJOR as u16;

/// Bit flags in the container header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContainerFlags(u32);

impl ContainerFlags {
    /// `content_size` in the header is meaningful.
    pub const CONTENT_SIZE: u32 = 1 << 0;
    /// The end marker carries a BLAKE3 hash of the whole decoded content.
    pub const CONTENT_CHECKSUM: u32 = 1 << 1;
    /// Every frame carries a BLAKE3 hash of its decoded content.
    pub const CHUNK_CHECKSUM: u32 = 1 << 2;

    /// All bits this version defines.
    pub const KNOWN: u32 = Self::CONTENT_SIZE | Self::CONTENT_CHECKSUM | Self::CHUNK_CHECKSUM;

    #[inline]
    pub fn bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn contains(self, f: u32) -> bool {
        self.0 & f == f
    }

    /// Build from raw bits, rejecting bits this major version does not define.
    pub fn from_bits_strict(bits: u32) -> Result<Self> {
        if bits & !Self::KNOWN != 0 {
            return Err(Error::InvalidHeader("unknown container flag bits set"));
        }
        Ok(Self(bits))
    }
}

/// Parsed container header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContainerHeader {
    pub version_major: u8,
    pub version_minor: u8,
    pub flags: ContainerFlags,
    /// Target number of *uncompressed* bytes per frame.
    pub chunk_size: u32,
    /// Largest match distance the encoder may emit. A decoder may clamp this
    /// down to the frame's own content size.
    pub window_size: u32,
    /// Total uncompressed size, or `None` when unknown (streaming input).
    pub content_size: Option<u64>,
}

impl ContainerHeader {
    /// Serialise into `w`. `header_len` is emitted as the current length, which
    /// is the only legal choice for this version.
    pub fn encode(&self, w: &mut ByteWriter) {
        debug_assert!(
            w.is_empty(),
            "header encoding must start from an empty writer"
        );
        w.bytes(&MAGIC);
        w.u8(self.version_major);
        w.u8(self.version_minor);
        w.u16le(HEADER_LEN_V1 as u16);
        w.u32le(self.flags.bits());
        w.u32le(self.chunk_size);
        w.u32le(self.window_size);
        w.u64le(self.content_size.unwrap_or(0));
    }

    /// Total encoded length of a v1 header.
    pub const fn encoded_len() -> usize {
        HEADER_LEN_V1
    }

    /// Parse a v1 header from `data`, which must contain exactly the header.
    pub fn decode(data: &[u8]) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let magic = r.take(MAGIC_LEN)?;
        if magic != MAGIC {
            return Err(Error::BadMagic {
                found: [magic[0], magic[1], magic[2], magic[3]],
            });
        }
        let version_major = r.u8()?;
        let version_minor = r.u8()?;
        if u16::from(version_major) > MAX_SUPPORTED_MAJOR {
            return Err(Error::UnsupportedVersion {
                major: u16::from(version_major),
                minor: u16::from(version_minor),
                max_supported: MAX_SUPPORTED_MAJOR,
            });
        }
        let header_len = usize::from(r.u16le()?);
        if !(crate::format::CONTAINER_PREAMBLE_LEN..=crate::format::MAX_CONTAINER_HEADER_LEN)
            .contains(&header_len)
        {
            return Err(Error::FieldOutOfRange {
                field: "header_len",
                value: header_len as u64,
            });
        }
        if data.len() < header_len {
            return Err(Error::UnexpectedEof {
                needed: header_len,
                available: data.len(),
            });
        }
        // A newer *minor* version may append fields this build does not know.
        // `header_len` tells us where they end, and the range check above proved
        // the whole header is present, so those trailing bytes are skipped
        // rather than treated as corruption. No action needed here.
        let flags = ContainerFlags::from_bits_strict(r.u32le()?)?;
        let chunk_size = r.u32le()?;
        let window_size = r.u32le()?;
        let content_size = r.u64le()?;

        if chunk_size == 0 {
            return Err(Error::FieldOutOfRange {
                field: "chunk_size",
                value: 0,
            });
        }
        if window_size == 0 {
            return Err(Error::FieldOutOfRange {
                field: "window_size",
                value: 0,
            });
        }
        if window_size > chunk_size {
            // Frames never exceed chunk_size, so a larger window is pointless
            // and would waste decoder memory.
            return Err(Error::FieldOutOfRange {
                field: "window_size",
                value: u64::from(window_size),
            });
        }
        let content_size = if flags.contains(ContainerFlags::CONTENT_SIZE) {
            Some(content_size)
        } else {
            if content_size != 0 {
                return Err(Error::InvalidHeader(
                    "content_size set without CONTENT_SIZE flag",
                ));
            }
            None
        };

        Ok(Self {
            version_major,
            version_minor,
            flags,
            chunk_size,
            window_size,
            content_size,
        })
    }
}

/// Encoded length of the fixed v1 container header.
pub const HEADER_LEN_V1: usize = crate::format::CONTAINER_PREAMBLE_LEN + 4 + 4 + 4 + 8;

/// End-of-stream marker magic: `END 1A`.
pub const END_MAGIC: [u8; 4] = *b"END\x1a";
pub const END_MAGIC_LEN: usize = 4;

/// Parsed end marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndMarker {
    /// Total decoded content size as recorded by the encoder.
    pub content_size: u64,
    /// Present when the container header sets `CONTENT_CHECKSUM`.
    pub content_hash: Option<[u8; crate::format::HASH_LEN]>,
}

impl EndMarker {
    pub fn encode(&self, w: &mut ByteWriter) {
        w.bytes(&END_MAGIC);
        w.u64le(self.content_size);
        if let Some(h) = self.content_hash {
            w.bytes(&h);
        }
    }

    pub fn decode(data: &[u8], expect_hash: bool) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let m = r.take(END_MAGIC_LEN)?;
        if m != END_MAGIC {
            return Err(Error::InvalidHeader("missing end-of-stream marker"));
        }
        Self::decode_tail(r.take(r.remaining())?, expect_hash)
    }

    /// Parse an end marker whose magic has already been consumed.
    ///
    /// Splitting this out lets a streaming reader dispatch on the 4 magic bytes
    /// it must read to tell a frame from the end marker, without then reading
    /// those same bytes a second time.
    pub fn decode_tail(data: &[u8], expect_hash: bool) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let content_size = r.u64le()?;
        let content_hash = if expect_hash {
            let h = r.take(crate::format::HASH_LEN)?;
            let mut out = [0u8; crate::format::HASH_LEN];
            out.copy_from_slice(h);
            Some(out)
        } else {
            None
        };
        Ok(Self {
            content_size,
            content_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ContainerHeader {
        ContainerHeader {
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            flags: ContainerFlags(
                ContainerFlags::CONTENT_SIZE
                    | ContainerFlags::CONTENT_CHECKSUM
                    | ContainerFlags::CHUNK_CHECKSUM,
            ),
            chunk_size: 1 << 20,
            window_size: 1 << 16,
            content_size: Some(12345),
        }
    }

    #[test]
    fn header_roundtrip() {
        let h = sample();
        let mut w = ByteWriter::new();
        h.encode(&mut w);
        let buf = w.into_vec();
        assert_eq!(buf.len(), ContainerHeader::encoded_len());
        assert_eq!(ContainerHeader::decode(&buf).unwrap(), h);
    }

    #[test]
    fn header_without_content_size() {
        let mut h = sample();
        h.flags = ContainerFlags(ContainerFlags::CONTENT_CHECKSUM);
        h.content_size = None;
        let mut w = ByteWriter::new();
        h.encode(&mut w);
        assert_eq!(ContainerHeader::decode(w.as_slice()).unwrap(), h);
    }

    #[test]
    fn bad_magic_detected() {
        let mut w = ByteWriter::new();
        sample().encode(&mut w);
        let mut buf = w.into_vec();
        buf[1] = b'X';
        assert!(matches!(
            ContainerHeader::decode(&buf),
            Err(Error::BadMagic { .. })
        ));
    }

    #[test]
    fn future_major_version_rejected() {
        let mut w = ByteWriter::new();
        sample().encode(&mut w);
        let mut buf = w.into_vec();
        buf[MAGIC_LEN] = 9;
        assert!(matches!(
            ContainerHeader::decode(&buf),
            Err(Error::UnsupportedVersion { major: 9, .. })
        ));
    }

    #[test]
    fn unknown_flag_bits_rejected() {
        let mut buf = ByteWriter::new();
        sample().encode(&mut buf);
        let mut v = buf.into_vec();
        // flags field starts right after the 8-byte preamble
        v[crate::format::CONTAINER_PREAMBLE_LEN] |= 0x80;
        assert!(matches!(
            ContainerHeader::decode(&v),
            Err(Error::InvalidHeader(_))
        ));
    }

    #[test]
    fn window_larger_than_chunk_rejected() {
        let mut h = sample();
        h.window_size = h.chunk_size + 1;
        let mut w = ByteWriter::new();
        h.encode(&mut w);
        assert!(ContainerHeader::decode(w.as_slice()).is_err());
    }

    #[test]
    fn end_marker_roundtrip() {
        let em = EndMarker {
            content_size: 42,
            content_hash: Some([7u8; 32]),
        };
        let mut w = ByteWriter::new();
        em.encode(&mut w);
        assert_eq!(EndMarker::decode(w.as_slice(), true).unwrap(), em);
        assert_eq!(w.len(), crate::format::END_MARKER_FIXED_LEN + 32);
    }
}
