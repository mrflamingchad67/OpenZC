//! Error types for the OpenZC engine.
//!
//! Every failure mode is represented explicitly so that a decoder can fail
//! cleanly on damaged input rather than silently producing wrong output.

use std::fmt;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A categorised OpenZC failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The leading magic bytes were not an OpenZC stream.
    BadMagic { found: [u8; 4] },
    /// The stream declared a format version this build cannot decode.
    UnsupportedVersion {
        major: u16,
        minor: u16,
        max_supported: u16,
    },
    /// A header field was structurally invalid.
    InvalidHeader(&'static str),
    /// A structural field held a value outside its permitted range.
    FieldOutOfRange { field: &'static str, value: u64 },
    /// The stream ended in the middle of a structure.
    UnexpectedEof { needed: usize, available: usize },
    /// The end-of-stream marker was never reached.
    MissingEndMarker,
    /// A chunk referenced content beyond what the frame header permits.
    CorruptChunk(&'static str),
    /// A referenced offset, length, or distance was invalid for the data.
    CorruptSequence(&'static str),
    /// BLAKE3 content verification failed.
    ChecksumMismatch { what: &'static str },
    /// A dictionary id was requested that the stream does not define.
    UnknownDictionary(u32),
    /// The decoder does not implement the pipeline id recorded in the frame.
    UnknownPipeline(u32),
    /// The frame mode recorded in the frame header is not supported.
    UnknownFrameMode(u8),
    /// A transform id recorded in the frame header is not supported.
    UnknownTransform(u8),
    /// The entropy model in the chunk payload is malformed.
    CorruptEntropyModel(&'static str),
    /// The input did not compress; the caller asked for strict mode.
    NotCompressible { original: u64, compressed: u64 },
    /// Wrapped I/O failure.
    Io(std::io::Error),
    /// Configuration was rejected.
    Config(String),
    /// Requested feature is not compiled in.
    Unsupported(&'static str),
}

impl Error {
    /// A short machine-friendly classification, useful for tests and for the
    /// CLI to pick an exit code and a message.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Error::BadMagic { .. } => ErrorKind::BadMagic,
            Error::UnsupportedVersion { .. } => ErrorKind::UnsupportedVersion,
            Error::InvalidHeader(_) => ErrorKind::InvalidHeader,
            Error::FieldOutOfRange { .. } => ErrorKind::InvalidHeader,
            Error::UnexpectedEof { .. } => ErrorKind::Truncated,
            Error::MissingEndMarker => ErrorKind::Truncated,
            Error::CorruptChunk(_) => ErrorKind::Corrupt,
            Error::CorruptSequence(_) => ErrorKind::Corrupt,
            Error::ChecksumMismatch { .. } => ErrorKind::Checksum,
            Error::UnknownDictionary(_) => ErrorKind::InvalidHeader,
            Error::UnknownPipeline(_) => ErrorKind::InvalidHeader,
            Error::UnknownFrameMode(_) => ErrorKind::InvalidHeader,
            Error::UnknownTransform(_) => ErrorKind::InvalidHeader,
            Error::CorruptEntropyModel(_) => ErrorKind::Corrupt,
            Error::NotCompressible { .. } => ErrorKind::NotCompressible,
            Error::Io(_) => ErrorKind::Io,
            Error::Config(_) | Error::Unsupported(_) => ErrorKind::Config,
        }
    }

    /// True when the error indicates the input stream was damaged or is not a
    /// valid OpenZC stream, as opposed to a local I/O or configuration problem.
    pub fn is_data_error(&self) -> bool {
        !matches!(self.kind(), ErrorKind::Io | ErrorKind::Config)
    }
}

/// Stable classification of [`Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    BadMagic,
    UnsupportedVersion,
    InvalidHeader,
    Truncated,
    Corrupt,
    Checksum,
    NotCompressible,
    Io,
    Config,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic { found } => write!(
                f,
                "not an OpenZC stream: bad magic {:02x?}, expected {:02x?}",
                found, crate::format::MAGIC
            ),
            Error::UnsupportedVersion { major, minor, max_supported } => write!(
                f,
                "unsupported OpenZC format version {major}.{minor} (this build decodes up to {max_supported})"
            ),
            Error::InvalidHeader(why) => write!(f, "invalid stream header: {why}"),
            Error::FieldOutOfRange { field, value } => {
                write!(f, "header field {field} out of range: {value}")
            }
            Error::UnexpectedEof { needed, available } => {
                write!(f, "unexpected end of stream: needed {needed} byte(s), {available} available")
            }
            Error::MissingEndMarker => write!(f, "stream ended without a valid end-of-stream marker"),
            Error::CorruptChunk(why) => write!(f, "corrupt chunk: {why}"),
            Error::CorruptSequence(why) => write!(f, "corrupt LZ sequence: {why}"),
            Error::ChecksumMismatch { what } => write!(f, "BLAKE3 integrity check failed for {what}"),
            Error::UnknownDictionary(id) => write!(f, "unknown dictionary id {id}"),
            Error::UnknownPipeline(id) => write!(f, "unknown pipeline id {id}"),
            Error::UnknownFrameMode(m) => write!(f, "unknown frame mode {m}"),
            Error::UnknownTransform(t) => write!(f, "unknown transform id {t}"),
            Error::CorruptEntropyModel(why) => write!(f, "corrupt entropy model: {why}"),
            Error::NotCompressible { original, compressed } => write!(
                f,
                "data did not compress ({original} -> {compressed} bytes) and strict mode is enabled"
            ),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Config(m) => write!(f, "invalid configuration: {m}"),
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        // Unwrap our own errors first. Anything that went through
        // `From<Error> for io::Error` — which is every error crossing the
        // `std::io::Read` boundary, so essentially every streaming path — arrives
        // here boxed inside an `io::Error`. Recovering it is what lets a caller
        // match on `UnknownPipeline` or `ChecksumMismatch` instead of seeing an
        // opaque I/O failure. The box is moved out rather than cloned, because
        // `io::Error` is not `Clone` and neither is this enum.
        if e.get_ref().is_some_and(|inner| inner.is::<Error>()) {
            // `get_ref` said the payload is one of ours, so the downcast cannot
            // fail; if it somehow did, reporting a plain I/O error is still honest.
            if let Some(ours) = e.into_inner().and_then(|b| b.downcast::<Error>().ok()) {
                return *ours;
            }
            return Error::Io(std::io::Error::other(
                "error payload could not be recovered",
            ));
        }

        // An unexpected EOF from the underlying reader is a truncated stream,
        // not a generic I/O failure. This distinction matters because callers
        // (OpenZE in particular) want to report damage accurately.
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::UnexpectedEof {
                needed: 0,
                available: 0,
            }
        } else {
            Error::Io(e)
        }
    }
}

impl From<Error> for std::io::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Io(io) => io,
            other => std::io::Error::new(std::io::ErrorKind::InvalidData, other),
        }
    }
}
