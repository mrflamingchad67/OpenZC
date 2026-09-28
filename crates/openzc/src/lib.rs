//! OpenZ Compressor — a modular, lossless, streamable compression engine.
//!
//! # Overview
//!
//! OpenZC is a standalone compression engine. It has no dependency on any
//! archive format, GUI, or CLI; every consumer drives it through the same
//! public API. The `openzc-cli` binary in this workspace is a thin front-end
//! over exactly the API documented here.
//!
//! The invariant the whole crate exists to uphold is:
//!
//! ```text
//! decompress(compress(input)) == input      (byte for byte)
//! ```
//!
//! # Quick start
//!
//! ```
//! use openzc::{Config, Level, OpenZc};
//!
//! let openzc = OpenZc::new(Config::new(Level::Default))?;
//!
//! let mut compressed = Vec::new();
//! openzc.compress(&b"hello hello hello world"[..], &mut compressed)?;
//!
//! let mut roundtrip = Vec::new();
//! openzc.decompress(compressed.as_slice(), &mut roundtrip)?;
//!
//! assert_eq!(roundtrip, b"hello hello hello world");
//! # Ok::<(), openzc::Error>(())
//! ```
//!
//! # Module map
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`config`] | Levels, strategies, window sizing, integrity policy |
//! | [`format`] | Versioned container and frame headers (the on-disk contract) |
//! | [`chunk`] | Bounded-memory input splitting |
//! | [`analyze`] | Cheap per-chunk characterisation |
//! | [`matchfinder`] | LZ match discovery |
//! | [`transform`] | Reversible pre-LZ transforms |
//! | [`entropy`] | Range coder and models |
//! | [`codec`] | Per-pipeline encoders and decoders |
//! | [`pipeline`] | Per-chunk strategy selection |
//! | [`integrity`] | BLAKE3 verification |
//! | [`stream`] | Streaming reader/writer over `Read`/`Write` |
//!
//! # Safety and robustness
//!
//! The crate contains no `unsafe` code. Decoders are fully bounds checked, so
//! a damaged stream produces an [`Error`] rather than a panic or an
//! out-of-bounds read. Integrity hashes are the final authority on whether
//! output is trustworthy.

#![deny(unsafe_code)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

pub mod analyze;
pub mod bytes;
pub mod chunk;
pub mod codec;
pub mod config;
pub mod entropy;
pub mod error;
pub mod format;
pub mod integrity;
pub mod matchfinder;
pub mod pipeline;
pub mod stream;
pub mod transform;

mod compress;
mod decompress;

pub use compress::{
    analyse, compress, compress_at_level, compress_slice, predict_pipeline, store, will_compress,
    CompressStats, Compressor, CountingWriter,
};
pub use config::{Checksum, Config, Level, Strategy, Threads, WindowConfig};
pub use decompress::{
    decompress, decompress_slice, transforms_used, DecompressStats, Decompressor, StreamFlags,
};
pub use error::{Error, ErrorKind, Result};
pub use format::{
    ContainerFlags, ContainerHeader, FrameFlags, FrameHeader, PipelineId, TransformId,
};

use std::io::{Read, Write};

/// Format version this build writes.
pub const FORMAT_VERSION: (u8, u8) = (format::VERSION_MAJOR, format::VERSION_MINOR);

/// A configured compressor and decompressor.
///
/// The type is cheap to clone and holds no I/O state, so one instance can be
/// reused for many streams and shared across threads. All actual work happens
/// in [`Compressor`] and [`Decompressor`], which own the stream position.
///
/// ```no_run
/// use std::fs::File;
/// use openzc::{Config, Level, OpenZc};
///
/// let openzc = OpenZc::new(Config::new(Level::High))?;
/// let src = File::open("input.bin")?;
/// let mut dst = File::create("input.ozc")?;
/// let stats = openzc.compress(src, &mut dst)?;
/// println!("{} -> {} bytes", stats.input_size, stats.output_size);
/// # Ok::<(), openzc::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct OpenZc {
    config: Config,
}

impl OpenZc {
    /// Create an engine from a configuration.
    ///
    /// The configuration is validated and normalised here, so a successful
    /// construction means every later operation has usable settings.
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config: config.validated()?,
        })
    }

    /// Build an engine for a compression level, leaving everything else at
    /// its default.
    pub fn with_level(level: Level) -> Result<Self> {
        Self::new(Config::new(level))
    }

    /// The validated configuration in use.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Compress everything from `src` into `dst`.
    ///
    /// Input size need not be known in advance: the reader is drained and the
    /// stream is finalised with `dst` flushed but not closed.
    pub fn compress<R: Read, W: Write>(&self, src: R, dst: W) -> Result<CompressStats> {
        compress::compress(src, dst, &self.config)
    }

    /// Decompress everything from `src` into `dst`.
    pub fn decompress<R: Read, W: Write>(&self, src: R, dst: &mut W) -> Result<DecompressStats> {
        Decompressor::with_reader(src, &self.config)?.copy_to(dst)
    }
}

/// Compress an in-memory buffer, returning the OpenZC stream.
///
/// Convenience wrapper for callers that already hold the whole input.
pub fn compress_to_vec(config: &Config, input: &[u8]) -> Result<Vec<u8>> {
    let openzc = OpenZc::new(config.clone())?;
    let mut out = Vec::new();
    openzc.compress(input, &mut out)?;
    Ok(out)
}

/// Decompress an OpenZC stream held in memory.
pub fn decompress_to_vec(config: &Config, input: &[u8]) -> Result<Vec<u8>> {
    let openzc = OpenZc::new(config.clone())?;
    let mut out = Vec::new();
    openzc.decompress(input, &mut out)?;
    Ok(out)
}
