//! Decompression entry point.
//!
//! [`Decompressor`] reads frames, verifies each against its BLAKE3 hash, decodes
//! it, and reverses any transform. It implements [`std::io::Read`], so it drops
//! into any streaming consumer without buffering the whole output.

use crate::codec::DictionaryTable;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::format::{ContainerFlags, PipelineId, TransformId, HASH_LEN};
use crate::integrity;
use crate::pipeline::Planner;
use crate::stream::{SliceReader, StreamReader};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::time::Instant;

/// What a decompression run did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompressStats {
    /// Uncompressed bytes produced.
    pub output_size: u64,
    /// Compressed bytes consumed, including headers.
    pub input_size: u64,
    /// Frames decoded.
    pub frames: u64,
    /// How many frames used each pipeline.
    pub pipelines: BTreeMap<PipelineId, u64>,
    /// Wall-clock time spent decompressing.
    pub elapsed: std::time::Duration,
    /// True when every frame's content hash was verified.
    pub verified: bool,
}

impl DecompressStats {
    /// Uncompressed megabytes per second.
    pub fn throughput_mbs(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        (self.output_size as f64 / (1024.0 * 1024.0)) / secs
    }
}

/// Streaming decompressor over any [`Read`] input.
#[derive(Debug)]
pub struct Decompressor<R: Read> {
    reader: StreamReader<R>,
    config: Config,
    planner: Planner,
    dictionaries: DictionaryTable,
    started: Instant,
    /// Decoded-but-unread output from the current frame.
    pending: Vec<u8>,
    pending_pos: usize,
    content_hash: Option<integrity::ContentHasher>,
    /// Compressed bytes consumed, including headers.
    input_size: u64,
    output_size: u64,
    pipelines: BTreeMap<PipelineId, u64>,
    frames: u64,
    ended: bool,
    verified: bool,
    expect_hash: bool,
    max_input: Option<u64>,
}

impl<R: Read> Decompressor<R> {
    /// Build a decompressor and read the container header.
    pub fn new(reader: R, config: &Config) -> Result<Self> {
        let config = config.validated()?;
        let expect_hash = config.checksum().chunk_hashes();
        let max_input = config.max_input_size;
        let content_hash = if config.checksum().content_hash() {
            Some(integrity::ContentHasher::new())
        } else {
            None
        };
        // The container header has a fixed length, so input accounting can start
        // without reading it back.
        let reader = StreamReader::new(reader, &config)?;
        let planner = Planner::new(&config);
        Ok(Self {
            planner,
            reader,
            config,
            dictionaries: DictionaryTable::new(),
            started: Instant::now(),
            pending: Vec::new(),
            pending_pos: 0,
            content_hash,
            input_size: crate::format::ContainerHeader::encoded_len() as u64,
            output_size: 0,
            pipelines: BTreeMap::new(),
            frames: 0,
            ended: false,
            verified: false,
            expect_hash,
            max_input,
        })
    }

    /// Build a decompressor, reading the container header immediately.
    pub fn with_reader(reader: R, config: &Config) -> Result<Self> {
        Self::new(reader, config)
    }

    /// The configuration in use.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Decode the next frame into the internal buffer, or return false at end.
    fn advance(&mut self) -> Result<bool> {
        if self.ended {
            return Ok(false);
        }
        match self.reader.next_frame()? {
            None => {
                self.ended = true;
                self.finish_checks()?;
                Ok(false)
            }
            Some(frame) => {
                let header = frame.header;
                if header.pipeline == PipelineId::Dictionary {
                    // Dictionary frames are decoded and registered, then
                    // consumed: they carry no user-visible content.
                    let content = self.planner.decode_payload(
                        header.pipeline,
                        header.transform,
                        &frame.payload,
                        header.content_size as usize,
                        &self.dictionaries,
                        0,
                    )?;
                    self.verify_frame_hash(&content, frame.content_hash)?;
                    let id = self.dictionaries.len() as u8 + 1;
                    self.dictionaries.insert(id, content);
                    return self.advance();
                }

                let content = self.planner.decode_payload(
                    header.pipeline,
                    header.transform,
                    &frame.payload,
                    header.content_size as usize,
                    &self.dictionaries,
                    header.dict_id,
                )?;
                self.verify_frame_hash(&content, frame.content_hash)?;
                // Account for the frame's on-wire bytes: fixed header, payload,
                // and the content hash when the policy writes one.
                self.input_size += frame.header.frame_len(self.expect_hash) as u64;

                if let Some(h) = self.content_hash.as_mut() {
                    h.update(&content);
                }
                self.output_size += content.len() as u64;
                *self.pipelines.entry(header.pipeline).or_insert(0) += 1;
                self.frames += 1;

                self.pending_pos = 0;
                self.pending = content;
                Ok(true)
            }
        }
    }

    /// Verify a frame's content hash when the policy calls for one.
    fn verify_frame_hash(
        &mut self,
        content: &[u8],
        expected: Option<[u8; HASH_LEN]>,
    ) -> Result<()> {
        if !self.expect_hash {
            return Ok(());
        }
        let expected = expected.ok_or(Error::CorruptChunk("frame is missing its content hash"))?;
        if integrity::hash(content) != expected {
            return Err(Error::ChecksumMismatch {
                what: "frame content",
            });
        }
        self.verified = true;
        Ok(())
    }

    /// Validate the end-of-stream state.
    fn finish_checks(&mut self) -> Result<()> {
        self.reader.ensure_finished()?;
        if let Some(max) = self.max_input {
            if self.output_size > max {
                return Err(Error::Config(format!(
                    "decompressed size {} exceeds the configured limit of {max}",
                    self.output_size
                )));
            }
        }
        if let Some(computed) = self
            .content_hash
            .as_ref()
            .map(integrity::ContentHasher::finalize)
        {
            self.reader.verify_content_hash(&computed)?;
        }
        if let Some(declared) = self.reader.content_size() {
            if declared != self.output_size {
                return Err(Error::InvalidHeader(
                    "decoded size disagrees with the end marker",
                ));
            }
        }
        Ok(())
    }

    /// Drain the decompressor into `dst`.
    pub fn copy_to<W: Write>(mut self, mut dst: W) -> Result<DecompressStats> {
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = self.read(&mut buf)?;
            if n == 0 {
                break;
            }
            dst.write_all(&buf[..n])?;
        }
        Ok(self.stats())
    }

    /// Statistics for a completed run.
    pub fn stats(&self) -> DecompressStats {
        DecompressStats {
            output_size: self.output_size,
            input_size: self.input_size,
            frames: self.frames,
            pipelines: self.pipelines.clone(),
            elapsed: self.started.elapsed(),
            verified: self.verified,
        }
    }

    /// Force the remaining stream to be read and validated.
    ///
    /// Useful when a caller only wants a prefix, but still wants the end marker
    /// and whole-content hash checked.
    pub fn finish(&mut self) -> Result<DecompressStats> {
        while self.advance()? {}
        Ok(self.stats())
    }
}

impl<'a> Decompressor<SliceReader<'a>> {
    /// Build a decompressor over an in-memory slice.
    pub fn from_slice(data: &'a [u8], config: &Config) -> Result<Self> {
        Self::new(SliceReader::new(data), config)
    }
}

impl<R: Read> Read for Decompressor<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        loop {
            if self.pending_pos < self.pending.len() {
                let n = out.len().min(self.pending.len() - self.pending_pos);
                out[..n].copy_from_slice(&self.pending[self.pending_pos..self.pending_pos + n]);
                self.pending_pos += n;
                return Ok(n);
            }
            match self.advance() {
                Ok(true) => continue,
                Ok(false) => return Ok(0),
                Err(e) => return Err(e.into()),
            }
        }
    }
}

/// Decompress a stream from a reader into a writer, reporting statistics.
pub fn decompress<R: Read, W: Write>(src: R, dst: W, config: &Config) -> Result<DecompressStats> {
    Decompressor::new(src, config)?.copy_to(dst)
}

/// Decompress an in-memory stream.
pub fn decompress_slice(input: &[u8], config: &Config) -> Result<(Vec<u8>, DecompressStats)> {
    let mut out = Vec::new();
    let d = Decompressor::from_slice(input, config)?;
    let stats = d.copy_to(&mut out)?;
    Ok((out, stats))
}

/// Header flags of a stream, for `info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamFlags {
    pub content_hashed: bool,
    pub chunk_hashed: bool,
}

impl StreamFlags {
    /// Read a stream's flags without decoding any content.
    pub fn of(input: &[u8], config: &Config) -> Result<Self> {
        let config = config.validated()?;
        let r = StreamReader::new(SliceReader::new(input), &config)?;
        let f = r.header().flags;
        Ok(Self {
            content_hashed: f.contains(ContainerFlags::CONTENT_CHECKSUM),
            chunk_hashed: f.contains(ContainerFlags::CHUNK_CHECKSUM),
        })
    }
}

/// Transform ids present in a stream, for `info`.
pub fn transforms_used(input: &[u8], config: &Config) -> Result<Vec<TransformId>> {
    let config = config.validated()?;
    let mut r = StreamReader::new(SliceReader::new(input), &config)?;
    let mut out = Vec::new();
    while let Some(f) = r.next_frame()? {
        if !out.contains(&f.header.transform) {
            out.push(f.header.transform);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::compress_slice;
    use crate::config::{Checksum, Level, WindowConfig};

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

    fn roundtrip(data: &[u8], cfg: &Config) {
        let (compressed, _) = compress_slice(data, cfg).unwrap();
        let (decoded, stats) = decompress_slice(&compressed, cfg).unwrap();
        assert_eq!(decoded, data, "roundtrip failed for {} bytes", data.len());
        assert_eq!(stats.output_size, data.len() as u64);
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(b"", &Config::default());
    }

    #[test]
    fn roundtrip_single_byte() {
        roundtrip(b"z", &Config::default());
    }

    #[test]
    fn roundtrip_every_level() {
        let data = "hello world ".repeat(1000).into_bytes();
        for level in Level::ALL {
            roundtrip(&data, &Config::new(level));
        }
    }

    #[test]
    fn roundtrip_multiple_chunks() {
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 97) as u8).collect();
        let cfg = Config::default().with_window(WindowConfig {
            chunk_size: 8192,
            window_size: 4096,
        });
        roundtrip(&data, &cfg);
        let (_, stats) = compress_slice(&data, &cfg).unwrap();
        assert!(stats.frames > 1, "expected several frames");
    }

    #[test]
    fn checksums_are_verified() {
        let data = "hello world ".repeat(500).into_bytes();
        let cfg = Config::default();
        let (compressed, _) = compress_slice(&data, &cfg).unwrap();
        let (_, stats) = decompress_slice(&compressed, &cfg).unwrap();
        assert!(stats.verified);
    }

    #[test]
    fn detects_corrupted_payload() {
        let data = "hello world ".repeat(2000).into_bytes();
        let cfg = Config::default();
        let (mut compressed, _) = compress_slice(&data, &cfg).unwrap();
        // Flip a byte in the middle of the stream.
        let mid = compressed.len() / 2;
        compressed[mid] ^= 0xFF;
        let r = decompress_slice(&compressed, &cfg);
        assert!(r.is_err(), "corruption must be detected");
    }

    #[test]
    fn detects_truncated_stream() {
        let data = "hello world ".repeat(2000).into_bytes();
        let cfg = Config::default();
        let (compressed, _) = compress_slice(&data, &cfg).unwrap();
        for cut in [1usize, compressed.len() / 2, compressed.len() - 1] {
            let r = decompress_slice(&compressed[..cut], &cfg);
            assert!(r.is_err(), "truncation at {cut} must be detected");
        }
    }

    #[test]
    fn detects_invalid_magic() {
        let mut data = b"NOTANOPENZCSTREAMATALL__".to_vec();
        data[0] = b'Z';
        assert!(matches!(
            decompress_slice(&data, &Config::default()),
            Err(Error::BadMagic { .. })
        ));
    }

    #[test]
    fn detects_unsupported_version() {
        let data = "hello ".repeat(100).into_bytes();
        let (mut compressed, _) = compress_slice(&data, &Config::default()).unwrap();
        // Byte 4 is version_major.
        compressed[4] = 99;
        assert!(matches!(
            decompress_slice(&compressed, &Config::default()),
            Err(Error::UnsupportedVersion { major: 99, .. })
        ));
    }

    #[test]
    fn detects_invalid_pipeline_id() {
        let data = "hello ".repeat(100).into_bytes();
        let (mut compressed, _) = compress_slice(&data, &Config::default()).unwrap();
        // Find the first frame and corrupt its pipeline id.
        let hdr_len = crate::format::ContainerHeader::encoded_len();
        let pos = hdr_len + crate::format::frame::FRAME_MAGIC_LEN;
        compressed[pos] = 200;
        assert!(matches!(
            decompress_slice(&compressed, &Config::default()),
            Err(Error::UnknownPipeline(200))
        ));
    }

    #[test]
    fn detects_missing_end_marker() {
        let data = "hello ".repeat(100).into_bytes();
        let (compressed, _) = compress_slice(&data, &Config::default()).unwrap();
        let cut = compressed.len() - crate::format::END_MARKER_FIXED_LEN - HASH_LEN;
        assert!(decompress_slice(&compressed[..cut], &Config::default()).is_err());
    }

    #[test]
    fn input_size_limit_is_enforced() {
        let data = "hello ".repeat(1000).into_bytes();
        let (compressed, _) = compress_slice(&data, &Config::default()).unwrap();
        let cfg = Config::default().with_max_input_size(10);
        let r = decompress_slice(&compressed, &cfg);
        assert!(matches!(r, Err(Error::Config(_))));
    }

    #[test]
    fn decompression_is_streaming() {
        // Reading a few bytes must work without decoding the whole stream.
        let data = "hello world ".repeat(10_000).into_bytes();
        let (compressed, _) = compress_slice(&data, &Config::default()).unwrap();
        let mut d = Decompressor::from_slice(&compressed, &Config::default()).unwrap();
        let mut buf = [0u8; 16];
        let n = d.read(&mut buf).unwrap();
        assert_eq!(n, 16);
        assert_eq!(&buf[..], &data[..16]);
    }

    #[test]
    fn stream_flags_readable() {
        let data = "hello ".repeat(100).into_bytes();
        let cfg = Config::default();
        let (compressed, _) = compress_slice(&data, &cfg).unwrap();
        let f = StreamFlags::of(&compressed, &cfg).unwrap();
        assert!(f.content_hashed);
        assert!(f.chunk_hashed);

        let (c2, _) =
            compress_slice(&data, &Config::default().with_checksum(Checksum::None)).unwrap();
        let f2 = StreamFlags::of(&c2, &Config::default()).unwrap();
        assert!(!f2.content_hashed);
        assert!(!f2.chunk_hashed);
    }

    #[test]
    fn transforms_listed() {
        let data = "hello ".repeat(100).into_bytes();
        let (compressed, _) = compress_slice(&data, &Config::default()).unwrap();
        let t = transforms_used(&compressed, &Config::default()).unwrap();
        assert!(t.contains(&TransformId::None));
    }

    #[test]
    fn random_input_roundtrips() {
        roundtrip(&pseudo_random(30_000, 7), &Config::default());
    }

    #[test]
    fn incompressible_input_roundtrips() {
        roundtrip(&pseudo_random(100_000, 31337), &Config::default());
    }
}
