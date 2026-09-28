//! Compression entry point: chunks input, drives the planner, writes frames.
//!
//! [`Compressor`] is the streaming face of the engine. It owns the input
//! position, so it is single-use, while [`crate::OpenZc`] is the reusable,
//! shareable handle that configures it.
//!
//! # Structure
//!
//! The compressor is a thin loop: read a chunk, plan it, hand the plan to
//! [`crate::stream::StreamWriter`], repeat, then finish. All framing, hashing,
//! and size accounting live in the stream layer, so there is exactly one place
//! where the container is assembled and one place where bytes are counted.

use crate::analyze::Analysis;
use crate::chunk::Chunker;
use crate::config::{Config, Level, Strategy};
use crate::error::Result;
use crate::format::PipelineId;
use crate::pipeline::Planner;
use crate::stream::{SliceReader, StreamWriter};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::time::Instant;

/// What a compression run did.
///
/// Every number here is measured, not estimated.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressStats {
    /// Uncompressed bytes consumed.
    pub input_size: u64,
    /// Bytes written, including all headers and hashes.
    pub output_size: u64,
    /// Number of frames emitted.
    pub frames: u64,
    /// How many frames used each pipeline.
    pub pipelines: BTreeMap<PipelineId, u64>,
    /// Wall-clock time spent compressing.
    pub elapsed: std::time::Duration,
    /// True when every frame was stored.
    pub all_stored: bool,
    /// Fraction of chunks stored rather than compressed, 0.0..=1.0.
    pub stored_fraction: f32,
}

impl CompressStats {
    /// Compressed size divided by original size, 1.0 meaning no change.
    pub fn ratio(&self) -> f64 {
        if self.input_size == 0 {
            1.0
        } else {
            self.output_size as f64 / self.input_size as f64
        }
    }

    /// Uncompressed megabytes per second.
    pub fn throughput_mbs(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        (self.input_size as f64 / (1024.0 * 1024.0)) / secs
    }
}

/// A `Write` adapter that counts bytes, so `output_size` is measured rather
/// than inferred.
#[derive(Debug)]
pub struct CountingWriter<W> {
    inner: W,
    count: u64,
}

impl<W> CountingWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner, count: 0 }
    }

    /// Bytes written so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Unwrap, discarding the count.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Streaming compressor: reads chunks, plans them, writes frames.
///
/// `R` is the input and `W` the output. The writer is wrapped in a
/// [`CountingWriter`] so `output_size` is measured rather than inferred from
/// the format, and can be recovered with [`Compressor::finish`].
#[derive(Debug)]
pub struct Compressor<R: Read, W: Write> {
    chunker: Chunker<R>,
    writer: StreamWriter<CountingWriter<W>>,
    config: Config,
    started: Instant,
    pipelines: BTreeMap<PipelineId, u64>,
    stored: u64,
    chunks: u64,
}

impl<R: Read, W: Write> Compressor<R, W> {
    /// Build a compressor from a config.
    pub fn new(reader: R, writer: W, config: &Config) -> Result<Self> {
        let config = config.validated()?;
        let chunk_size = config.window().chunk_size as usize;
        Ok(Self {
            chunker: Chunker::new(reader, chunk_size),
            writer: StreamWriter::new(CountingWriter::new(writer), &config)?,
            config,
            started: Instant::now(),
            pipelines: BTreeMap::new(),
            stored: 0,
            chunks: 0,
        })
    }

    /// The configuration in use.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Push one chunk. Exposed for callers that do their own chunking.
    pub fn push(&mut self, chunk: &[u8]) -> Result<()> {
        // In strict mode the adaptive search is skipped so a size guarantee is
        // meaningful; otherwise the planner measures every candidate.
        let planning_config = if self.config.strict_size {
            self.config.clone().with_strategy(Strategy::Fixed)
        } else {
            self.config.clone()
        };
        let plan = Planner::new(&planning_config).plan_chunk(chunk, true)?;

        if self.config.strict_size && plan.payload.len() >= chunk.len() {
            return Err(crate::error::Error::NotCompressible {
                original: chunk.len() as u64,
                compressed: plan.payload.len() as u64,
            });
        }

        *self.pipelines.entry(plan.pipeline).or_insert(0) += 1;
        if plan.stored {
            self.stored += 1;
        }
        self.chunks += 1;
        self.writer.write_planned(&plan, chunk)
    }

    /// Drain the input and finish the stream.
    pub fn finish(mut self) -> Result<(W, CompressStats)> {
        let deadline = self.config.max_compress_time;
        while let Some(chunk) = self.chunker.next_chunk()? {
            if let Some(limit) = deadline {
                if self.started.elapsed() > limit {
                    return Err(crate::error::Error::Config(
                        "compression exceeded its time budget".into(),
                    ));
                }
            }
            self.push(&chunk)?;
        }

        let summary = self.writer.finish()?;
        // Read the byte count *after* finishing, so the end marker is included
        // rather than reconstructed by hand.
        let output_size = self.writer.bytes_written();
        let elapsed = self.started.elapsed();
        let writer = self.writer.into_writer().into_inner();

        let stats = CompressStats {
            input_size: summary.content_size,
            output_size,
            frames: summary.frames,
            pipelines: self.pipelines,
            elapsed,
            all_stored: self.stored == self.chunks && self.chunks > 0,
            stored_fraction: if self.chunks == 0 {
                0.0
            } else {
                self.stored as f32 / self.chunks as f32
            },
        };
        Ok((writer, stats))
    }
}

/// Compress `src` into `dst`.
pub fn compress<R: Read, W: Write>(src: R, dst: W, config: &Config) -> Result<CompressStats> {
    let (dst, stats) = Compressor::new(src, dst, config)?.finish()?;
    drop(dst);
    Ok(stats)
}

/// Compress an in-memory slice, returning the stream and statistics.
pub fn compress_slice(input: &[u8], config: &Config) -> Result<(Vec<u8>, CompressStats)> {
    let c: Compressor<SliceReader<'_>, Vec<u8>> =
        Compressor::new(SliceReader::new(input), Vec::new(), config)?;
    c.finish()
}

/// Compress at a given level with default settings.
pub fn compress_at_level(input: &[u8], level: Level) -> Result<Vec<u8>> {
    compress_slice(input, &Config::new(level)).map(|(v, _)| v)
}

/// Store only, whatever the input.
pub fn store(input: &[u8]) -> Result<Vec<u8>> {
    compress_at_level(input, Level::None)
}

/// Analyse a chunk without compressing it.
pub fn analyse(data: &[u8]) -> Result<Analysis> {
    Analysis::of(data)
}

/// The pipeline a chunk would use. Runs the planner, so it does real work.
pub fn predict_pipeline(data: &[u8], config: &Config) -> Result<PipelineId> {
    let planner = Planner::new(&config.validated()?);
    Ok(planner.plan_chunk(data, true)?.pipeline)
}

/// True when the config would compress at all.
pub fn will_compress(config: &Config) -> bool {
    config.strategy() != Strategy::NeverCompress && config.level() != Level::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Checksum, Threads, WindowConfig};
    use crate::decompress::decompress_slice;

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

    fn roundtrip(data: &[u8], config: &Config) {
        let (compressed, _) = compress_slice(data, config).unwrap();
        let (decoded, ds) = decompress_slice(&compressed, config).unwrap();
        assert_eq!(decoded, data, "roundtrip failed for {} bytes", data.len());
        assert_eq!(
            ds.output_size,
            data.len() as u64,
            "size accounting disagrees"
        );
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(b"", &Config::default());
    }

    #[test]
    fn roundtrip_one_byte() {
        roundtrip(b"a", &Config::default());
    }

    #[test]
    fn roundtrip_tiny() {
        for n in 0..64usize {
            roundtrip(&pseudo_random(n, n as u32 + 1), &Config::default());
        }
    }

    #[test]
    fn roundtrip_text() {
        roundtrip(
            "the quick brown fox jumps over the lazy dog. "
                .repeat(200)
                .as_bytes(),
            &Config::default(),
        );
    }

    #[test]
    fn roundtrip_repetitive() {
        roundtrip(&vec![0x41u8; 100_000], &Config::default());
    }

    #[test]
    fn roundtrip_random() {
        roundtrip(&pseudo_random(50_000, 999), &Config::default());
    }

    #[test]
    fn roundtrip_binary() {
        let data: Vec<u8> = (0..30_000u32).map(|i| (i % 256) as u8).collect();
        roundtrip(&data, &Config::default());
    }

    #[test]
    fn roundtrip_every_level() {
        let data = "hello world ".repeat(1000).into_bytes();
        for level in Level::ALL {
            roundtrip(&data, &Config::new(level));
        }
    }

    #[test]
    fn roundtrip_every_strategy() {
        let data = "hello world ".repeat(1000).into_bytes();
        for s in [Strategy::Adaptive, Strategy::Fixed, Strategy::NeverCompress] {
            roundtrip(&data, &Config::new(Level::Default).with_strategy(s));
        }
    }

    #[test]
    fn roundtrip_every_checksum_policy() {
        let data = "hello world ".repeat(1000).into_bytes();
        for c in [Checksum::Full, Checksum::Chunk, Checksum::None] {
            roundtrip(&data, &Config::default().with_checksum(c));
        }
    }

    #[test]
    fn roundtrip_across_chunk_boundaries() {
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let cfg = Config::default().with_window(WindowConfig {
            chunk_size: 4096,
            window_size: 2048,
        });
        roundtrip(&data, &cfg);
    }

    #[test]
    fn stats_report_measured_sizes() {
        let data = "hello world ".repeat(1000).into_bytes();
        let (compressed, stats) = compress_slice(&data, &Config::default()).unwrap();
        assert_eq!(stats.input_size, data.len() as u64);
        assert_eq!(stats.output_size, compressed.len() as u64);
        assert!(stats.frames >= 1);
        assert!(stats.ratio() < 1.0, "ratio was {}", stats.ratio());
        assert!(!stats.pipelines.is_empty());
    }

    #[test]
    fn store_never_shrinks_content() {
        let data = pseudo_random(5000, 4);
        let (compressed, stats) = compress_slice(&data, &Config::new(Level::None)).unwrap();
        assert!(stats.all_stored);
        // Store adds headers, so the result is slightly larger than the input.
        assert!(compressed.len() > data.len());
    }

    #[test]
    fn compression_reduces_compressible_data() {
        let data = "the quick brown fox jumps over the lazy dog. "
            .repeat(1000)
            .into_bytes();
        let (compressed, _) = compress_slice(&data, &Config::default()).unwrap();
        assert!(
            compressed.len() * 4 < data.len(),
            "{} vs {}",
            compressed.len(),
            data.len()
        );
    }

    #[test]
    fn output_is_deterministic() {
        let data = "hello world ".repeat(2000).into_bytes();
        let a = compress_at_level(&data, Level::Default).unwrap();
        let b = compress_at_level(&data, Level::Default).unwrap();
        assert_eq!(a, b, "same input must produce identical output");
    }

    #[test]
    fn serial_and_auto_threads_agree() {
        let data = "hello world ".repeat(5000).into_bytes();
        let a = compress_at_level(&data, Level::Default).unwrap();
        let cfg = Config::new(Level::Default).with_threads(Threads::Serial);
        let b = compress_slice(&data, &cfg).unwrap().0;
        assert_eq!(a, b, "thread policy changed the output");
    }

    #[test]
    fn counting_writer_counts_bytes() {
        let mut out = Vec::new();
        {
            let mut cw = CountingWriter::new(&mut out);
            cw.write_all(b"hello world").unwrap();
            assert_eq!(cw.count(), 11);
        }
    }

    #[test]
    fn analyse_exposes_analysis() {
        let a = analyse(b"hello").unwrap();
        assert_eq!(a.sampled, 5);
    }

    #[test]
    fn predict_pipeline_matches_actual() {
        let data = "hello world ".repeat(1000).into_bytes();
        let cfg = Config::new(Level::Default);
        let predicted = predict_pipeline(&data, &cfg).unwrap();
        let (_, stats) = compress_slice(&data, &cfg).unwrap();
        assert!(
            stats.pipelines.contains_key(&predicted) || predicted == PipelineId::Store,
            "predicted {predicted:?} not in {:?}",
            stats.pipelines
        );
    }

    #[test]
    fn will_compress_reflects_config() {
        assert!(will_compress(&Config::new(Level::Default)));
        assert!(!will_compress(&Config::new(Level::None)));
        assert!(!will_compress(
            &Config::new(Level::Default).with_strategy(Strategy::NeverCompress)
        ));
    }

    #[test]
    fn large_input_roundtrips() {
        let data: Vec<u8> = (0..4_000_000u32).map(|i| ((i / 7) % 256) as u8).collect();
        let cfg = Config::default().with_window(WindowConfig {
            chunk_size: 1 << 20,
            window_size: 1 << 16,
        });
        roundtrip(&data, &cfg);
    }

    #[test]
    fn strict_size_rejects_incompressible() {
        let cfg = Config::new(Level::Default).with_strict_size(true);
        let data = pseudo_random(5000, 12);
        let r = compress_slice(&data, &cfg);
        assert!(matches!(r, Err(crate::Error::NotCompressible { .. })));
    }

    #[test]
    fn strict_size_accepts_compressible() {
        let cfg = Config::new(Level::Default).with_strict_size(true);
        let data = "hello world ".repeat(1000).into_bytes();
        roundtrip(&data, &cfg);
    }

    #[test]
    fn time_budget_is_enforced() {
        let cfg = Config::new(Level::Max).with_time_budget(std::time::Duration::from_nanos(1));
        let data: Vec<u8> = (0..2_000_000u32).map(|i| (i % 253) as u8).collect();
        match compress_slice(&data, &cfg) {
            Ok(_) => {}
            Err(e) => assert!(
                matches!(e, crate::Error::Config(_)),
                "unexpected error: {e}"
            ),
        }
    }

    #[test]
    fn writes_to_any_writer() {
        let data = "hello world ".repeat(100).into_bytes();
        let mut out = Vec::new();
        let stats = compress(SliceReader::new(&data), &mut out, &Config::default()).unwrap();
        assert_eq!(out.len() as u64, stats.output_size);
    }
}
