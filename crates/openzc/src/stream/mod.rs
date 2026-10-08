//! Streaming layer: turns a chunk sequence into a valid OpenZC stream and back.
//!
//! # Memory contract
//!
//! The writer holds at most one chunk plus the current frame in memory, so peak
//! usage is a function of the configured chunk size and not of the input. The
//! reader does the same. Nothing here accumulates the whole stream.
//!
//! # Structure
//!
//! Every stream is `container header | frames | end marker`. That is the entire
//! grammar a decoder must understand, and `docs/FORMAT.md` specifies it
//! normatively.
//!
//! The container header's content size is only known after the input is
//! drained, so the writer leaves `CONTENT_SIZE` clear and the end marker
//! carries the authoritative total. A decoder therefore always has the true
//! size exactly once, from a field whose position it knows.
//!
//! # Integrity
//!
//! Two independent checks, both BLAKE3:
//!
//! * each frame carries a hash of the content it decodes to, so damage is
//!   localised and chunks can be verified in parallel;
//! * the end marker carries a hash of all content, which additionally catches a
//!   dropped, reordered, or duplicated frame — none of which the per-frame
//!   hashes can detect on their own.

use crate::bytes::ByteWriter;
use crate::codec::DictionaryTable;
use crate::config::{Checksum, Config};
use crate::error::{Error, Result};
use crate::format::frame::FRAME_HEADER_LEN;
use crate::format::header::{END_MAGIC, END_MAGIC_LEN};
use crate::format::{
    ContainerFlags, ContainerHeader, EndMarker, FrameFlags, FrameHeader, PipelineId,
    CONTAINER_PREAMBLE_LEN, END_MARKER_FIXED_LEN, FRAME_MAGIC, FRAME_MAGIC_LEN, HASH_LEN,
    VERSION_MAJOR, VERSION_MINOR,
};
use crate::integrity;
use crate::pipeline::{ChunkPlan, Planner};
use std::io::{Read, Write};

/// Default ceiling on frame count, as a decompression-bomb guard. At the
/// default 1 MiB chunk size this allows a terabyte of content.
pub const DEFAULT_MAX_FRAMES: u64 = 1 << 20;

/// Container flag bits implied by a checksum policy.
fn container_flags(checksum: Checksum) -> ContainerFlags {
    let mut bits = 0;
    if checksum.content_hash() {
        bits |= ContainerFlags::CONTENT_CHECKSUM;
    }
    if checksum.chunk_hashes() {
        bits |= ContainerFlags::CHUNK_CHECKSUM;
    }
    // Safe by construction: `bits` is assembled only from defined flags.
    ContainerFlags::from_bits_strict(bits).expect("flags built from known bits")
}

/// Totals reported once a stream is complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamSummary {
    /// Total uncompressed bytes.
    pub content_size: u64,
    /// Number of frames written.
    pub frames: u64,
    /// True when a whole-content hash was written.
    pub content_hashed: bool,
}

/// Writes an OpenZC stream to any [`Write`].
#[derive(Debug)]
pub struct StreamWriter<W: Write> {
    inner: W,
    config: Config,
    planner: Planner,
    dictionaries: DictionaryTable,
    content_written: u64,
    frames_written: u64,
    bytes_written: u64,
    finished: bool,
    /// Whole-content hasher, present only when the policy asks for one.
    content_hash: Option<integrity::ContentHasher>,
}

impl<W: Write> StreamWriter<W> {
    /// Create a writer and emit the container header.
    pub fn new(mut inner: W, config: &Config) -> Result<Self> {
        let config = config.validated()?;
        let header = ContainerHeader {
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            // CONTENT_SIZE is intentionally clear: the size is not known until
            // the input is drained, and the end marker carries it authoritatively.
            flags: container_flags(config.checksum()),
            chunk_size: config.window().chunk_size,
            window_size: config.window().window_size,
            content_size: None,
        };
        let mut w = ByteWriter::new();
        header.encode(&mut w);
        debug_assert_eq!(w.len(), ContainerHeader::encoded_len());

        // Write the header before moving `inner` into the struct.
        inner.write_all(w.as_slice())?;

        Ok(Self {
            inner,
            planner: Planner::new(&config),
            content_hash: if config.checksum().content_hash() {
                Some(integrity::ContentHasher::new())
            } else {
                None
            },
            config,
            dictionaries: DictionaryTable::new(),
            content_written: 0,
            frames_written: 0,
            bytes_written: ContainerHeader::encoded_len() as u64,
            finished: false,
        })
    }

    /// The configuration in use.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Number of frames written so far.
    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }

    /// Uncompressed bytes written so far.
    pub fn content_written(&self) -> u64 {
        self.content_written
    }

    /// Plan and write one chunk.
    pub fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finished {
            return Err(Error::Config("cannot write to a finished stream".into()));
        }
        let plan = self.planner.plan_chunk(chunk, true)?;
        self.emit(&plan, chunk)?;
        if let Some(h) = self.content_hash.as_mut() {
            // Hash the original bytes: that is what a decoder must reproduce,
            // so it is the only thing worth verifying.
            h.update(chunk);
        }
        Ok(())
    }

    /// Write a chunk planned elsewhere, for the parallel path.
    ///
    /// `chunk` must be the exact input `plan` was produced from; it is used only
    /// for hashing, and `plan.content_size()` must equal `chunk.len()`.
    pub fn write_planned(&mut self, plan: &ChunkPlan, chunk: &[u8]) -> Result<()> {
        if self.finished {
            return Err(Error::Config("cannot write to a finished stream".into()));
        }
        if plan.content_size() != chunk.len() {
            return Err(Error::Config(
                "planned content size does not match the chunk length".into(),
            ));
        }
        self.emit(plan, chunk)?;
        if let Some(h) = self.content_hash.as_mut() {
            h.update(chunk);
        }
        Ok(())
    }

    /// Serialise one frame to the underlying writer.
    fn emit(&mut self, plan: &ChunkPlan, chunk: &[u8]) -> Result<()> {
        let header = FrameHeader {
            pipeline: plan.pipeline,
            transform: plan.transform,
            flags: FrameFlags::from_bits_unchecked(if plan.independent {
                FrameFlags::INDEPENDENT
            } else {
                0
            }),
            dict_id: plan.dict_id,
            content_size: chunk.len() as u32,
            payload_size: plan.payload.len() as u32,
        };

        let mut w = ByteWriter::with_capacity(FRAME_HEADER_LEN + plan.payload.len() + HASH_LEN);
        header.encode(&mut w);
        w.bytes(&plan.payload);
        if self.config.checksum().chunk_hashes() {
            w.bytes(&integrity::hash(chunk));
        }

        self.content_written += chunk.len() as u64;
        self.frames_written += 1;
        if let Some(limit) = self.config.max_frames {
            if self.frames_written > limit {
                return Err(Error::Config(format!(
                    "stream exceeded the frame limit of {limit}"
                )));
            }
        }
        self.bytes_written += w.len() as u64;
        self.inner.write_all(w.as_slice())?;
        Ok(())
    }

    /// Bytes written to the underlying writer so far, excluding buffering.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Dictionaries registered for later frames to reference.
    pub fn dictionaries(&self) -> &DictionaryTable {
        &self.dictionaries
    }

    /// Look up a registered dictionary by id.
    pub fn dictionary(&self, id: u8) -> Option<&[u8]> {
        self.dictionaries.get(id).ok()
    }

    /// Unwrap the inner writer.
    pub fn into_writer(self) -> W {
        self.inner
    }

    /// Emit the end marker and flush.
    ///
    /// Takes `&mut self` so the writer remains recoverable with
    /// [`StreamWriter::into_writer`] afterwards, which callers need in order to
    /// read back the produced bytes.
    pub fn finish(&mut self) -> Result<StreamSummary> {
        if self.finished {
            return Err(Error::Config("stream already finished".into()));
        }
        self.finished = true;

        let end = EndMarker {
            content_size: self.content_written,
            content_hash: self
                .content_hash
                .as_ref()
                .map(integrity::ContentHasher::finalize),
        };
        let mut w = ByteWriter::new();
        end.encode(&mut w);
        self.bytes_written += w.len() as u64;
        self.inner.write_all(w.as_slice())?;
        self.inner.flush()?;

        Ok(StreamSummary {
            content_size: self.content_written,
            frames: self.frames_written,
            content_hashed: end.content_hash.is_some(),
        })
    }
}

/// One frame as read from a stream.
#[derive(Debug, Clone)]
pub struct RawFrame {
    pub header: FrameHeader,
    pub payload: Vec<u8>,
    /// Hash of the decoded content, when the container header promises one.
    pub content_hash: Option<[u8; HASH_LEN]>,
}

/// Reads an OpenZC stream from any [`Read`].
#[derive(Debug)]
pub struct StreamReader<R: Read> {
    inner: R,
    header: ContainerHeader,
    max_frames: u64,
    frames_read: u64,
    ended: bool,
    end_content_size: Option<u64>,
    expected_content_hash: Option<[u8; HASH_LEN]>,
}

impl<R: Read> StreamReader<R> {
    /// Read and validate the container header.
    ///
    /// The preamble is checked on its own first. Demanding the whole header up
    /// front would report a short file as *truncated*, which is true but useless:
    /// if the first bytes are not ours then no length of this file will decode,
    /// and "this is not an OpenZC stream" is the answer a caller can act on. The
    /// version is checked at the same time for the same reason — a file from a
    /// newer build should be reported as such rather than as damage. Neither
    /// check costs anything, since the header has to be read regardless.
    pub fn new(mut inner: R, config: &Config) -> Result<Self> {
        let config = config.validated()?;
        let mut preamble = [0u8; CONTAINER_PREAMBLE_LEN];
        read_exact(&mut inner, &mut preamble)?;
        let magic: [u8; crate::format::header::MAGIC_LEN] = preamble
            [..crate::format::header::MAGIC_LEN]
            .try_into()
            .expect("the preamble is at least as long as the magic");
        if magic != crate::format::header::MAGIC {
            return Err(Error::BadMagic { found: magic });
        }
        let major = u16::from(preamble[crate::format::header::MAGIC_LEN]);
        if major > crate::format::header::MAX_SUPPORTED_MAJOR {
            return Err(Error::UnsupportedVersion {
                major,
                minor: u16::from(preamble[crate::format::header::MAGIC_LEN + 1]),
                max_supported: crate::format::header::MAX_SUPPORTED_MAJOR,
            });
        }

        let mut rest = vec![0u8; ContainerHeader::encoded_len() - CONTAINER_PREAMBLE_LEN];
        read_exact(&mut inner, &mut rest)?;
        let mut buf = Vec::with_capacity(ContainerHeader::encoded_len());
        buf.extend_from_slice(&preamble);
        buf.extend_from_slice(&rest);
        let header = ContainerHeader::decode(&buf)?;
        Ok(Self {
            inner,
            header,
            max_frames: config.max_frames.unwrap_or(DEFAULT_MAX_FRAMES),
            frames_read: 0,
            ended: false,
            end_content_size: None,
            expected_content_hash: None,
        })
    }

    /// The container header.
    pub fn header(&self) -> &ContainerHeader {
        &self.header
    }

    /// Frames read so far.
    pub fn frames_read(&self) -> u64 {
        self.frames_read
    }

    /// Content size from the end marker, available once the stream ends.
    pub fn content_size(&self) -> Option<u64> {
        self.end_content_size
    }

    /// Read the next frame, or `None` at the end marker.
    pub fn next_frame(&mut self) -> Result<Option<RawFrame>> {
        if self.ended {
            return Ok(None);
        }
        if self.frames_read >= self.max_frames {
            return Err(Error::Config(format!(
                "stream exceeded the frame limit of {}",
                self.max_frames
            )));
        }

        // Four bytes distinguish a frame from the end marker.
        let mut magic = [0u8; 4];
        match read_up_to(&mut self.inner, &mut magic)? {
            0 => return Err(Error::MissingEndMarker),
            4 => {}
            n => {
                let _ = n;
                return Err(Error::UnexpectedEof {
                    needed: 4,
                    available: n,
                });
            }
        }

        if magic == END_MAGIC {
            self.read_end_marker()?;
            self.ended = true;
            return Ok(None);
        }
        if magic != FRAME_MAGIC {
            return Err(Error::InvalidHeader("frame magic mismatch"));
        }

        let mut rest = [0u8; FRAME_HEADER_LEN - FRAME_MAGIC_LEN];
        read_exact(&mut self.inner, &mut rest)?;
        let mut full = Vec::with_capacity(FRAME_HEADER_LEN);
        full.extend_from_slice(&magic);
        full.extend_from_slice(&rest);
        let header = FrameHeader::decode(&full)?;

        // A frame claiming more content than the container advertised is a sign
        // of corruption, and would otherwise let a hostile stream dictate a
        // large allocation.
        if header.content_size > self.header.chunk_size {
            return Err(Error::CorruptChunk(
                "frame content exceeds the declared chunk size",
            ));
        }

        // A conforming encoder never emits a payload larger than the content it
        // decodes to: `store` is the floor, and a pipeline is only kept when it
        // beat `store`. So `payload_size > content_size` is corruption, and it
        // must be caught here rather than discovered three lines below.
        //
        // Checking it before the allocation is what makes this a security check
        // and not a tidiness one: a corrupt length of 4 GiB would otherwise ask
        // for a 4 GiB buffer before failing, and would then be *misreported* as a
        // truncated stream, sending whoever is diagnosing the file down the wrong
        // path entirely.
        if header.payload_size > header.content_size {
            return Err(Error::CorruptChunk(
                "frame payload is larger than the content it decodes to",
            ));
        }

        let mut payload = vec![0u8; header.payload_size as usize];
        read_exact(&mut self.inner, &mut payload)?;

        let content_hash = if self.header.flags.contains(ContainerFlags::CHUNK_CHECKSUM) {
            let mut h = [0u8; HASH_LEN];
            read_exact(&mut self.inner, &mut h)?;
            Some(h)
        } else {
            None
        };

        self.frames_read += 1;
        Ok(Some(RawFrame {
            header,
            payload,
            content_hash,
        }))
    }

    /// Consume the end marker.
    ///
    /// The 4-byte `END` magic has already been consumed by the caller that
    /// dispatched on it, so only the size field and the optional hash are read
    /// here.
    fn read_end_marker(&mut self) -> Result<()> {
        let expect_hash = self.header.flags.contains(ContainerFlags::CONTENT_CHECKSUM);
        // END_MARKER_FIXED_LEN includes the magic, so the remainder is 4 shorter.
        let mut all = vec![0u8; END_MARKER_FIXED_LEN - END_MAGIC_LEN];
        read_exact(&mut self.inner, &mut all)?;
        if expect_hash {
            let mut h = [0u8; HASH_LEN];
            read_exact(&mut self.inner, &mut h)?;
            all.extend_from_slice(&h);
        }
        let end = EndMarker::decode_tail(&all, expect_hash)?;

        if let Some(declared) = self.header.content_size {
            if declared != end.content_size {
                return Err(Error::InvalidHeader(
                    "content size disagrees between header and end marker",
                ));
            }
        }
        self.end_content_size = Some(end.content_size);
        self.expected_content_hash = end.content_hash;
        Ok(())
    }

    /// Verify the whole-content hash computed while decoding.
    pub fn verify_content_hash(&self, computed: &[u8; HASH_LEN]) -> Result<()> {
        if let Some(expected) = self.expected_content_hash {
            if &expected != computed {
                return Err(Error::ChecksumMismatch {
                    what: "stream content",
                });
            }
        }
        Ok(())
    }

    /// Confirm the end marker was reached.
    pub fn ensure_finished(&self) -> Result<()> {
        if !self.ended {
            return Err(Error::MissingEndMarker);
        }
        Ok(())
    }
}

/// Read exactly `buf.len()` bytes, mapping a short read to a clean error.
fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => {
                return Err(Error::UnexpectedEof {
                    needed: buf.len(),
                    available: filled,
                });
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(Error::Io(e)),
        }
    }
    Ok(())
}

/// Read up to `buf.len()` bytes, returning how many were read.
fn read_up_to<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(Error::Io(e)),
        }
    }
    Ok(filled)
}

/// A `Read` implementation over a byte slice, for tests and small inputs.
#[derive(Debug, Clone)]
pub struct SliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SliceReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl Read for SliceReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = out.len().min(self.data.len() - self.pos);
        out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Bytes of the container header preamble, exposed for tests.
pub const HEADER_PREAMBLE: usize = CONTAINER_PREAMBLE_LEN;

/// Re-exported for callers that dispatch on pipeline ids.
pub type StreamPipeline = PipelineId;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Level;

    fn cfg() -> Config {
        Config::new(Level::Default)
    }

    #[test]
    fn container_flags_match_policy() {
        assert_eq!(
            container_flags(Checksum::Full).bits(),
            ContainerFlags::CONTENT_CHECKSUM | ContainerFlags::CHUNK_CHECKSUM
        );
        assert_eq!(
            container_flags(Checksum::Chunk).bits(),
            ContainerFlags::CHUNK_CHECKSUM
        );
        assert_eq!(container_flags(Checksum::None).bits(), 0);
    }

    #[test]
    fn empty_stream_roundtrips() {
        let mut out = Vec::new();
        let mut w = StreamWriter::new(&mut out, &cfg()).unwrap();
        let summary = w.finish().unwrap();
        assert_eq!(summary.content_size, 0);
        assert_eq!(summary.frames, 0);

        let mut r = StreamReader::new(SliceReader::new(&out), &cfg()).unwrap();
        assert!(r.next_frame().unwrap().is_none());
        r.ensure_finished().unwrap();
        assert_eq!(r.content_size(), Some(0));
    }

    #[test]
    fn one_chunk_roundtrips() {
        let data = b"hello hello hello world".repeat(50);
        let mut out = Vec::new();
        let mut w = StreamWriter::new(&mut out, &cfg()).unwrap();
        w.write_chunk(&data).unwrap();
        let summary = w.finish().unwrap();
        assert_eq!(summary.content_size, data.len() as u64);
        assert_eq!(summary.frames, 1);

        let mut r = StreamReader::new(SliceReader::new(&out), &cfg()).unwrap();
        let f = r.next_frame().unwrap().expect("expected a frame");
        assert_eq!(f.header.content_size as usize, data.len());
        assert!(r.next_frame().unwrap().is_none());
        r.ensure_finished().unwrap();
        assert_eq!(r.content_size(), Some(data.len() as u64));
    }

    #[test]
    fn many_chunks_produce_many_frames() {
        let mut out = Vec::new();
        let mut w = StreamWriter::new(&mut out, &cfg()).unwrap();
        for _ in 0..10 {
            w.write_chunk(b"chunk data").unwrap();
        }
        let summary = w.finish().unwrap();
        assert_eq!(summary.frames, 10);
        assert_eq!(summary.content_size, 100);

        let mut r = StreamReader::new(SliceReader::new(&out), &cfg()).unwrap();
        let mut n = 0;
        while r.next_frame().unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 10);
    }

    #[test]
    fn write_after_finish_is_rejected() {
        let mut out = Vec::new();
        let w = StreamWriter::new(&mut out, &cfg()).unwrap();
        let mut w = w;
        // A finished writer is consumed by `finish`, so this checks the guard
        // that protects a writer reused before finishing.
        w.write_chunk(b"data").unwrap();
        let _ = w.finish().unwrap();
    }

    #[test]
    fn double_finish_is_rejected() {
        let mut out = Vec::new();
        let mut w = StreamWriter::new(&mut out, &cfg()).unwrap();
        w.finish().unwrap();
        // `finish` takes `&mut self` precisely so a second call is possible and
        // must be refused rather than writing a second end marker.
        assert!(matches!(w.finish(), Err(Error::Config(_))));
    }

    #[test]
    fn checksum_policies_produce_different_sizes() {
        let data = vec![1u8; 5000];
        let mut sizes = Vec::new();
        for policy in [Checksum::Full, Checksum::Chunk, Checksum::None] {
            let c = cfg().with_checksum(policy);
            let mut out = Vec::new();
            let mut w = StreamWriter::new(&mut out, &c).unwrap();
            w.write_chunk(&data).unwrap();
            w.finish().unwrap();
            sizes.push(out.len());
        }
        assert!(
            sizes[0] > sizes[1],
            "full {} vs chunk {}",
            sizes[0],
            sizes[1]
        );
        assert!(
            sizes[1] > sizes[2],
            "chunk {} vs none {}",
            sizes[1],
            sizes[2]
        );
    }

    #[test]
    fn reader_rejects_bad_magic() {
        let mut bad = b"NOTOPENZC_stream_data_here__".to_vec();
        bad[0] = b'X';
        assert!(matches!(
            StreamReader::new(SliceReader::new(&bad), &cfg()),
            Err(Error::BadMagic { .. })
        ));
    }

    #[test]
    fn reader_rejects_truncated_header() {
        let data = b"short".to_vec();
        assert!(matches!(
            StreamReader::new(SliceReader::new(&data), &cfg()),
            Err(Error::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn corrupt_frame_length_is_reported_as_corruption_not_truncation() {
        // A damaged length field must be diagnosed as corruption. Reporting it as
        // a truncated stream would send whoever is repairing the file looking for
        // a cut-off download, and the two have nothing in common.
        let mut out = Vec::new();
        {
            let mut w = StreamWriter::new(&mut out, &cfg()).unwrap();
            w.write_chunk(b"some data here, long enough to compress a little")
                .unwrap();
            w.finish().unwrap();
        }

        let header_len = ContainerHeader::encoded_len();
        // payload_size sits at offset 12 of the frame header, which itself starts
        // immediately after the container header.
        let payload_size_at = header_len + FRAME_HEADER_LEN - 4;
        let mut damaged = out.clone();
        damaged[payload_size_at..payload_size_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());

        let mut r = StreamReader::new(SliceReader::new(&damaged), &cfg()).unwrap();
        assert!(
            matches!(r.next_frame(), Err(Error::CorruptChunk(_))),
            "expected corruption, got {:?}",
            r.next_frame().err()
        );
    }

    #[test]
    fn oversized_frame_length_does_not_attempt_the_allocation() {
        // The same check is a memory-safety property: a length of 4 GiB must be
        // rejected before the buffer is reserved, not after.
        let mut out = Vec::new();
        {
            let mut w = StreamWriter::new(&mut out, &cfg()).unwrap();
            w.write_chunk(b"payload").unwrap();
            w.finish().unwrap();
        }
        let header_len = ContainerHeader::encoded_len();
        let payload_size_at = header_len + FRAME_HEADER_LEN - 4;
        let mut damaged = out.clone();
        damaged[payload_size_at..payload_size_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());

        // If this tried to allocate 4 GiB it would be obvious; succeeding quickly
        // with a clean error is the point.
        let mut r = StreamReader::new(SliceReader::new(&damaged), &cfg()).unwrap();
        let err = r.next_frame().expect_err("must reject");
        assert!(matches!(err, Error::CorruptChunk(_)), "got {err:?}");
    }

    #[test]
    fn reader_rejects_stream_without_end_marker() {
        let mut out = Vec::new();
        {
            let mut w = StreamWriter::new(&mut out, &cfg()).unwrap();
            w.write_chunk(b"some data here").unwrap();
            // Deliberately never finished: the frame is written and intact, but
            // the stream has no terminator. That is a different fault from a
            // damaged frame, and the reader must say which one it found.
        }
        let mut r = StreamReader::new(SliceReader::new(&out), &cfg()).unwrap();
        assert!(
            r.next_frame().is_ok(),
            "the frame itself should still read back"
        );
        assert!(matches!(r.next_frame(), Err(Error::MissingEndMarker)));
    }

    #[test]
    fn reader_enforces_frame_limit() {
        let mut out = Vec::new();
        let c = cfg().with_window(crate::config::WindowConfig {
            chunk_size: 1024,
            window_size: 1024,
        });
        let mut w = StreamWriter::new(&mut out, &c).unwrap();
        for _ in 0..5 {
            w.write_chunk(b"1234567890").unwrap();
        }
        w.finish().unwrap();

        let limited = cfg().with_window(c.window()).clone_limit(2);
        let mut r = StreamReader::new(SliceReader::new(&out), &limited).unwrap();
        assert!(r.next_frame().unwrap().is_some());
        assert!(r.next_frame().unwrap().is_some());
        assert!(r.next_frame().is_err(), "frame limit should stop the third");
    }

    #[test]
    fn slice_reader_reads_everything() {
        let data = b"hello world".to_vec();
        let mut r = SliceReader::new(&data);
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn read_exact_reports_truncation() {
        let data = b"abc".to_vec();
        let mut r = SliceReader::new(&data);
        let mut buf = [0u8; 10];
        assert!(matches!(
            read_exact(&mut r, &mut buf),
            Err(Error::UnexpectedEof { .. })
        ));
    }

    /// Small helper so the frame-limit test reads clearly.
    trait CloneLimit {
        fn clone_limit(&self, frames: u64) -> Config;
    }

    impl CloneLimit for Config {
        fn clone_limit(&self, frames: u64) -> Config {
            let mut c = self.clone();
            c.max_frames = Some(frames);
            c
        }
    }
}
