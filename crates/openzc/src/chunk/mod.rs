//! Bounded-memory input chunking.
//!
//! The chunker is the component that makes streaming work: it never holds more
//! than one chunk plus a small carry-over buffer, so peak memory is a function
//! of the configured chunk size and not of the input size.
//!
//! Chunk boundaries are deterministic — always at multiples of `chunk_size`
//! except the last — which is what lets the same input produce byte-identical
//! output on every run and every platform. That determinism is a test
//! requirement, not an optimisation.

use std::io::{Read, Result};

/// Splits a byte stream into fixed-size chunks without unbounded buffering.
#[derive(Debug)]
pub struct Chunker<R> {
    reader: R,
    chunk_size: usize,
    /// Leftover bytes from a previous short read.
    carry: Vec<u8>,
    /// Total bytes consumed so far.
    consumed: u64,
    finished: bool,
}

impl<R: Read> Chunker<R> {
    /// Wrap `reader`, emitting chunks of at most `chunk_size` bytes.
    pub fn new(reader: R, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk size must be positive");
        Self {
            reader,
            chunk_size,
            carry: Vec::with_capacity(chunk_size.min(1 << 20)),
            consumed: 0,
            finished: false,
        }
    }

    /// Total bytes read from the underlying reader so far.
    pub fn bytes_read(&self) -> u64 {
        self.consumed
    }

    /// True once the underlying reader has signalled end of input.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Fill `out` from the underlying reader, bounded by `out.len()`.
    fn fill(&mut self, out: &mut [u8]) -> Result<usize> {
        let mut filled = 0;
        while filled < out.len() {
            match self.reader.read(&mut out[filled..])? {
                0 => {
                    self.finished = true;
                    break;
                }
                n => filled += n,
            }
        }
        self.consumed += filled as u64;
        Ok(filled)
    }

    /// Produce the next chunk, or `None` at end of input.
    ///
    /// An empty `Vec` is never returned for a non-empty stream, and a stream
    /// with no bytes at all yields a single `None`.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        if self.finished && self.carry.is_empty() {
            return Ok(None);
        }

        let mut buf = std::mem::take(&mut self.carry);
        buf.clear();
        buf.resize(self.chunk_size, 0);

        let filled = self.fill(&mut buf)?;
        buf.truncate(filled);

        if buf.is_empty() {
            self.carry = buf;
            return Ok(None);
        }
        Ok(Some(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader that returns at most `n` bytes per call, to prove the chunker
    /// copes with short reads and does not depend on read granularity.
    struct DribbleReader {
        data: Vec<u8>,
        pos: usize,
        max: usize,
    }

    impl Read for DribbleReader {
        fn read(&mut self, out: &mut [u8]) -> Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let n = out.len().min(self.max).min(self.data.len() - self.pos);
            out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    fn chunk_all(data: &[u8], chunk_size: usize, max_read: usize) -> Vec<Vec<u8>> {
        let mut c = Chunker::new(
            DribbleReader {
                data: data.to_vec(),
                pos: 0,
                max: max_read,
            },
            chunk_size,
        );
        let mut out = Vec::new();
        while let Some(chunk) = c.next_chunk().unwrap() {
            out.push(chunk);
        }
        out
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert!(chunk_all(&[], 1024, 1024).is_empty());
    }

    #[test]
    fn exact_multiple_splits_cleanly() {
        let data = vec![7u8; 4096];
        let chunks = chunk_all(&data, 1024, 1024);
        assert_eq!(chunks.len(), 4);
        assert!(chunks.iter().all(|c| c.len() == 1024));
    }

    #[test]
    fn remainder_becomes_final_chunk() {
        let data = vec![7u8; 2500];
        let chunks = chunk_all(&data, 1024, 1024);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2].len(), 2500 - 2048);
    }

    #[test]
    fn short_reads_do_not_change_boundaries() {
        // One byte per read, then mid-size reads: the chunk boundaries and the
        // reassembled content must be identical.
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 256) as u8).collect();
        for max_read in [1, 2, 7, 512, 5000, 100_000] {
            let chunks = chunk_all(&data, 1024, max_read);
            let flat: Vec<u8> = chunks.concat();
            assert_eq!(flat, data, "max_read = {max_read}");
            assert!(
                chunks.iter().all(|c| c.len() <= 1024),
                "max_read = {max_read}"
            );
        }
    }

    #[test]
    fn single_byte_chunk() {
        let chunks = chunk_all(b"abc", 1, 1);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks.concat(), b"abc");
    }

    #[test]
    fn total_bytes_read_is_exact() {
        let data = vec![1u8; 10_000];
        let mut c = Chunker::new(&data[..], 512);
        let mut total = 0usize;
        while let Some(chunk) = c.next_chunk().unwrap() {
            total += chunk.len();
        }
        assert_eq!(total, data.len());
        assert_eq!(c.bytes_read(), data.len() as u64);
    }

    #[test]
    fn next_chunk_after_end_returns_none() {
        let mut c = Chunker::new(&b"xy"[..], 16);
        assert!(c.next_chunk().unwrap().is_some());
        assert!(c.next_chunk().unwrap().is_none());
        assert!(c.next_chunk().unwrap().is_none());
    }
}
