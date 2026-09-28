//! Small, allocation-light helpers for serialising and parsing byte streams.
//!
//! Every parser in this crate is *fully bounds checked*: no `unsafe`, no
//! unchecked slicing. A malformed stream therefore produces an [`Error`] rather
//! than a panic or an out-of-bounds read. That property is what allows the
//! integrity hash to be the single authority on "is this stream damaged?".
//!
//! [`Error`]: crate::Error

use crate::error::{Error, Result};

/// Bounds-checked little-endian cursor over a byte slice.
#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    #[inline]
    fn need(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            Err(Error::UnexpectedEof {
                needed: n,
                available: self.remaining(),
            })
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    #[inline]
    pub fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    #[inline]
    pub fn u16le(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    #[inline]
    pub fn u24le(&mut self) -> Result<u32> {
        let b = self.take(3)?;
        Ok(u32::from(b[0]) | (u32::from(b[1]) << 8) | (u32::from(b[2]) << 16))
    }

    #[inline]
    pub fn u32le(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    pub fn u64le(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// LEB128-style unsigned varint. Unused high bits are rejected so that a
    /// corrupt stream cannot produce a value that later overflows a length.
    pub fn varint(&mut self) -> Result<u64> {
        let mut out: u64 = 0;
        let mut shift = 0u32;
        loop {
            let b = self.u8()?;
            if shift >= 64 {
                return Err(Error::CorruptChunk("varint overflow"));
            }
            let payload = u64::from(b & 0x7F);
            if shift == 63 && payload > 1 {
                return Err(Error::CorruptChunk("varint overflow"));
            }
            out |= payload << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
        }
    }
}

/// Append-only little-endian writer.
#[derive(Debug, Default, Clone)]
pub struct ByteWriter {
    out: Vec<u8>,
}

impl ByteWriter {
    #[inline]
    pub fn new() -> Self {
        Self { out: Vec::new() }
    }

    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            out: Vec::with_capacity(n),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.out.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.out
    }

    #[inline]
    pub fn into_vec(self) -> Vec<u8> {
        self.out
    }

    #[inline]
    pub fn clear(&mut self) {
        self.out.clear();
    }

    #[inline]
    pub fn u8(&mut self, v: u8) {
        self.out.push(v);
    }

    #[inline]
    pub fn u16le(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn u24le(&mut self, v: u32) {
        self.out.push(v as u8);
        self.out.push((v >> 8) as u8);
        self.out.push((v >> 16) as u8);
    }

    #[inline]
    pub fn u32le(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn u64le(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn bytes(&mut self, v: &[u8]) {
        self.out.extend_from_slice(v);
    }

    /// LEB128-style unsigned varint, matching [`ByteReader::varint`].
    pub fn varint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                self.out.push(b);
                return;
            }
            self.out.push(b | 0x80);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        for v in [
            0u64,
            1,
            127,
            128,
            300,
            16383,
            16384,
            u32::MAX as u64,
            u64::MAX,
        ] {
            let mut w = ByteWriter::new();
            w.varint(v);
            let buf = w.into_vec();
            let mut r = ByteReader::new(&buf);
            assert_eq!(r.varint().unwrap(), v, "value {v}");
            assert!(r.is_empty());
        }
    }

    #[test]
    fn varint_rejects_overflow() {
        let buf = [0xFFu8; 12];
        let mut r = ByteReader::new(&buf);
        assert!(r.varint().is_err());
    }

    #[test]
    fn reader_is_bounds_checked() {
        let mut r = ByteReader::new(&[1, 2, 3]);
        assert_eq!(r.u8().unwrap(), 1);
        assert_eq!(r.u16le().unwrap(), 0x0302);
        assert!(r.u8().is_err());
        assert!(r.take(1).is_err());
    }

    #[test]
    fn integers_roundtrip() {
        let mut w = ByteWriter::new();
        w.u8(0xAB);
        w.u16le(0xBEEF);
        w.u24le(0xDEADBE);
        w.u32le(0xFEED_FACE);
        w.u64le(u64::MAX - 3);
        let buf = w.into_vec();
        let mut r = ByteReader::new(&buf);
        assert_eq!(r.u8().unwrap(), 0xAB);
        assert_eq!(r.u16le().unwrap(), 0xBEEF);
        assert_eq!(r.u24le().unwrap(), 0xDEADBE);
        assert_eq!(r.u32le().unwrap(), 0xFEED_FACE);
        assert_eq!(r.u64le().unwrap(), u64::MAX - 3);
        assert!(r.is_empty());
    }
}
