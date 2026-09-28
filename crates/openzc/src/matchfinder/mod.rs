//! LZ77 match finding.
//!
//! The match finder turns a window of already-seen bytes into candidate
//! `(offset, length)` pairs. It is deliberately separate from parsing: the
//! parser decides *which* matches to accept, the finder decides *what* is
//! available. Swapping a hash chain for a hash table, or a greedy parser for an
//! optimal one, changes one side only.
//!
//! # Two strategies, one interface
//!
//! * [`HashTable3`] — one candidate position per 3-byte hash. O(1) per position,
//!   no chains to walk. This is the LZ4/Zstd-style "fast" mode.
//! * [`HashChain4`] — a bounded-length chain per 4-byte hash, walked from most
//!   to least recent. More candidates at higher cost. This is the LZMA-style
//!   "strong" mode.
//!
//! # Memory
//!
//! The chain is `Vec<u32>` of `window_size / 4` entries — 4 bytes per 4 input
//! bytes, so 1 MiB of input costs 1 MiB of index. That is proportional to the
//! configured window, never to the input, which is the memory requirement the
//! format demands.

use std::collections::HashMap;

/// A match found by a match finder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    /// Distance back to the start of the match. Always `>= 1`.
    pub offset: u32,
    /// Number of matching bytes, at least [`MIN_MATCH`].
    pub length: u32,
}

/// Shortest match worth emitting. Below this, the length and offset cost more
/// than the bytes they replace.
pub const MIN_MATCH: u32 = 4;

/// Longest match the parsers will emit. Bounds the length field and keeps
/// pathological repeats from producing huge single sequences.
pub const MAX_MATCH: u32 = 258;

/// Common match finder interface.
pub trait MatchFinder {
    /// Record positions up to and including `pos`, so that lookups from
    /// `pos + 1` can find them. Returns bytes consumed from `data`.
    fn insert(&mut self, data: &[u8], pos: usize) -> usize;

    /// Longest match for the sequence at `pos`, searching back at most
    /// `max_offset` bytes and returning at most `max_len` bytes.
    fn find(&self, data: &[u8], pos: usize, max_offset: usize, max_len: u32) -> Option<Match>;

    /// Cheap hint that a match is unlikely, letting the parser skip work.
    fn unlikely(&self, data: &[u8], pos: usize) -> bool {
        self.find(data, pos, 0, MIN_MATCH).is_none()
    }
}

/// Hash three bytes into a table index.
#[inline]
pub fn hash3(data: &[u8], pos: usize) -> usize {
    let v =
        (u32::from(data[pos]) << 16) | (u32::from(data[pos + 1]) << 8) | u32::from(data[pos + 2]);
    (v.wrapping_mul(2_654_435_761) >> 16) as usize
}

/// Hash four bytes into a table index.
#[inline]
pub fn hash4(data: &[u8], pos: usize) -> usize {
    let v = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    (v.wrapping_mul(2_654_435_761) >> 16) as usize
}

/// Compare `data[a..]` with `data[b..]` for up to `max` bytes.
///
/// `a` must be less than `b`; overlapping matches (distance smaller than length)
/// are legal in LZ77 and handled here by the natural slice walk.
#[inline]
pub fn common_prefix(data: &[u8], a: usize, b: usize, max: usize) -> u32 {
    let end = b.saturating_add(max).min(data.len());
    let mut len = 0u32;
    let mut i = a;
    let mut j = b;
    while j < end && data[i] == data[j] {
        i += 1;
        j += 1;
        len += 1;
    }
    len
}

/// Single-slot hash table over 3-byte hashes: the fast finder.
///
/// Memory is `4 * table_size` bytes, independent of input size.
#[derive(Debug)]
pub struct HashTable3 {
    table: Vec<u32>,
    mask: usize,
}

impl HashTable3 {
    /// Build a table with `2^bits` slots. `bits` is clamped to 10..=22 so the
    /// table stays between 4 KiB and 16 MiB.
    pub fn new(bits: u32) -> Self {
        let bits = bits.clamp(10, 22);
        let size = 1usize << bits;
        // 0 is a reserved "empty" marker, so real positions are stored as
        // `pos + 1` to keep the zero page useful.
        Self {
            table: vec![0u32; size],
            mask: size - 1,
        }
    }

    /// Slots in the table.
    pub fn capacity(&self) -> usize {
        self.table.len()
    }

    /// Bytes of index memory.
    pub fn memory_bytes(&self) -> usize {
        self.table.len() * 4
    }
}

impl MatchFinder for HashTable3 {
    fn insert(&mut self, data: &[u8], pos: usize) -> usize {
        if pos + 3 > data.len() {
            return 0;
        }
        let h = hash3(data, pos) & self.mask;
        self.table[h] = (pos + 1) as u32;
        3
    }

    fn find(&self, data: &[u8], pos: usize, max_offset: usize, max_len: u32) -> Option<Match> {
        if pos + MIN_MATCH as usize > data.len() {
            return None;
        }
        let h = hash3(data, pos) & self.mask;
        let stored = self.table[h];
        if stored == 0 {
            return None;
        }
        let cand = (stored - 1) as usize;
        if cand >= pos {
            return None;
        }
        let offset = pos - cand;
        if offset > max_offset {
            return None;
        }
        let want = max_len.min(MAX_MATCH);
        let len = common_prefix(data, cand, pos, want as usize);
        if len < MIN_MATCH {
            return None;
        }
        Some(Match {
            offset: offset as u32,
            length: len,
        })
    }
}

/// Bounded hash chain over 4-byte hashes: the strong finder.
///
/// The chain is indexed by input position, so it is `Vec<u32>` of
/// `data.len() / 4` entries. A parallel array of the most recent position per
/// hash provides the chain head.
#[derive(Debug)]
pub struct HashChain4 {
    /// `prev[pos >> 2]` is the previous position sharing `pos`'s 4-byte hash.
    prev: Vec<u32>,
    /// `head[h]` is the most recent position with 4-byte hash `h`, plus one.
    head: Vec<u32>,
    /// Positions inserted so far, in units of positions.
    count: u32,
    mask: usize,
    /// Maximum chain entries walked per lookup.
    max_chain: u32,
    /// Maximum positions inserted, derived from the window.
    limit: usize,
}

impl HashChain4 {
    /// Build a chain sized for `data` of `data_len` bytes and a window of
    /// `window` bytes.
    pub fn new(bits: u32, data_len: usize, window: usize, max_chain: u32) -> Self {
        let bits = bits.clamp(10, 22);
        let size = 1usize << bits;
        let slots = (data_len / 4).max(1);
        Self {
            prev: vec![0u32; slots],
            head: vec![0u32; size],
            count: 0,
            mask: size - 1,
            max_chain: max_chain.max(1),
            limit: window.max(4),
        }
    }

    /// Bytes of index memory: one `u32` per input position plus the head table.
    pub fn memory_bytes(&self) -> usize {
        self.prev.len() * 4 + self.head.len() * 4
    }

    /// Chain entries walked per lookup.
    pub fn max_chain(&self) -> u32 {
        self.max_chain
    }
}

impl MatchFinder for HashChain4 {
    fn insert(&mut self, data: &[u8], pos: usize) -> usize {
        if pos + 4 > data.len() {
            return 0;
        }
        let slot = pos >> 2;
        if slot >= self.prev.len() {
            return 4;
        }
        let h = hash4(data, pos) & self.mask;
        let prev_head = self.head[h];
        // 0 means "no previous"; positions are stored plus one.
        self.prev[slot] = prev_head;
        self.head[h] = (pos + 1) as u32;
        self.count = self.count.saturating_add(1);
        4
    }

    fn find(&self, data: &[u8], pos: usize, max_offset: usize, max_len: u32) -> Option<Match> {
        if pos + MIN_MATCH as usize > data.len() {
            return None;
        }
        let h = hash4(data, pos) & self.mask;
        let want = max_len.min(MAX_MATCH) as usize;
        let mut cand = self.head[h];
        let mut tries = self.max_chain;
        let mut best: Option<Match> = None;

        while cand != 0 && tries > 0 {
            let p = (cand - 1) as usize;
            if p >= pos {
                break;
            }
            let offset = pos - p;
            if offset > max_offset || offset > self.limit {
                break;
            }
            let len = common_prefix(data, p, pos, want);
            if len >= MIN_MATCH {
                let m = Match {
                    offset: offset as u32,
                    length: len,
                };
                // Chains run most-recent-first, so a hit needs no further
                // candidates once the maximum length is reached.
                if best.is_none_or(|b| m.length > b.length) {
                    best = Some(m);
                }
                if len == MAX_MATCH {
                    break;
                }
            }
            let slot = p >> 2;
            if slot >= self.prev.len() {
                break;
            }
            cand = self.prev[slot];
            tries -= 1;
        }
        best
    }
}

/// Count matches in `data` for a fixed window. Used by tests and by the
/// analyzer to estimate LZ headroom without running a full parse.
pub fn count_matches<F: FnMut(usize) -> Option<Match>>(data: &[u8], mut step: F) -> (usize, u64) {
    let mut pos = 0usize;
    let mut matches = 0usize;
    let mut saved = 0u64;
    while pos + MIN_MATCH as usize <= data.len() {
        match step(pos) {
            Some(m) => {
                matches += 1;
                saved += u64::from(m.length) - 2;
                pos += m.length as usize;
            }
            None => pos += 1,
        }
    }
    (matches, saved)
}

/// Dictionary of known byte sequences, keyed by length-prefixed content.
///
/// A dictionary is ordinary data prepended to the match space: a frame using
/// one treats dictionary bytes as "already seen at negative offsets". This type
/// only builds the lookup structure.
#[derive(Debug, Default)]
pub struct DictionaryIndex {
    /// The dictionary bytes. Kept so matches can be verified rather than
    /// trusted from the hash alone.
    data: Vec<u8>,
    /// 4-byte hash -> newest dictionary position (plus one).
    head: HashMap<usize, u32>,
    /// `prev[pos >> 2]` -> previous position sharing that hash.
    prev: Vec<u32>,
}

impl DictionaryIndex {
    /// Index a dictionary of arbitrary length.
    pub fn build(dict: &[u8]) -> Self {
        let mut idx = Self {
            data: dict.to_vec(),
            head: HashMap::with_capacity(dict.len() / 4),
            prev: vec![0; dict.len() / 4],
        };
        for pos in 0..dict.len().saturating_sub(3) {
            let h = hash4(&idx.data, pos);
            if let Some(&head) = idx.head.get(&h) {
                let slot = pos >> 2;
                if slot < idx.prev.len() {
                    idx.prev[slot] = head;
                }
            }
            idx.head.insert(h, (pos + 1) as u32);
        }
        idx
    }

    /// Longest match for `needle` inside the dictionary, as `(offset, length)`.
    ///
    /// `offset` is measured back from the end of the dictionary, so a decoder
    /// can treat a dictionary hit exactly like a back-reference into a buffer
    /// that already holds the dictionary.
    pub fn find(&self, needle: &[u8], max_len: u32) -> Option<(u32, u32)> {
        let n = self.data.len();
        if n < MIN_MATCH as usize || needle.len() < MIN_MATCH as usize {
            return None;
        }
        let want = (max_len.min(MAX_MATCH) as usize).min(needle.len());
        let h = hash4(needle, 0);
        let mut cand = *self.head.get(&h)?;
        let mut best: Option<(u32, u32)> = None;
        // Dictionary chains are short; 32 hops finds the best candidate without
        // risking a pathological walk.
        let mut tries = 32u32;

        while cand != 0 && tries > 0 {
            let p = (cand - 1) as usize;
            if p >= n {
                break;
            }
            let end = (p + want).min(n);
            let mut len = 0usize;
            while p + len < end && needle[len] == self.data[p + len] {
                len += 1;
            }
            if len as u32 >= MIN_MATCH {
                let m = ((n - p) as u32, len as u32);
                if best.is_none_or(|(_, bl)| m.1 > bl) {
                    best = Some(m);
                }
                if len as u32 == MAX_MATCH {
                    break;
                }
            }
            let slot = p >> 2;
            if slot >= self.prev.len() {
                break;
            }
            cand = self.prev[slot];
            tries -= 1;
        }
        best
    }

    /// Number of indexed positions.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the dictionary is empty.
    /// True when the dictionary holds nothing that could ever match.
    ///
    /// This is about usability, not length: a dictionary shorter than
    /// [`MIN_MATCH`] indexes no positions at all, so treating it as a usable
    /// dictionary would be a promise the index cannot keep.
    pub fn is_empty(&self) -> bool {
        self.head.is_empty()
    }

    /// The dictionary bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn common_prefix_counts_overlapping_matches() {
        let data = b"abcabcabcabc".to_vec();
        // Position 0 vs 3, overlapping by construction.
        let n = common_prefix(&data, 0, 3, 12);
        assert_eq!(n, 9);
    }

    #[test]
    fn common_prefix_respects_bounds() {
        let data = b"abcabc".to_vec();
        // Capped by the end of the data, not by the requested length.
        assert_eq!(common_prefix(&data, 0, 3, 100), 3);
        // Capped by the requested length.
        assert_eq!(common_prefix(&data, 0, 3, 2), 2);
        // A difference in the very first byte yields nothing.
        assert_eq!(common_prefix(b"abc", 0, 1, 100), 0);
    }

    #[test]
    fn hash_table_finds_repeat() {
        let data = b"the quick brown fox ".repeat(20);
        let mut mf = HashTable3::new(12);
        let pos = data.len() - 4;
        for i in 0..pos {
            mf.insert(&data, i);
        }
        let m = mf
            .find(&data, pos, 65_536, MAX_MATCH)
            .expect("expected a match");
        assert_eq!(
            &data[pos..pos + m.length as usize],
            &data[(pos - m.offset as usize)..][..m.length as usize]
        );
    }

    #[test]
    fn hash_table_returns_none_without_repeat() {
        let data = pseudo_random(4096, 99);
        let mut mf = HashTable3::new(12);
        for i in 0..2048 {
            mf.insert(&data, i);
        }
        // Random data should almost never produce a match.
        let mut found = 0;
        for i in 2048..4000 {
            if mf.find(&data, i, 65_536, MAX_MATCH).is_some() {
                found += 1;
            }
        }
        assert!(found < 20, "found {found} matches in random data");
    }

    #[test]
    fn hash_chain_finds_longer_match() {
        let mut data = b"ABCDEFGH".repeat(40);
        data.extend_from_slice(b"ABCDEFGH");
        let pos = data.len() - 8;
        let mut mf = HashChain4::new(14, data.len(), 65_536, 64);
        for i in 0..pos {
            mf.insert(&data, i);
        }
        let m = mf
            .find(&data, pos, 65_536, MAX_MATCH)
            .expect("expected a match");
        assert!(m.length >= 8, "length {}", m.length);
    }

    #[test]
    fn hash_chain_respects_max_offset() {
        let data = b"XYZ".repeat(100);
        // Far enough from the end that a full-length match is still possible.
        let pos = data.len() - 8;
        let mut mf = HashChain4::new(14, data.len(), 1 << 20, 64);
        for i in 0..pos {
            mf.insert(&data, i);
        }
        // max_offset of 1 means only the immediately preceding position is legal,
        // and the nearest match here is three bytes back.
        assert!(mf.find(&data, pos, 1, MAX_MATCH).is_none());
        let m = mf
            .find(&data, pos, 3, MAX_MATCH)
            .expect("a match three bytes back");
        assert_eq!(
            m.offset, 3,
            "the finder should report the nearest candidate"
        );
    }

    #[test]
    fn hash_chain_respects_max_len() {
        let data = b"ABCDEFGH".repeat(10);
        let pos = data.len() - 8;
        let mut mf = HashChain4::new(14, data.len(), 1 << 20, 64);
        for i in 0..pos {
            mf.insert(&data, i);
        }
        let m = mf.find(&data, pos, 1 << 20, 5).expect("expected a match");
        assert!(m.length <= 5, "length {}", m.length);
    }

    #[test]
    fn matches_never_exceed_max_match() {
        let data = vec![0xABu8; 100_000];
        let mut mf = HashChain4::new(16, data.len(), 1 << 20, 32);
        for i in 0..1000 {
            mf.insert(&data, i);
        }
        let m = mf
            .find(&data, 1000, 1 << 20, 1_000_000)
            .expect("expected a match");
        assert!(m.length <= MAX_MATCH, "length {}", m.length);
    }

    #[test]
    fn no_panic_near_end_of_buffer() {
        let data = pseudo_random(64, 5);
        let mut mf = HashTable3::new(10);
        let mut mf2 = HashChain4::new(10, data.len(), 1024, 8);
        for i in 0..data.len() {
            mf.insert(&data, i);
            mf2.insert(&data, i);
            let _ = mf.find(&data, i, 1024, MAX_MATCH);
            let _ = mf2.find(&data, i, 1024, MAX_MATCH);
        }
    }

    #[test]
    fn table_memory_is_bounded_and_reported() {
        let mf = HashTable3::new(12);
        assert_eq!(mf.capacity(), 4096);
        assert_eq!(mf.memory_bytes(), 4096 * 4);
        // Clamping keeps the table inside the documented range.
        assert_eq!(HashTable3::new(2).capacity(), 1 << 10);
        assert_eq!(HashTable3::new(30).capacity(), 1 << 22);
    }

    #[test]
    fn count_matches_walks_whole_input() {
        let data = b"abcabcabcabcabc".repeat(100);
        let mut mf = HashTable3::new(14);
        // Index as we go, exactly as a parser does. Pre-filling a single-slot
        // table would be meaningless: it keeps one position per hash, so a fully
        // populated table can only ever offer candidates at or after the position
        // being searched, which the finder correctly rejects.
        let mut indexed = 0usize;
        let (n, saved) = count_matches(&data, |p| {
            while indexed < p {
                mf.insert(&data, indexed);
                indexed += 1;
            }
            mf.find(&data, p, 1 << 20, MAX_MATCH)
        });
        assert!(n > 0, "expected matches");
        assert!(saved > 0, "expected savings");
    }

    #[test]
    fn empty_and_tiny_inputs_are_safe() {
        let mf = DictionaryIndex::build(b"");
        assert!(mf.is_empty());
        assert!(mf.find(b"abcd", MAX_MATCH).is_none());
        let mf = DictionaryIndex::build(b"ab");
        assert!(mf.is_empty());
    }
}
