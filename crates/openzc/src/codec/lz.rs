//! Byte-oriented LZ pipeline (the "fast" mode).
//!
//! The token stream is designed so that decoding is a tight loop of `memcpy`
//! and slice fills with no arithmetic-heavy per-byte work — the LZ4 shape. No
//! entropy coder, no context model, no divisions. That is the whole point of
//! having this pipeline: a bound on decode cost that no input can push past.
//!
//! # Stream layout
//!
//! A sequence of blocks. Each block is:
//!
//! ```text
//! <varint literal_count> <literal_count bytes> <varint offset_minus_1> <varint match_len_minus_4>
//! ```
//!
//! except that a block with no following match ends the stream after its
//! literals, marked by the high bit of the length varint. That terminator is
//! what lets a decoder know when to stop without a separate count, so a chunk
//! costs one extra byte rather than a fixed header.
//!
//! # Why offset is stored minus one
//!
//! Distance zero is meaningless, so storing `offset - 1` wastes nothing on the
//! low end and keeps zero a reserved "no match" value in the varint space.

use crate::analyze::Analysis;
use crate::bytes::{ByteReader, ByteWriter};
use crate::codec::{Codec, DecodeContext, EncodeContext, MatchContext};
use crate::config::Level;
use crate::error::{Error, Result};
use crate::format::PipelineId;
use crate::matchfinder::{MatchFinder, MAX_MATCH, MIN_MATCH};

/// Bit set on a literal-count varint to mark the final block.
const LAST_BLOCK: u64 = 1 << 63;

/// How to choose among candidate matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseStrategy {
    /// Take the first match that qualifies. Fastest, weakest.
    Greedy,
    /// Take a match, then check whether a slightly later match is long enough
    /// to be worth restarting for. The classic speed/ratio compromise.
    Lazy,
    /// Consider every position. Best ratio, slowest.
    Optimal,
}

impl ParseStrategy {
    /// Strategy implied by a compression level.
    pub fn for_level(level: Level) -> Self {
        match level {
            Level::None | Level::Fast => ParseStrategy::Greedy,
            Level::Default => ParseStrategy::Lazy,
            Level::High | Level::Max => ParseStrategy::Lazy,
        }
    }
}

/// One parsed block: literals followed by an optional back-reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    /// Index of the first literal in the input.
    lit_start: u32,
    lit_len: u32,
    /// `None` for the final literal-only block.
    m: Option<(u32, u32)>,
}

/// Fast LZ codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct LzFastCodec;

impl LzFastCodec {
    pub fn new() -> Self {
        Self
    }

    /// Parse `input` into blocks.
    fn parse(
        &self,
        input: &[u8],
        finder: &mut dyn MatchFinder,
        strategy: ParseStrategy,
        window: usize,
    ) -> Vec<Block> {
        let n = input.len();
        let mut blocks: Vec<Block> = Vec::new();
        let mut pos = 0usize;
        let mut lit_start = 0usize;

        while pos + MIN_MATCH as usize <= n {
            // Search *before* inserting `pos`. The finder keys on the bytes at
            // the position being searched, so inserting first would make `pos`
            // its own match and the finder would (correctly) reject it.
            let m = finder.find(input, pos, window, MAX_MATCH);
            let _ = finder.insert(input, pos);

            let m = match m {
                Some(m) => m,
                None => {
                    pos += 1;
                    continue;
                }
            };

            // Lazy parsing: if the next position offers a meaningfully longer
            // match, emit the byte in between as a literal and take that match
            // instead. `start` tracks where the accepted match actually begins,
            // which is not always `pos` — getting this wrong silently shifts the
            // match by one byte.
            let mut chosen = m;
            let mut start = pos;
            if strategy == ParseStrategy::Lazy && m.length < MAX_MATCH {
                let next = pos + 1;
                if next + MIN_MATCH as usize <= n {
                    // Deliberately not inserted here: this is a speculative
                    // lookahead, and the single-slot table would lose the older,
                    // better candidate to it. If the match is taken, the indexing
                    // loop below inserts it anyway.
                    let m2 = finder.find(input, next, window, MAX_MATCH);
                    if let Some(m2) = m2 {
                        if m2.length > m.length + 1 {
                            pos += 1;
                            continue;
                        }
                        if m2.length > m.length {
                            chosen = m2;
                            start = next;
                        }
                    }
                }
            }

            let lit_len = (start - lit_start) as u32;
            let match_len = chosen.length;
            blocks.push(Block {
                lit_start: lit_start as u32,
                lit_len,
                m: Some((chosen.offset, match_len)),
            });

            // Index everything the match covered so later positions can find it.
            for i in start..start + match_len as usize {
                let _ = finder.insert(input, i);
            }
            pos = start + match_len as usize;
            lit_start = pos;
        }

        // A literal-only block terminates the stream, so one must always be
        // present — even with zero literals when the input ends exactly on a
        // match boundary. Without it the decoder would read past the payload
        // looking for the next token.
        match blocks.last() {
            Some(last) if last.m.is_none() => {}
            _ => blocks.push(Block {
                lit_start: lit_start as u32,
                lit_len: (n - lit_start) as u32,
                m: None,
            }),
        }
        blocks
    }

    /// Serialise blocks into the token stream.
    fn serialise(&self, input: &[u8], blocks: &[Block]) -> Vec<u8> {
        let mut w = ByteWriter::with_capacity(input.len() / 2 + 16);
        for (i, b) in blocks.iter().enumerate() {
            let is_last = b.m.is_none() && i + 1 == blocks.len();
            if is_last {
                w.varint(u64::from(b.lit_len) | LAST_BLOCK);
            } else {
                w.varint(u64::from(b.lit_len));
            }
            let s = b.lit_start as usize;
            let e = s + b.lit_len as usize;
            w.bytes(&input[s..e]);
            if let Some((offset, len)) = b.m {
                debug_assert!(offset >= 1);
                w.varint(u64::from(offset - 1));
                w.varint(u64::from(len - MIN_MATCH));
            }
        }
        w.into_vec()
    }

    /// Replay a token stream into `out`.
    fn replay(&self, payload: &[u8], out: &mut [u8], window: usize) -> Result<()> {
        let target = out.len();
        let mut r = ByteReader::new(payload);
        let mut pos = 0usize;

        loop {
            let raw = r.varint()?;
            let is_last = raw & LAST_BLOCK != 0;
            let lit_len = (raw & !LAST_BLOCK) as usize;

            if lit_len > target - pos {
                return Err(Error::CorruptSequence("literal run overruns chunk"));
            }
            let lits = r.take(lit_len)?;
            out[pos..pos + lit_len].copy_from_slice(lits);
            pos += lit_len;

            if is_last {
                // The terminator must coincide with the end of the chunk;
                // otherwise the stream is malformed.
                if pos != target {
                    return Err(Error::CorruptSequence("terminator before end of chunk"));
                }
                break;
            }

            let offset = r.varint()? as usize + 1;
            let match_len = r.varint()? as usize + MIN_MATCH as usize;

            if offset > pos {
                return Err(Error::CorruptSequence(
                    "match distance exceeds output so far",
                ));
            }
            if offset > window {
                return Err(Error::CorruptSequence("match distance exceeds window"));
            }
            if match_len > target - pos {
                return Err(Error::CorruptSequence("match overruns chunk"));
            }

            // A match may reach back into the bytes it is writing, so this copy
            // must run *forward*, one byte at a time: with an offset shorter than
            // the length, each byte has to see the value the previous one just
            // wrote. A `copy_within` would use memmove semantics and copy the
            // not-yet-written bytes instead, which is right for a run of one byte
            // (where every value is equal) and wrong for every other overlap.
            let (head, tail) = out.split_at_mut(pos);
            let src = pos - offset;
            // Bytes that lie entirely in the already-decoded region can be copied
            // in one go; only the part that overlaps the destination is
            // order-dependent.
            let bulk = offset.min(match_len);
            tail[..bulk].copy_from_slice(&head[src..src + bulk]);
            for i in bulk..match_len {
                tail[i] = tail[i - offset];
            }
            pos += match_len;
        }

        if pos != target {
            return Err(Error::CorruptSequence(
                "token stream ended before chunk end",
            ));
        }
        Ok(())
    }
}

impl Codec for LzFastCodec {
    fn id(&self) -> PipelineId {
        PipelineId::LzFast
    }

    fn name(&self) -> &'static str {
        "lz-fast"
    }

    /// Skip on data with no repetition at all: there is nothing for LZ to find,
    /// and `Store` is strictly better.
    fn candidate(&self, analysis: &Analysis, _level: Level) -> bool {
        !analysis.looks_incompressible()
    }

    fn encode(&self, ctx: &EncodeContext<'_>) -> Result<Vec<u8>> {
        let input = ctx.input;
        // Note: there is deliberately no "too short, just copy the bytes" fast
        // path. The payload must always be a valid token stream, because the
        // frame header records only the pipeline id — a decoder has no way to
        // discover that this one frame happens to hold raw bytes. `Store` is the
        // pipeline for "don't compress", and the planner picks it when it is
        // genuinely the smaller option.
        let window = input.len();
        // The level decides both how hard the finder searches and how greedily
        // the parser takes matches, so effort and ratio move together.
        let strategy = if ctx.independent {
            // A frame that references only itself is decoded in isolation, so
            // spending time on lookahead cannot be amortised across a window.
            ParseStrategy::Greedy
        } else {
            ParseStrategy::for_level(ctx.level)
        };
        let mut mc = MatchContext::build(
            ctx.level,
            input.len(),
            window,
            ctx.dictionary.map(|d| d.bytes()),
        );
        let blocks = self.parse(input, mc.finder.as_mut(), strategy, window);
        Ok(self.serialise(input, &blocks))
    }

    fn decode(&self, ctx: &DecodeContext<'_>, out: &mut [u8]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        self.replay(ctx.payload, out, out.len())
    }
}

/// Exposed for the pipeline's cost model: the number of tokens a given input
/// produces, used to predict whether the token stream will fit in the window.
pub fn estimate_tokens(input: &[u8], finder: &mut dyn MatchFinder, window: usize) -> usize {
    let n = input.len();
    let mut pos = 0usize;
    let mut tokens = 0usize;
    while pos + MIN_MATCH as usize <= n {
        // Same ordering rule as the parser: search, then insert.
        let m = finder.find(input, pos, window, MAX_MATCH);
        let _ = finder.insert(input, pos);
        match m {
            Some(m) => {
                tokens += 1;
                for i in pos..pos + m.length as usize {
                    let _ = finder.insert(input, i);
                }
                pos += m.length as usize;
            }
            None => pos += 1,
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::TransformId;

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

    fn encode(data: &[u8]) -> Vec<u8> {
        let c = LzFastCodec::new();
        let a = Analysis::of(data).unwrap();
        let ctx = EncodeContext {
            input: data,
            analysis: &a,
            level: Level::Default,
            dictionary: None,
            independent: false,
        };
        c.encode(&ctx).unwrap()
    }

    fn roundtrip(data: &[u8]) {
        let payload = encode(data);
        let mut out = vec![0u8; data.len()];
        let c = LzFastCodec::new();
        let dec = DecodeContext {
            payload: &payload,
            content_size: data.len(),
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        c.decode(&dec, &mut out)
            .unwrap_or_else(|e| panic!("decode failed for {} bytes: {e}", data.len()));
        assert_eq!(out, data, "roundtrip mismatch for {} bytes", data.len());
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(b"");
    }

    #[test]
    fn roundtrip_tiny_inputs() {
        for n in 0..80usize {
            roundtrip(&pseudo_random(n, n as u32 + 1));
        }
    }

    #[test]
    fn roundtrip_text() {
        let text = "the quick brown fox jumps over the lazy dog. ";
        let data = text.repeat(200).into_bytes();
        roundtrip(&data);
    }

    #[test]
    fn roundtrip_repetitive() {
        roundtrip(&vec![0x41u8; 100_000]);
        roundtrip(&b"ABCDEFGH".repeat(5000));
    }

    #[test]
    fn roundtrip_random() {
        roundtrip(&pseudo_random(50_000, 1234));
    }

    #[test]
    fn roundtrip_source_like() {
        let mut data = Vec::new();
        for i in 0..3000 {
            data.extend_from_slice(
                format!("    let value_{i} = compute(&self, {i})?;\n").as_bytes(),
            );
        }
        roundtrip(&data);
    }

    #[test]
    fn roundtrip_overlapping_matches() {
        // A run is the overlapping case: offset 1, long length.
        roundtrip(&[7u8; 70_000]);
    }

    #[test]
    fn roundtrip_overlapping_matches_with_varied_bytes() {
        // The case that separates a correct forward copy from a memmove: a short
        // repeating period, where the match reaches back into the bytes it is
        // writing and every byte value differs. A constant run cannot catch this,
        // because memmove and forward-copy agree when every byte is equal.
        for period in 2usize..=17 {
            let unit: Vec<u8> = (0..period).map(|i| (i * 37 + 11) as u8).collect();
            let mut data = unit.clone();
            while data.len() < 40_000 {
                data.extend_from_slice(&unit);
            }
            roundtrip(&data);
        }
    }

    #[test]
    fn decode_reproduces_an_overlapping_match_forward() {
        // Pin the semantics directly: over the literals "abcdef", a match with
        // offset 3 and length 12 extends the period, so the result is
        // "abcdef" followed by "def" four times. A memmove-style copy would
        // instead re-read the not-yet-written tail and produce garbage there.
        let mut w = ByteWriter::new();
        w.varint(6); // six literals
        w.bytes(b"abcdef");
        w.varint(2); // offset - 1, so offset 3
        w.varint(8); // length - MIN_MATCH, so length 12
        w.varint(LAST_BLOCK);
        let payload = w.into_vec();

        let mut out = vec![0u8; 18];
        let c = LzFastCodec::new();
        let dec = DecodeContext {
            payload: &payload,
            content_size: 18,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        c.decode(&dec, &mut out).unwrap();
        assert_eq!(&out[..], b"abcdefdefdefdefdef");
    }

    #[test]
    fn roundtrip_various_lengths() {
        let base: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
        for n in [1usize, 3, 4, 5, 63, 64, 65, 1000, 4095, 4096, 4097, 20_000] {
            roundtrip(&base[..n]);
        }
    }

    #[test]
    fn repetitive_data_shrinks() {
        // Collapsing runs to a handful of bytes is the RLE pipeline's job, not
        // this one's — LZ pays a token per [`MAX_MATCH`] bytes, so a 50 KB run has
        // a hard floor of `len / MAX_MATCH` tokens no matter how well it parses.
        // What LZ must guarantee is that the floor, not the input, sets the size.
        let data = vec![3u8; 50_000];
        let payload = encode(&data);
        let min_tokens = data.len() / MAX_MATCH as usize;
        let budget = min_tokens * 5 + 16;
        assert!(
            payload.len() < budget,
            "payload {} bytes, budget {}",
            payload.len(),
            budget
        );
        assert!(
            payload.len() * 50 < data.len(),
            "ratio too poor: {} of {}",
            payload.len(),
            data.len()
        );
    }

    #[test]
    fn text_compresses() {
        let text = "the quick brown fox jumps over the lazy dog. "
            .repeat(500)
            .into_bytes();
        let payload = encode(&text);
        assert!(
            payload.len() < text.len() / 4,
            "payload {} vs {}",
            payload.len(),
            text.len()
        );
    }

    #[test]
    fn decode_rejects_offset_beyond_output() {
        let mut w = ByteWriter::new();
        w.varint(0); // no literals
        w.varint(9999); // offset - 1, far beyond the 8 bytes decoded so far
        w.varint(0);
        let payload = w.into_vec();
        let c = LzFastCodec::new();
        let mut out = vec![0u8; 8];
        let dec = DecodeContext {
            payload: &payload,
            content_size: 8,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        assert!(c.decode(&dec, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_match_overrunning_chunk() {
        let mut w = ByteWriter::new();
        w.varint(4);
        w.bytes(b"abcd");
        w.varint(0); // offset 1
        w.varint(9999); // match length way past the end
        let payload = w.into_vec();
        let c = LzFastCodec::new();
        let mut out = vec![0u8; 8];
        let dec = DecodeContext {
            payload: &payload,
            content_size: 8,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        assert!(c.decode(&dec, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_literals_overrunning_chunk() {
        let mut w = ByteWriter::new();
        w.varint(100);
        w.bytes(b"only ten..");
        let payload = w.into_vec();
        let c = LzFastCodec::new();
        let mut out = vec![0u8; 10];
        let dec = DecodeContext {
            payload: &payload,
            content_size: 10,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        assert!(c.decode(&dec, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_missing_terminator() {
        // A block that claims a match but the payload ends first.
        let mut w = ByteWriter::new();
        w.varint(0);
        w.varint(0);
        w.varint(0);
        let payload = w.into_vec();
        let c = LzFastCodec::new();
        let mut out = vec![0u8; 8];
        let dec = DecodeContext {
            payload: &payload,
            content_size: 8,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        assert!(c.decode(&dec, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_early_terminator() {
        // A last-block marker that fires before the chunk is filled.
        let mut w = ByteWriter::new();
        w.varint(2 | LAST_BLOCK);
        w.bytes(b"ab");
        let payload = w.into_vec();
        let c = LzFastCodec::new();
        let mut out = vec![0u8; 8];
        let dec = DecodeContext {
            payload: &payload,
            content_size: 8,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        assert!(c.decode(&dec, &mut out).is_err());
    }

    #[test]
    fn decode_never_panics_on_garbage() {
        // Systematic single-byte corruption across a valid stream.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(50);
        let payload = encode(&data);
        let c = LzFastCodec::new();
        for i in 0..payload.len().min(400) {
            for delta in [0x01u8, 0x80, 0xFF] {
                let mut bad = payload.clone();
                bad[i] = bad[i].wrapping_add(delta);
                let mut out = vec![0u8; data.len()];
                let dec = DecodeContext {
                    payload: &bad,
                    content_size: data.len(),
                    transform: TransformId::None,
                    dictionary: None,
                    independent: true,
                };
                // Must either fail cleanly or produce something; never panic.
                let _ = c.decode(&dec, &mut out);
            }
        }
    }

    #[test]
    fn strategy_selection_by_level() {
        assert_eq!(ParseStrategy::for_level(Level::Fast), ParseStrategy::Greedy);
        assert_eq!(
            ParseStrategy::for_level(Level::Default),
            ParseStrategy::Lazy
        );
        assert_eq!(ParseStrategy::for_level(Level::Max), ParseStrategy::Lazy);
    }

    #[test]
    fn parse_handles_input_without_any_match() {
        let data = pseudo_random(1000, 77);
        let mut mf = crate::matchfinder::HashTable3::new(14);
        let blocks = LzFastCodec::new().parse(&data, &mut mf, ParseStrategy::Greedy, data.len());
        // A single trailing literal block is the expected shape.
        assert!(blocks.iter().all(|b| b.m.is_none()));
        assert_eq!(
            blocks.last().unwrap().lit_start as usize + blocks.last().unwrap().lit_len as usize,
            data.len()
        );
    }

    #[test]
    fn estimate_tokens_counts_blocks() {
        let data = vec![1u8; 10_000];
        let mut mf = crate::matchfinder::HashTable3::new(14);
        let t = estimate_tokens(&data, &mut mf, data.len());
        assert!(t > 0);
    }
}
