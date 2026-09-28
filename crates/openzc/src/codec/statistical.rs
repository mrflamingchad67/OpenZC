//! Statistical pipeline: transform, LZ, context model, and entropy coding.
//!
//! This is the engine's default for compressible data. It is the only pipeline
//! that models context, and therefore the only one that adapts within a chunk.
//!
//! # Design
//!
//! The LZ stage reuses the same match finder and parsing idea as
//! [`super::lz`], but instead of byte-oriented tokens it drives one range coder
//! with four kinds of symbol:
//!
//! * **Sequence symbols** — one per `(literal run length, match length, offset
//!   width)` triple, packed into a 16-bit symbol. Offsets and over-long lengths
//!   escape to [`Symbol::Extra`], which costs one extra symbol but keeps every
//!   value inside a modelled alphabet.
//! * **Literal symbols** — one per literal byte, drawn from one of 16 order-1
//!   contexts keyed on the previous literal's high nibble. This is the
//!   "context modelling" stage, and it is where the ratio advantage over
//!   [`super::lz`] comes from on text and source code.
//! * **Extra symbols** — the low-order bytes of an over-long length or a
//!   multi-byte offset.
//!
//! # Why every value is a symbol
//!
//! An earlier design spliced raw bits straight into the coded stream through a
//! side channel. That is faster, but it creates a second read position which can
//! desynchronise from the coder, and the result is a frame that decodes to
//! *plausible but wrong* bytes rather than an error. Coding extras as ordinary
//! symbols against a uniform 256-way model costs the same eight bits and keeps a
//! single, provably consistent stream.
//!
//! # Determinism
//!
//! Models are trained from the chunk's own statistics before encoding and
//! rebuilt identically at decode time from the transmitted counts. No adaptive
//! state crosses a frame boundary, so a frame is decodable in isolation — which
//! is what makes parallel chunk decoding possible.

use crate::analyze::Analysis;
use crate::bytes::{ByteReader, ByteWriter};
use crate::codec::{Codec, DecodeContext, EncodeContext, MatchContext};
use crate::config::Level;
use crate::entropy::{self, Decoder, Encoder, Model};
use crate::error::{Error, Result};
use crate::format::PipelineId;
use crate::matchfinder::{MatchFinder, MAX_MATCH, MIN_MATCH};

/// Contexts for the literal model, keyed on the previous literal's high nibble.
const LIT_CONTEXTS: usize = 16;
/// Symbols per literal context.
const LIT_SYMBOLS: usize = 256;

/// Alphabet size of the sequence model.
///
/// Layout of a sequence symbol:
/// ```text
/// bit  10     : a match follows
/// bits 8..=9  : offset width class
/// bits 4..=7  : match length field
/// bits 0..=3  : literal run length field
/// ```
const SEQ_SYMBOLS: usize = 0x800;

/// Set in a sequence symbol when a match follows the literal run.
const SEQ_HAS_MATCH: u16 = 0x400;
/// Mask for the offset-width class.
const SEQ_OFF_CLASS_MASK: u16 = 0x03;
/// Bit position of the offset-width class.
const SEQ_OFF_CLASS_SHIFT: u32 = 8;
/// Literal-length field value meaning "an extra symbol carries the real length".
const SEQ_LIT_LEN_EXT: usize = 15;
/// Match-length field value meaning "an extra symbol carries the real length".
const SEQ_MATCH_LEN_EXT: usize = 15;
/// Mask for the match-length field.
const SEQ_MATCH_LEN_MASK: u16 = 0x0F;
/// Mask for the literal-length field.
const SEQ_LIT_LEN_MASK: u16 = 0x0F;

/// The uniform 256-way model used for escaped bytes.
///
/// These bytes are lengths and offset remainders with no useful structure, so an
/// adaptive model would add table cost without saving bits.
fn byte_model() -> Result<Model> {
    entropy::uniform_model(LIT_SYMBOLS)
}

/// Bytes of model table this codec writes per chunk, in the worst case.
///
/// A dense encoding of the tables above — the sequence model's
/// `2 + (SEQ_SYMBOLS - 1) * 2` bytes plus 16 literal tables of
/// `2 + (LIT_SYMBOLS - 1) * 2` — comes to about 12 KiB, which is why the tables
/// are written sparsely instead (see [`entropy::write_sparse_table`]). This
/// constant is the ceiling, kept as the bound the design is reasoned about;
/// the cost actually paid is set by how many symbols the trained model puts
/// above the frequency floor.
pub const fn worst_case_table_bytes() -> usize {
    2 + (SEQ_SYMBOLS - 1) * 2 + LIT_CONTEXTS * (2 + (LIT_SYMBOLS - 1) * 2)
}

/// Smallest chunk for which this codec is worth attempting.
///
/// Sparse tables cost a few hundred bytes on real data rather than the ~12 KiB a
/// dense encoding would, so the model is affordable well below this size. What
/// still argues for a floor is convergence: a context model needs enough
/// sequences before its frequencies mean anything, and a chunk too small to
/// train one is better served by the fast pipeline's simpler decisions.
///
/// This is a *measured* constant, not a guess: see the
/// `model_tables_are_sparse` and `model_tables_pay_for_themselves` tests, which
/// guard both the sparsity and the amortisation claim. The planner still
/// measures the real result before committing, so a wrong threshold costs
/// ratio, never correctness.
pub const MIN_USEFUL_CHUNK: usize = 48 * 1024;

/// Statistical codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct StatisticalCodec;

impl StatisticalCodec {
    pub fn new() -> Self {
        Self
    }

    /// Walk `input`, emitting one [`Sequence`] per LZ block.
    fn parse(&self, input: &[u8], finder: &mut dyn MatchFinder, window: usize) -> Vec<Sequence> {
        let mut out: Vec<Sequence> = Vec::new();
        let mut pos = 0usize;
        let mut lit_start = 0usize;

        while pos + MIN_MATCH as usize <= input.len() {
            // Search *before* inserting `pos`, or the finder would find `pos`
            // matching itself and reject it. Same ordering requirement as the
            // fast pipeline's parser.
            let m = finder.find(input, pos, window, MAX_MATCH);
            let _ = finder.insert(input, pos);
            let m = match m {
                Some(m) => m,
                None => {
                    pos += 1;
                    continue;
                }
            };
            out.push(Sequence {
                lit_start: lit_start as u32,
                lit_len: (pos - lit_start) as u32,
                m: Some((m.offset, m.length)),
            });
            for i in pos..pos + m.length as usize {
                let _ = finder.insert(input, i);
            }
            pos += m.length as usize;
            lit_start = pos;
        }

        // A literal-only sequence terminates the stream, so one must always be
        // present — even with zero literals when the input ends on a match.
        if lit_start < input.len() || out.last().is_none_or(|s| s.m.is_some()) {
            out.push(Sequence {
                lit_start: lit_start as u32,
                lit_len: (input.len() - lit_start) as u32,
                m: None,
            });
        }
        out
    }
}

/// One LZ block: a literal run followed by an optional back-reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sequence {
    lit_start: u32,
    lit_len: u32,
    /// `Some((offset, length))` when a match follows the literals.
    m: Option<(u32, u32)>,
}

/// The literals belonging to a sequence.
#[inline]
fn literal_slice<'a>(input: &'a [u8], s: &Sequence) -> &'a [u8] {
    let start = s.lit_start as usize;
    &input[start..start + s.lit_len as usize]
}

/// Context for the literal at `i`: the previous literal's high nibble.
///
/// The first literal of a run uses context 0. Keying on the previous *literal*
/// rather than the previous output byte keeps the model identical whether or not
/// a match intervened, which is what lets a sequence boundary be decoded without
/// replaying match state.
#[inline]
fn literal_context(lits: &[u8], i: usize) -> usize {
    if i == 0 {
        0
    } else {
        usize::from(lits[i - 1] >> 4).min(LIT_CONTEXTS - 1)
    }
}

/// Symbol carrying the high byte of an escaped literal-run length.
/// Reserved for future use; kept so the escape encoding has a named constant.
const _EXT_RESERVED: u8 = 0;

/// Number of low-order literal-length bytes an escape carries.
const LIT_LEN_EXT_BYTES: usize = 2;
/// Number of bytes an offset costs, from its magnitude.
#[inline]
fn offset_width(offset: u32) -> usize {
    match offset {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0xFF_FFFF => 3,
        _ => 4,
    }
}

/// Train the sequence model, returning raw observation counts.
///
/// A frame with one distinct sequence must still be encodable, so the plain
/// no-match packings are given a count: without them the model would assign them
/// the floor, and a symbol that happens to be coded first would cost far more
/// than the others.
fn train_sequence(seqs: &[Sequence]) -> Vec<u32> {
    let mut counts = vec![0u32; SEQ_SYMBOLS];
    for s in seqs {
        counts[usize::from(pack(s).0)] += 1;
    }
    for (i, c) in counts.iter_mut().enumerate() {
        if *c == 0 && i < SEQ_LIT_LEN_EXT {
            *c = 1;
        }
    }
    counts
}

/// Train the per-context literal models, returning raw observation counts.
///
/// Each context gets its **own** table. Pooling the counts into one large table
/// and slicing it afterwards would leave each slice summing to only a fraction of
/// the total, which is not a valid distribution — every model must carry the full
/// mass.
///
/// The counts are normalised separately on each side by
/// [`entropy::normalise_trained`], which is what keeps a context honest: most
/// contexts see only a fraction of the byte alphabet, and a naive normalisation
/// would hand the unseen majority most of the probability mass.
fn train_literals(input: &[u8], seqs: &[Sequence]) -> Vec<Vec<u32>> {
    let mut per_context: Vec<Vec<u32>> = vec![vec![0u32; LIT_SYMBOLS]; LIT_CONTEXTS];
    for s in seqs {
        let lits = literal_slice(input, s);
        for (i, &b) in lits.iter().enumerate() {
            let ctx = literal_context(lits, i);
            per_context[ctx][usize::from(b)] += 1;
        }
    }
    per_context
}

/// Normalise a trained count table into a coder model.
///
/// Both alphabets are compile-time constants well inside the coder's limits, so
/// this cannot fail; surfacing that as an error rather than an assertion keeps
/// the assumption checked instead of asserted.
fn model_for(counts: &mut Vec<u32>) -> Result<Model> {
    let freqs = entropy::normalise_trained(counts).ok_or(Error::Unsupported(
        "trained table is outside the coder's limits",
    ))?;
    entropy::model_from(&freqs)
}

/// Pack a sequence into its 16-bit symbol.
///
/// Returns `(symbol, escaped literal length, escaped match length)`. The
/// escapes are `None` when the corresponding field is coded directly.
fn pack(s: &Sequence) -> (u16, Option<u32>, Option<u32>) {
    let (lit_field, lit_ext) = if s.lit_len as usize >= SEQ_LIT_LEN_EXT {
        (SEQ_LIT_LEN_EXT, Some(s.lit_len))
    } else {
        (s.lit_len as usize, None)
    };

    match s.m {
        None => ((lit_field as u16) & SEQ_LIT_LEN_MASK, lit_ext, None),
        Some((offset, mlen)) => {
            let ml = mlen - MIN_MATCH;
            let (ml_field, ml_ext) = if ml as usize >= SEQ_MATCH_LEN_EXT {
                (SEQ_MATCH_LEN_EXT, Some(ml))
            } else {
                (ml as usize, None)
            };
            let off_class = (offset_width(offset) as u16 - 1) & SEQ_OFF_CLASS_MASK;
            let sym = SEQ_HAS_MATCH
                | (off_class << SEQ_OFF_CLASS_SHIFT)
                | ((ml_field as u16) << 4)
                | ((lit_field as u16) & SEQ_LIT_LEN_MASK);
            (sym, lit_ext, ml_ext)
        }
    }
}

/// Literal-run length implied by a sequence symbol and its escape.
fn literal_len(sym: u16, ext: Option<u32>) -> usize {
    match ext {
        Some(v) => v as usize,
        None => usize::from(sym & SEQ_LIT_LEN_MASK),
    }
}

/// Match length implied by a sequence symbol and its escape.
fn match_len(sym: u16, ext: Option<u32>) -> usize {
    let field = usize::from((sym >> 4) & SEQ_MATCH_LEN_MASK);
    let base = if field == SEQ_MATCH_LEN_EXT { 0 } else { field };
    base + MIN_MATCH as usize + ext.unwrap_or(0) as usize
}

/// Fixed prefix before the model tables: the coded length.
const PREFIX_LEN: usize = 4;

impl Codec for StatisticalCodec {
    fn id(&self) -> PipelineId {
        PipelineId::Statistical
    }

    fn name(&self) -> &'static str {
        "statistical"
    }

    /// Needs enough data to amortise the model tables, and something to model.
    fn candidate(&self, analysis: &Analysis, _level: Level) -> bool {
        if analysis.looks_incompressible() {
            return false;
        }
        analysis.sampled >= MIN_USEFUL_CHUNK
    }

    fn encode(&self, ctx: &EncodeContext<'_>) -> Result<Vec<u8>> {
        let input = ctx.input;
        let mut w = ByteWriter::with_capacity(input.len() / 2 + 8192);

        if input.is_empty() {
            w.u32le(0);
            return Ok(w.into_vec());
        }

        let window = input.len();
        let mut mc = MatchContext::build(ctx.level, input.len(), window, None);
        let seqs = self.parse(input, mc.finder.as_mut(), window);

        // Models are built first because coding needs them, but they are written
        // *after* the coded stream: the payload layout is
        // `coded_len | coded | tables`, so a decoder can size the coded section
        // before it has to parse anything else.
        //
        // What is written is the raw counts, not the frequencies: the decoder
        // derives identical frequencies from identical counts, which is both
        // smaller and exact.
        let mut seq_counts = train_sequence(&seqs);
        let mut lit_counts = train_literals(input, &seqs);
        let mut tables = ByteWriter::with_capacity(1024);
        entropy::write_counts(&mut tables, &seq_counts);
        for table in &lit_counts {
            entropy::write_counts(&mut tables, table);
        }

        let seq_model = model_for(&mut seq_counts)?;
        let mut lit_models = Vec::with_capacity(LIT_CONTEXTS);
        for table in &mut lit_counts {
            lit_models.push(model_for(table)?);
        }
        let byte = byte_model()?;

        let mut enc = Encoder::new();
        for s in &seqs {
            let (sym, lit_ext, ml_ext) = pack(s);

            // The decoder reads the extended literal-run length immediately
            // after the sequence symbol, so it must be emitted there.
            enc.encode_symbol(u32::from(sym), &seq_model)?;
            if let Some(v) = lit_ext {
                for k in 0..LIT_LEN_EXT_BYTES {
                    enc.encode_symbol((v >> (8 * k)) & 0xFF, &byte)?;
                }
            }

            let lits = literal_slice(input, s);
            for (i, &b) in lits.iter().enumerate() {
                let c = literal_context(lits, i);
                enc.encode_symbol(u32::from(b), &lit_models[c])?;
            }

            if let Some((offset, _)) = s.m {
                if let Some(v) = ml_ext {
                    enc.encode_symbol(v & 0xFF, &byte)?;
                }
                for k in 0..offset_width(offset) {
                    enc.encode_symbol((offset >> (8 * k)) & 0xFF, &byte)?;
                }
            }
        }

        let coded = enc.finish()?;
        w.u32le(coded.len() as u32);
        w.bytes(&coded);
        w.bytes(tables.as_slice());
        Ok(w.into_vec())
    }

    fn decode(&self, ctx: &DecodeContext<'_>, out: &mut [u8]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let mut r = ByteReader::new(ctx.payload);
        let prefix = r.take(PREFIX_LEN)?;
        let coded_len = u32::from_le_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]) as usize;
        if coded_len > r.remaining() {
            return Err(Error::CorruptEntropyModel(
                "coded stream longer than payload",
            ));
        }
        let coded = r.take(coded_len)?;
        let mut seq_counts = entropy::read_counts(&mut r)?;
        let mut lit_counts = Vec::with_capacity(LIT_CONTEXTS);
        for _ in 0..LIT_CONTEXTS {
            lit_counts.push(entropy::read_counts(&mut r)?);
        }

        if seq_counts.len() != SEQ_SYMBOLS {
            return Err(Error::CorruptEntropyModel(
                "sequence model has the wrong size",
            ));
        }
        for table in &lit_counts {
            if table.len() != LIT_SYMBOLS {
                return Err(Error::CorruptEntropyModel(
                    "literal model has the wrong size",
                ));
            }
        }

        let seq_model = model_for(&mut seq_counts)?;
        let mut lit_models = Vec::with_capacity(LIT_CONTEXTS);
        for table in &mut lit_counts {
            lit_models.push(model_for(table)?);
        }
        let byte = byte_model()?;

        let mut dec = Decoder::new(coded)?;
        let target = out.len();
        let mut pos = 0usize;

        loop {
            let sym = dec.decode_symbol(&seq_model)? as u16;
            let has_match = sym & SEQ_HAS_MATCH != 0;

            let lit_ext = if usize::from(sym & SEQ_LIT_LEN_MASK) == SEQ_LIT_LEN_EXT {
                let mut val = 0u32;
                for k in 0..LIT_LEN_EXT_BYTES {
                    val |= dec.decode_symbol(&byte)? << (8 * k);
                }
                Some(val)
            } else {
                None
            };

            let lit_len = literal_len(sym, lit_ext);
            if lit_len > target - pos {
                return Err(Error::CorruptSequence("literal run overruns chunk"));
            }

            // Literals are decoded and written as they arrive. The context comes
            // from the previous literal of this run, which is the byte just
            // written to `out`.
            for i in 0..lit_len {
                let c = if i == 0 {
                    0
                } else {
                    usize::from(out[pos - 1] >> 4).min(LIT_CONTEXTS - 1)
                };
                let b = dec.decode_symbol(&lit_models[c])?;
                if b > 255 {
                    return Err(Error::CorruptSequence("literal outside the byte range"));
                }
                out[pos] = b as u8;
                pos += 1;
            }

            if !has_match {
                if pos != target {
                    return Err(Error::CorruptSequence("terminator before end of chunk"));
                }
                break;
            }

            let ml_ext = if usize::from((sym >> 4) & SEQ_MATCH_LEN_MASK) == SEQ_MATCH_LEN_EXT {
                Some(dec.decode_symbol(&byte)?)
            } else {
                None
            };
            let mlen = match_len(sym, ml_ext);
            let mlen = mlen.min(MAX_MATCH as usize);

            // The offset width class is carried in the sequence symbol, so the
            // decoder consumes exactly as many bytes as the encoder wrote.
            let off_class = usize::from((sym >> SEQ_OFF_CLASS_SHIFT) & SEQ_OFF_CLASS_MASK) + 1;
            let mut offset = 0u32;
            for k in 0..off_class {
                offset |= dec.decode_symbol(&byte)? << (8 * k);
            }

            let offset = offset as usize;
            if offset == 0 || offset > pos {
                return Err(Error::CorruptSequence(
                    "match distance exceeds output so far",
                ));
            }
            if mlen > target - pos {
                return Err(Error::CorruptSequence("match overruns chunk"));
            }

            // The copy must run forward, one byte at a time, wherever the match
            // reaches back into the bytes it is writing — see the fast pipeline
            // for why memmove semantics would be wrong here.
            let (head, tail) = out.split_at_mut(pos);
            let src = pos - offset;
            let bulk = offset.min(mlen);
            tail[..bulk].copy_from_slice(&head[src..src + bulk]);
            for i in bulk..mlen {
                tail[i] = tail[i - offset];
            }
            pos += mlen;
        }

        if pos != target {
            return Err(Error::CorruptSequence("stream ended before chunk end"));
        }
        Ok(())
    }
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
        let c = StatisticalCodec::new();
        let a = Analysis::of(data).unwrap();
        let ctx = EncodeContext {
            input: data,
            analysis: &a,
            level: Level::Default,
            dictionary: None,
            independent: true,
        };
        c.encode(&ctx).unwrap()
    }

    /// Split a payload into `(coded bytes, table bytes)` per the documented
    /// `coded_len | coded | tables` layout.
    fn split_payload(payload: &[u8]) -> (usize, usize) {
        let len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        assert!(
            len <= payload.len() - PREFIX_LEN,
            "coded length overruns payload"
        );
        (len, payload.len() - PREFIX_LEN - len)
    }

    fn roundtrip(data: &[u8]) -> Vec<u8> {
        let payload = encode(data);
        let mut out = vec![0u8; data.len()];
        let c = StatisticalCodec::new();
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
        payload
    }

    #[test]
    fn roundtrip_empty() {
        let payload = encode(b"");
        assert_eq!(payload.len(), PREFIX_LEN);
    }

    #[test]
    fn roundtrip_small() {
        // Correctness is independent of size: even chunks far below the
        // break-even point must round-trip, because the encoder and decoder are
        // driven by the same layout. Only *usefulness* is size-gated.
        for n in [1usize, 2, 3, 4, 8, 16, 100, 1000, 4095, 4096, 4097] {
            roundtrip(&pseudo_random(n, n as u32 + 1));
        }
    }

    #[test]
    fn roundtrip_text() {
        roundtrip(
            "the quick brown fox jumps over the lazy dog. "
                .repeat(1000)
                .as_bytes(),
        );
    }

    #[test]
    fn roundtrip_repetitive() {
        roundtrip(&vec![0x41u8; 50_000]);
    }

    #[test]
    fn roundtrip_random() {
        roundtrip(&pseudo_random(20_000, 4242));
    }

    #[test]
    fn roundtrip_source_like() {
        let mut data = Vec::new();
        for i in 0..2000 {
            data.extend_from_slice(
                format!("    let value_{i} = compute(&self, {i})?;\n").as_bytes(),
            );
        }
        roundtrip(&data);
    }

    #[test]
    fn roundtrip_with_long_literal_runs() {
        // A long non-matching stretch forces the extended literal-length escape.
        let mut data = pseudo_random(70_000, 11);
        for b in data.iter_mut().take(5000) {
            *b = b'x';
        }
        roundtrip(&data);
    }

    #[test]
    fn roundtrip_various() {
        let base: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
        for n in [4usize, 5, 63, 64, 1000, 4096, 9000, 20_000] {
            roundtrip(&base[..n]);
        }
    }

    #[test]
    fn model_tables_are_sparse() {
        // The tables are what decide whether this pipeline is viable at all: a
        // dense encoding of both models is ~12 KiB of fixed per-chunk cost, which
        // no amount of modelling repays on a normal chunk. Guard that the sparse
        // encoding actually stays sparse on real data.
        let data = "the quick brown fox jumps over the lazy dog. ".repeat(6000);
        let payload = encode(data.as_bytes());
        let (coded, tables) = split_payload(&payload);

        let bound = worst_case_table_bytes();
        assert!(
            bound > 8 * 1024,
            "dense bound should be ~12 KiB, got {bound}"
        );
        assert!(
            tables * 20 < bound,
            "tables were {tables} bytes against a {bound} byte dense bound"
        );
        // Every table is at least a two-byte header plus its first count, so a
        // chunk that produces no sequences at all cannot beat this.
        assert!(
            tables >= 2 * (1 + LIT_CONTEXTS),
            "tables were only {tables} bytes"
        );
        assert!(coded > 0, "coded stream was empty");
    }

    #[test]
    fn model_tables_pay_for_themselves() {
        // The point of the context model: on a chunk large enough to amortise the
        // tables, the whole payload — tables included — must beat the fast LZ
        // pipeline, not merely the coded portion.
        let data = "the quick brown fox jumps over the lazy dog. ".repeat(6000);
        let lzf = super::super::lz::LzFastCodec::new();
        let a = Analysis::of(data.as_bytes()).unwrap();
        let ctx = EncodeContext {
            input: data.as_bytes(),
            analysis: &a,
            level: Level::Default,
            dictionary: None,
            independent: true,
        };
        let fast = lzf.encode(&ctx).unwrap();
        let stat = encode(data.as_bytes());
        assert!(
            stat.len() < fast.len(),
            "statistical {} was not smaller than lz-fast {}",
            stat.len(),
            fast.len()
        );
    }

    #[test]
    fn repetitive_data_shrinks() {
        assert!(encode(&vec![9u8; 50_000]).len() < 1000);
    }

    #[test]
    fn decode_rejects_bad_prefix_length() {
        let c = StatisticalCodec::new();
        let mut payload = encode(b"hello world hello world hello world");
        payload[0..4].copy_from_slice(&999_999u32.to_le_bytes());
        let mut out = vec![0u8; 35];
        let dec = DecodeContext {
            payload: &payload,
            content_size: 35,
            transform: TransformId::None,
            dictionary: None,
            independent: true,
        };
        assert!(c.decode(&dec, &mut out).is_err());
    }

    #[test]
    fn decode_never_panics_on_truncated_payload() {
        let c = StatisticalCodec::new();
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(30);
        let payload = encode(&data);
        let mut out = vec![0u8; data.len()];
        for cut in [1usize, 4, 8, 20, payload.len() - 1] {
            let dec = DecodeContext {
                payload: &payload[..cut],
                content_size: data.len(),
                transform: TransformId::None,
                dictionary: None,
                independent: true,
            };
            let _ = c.decode(&dec, &mut out);
        }
    }

    #[test]
    fn decode_never_panics_on_garbage() {
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(30);
        let payload = encode(&data);
        let c = StatisticalCodec::new();
        for i in 0..payload.len().min(300) {
            for delta in [1u8, 0x40, 0x80, 0xFF] {
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
                let _ = c.decode(&dec, &mut out);
            }
        }
    }

    #[test]
    fn candidate_rejects_tiny_and_incompressible() {
        let c = StatisticalCodec::new();
        assert!(!c.candidate(&Analysis::of(&[1u8; 10]).unwrap(), Level::Default));
        // Below the break-even the tables would dominate, so it is not even
        // worth attempting.
        assert!(!c.candidate(&Analysis::of(&[7u8; 1024]).unwrap(), Level::Default));
        let random = pseudo_random(8192, 5);
        assert!(!c.candidate(&Analysis::of(&random).unwrap(), Level::Default));
        let text = "hello world ".repeat(8000);
        assert!(c.candidate(&Analysis::of(text.as_bytes()).unwrap(), Level::Default));
    }

    #[test]
    fn packing_roundtrips_lengths() {
        for (lit_len, m) in [
            (0usize, None),
            (5, None),
            (15, Some((1u32, 4u32))),
            (14, Some((300, 100))),
            (20, Some((70_000, 258))),
        ] {
            let s = Sequence {
                lit_start: 0,
                lit_len: lit_len as u32,
                m,
            };
            let (sym, lit_ext, ml_ext) = pack(&s);
            assert_eq!(literal_len(sym, lit_ext), lit_len, "lit_len {lit_len}");
            assert_eq!(sym & SEQ_HAS_MATCH != 0, m.is_some());
            if let Some((_, mlen)) = m {
                assert_eq!(
                    match_len(sym, ml_ext) as u32,
                    mlen,
                    "match len for {lit_len}"
                );
            }
        }
    }

    #[test]
    fn models_have_expected_sizes() {
        let data = b"hello hello hello world world world".repeat(40);
        let mut mc = MatchContext::build(Level::Default, data.len(), data.len(), None);
        let seqs = StatisticalCodec::new().parse(&data, mc.finder.as_mut(), data.len());
        assert_eq!(train_sequence(&seqs).len(), SEQ_SYMBOLS);
        let literals = train_literals(&data, &seqs);
        assert_eq!(literals.len(), LIT_CONTEXTS);
        assert!(literals.iter().all(|t| t.len() == LIT_SYMBOLS));
    }

    #[test]
    fn every_context_model_is_a_valid_distribution() {
        // Each context carries the full probability mass; a pooled table sliced
        // per context would not, and the coder rejects it.
        let data = b"the quick brown fox jumps over the lazy dog ".repeat(200);
        let mut mc = MatchContext::build(Level::Default, data.len(), data.len(), None);
        let seqs = StatisticalCodec::new().parse(&data, mc.finder.as_mut(), data.len());
        for mut table in train_literals(&data, &seqs) {
            let freqs = entropy::normalise_trained(&mut table).expect("context table normalises");
            let sum: u64 = freqs.iter().map(|&c| u64::from(c)).sum();
            assert_eq!(
                sum,
                u64::from(entropy::TOTAL),
                "context table must be normalised"
            );
            assert!(freqs.iter().all(|&c| c > 0), "no zero frequencies");
            assert!(entropy::model_from(&freqs).is_ok());
        }
    }

    #[test]
    fn unobserved_symbols_stay_at_the_floor() {
        // The reason `normalise_trained` exists: a context that only ever saw a
        // handful of byte values must not hand the unseen majority most of the
        // probability mass, which is what a plain scaling normalisation does.
        let mut counts = vec![0u32; 256];
        counts[b'a' as usize] = 45;
        counts[b' ' as usize] = 5;
        let freqs = entropy::normalise_trained(&mut counts).unwrap();

        let unobserved: u64 = {
            let mut mass = 0u64;
            for (i, &c) in freqs.iter().enumerate() {
                if i != b'a' as usize && i != b' ' as usize {
                    mass += u64::from(c);
                }
            }
            mass
        };
        let total = u64::from(entropy::TOTAL);
        assert_eq!(
            unobserved, 254,
            "every unseen symbol should hold exactly one unit"
        );
        assert!(
            u64::from(freqs[b'a' as usize]) * 4 > total / 2,
            "the dominant symbol lost its mass"
        );
    }
}
