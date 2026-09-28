//! Entropy stage: frequency models plus a range coder.
//!
//! # Why a library primitive here
//!
//! The entropy stage is the one part of a compressor where a subtle
//! fixed-point or normalisation bug does not produce garbage — it produces
//! bytes that *decode cleanly to the wrong thing*. That failure mode is
//! expensive to test for and easy to introduce, so this stage is built on
//! [`constriction`], a research-grade implementation whose probability models
//! are constructed to be exactly invertible.
//!
//! Everything above it is OpenZC's own: the container format, the LZ stage, the
//! order-1 context model, and the adaptive pipeline selection. This module owns
//! the model *representation* and the coder, not the design.
//!
//! # Model representation
//!
//! Every model here is a contiguous categorical distribution over
//! `0..alphabet` with 16-bit fixed-point precision, normalised to
//! [`TOTAL`]. Alphabet sizes are bounded by [`MAX_ALPHABET`], and
//! [`normalise`] guarantees two properties the coder relies on:
//!
//! * the frequencies sum to exactly `TOTAL`, and
//! * no frequency is zero, so every symbol in the support is encodable.
//!
//! A zero-frequency symbol would be unrepresentable and would turn a modelling
//! mistake into a decode failure; [`normalise`] floors instead of rejecting.

use constriction::stream::model::ContiguousCategoricalEntropyModel;
use constriction::stream::queue::{DefaultRangeDecoder, DefaultRangeEncoder};
use constriction::stream::{Decode, Encode};

use crate::bytes::{ByteReader, ByteWriter};
use crate::error::{Error, Result};

/// Fixed-point precision of every model: frequencies sum to `1 << 16`.
///
/// The library's default preset uses 24-bit precision. 16 bits is chosen
/// deliberately: it makes [`TOTAL`] fit the `u16` table encoding below, which
/// is worth far more than the extra precision — a large alphabet's model table
/// is a fixed per-chunk cost, and at 24 bits it would quadruple.
pub const PRECISION: usize = 16;

/// Fixed-point precision as the coder's models express it.
const TOTAL_LOG2: u32 = PRECISION as u32;

/// Total frequency mass of every model.
pub const TOTAL: u32 = 1 << TOTAL_LOG2;

/// Largest alphabet accepted. Bounds the model's table so a corrupt length in a
/// stream cannot request an unbounded allocation.
pub const MAX_ALPHABET: usize = 4096;

/// A frequency table: one count per symbol, summing to [`TOTAL`].
pub type Frequencies = Vec<u32>;

/// A categorical model over the contiguous alphabet `0..len()`.
pub type Model = ContiguousCategoricalEntropyModel<u32, Vec<u32>, PRECISION>;

/// Normalise `counts` into a valid [`Frequencies`] table.
///
/// Every entry is floored at one before scaling, then the rounding residue is
/// corrected in bulk so the sum is exact. Exactness matters: the coder rejects a
/// model whose frequencies do not sum to `TOTAL`, and a table that is "close
/// enough" would turn a modelling choice into a decode failure.
///
/// Returns `None` if the alphabet is empty or larger than [`TOTAL`], where a
/// non-zero frequency per symbol is impossible.
pub fn normalise(counts: &mut Vec<u32>) -> Option<Frequencies> {
    let n = counts.len();
    let target = u64::from(TOTAL);
    if n == 0 || n as u64 > target {
        return None;
    }

    let mut total: u64 = 0;
    for c in counts.iter_mut() {
        if *c == 0 {
            *c = 1;
        }
        total += u64::from(*c);
    }

    let mut assigned: u64 = 0;
    for c in counts.iter_mut() {
        let want = (u64::from(*c) * target) / total;
        let v = want.max(1);
        *c = v as u32;
        assigned += v;
    }

    // Order by current count so the residue lands where a unit is worth least.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]));

    match assigned.cmp(&target) {
        std::cmp::Ordering::Less => {
            let mut diff = target - assigned;
            let mut k = 0usize;
            while diff > 0 {
                counts[order[k % n]] += 1;
                diff -= 1;
                k += 1;
            }
        }
        std::cmp::Ordering::Greater => {
            // The minimum-frequency floor can overshoot; take units back, never
            // dropping a symbol to zero.
            let mut diff = assigned - target;
            while diff > 0 {
                let mut moved = false;
                for &i in &order {
                    if diff == 0 {
                        break;
                    }
                    if counts[i] > 1 {
                        counts[i] -= 1;
                        diff -= 1;
                        moved = true;
                    }
                }
                if !moved {
                    return None;
                }
            }
        }
        std::cmp::Ordering::Equal => {}
    }

    // Verify rather than assume.
    let sum: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    if sum != target {
        return None;
    }
    Some(std::mem::take(counts))
}

/// Normalise trained `counts` into a table that keeps unobserved symbols at the
/// floor.
///
/// [`normalise`] treats a zero count as one and then *scales* it, which is
/// correct for a hand-built table but wrong for a trained one. The scale factor
/// is `TOTAL / sum(counts)`, so an unobserved symbol ends up holding
/// `TOTAL / sum(counts)` units — for a table trained on 45 literals that is 256
/// units each, and 211 unobserved symbols swallow 82% of the mass. The model
/// then codes data it has never seen, and the table has no sparse symbols left to
/// elide.
///
/// So the floor is reserved instead: every symbol gets one unit, and the
/// remaining `TOTAL - n` is shared out in proportion to the observed counts. An
/// unobserved symbol ends up at exactly one, which both models the truth and
/// makes the table sparse (see [`write_sparse_table`]).
///
/// Returns `None` for an empty or oversized alphabet.
pub fn normalise_trained(counts: &mut Vec<u32>) -> Option<Frequencies> {
    let n = counts.len();
    if n == 0 || n as u64 > u64::from(TOTAL) {
        return None;
    }

    let observed: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    if observed == 0 {
        // Nothing was seen, so the table is uniform by definition.
        return normalise(counts);
    }

    let free = u64::from(TOTAL) - n as u64;
    let mut assigned = n as u64;
    let mut order: Vec<(usize, u64)> = Vec::with_capacity(n);
    for (i, c) in counts.iter_mut().enumerate() {
        if *c == 0 {
            *c = 1;
            continue;
        }
        let share = (u64::from(*c) * free) / observed;
        *c = 1 + share as u32;
        assigned += share;
        order.push((i, share));
    }

    // Integer division always rounds down, so the residue is spread over the
    // least significant symbols first — that is where a unit costs the model
    // least. `order` is non-empty because `observed > 0`.
    order.sort_unstable_by_key(|&(_, share)| share);
    let mut diff = u64::from(TOTAL) - assigned;
    let mut k = 0usize;
    while diff > 0 {
        counts[order[k % order.len()].0] += 1;
        diff -= 1;
        k += 1;
    }

    let sum: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    if sum != u64::from(TOTAL) {
        return None;
    }
    Some(std::mem::take(counts))
}

/// Build a coder model from a normalised table.
pub fn model_from(freqs: &[u32]) -> Result<Model> {
    if freqs.is_empty() || freqs.len() > MAX_ALPHABET {
        return Err(Error::CorruptEntropyModel("alphabet size out of range"));
    }
    Model::from_nonzero_fixed_point_probabilities(freqs.iter().copied(), false)
        .map_err(|_| Error::CorruptEntropyModel("frequencies are not a valid distribution"))
}

/// A uniform model over `0..alphabet`.
///
/// Used for fields with no useful structure, where an adaptive model would only
/// add table overhead.
pub fn uniform_model(alphabet: usize) -> Result<Model> {
    if alphabet == 0 || alphabet > MAX_ALPHABET {
        return Err(Error::CorruptEntropyModel("alphabet size out of range"));
    }
    let mut freqs = vec![1u32; alphabet];
    let normalised =
        normalise(&mut freqs).ok_or(Error::CorruptEntropyModel("cannot build a uniform model"))?;
    model_from(&normalised)
}
/// Serialise a table's training counts, sparse by construction.
///
/// # Why counts, not frequencies
///
/// The obvious encoding is to normalise on the way out and store the
/// frequencies. That costs `2 * n` bytes per table — 4 KiB for the sequence
/// model and 512 bytes for each of sixteen literal contexts, a fixed ~12 KiB bill
/// per chunk that no amount of good compression repays on a small chunk.
///
/// The frequencies are a pure function of the counts, and both sides already run
/// the same [`normalise_trained`] on the same input, so storing frequencies
/// stores a redundant derivation. Storing the *counts* is both smaller and exact:
/// counts sum to the number of observations, so each is a small integer, and the
/// unobserved majority costs nothing at all.
///
/// Layout:
/// ```text
/// <u16le alphabet> <varint present> (<varint delta> <varint count>) x present
/// ```
///
/// `delta` is the gap from the previous symbol, starting from zero, so observed
/// symbols in a dense alphabet cost one byte each. The reconstruction is exact,
/// which is the property that matters: the encoder's model and the decoder's
/// model are identical by construction rather than by coincidence.
pub fn write_counts(w: &mut ByteWriter, counts: &[u32]) {
    debug_assert!(!counts.is_empty());
    let present = counts.iter().filter(|&&c| c > 0).count();

    w.u16le(counts.len() as u16);
    w.varint(present as u64);
    // `prev` is the one-past position of the last stored symbol, so the delta
    // below is always at least one and consecutive symbols cost a single byte
    // however far into the alphabet they sit.
    let mut prev = 0u64;
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let idx = i as u64;
        w.varint(idx + 1 - prev);
        w.varint(u64::from(c));
        prev = idx + 1;
    }
}

/// Bytes needed to serialise a table with `present` non-zero counts, assuming
/// each needs a one-byte delta and a one-byte value.
///
/// The alphabet size does not appear: unobserved symbols cost nothing, so a
/// table's size is set entirely by how much of it the data actually used.
pub fn counts_size(present: usize) -> usize {
    3 + present * 2
}

/// Parse a sparsely serialised count table.
///
/// Validation covers exactly what a corrupt stream could break: the alphabet
/// must be a size the coder can build, the implied positions must stay inside
/// it, and a stored count must be non-zero, since a zero count would be
/// non-canonical — the encoder never writes one, and accepting it would let two
/// different byte strings mean the same table.
///
/// There is deliberately no check on the *sum* of the counts: unlike a frequency
/// table, a count table is normalised after parsing, and both sides derive the
/// same model from the same counts.
pub fn read_counts(r: &mut ByteReader<'_>) -> Result<Vec<u32>> {
    let n = usize::from(r.u16le()?);
    if n == 0 {
        return Err(Error::CorruptEntropyModel("empty alphabet"));
    }
    if n > MAX_ALPHABET {
        return Err(Error::CorruptEntropyModel("alphabet too large"));
    }
    let present = r.varint()? as usize;
    if present > n {
        return Err(Error::CorruptEntropyModel("more counts than symbols"));
    }

    let mut counts = vec![0u32; n];
    let mut idx = 0u64;
    for _ in 0..present {
        // `delta` is one more than the gap, so a run of consecutive observed
        // symbols costs one byte however far into the alphabet it reaches.
        let delta = r.varint()?;
        if delta == 0 {
            return Err(Error::CorruptEntropyModel("count symbols must advance"));
        }
        idx += delta;
        if idx > n as u64 {
            return Err(Error::CorruptEntropyModel("count symbol past the alphabet"));
        }
        let count = r.varint()?;
        if count == 0 {
            return Err(Error::CorruptEntropyModel("zero count is not canonical"));
        }
        counts[(idx - 1) as usize] =
            u32::try_from(count).map_err(|_| Error::CorruptEntropyModel("count out of range"))?;
    }
    Ok(counts)
}

/// Range-coder encoder.
///
/// Symbols are emitted one at a time with the model supplied per symbol, which
/// is what allows a context model to change the distribution mid-stream.
#[derive(Debug)]
pub struct Encoder {
    inner: DefaultRangeEncoder,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            inner: DefaultRangeEncoder::new(),
        }
    }

    /// Encode one symbol against `model`.
    pub fn encode_symbol(&mut self, symbol: u32, model: &Model) -> Result<()> {
        self.inner
            .encode_symbol(symbol as usize, model)
            .map_err(|_| Error::CorruptEntropyModel("symbol outside the model"))
    }

    /// Seal the stream and return its bytes.
    pub fn finish(self) -> Result<Vec<u8>> {
        let words = self
            .inner
            .into_compressed()
            .map_err(|_| Error::CorruptEntropyModel("could not seal the entropy stream"))?;
        Ok(words_to_bytes(&words))
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Range-coder decoder.
#[derive(Debug)]
pub struct Decoder {
    inner: DefaultRangeDecoder,
}

impl Decoder {
    /// Build a decoder over raw coded bytes.
    pub fn new(data: &[u8]) -> Result<Self> {
        let words = bytes_to_words(data)?;
        DefaultRangeDecoder::from_compressed(words)
            .map(|inner| Self { inner })
            .map_err(|_| Error::CorruptEntropyModel("malformed entropy stream"))
    }

    /// Decode one symbol against `model`.
    pub fn decode_symbol(&mut self, model: &Model) -> Result<u32> {
        self.inner
            .decode_symbol(model)
            .map(|s| s as u32)
            .map_err(|_| Error::CorruptEntropyModel("coded stream does not match the model"))
    }
}

/// Pack the coder's 32-bit words into bytes, little-endian.
fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut w = ByteWriter::with_capacity(words.len() * 4 + 4);
    w.u32le(words.len() as u32);
    for &v in words {
        w.u32le(v);
    }
    w.into_vec()
}

/// Unpack coder words from bytes, little-endian.
///
/// A trailing partial word is dropped, so a truncated or over-long stream is
/// rejected rather than silently misread.
fn bytes_to_words(data: &[u8]) -> Result<Vec<u32>> {
    if data.len() < 4 {
        return Err(Error::CorruptEntropyModel("entropy stream is too short"));
    }
    let mut r = ByteReader::new(data);
    let n = r.u32le()? as usize;
    if n > (r.remaining() / 4) {
        return Err(Error::CorruptEntropyModel("entropy stream is truncated"));
    }
    let mut words = Vec::with_capacity(n);
    for _ in 0..n {
        words.push(r.u32le()?);
    }
    Ok(words)
}

/// Encode a symbol sequence against one shared model.
pub fn encode_symbols(freqs: &[u32], symbols: &[u32]) -> Result<Vec<u8>> {
    let model = model_from(freqs)?;
    let mut enc = Encoder::new();
    for &s in symbols {
        enc.encode_symbol(s, &model)?;
    }
    enc.finish()
}

/// Decode exactly `n` symbols against one shared model.
pub fn decode_symbols(freqs: &[u32], data: &[u8], n: usize) -> Result<Vec<u32>> {
    let model = model_from(freqs)?;
    let mut dec = Decoder::new(data)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(dec.decode_symbol(&model)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG so the tests need no RNG dependency.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    fn counts_of(syms: &[u32], alphabet: usize) -> Vec<u32> {
        let mut c = vec![0u32; alphabet];
        for &s in syms {
            c[s as usize] += 1;
        }
        c
    }

    #[test]
    fn normalise_sums_to_total_with_no_zeros() {
        for n in [1usize, 2, 3, 16, 256, 1000] {
            let mut c: Vec<u32> = (0..n).map(|i| (i as u32 % 7) + 1).collect();
            let f = normalise(&mut c).unwrap();
            assert_eq!(f.len(), n);
            let sum: u64 = f.iter().map(|&x| u64::from(x)).sum();
            assert_eq!(sum, u64::from(TOTAL), "alphabet {n}");
            assert!(f.iter().all(|&x| x > 0), "alphabet {n}");
        }
    }

    #[test]
    fn normalise_handles_all_zero_counts() {
        let mut c = vec![0u32; 8];
        let f = normalise(&mut c).unwrap();
        assert!(f.iter().all(|&x| x > 0));
        assert_eq!(
            f.iter().map(|&x| u64::from(x)).sum::<u64>(),
            u64::from(TOTAL)
        );
    }

    #[test]
    fn normalise_rejects_empty_and_oversized() {
        assert!(normalise(&mut Vec::new()).is_none());
        // Above `TOTAL` there is no room for a non-zero count per symbol.
        assert!(normalise(&mut vec![1u32; TOTAL as usize + 1]).is_none());
    }

    #[test]
    fn model_from_rejects_oversized_alphabet() {
        assert!(model_from(&[]).is_err());
        let big = vec![1u32; MAX_ALPHABET + 1];
        assert!(model_from(&big).is_err());
    }

    #[test]
    fn roundtrip_uniform() {
        let alphabet = 256usize;
        let mut s = 42u64;
        let syms: Vec<u32> = (0..10_000)
            .map(|_| (lcg(&mut s) % alphabet as u64) as u32)
            .collect();
        let freqs = normalise(&mut counts_of(&syms, alphabet)).unwrap();
        let enc = encode_symbols(&freqs, &syms).unwrap();
        assert_eq!(decode_symbols(&freqs, &enc, syms.len()).unwrap(), syms);
    }

    #[test]
    fn roundtrip_skewed_with_rare_symbol() {
        let alphabet = 256usize;
        let mut s = 7u64;
        let syms: Vec<u32> = (0..20_000)
            .map(|_| {
                if lcg(&mut s) % 1000 == 0 {
                    255
                } else {
                    (lcg(&mut s) % 8) as u32
                }
            })
            .collect();
        let freqs = normalise(&mut counts_of(&syms, alphabet)).unwrap();
        let enc = encode_symbols(&freqs, &syms).unwrap();
        assert_eq!(decode_symbols(&freqs, &enc, syms.len()).unwrap(), syms);
    }

    #[test]
    fn roundtrip_single_symbol_alphabet() {
        let syms = vec![0u32; 1000];
        let freqs = normalise(&mut counts_of(&syms, 1)).unwrap();
        let enc = encode_symbols(&freqs, &syms).unwrap();
        assert_eq!(decode_symbols(&freqs, &enc, syms.len()).unwrap(), syms);
    }

    #[test]
    fn roundtrip_every_alphabet_size() {
        for alphabet in [2usize, 3, 17, 255, 256, 257, 1024] {
            let mut s = alphabet as u64;
            let syms: Vec<u32> = (0..5_000)
                .map(|_| (lcg(&mut s) % alphabet as u64) as u32)
                .collect();
            let freqs = normalise(&mut counts_of(&syms, alphabet)).unwrap();
            let enc = encode_symbols(&freqs, &syms).unwrap();
            assert_eq!(
                decode_symbols(&freqs, &enc, syms.len()).unwrap(),
                syms,
                "alphabet {alphabet}"
            );
        }
    }

    #[test]
    fn skewed_distribution_compresses() {
        // 90% one symbol must cost well under 8 bits per symbol.
        let mut freqs = vec![0u32; 256];
        freqs[0] = 900_000;
        for c in freqs.iter_mut().skip(1) {
            *c = 1000;
        }
        let freqs = normalise(&mut freqs).unwrap();
        let syms = vec![0u32; 9000];
        let enc = encode_symbols(&freqs, &syms).unwrap();
        let bits = enc.len() as f64 * 8.0 / syms.len() as f64;
        assert!(bits < 0.5, "{bits} bits per symbol");
    }

    #[test]
    fn empty_sequence_roundtrips() {
        let freqs = normalise(&mut counts_of(&[], 2)).unwrap();
        let enc = encode_symbols(&freqs, &[]).unwrap();
        assert_eq!(decode_symbols(&freqs, &enc, 0).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn model_can_change_between_symbols() {
        // This is the property the context-modelling stage depends on.
        let a = uniform_model(2).unwrap();
        let b = uniform_model(256).unwrap();
        let mut enc = Encoder::new();
        enc.encode_symbol(0, &a).unwrap();
        enc.encode_symbol(200, &b).unwrap();
        enc.encode_symbol(1, &a).unwrap();
        let bytes = enc.finish().unwrap();

        let mut dec = Decoder::new(&bytes).unwrap();
        assert_eq!(dec.decode_symbol(&a).unwrap(), 0);
        assert_eq!(dec.decode_symbol(&b).unwrap(), 200);
        assert_eq!(dec.decode_symbol(&a).unwrap(), 1);
    }

    #[test]
    fn count_serialisation_roundtrips_exactly() {
        // The property the format depends on: counts must survive the round trip
        // unchanged, because both sides derive the model from them.
        for n in [1usize, 2, 17, 256, 300, MAX_ALPHABET] {
            for pattern in 0..4 {
                let mut c = vec![0u32; n];
                for (i, v) in c.iter_mut().enumerate() {
                    *v = match pattern {
                        0 => 1, // dense
                        1 => {
                            if i % 3 == 0 {
                                7
                            } else {
                                0
                            }
                        } // sparse
                        2 => {
                            if i == 0 {
                                1
                            } else {
                                0
                            }
                        } // single symbol
                        _ => (i as u32 % 11) + 1, // uneven
                    };
                }
                let mut w = ByteWriter::new();
                write_counts(&mut w, &c);
                let buf = w.into_vec();
                let mut r = ByteReader::new(&buf);
                assert_eq!(read_counts(&mut r).unwrap(), c, "n={n} pattern={pattern}");
                assert!(
                    r.is_empty(),
                    "n={n} pattern={pattern} left {} bytes",
                    r.remaining()
                );
            }
        }
    }

    #[test]
    fn count_serialisation_is_sparse() {
        // The whole point of storing counts: an alphabet the data barely touched
        // must cost almost nothing. `counts_size` assumes one-byte deltas, so this
        // is a bound rather than an exact figure — the one-byte-delta case is
        // pinned by `consecutive_counts_cost_one_byte_each`.
        let mut c = vec![0u32; 256];
        for i in [7usize, 8, 9, 200] {
            c[i] = 3;
        }
        let mut w = ByteWriter::new();
        write_counts(&mut w, &c);
        let buf = w.into_vec();
        assert!(
            buf.len() <= counts_size(4) + 4,
            "{} bytes for four counts",
            buf.len()
        );
        assert!(
            buf.len() * 10 < 2 + 255 * 2,
            "dense would be {} bytes",
            2 + 255 * 2
        );
    }

    #[test]
    fn consecutive_counts_cost_one_byte_each() {
        // A dense alphabet must not pay for a delta per byte it does not skip.
        let c: Vec<u32> = (0..64).map(|_| 1).collect();
        let mut w = ByteWriter::new();
        write_counts(&mut w, &c);
        assert_eq!(w.as_slice().len(), counts_size(64));
    }

    #[test]
    fn read_counts_rejects_non_canonical_streams() {
        // A delta of zero would map two counts onto one symbol.
        let mut w = ByteWriter::new();
        w.u16le(4);
        w.varint(1);
        w.varint(0);
        w.varint(5);
        assert!(matches!(
            read_counts(&mut ByteReader::new(w.as_slice())),
            Err(Error::CorruptEntropyModel(_))
        ));

        // A zero count is never written, so accepting one would let two different
        // byte strings mean the same table.
        let mut w = ByteWriter::new();
        w.u16le(4);
        w.varint(1);
        w.varint(1);
        w.varint(0);
        assert!(matches!(
            read_counts(&mut ByteReader::new(w.as_slice())),
            Err(Error::CorruptEntropyModel(_))
        ));

        // Positions must stay inside the declared alphabet.
        let mut w = ByteWriter::new();
        w.u16le(4);
        w.varint(2);
        w.varint(3);
        w.varint(1);
        w.varint(3);
        w.varint(1);
        assert!(matches!(
            read_counts(&mut ByteReader::new(w.as_slice())),
            Err(Error::CorruptEntropyModel(_))
        ));

        // More counts than symbols.
        let mut w = ByteWriter::new();
        w.u16le(2);
        w.varint(3);
        assert!(matches!(
            read_counts(&mut ByteReader::new(w.as_slice())),
            Err(Error::CorruptEntropyModel(_))
        ));

        // Empty and oversized alphabets.
        let mut w = ByteWriter::new();
        w.u16le(0);
        assert!(matches!(
            read_counts(&mut ByteReader::new(w.as_slice())),
            Err(Error::CorruptEntropyModel(_))
        ));
        let mut w = ByteWriter::new();
        w.u16le(u16::MAX);
        assert!(matches!(
            read_counts(&mut ByteReader::new(w.as_slice())),
            Err(Error::CorruptEntropyModel(_))
        ));
    }

    #[test]
    fn read_counts_never_panics_on_truncated_input() {
        let mut c: Vec<u32> = (0..64).map(|i| (i % 5) as u32 * 3).collect();
        c[40] = 9;
        let mut w = ByteWriter::new();
        write_counts(&mut w, &c);
        let buf = w.into_vec();
        for cut in 0..buf.len() {
            let _ = read_counts(&mut ByteReader::new(&buf[..cut]));
        }
    }

    #[test]
    fn trained_normalisation_beats_plain_scaling_on_unseen_symbols() {
        // The bug `normalise_trained` exists to prevent: scaling a table trained
        // on a handful of observations hands the unseen majority most of the mass.
        let mut counts = vec![0u32; 256];
        counts[b'a' as usize] = 45;
        counts[b' ' as usize] = 5;
        let unseen = |f: &Frequencies| -> u64 {
            let mut mass = 0u64;
            for (i, &c) in f.iter().enumerate() {
                if i != b'a' as usize && i != b' ' as usize {
                    mass += u64::from(c);
                }
            }
            mass
        };

        let mut naive = counts.clone();
        let naive = normalise(&mut naive).unwrap();
        let mut trained = counts.clone();
        let trained = normalise_trained(&mut trained).unwrap();

        assert_eq!(
            unseen(&trained),
            254,
            "each unseen symbol should hold one unit"
        );
        let naive_mass = unseen(&naive);
        assert!(
            naive_mass > u64::from(TOTAL) / 2,
            "scaling was expected to swamp the unseen symbols, but gave {naive_mass}"
        );
    }

    #[test]
    fn trained_normalisation_is_a_valid_distribution() {
        for n in [1usize, 2, 16, 256, MAX_ALPHABET] {
            for observed in [0usize, 1, 2, n / 2, n] {
                let mut c = vec![0u32; n];
                for (i, v) in c.iter_mut().take(observed.min(n)).enumerate() {
                    *v = (i as u32 % 7) + 1;
                }
                let f = normalise_trained(&mut c).expect("valid alphabet");
                let sum: u64 = f.iter().map(|&x| u64::from(x)).sum();
                assert_eq!(sum, u64::from(TOTAL), "n={n} observed={observed}");
                assert!(f.iter().all(|&x| x > 0), "n={n} observed={observed}");
                assert!(model_from(&f).is_ok(), "n={n} observed={observed}");
            }
        }
    }

    #[test]
    fn trained_normalisation_rejects_empty_and_oversized() {
        assert!(normalise_trained(&mut Vec::new()).is_none());
        assert!(normalise_trained(&mut vec![1u32; TOTAL as usize + 1]).is_none());
    }

    #[test]
    fn decoder_never_panics_on_truncated_stream() {
        let mut c: Vec<u32> = (0..256).map(|i| (i as u32 % 17) + 1).collect();
        let freqs = normalise(&mut c).unwrap();
        let mut s = 3u64;
        let syms: Vec<u32> = (0..500).map(|_| (lcg(&mut s) % 256) as u32).collect();
        let enc = encode_symbols(&freqs, &syms).unwrap();
        for cut in 0..enc.len() {
            let _ = decode_symbols(&freqs, &enc[..cut], 500);
        }
    }

    #[test]
    fn decoder_rejects_malformed_word_stream() {
        assert!(Decoder::new(&[]).is_err());
        assert!(Decoder::new(&[0, 0, 0]).is_err());
        // Declares more words than are present.
        assert!(Decoder::new(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0]).is_err());
    }

    #[test]
    fn word_byte_roundtrip() {
        let words = vec![0u32, 1, 0xFFFF_FFFF, 0x1234_5678];
        let bytes = words_to_bytes(&words);
        assert_eq!(bytes_to_words(&bytes).unwrap(), words);
    }
}
