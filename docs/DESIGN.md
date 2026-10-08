# OpenZC design

Why the engine is built the way it is. The format itself is specified in
[`FORMAT.md`](FORMAT.md); this document is about the choices behind the code and
the reasoning that produced them, including the choices that turned out to be
wrong.

## The central idea

**Measure, don't guess.**

A compressor that picks its algorithm from a heuristic is betting. If the
heuristic is wrong, the file is bigger than it needed to be and nobody can tell,
because nothing measured it. OpenZC instead runs the plausible candidates on each
chunk of input and keeps whichever one actually produced the smallest output.

This has one overriding benefit: **a chunk can never grow**. `store` is always
measured as the last candidate, and a candidate is only kept if it is strictly
smaller. So "never make a file bigger" is a guarantee with a proof, not a hope
about the quality of a classifier.

The cost is that some chunks are compressed twice. That is the price of the
guarantee, and it is paid deliberately.

## Layers

```text
                    compress() / decompress()          public API
                          |
                    chunk::Chunker                    bounded memory
                          |
              +-----------+-----------+
              |                       |
        analyze::Analysis        format::            cheap statistics,  on-wire layout
              |                ContainerHeader
              |                FrameHeader
              |                EndMarker
              |
          pipeline::Planner                  candidate selection
              |
    +---------+---------+---------+
    |         |         |         |
  store      rle     lz-fast  statistical        codec/*
                          |
              matchfinder::MatchFinder          candidates, not decisions
                          |
                  entropy:: (constriction)      proven primitive
```

Each layer knows only the one below it. A codec cannot see the stream, and the
stream cannot see a codec's internals, which is what keeps them independently
testable.

## Decisions, and why

### Independent frames

Every frame is decodable without any other frame. This costs ratio — no frame can
reference a match in the previous one — and buys three things:

* frames decode in parallel, on any number of threads;
* damage is localised to the frame that contains it, and a decoder can skip
  forward to the next one;
* `info` can report every frame's pipeline, transform and sizes without decoding
  a single content byte.

The cost is measured, not assumed: the chunker sizes frames so that a frame's
window is worth having within one frame's worth of data.

### Two halves of the codec trait, deliberately unequal

```rust
fn encode(&self, ctx: &EncodeContext<'_>) -> Result<Vec<u8>>;   // may be slow
fn decode(&self, ctx: &DecodeContext<'_>, out: &mut [u8]) -> Result<()>;  // must be fast
```

Encoding is allowed to be slow, expensive, and adaptive: it may try several
strategies, build statistics, and search. Decoding is a direct inverse with no
search and no allocation beyond the output buffer. Anything that would make
decoding slower is a ratio decision, not a correctness one.

### `content_size` is always known by the decoder

The frame header records how many bytes the frame decodes to. This is what lets a
decoder allocate exactly once, and — more importantly — refuse to write more than
the frame claims. A decoder that trusted the payload to end when it ended would be
one corrupt length away from writing into memory it does not own.

It also means a decoder never has to guess whether it is finished, which is the
root of an entire class of trailing-garbage bugs.

### Errors are typed, and never panic on bad input

`Error` distinguishes wrong magic, unsupported version, truncation, structural
corruption, checksum mismatch and bad configuration. These are not decoration:
they have different remedies. Somebody handed a corrupt file needs to know whether
to re-download it, try another decoder version, or accept that the file is lost.
Collapsing all of them into "failed" makes a damaged file undiagnosable.

The `panic = "abort"` profile option is deliberately **not** set, for the same
reason: a decoder that panics on malformed input cannot be tested for that
behaviour, and `catch_unwind` is what the corruption tests and fuzz targets rely
on.

### The entropy stage is a dependency, not our code

`constriction` is used for the range coder and the probability models, and nothing
else. The reasoning is specific: the entropy stage is the one place where a subtle
fixed-point or normalisation bug does not produce garbage — it produces bytes that
decode *cleanly to the wrong thing*. That failure mode is expensive to test for
and easy to introduce, so it is delegated to a research-grade implementation whose
models are constructed to be exactly invertible.

The context model, the LZ stage, the format and the pipeline selection are all
OpenZC's own.

### The level is carried in the encode context

`EncodeContext` carries the `Level`, rather than a codec reading it from a global.
A codec's cost is then a function of its inputs and nothing else.

This was a real bug once: the codecs hardcoded `Level::Default` internally, so
every level produced byte-identical output and the level knob was a lie. It is
now pinned by a test that requires `max` to beat `fast` on data where a deeper
chain pays, and another that requires every level to decode correctly.

## What the analysis stage is for

`analyze` computes Shannon entropy, byte distribution, longest run, and trigram
reuse, sampling at most 64 KiB per chunk. It exists to **prune**, not to decide.
A wrong guess costs ratio (a candidate that would have lost is not run); it can
never cost correctness, because the output is always validated by measurement.

The trigram statistic deserves a note, because it contains a trap that produced a
real bug. It was first computed with a 14-bit tag set. For a sample of 8 KiB
positions drawn into 16 384 slots, the birthday bound predicts roughly 2 000 false
hits — so uniformly random data scored a 25% "repetition ratio" and was classified
as compressible. The fix was to count *exact* 24-bit trigram repeats over a
bounded, evenly spread set of positions, and subtract nothing because exact
matching introduces no false positives. The lesson generalises: any "have I seen
this before" statistic built on a truncated hash is measuring the hash table, not
the data.

## Model tables: transmit counts, not frequencies

The statistical pipeline trains a sequence model and 16 order-1 literal models per
chunk, and the frequencies must be identical on both sides.

The first implementation transmitted normalised frequencies, densely. That cost
**12 KiB per chunk** — about 4 KiB for the 2048-symbol sequence table and 512
bytes for each of sixteen 256-symbol literal tables — which the model could not
repay on any ordinary chunk. A 270 KB text chunk spent 12 288 bytes on tables to
save 2 091 bytes over the fast LZ pipeline: a net loss.

Two changes fixed it, and both are worth more than the size saving alone.

**Transmit counts, not frequencies.** Frequencies are a pure function of the
counts, and both sides already run the same normalisation on the same input, so
sending frequencies sends a redundant derivation. Counts are small integers, and
unobserved symbols cost nothing. Tables went from 12 288 bytes to 109.

**Reserve the floor rather than scaling it.** The subtler bug. `normalise` treats
an unobserved symbol as count 1 and then *scales* it by `TOTAL / sum(counts)`. For
a literal context trained on 45 literals, that is a factor of 256 — so all 211
unobserved symbols took 82% of the probability mass between them, and the model
coded data it had never seen. `normalise_trained` instead gives every symbol one
unit and shares only the remainder among observed symbols.

Together these took a 270 KB text chunk from **14 948 bytes to 2 517**, against
4 751 for the fast LZ pipeline. The statistical pipeline is now the best choice on
text, which is the only reason it exists.

## The LZ overlap trap

The single easiest way to write a subtly broken LZ decoder is to get overlapping
matches wrong. When `offset < match_length`, the copy must run **forward, one byte
at a time**, so each byte sees the value the byte before it just wrote. This
extends a repeating period: over the literals `abcdef`, an offset of 3 and a
length of 12 yields `abcdefdefdefdefdef`.

`copy_within` looks like the obvious answer and is **wrong** — it has memmove
semantics and re-reads the not-yet-written tail. It was introduced here by
following a Clippy `needless_range_loop` suggestion, and it passed the existing
overlap test because that test used a constant byte, where memmove and a forward
copy agree. A 123-byte input decoded to a trailing `\0`, and every 512 KiB corpus
failed its checksum.

The regression test now pins the exact byte output, and a second test sweeps
repeating periods 2 through 17 — the varied case, not the constant one.

This is recorded because the lesson is not "be careful with LZ". It is that **a
test whose input makes two different implementations agree cannot detect the
difference between them.**

## Rejected and deferred

* **Dictionary priming.** Specified in the format (`dict_id`, pipeline 5) but not
  emitted. The field and its validation rules are in place so that adding it is not
  a breaking change, but shipping a half-used field would be worse than shipping
  it whole later.
* **A transform search.** `Planner::transform_for` currently always returns
  `None`, so `delta` is never selected by the adaptive path even though it is fully
  implemented and tested. Delta is a large win on some numeric data and the
  selection logic is a small piece of work.
* **GPU acceleration.** Deliberately out of scope. It is the only plausible answer
  to "much faster", and it would make the decoder harder to audit — which is the
  wrong trade for a format whose main claim is that damage is detectable.
* **Lossy codecs.** A general-purpose lossless compressor cannot improve a
  well-compressed PNG; that needs AVIF or JPEG-XL, which breaks the contract that
  makes the format trustworthy. If that is wanted, it belongs in a different tool.

## Testing strategy

220 unit tests, organised so that each guards a *property* rather than a value:

* round trips across every pipeline, level, strategy and checksum policy;
* every decoder's behaviour on truncated, corrupted and garbage input;
* structural invariants (a chunk never grows; a frame is bounded by its declared
  content size; a table is a valid distribution);
* properties that must not drift (levels must change the search; tables must stay
  sparse; statistics must classify random data as incompressible).

CI runs the same four-command gate as `scripts/publish.ps1`, plus an MSRV job and
a downstream-consumer job that smoke-tests a real round trip and asserts that a
corrupted stream is *rejected*. Fuzz targets are the next step; the decoder is
written to return errors rather than panic precisely so that fuzzing is the right
tool for it.

## Where the performance is

Measured on 512 KiB generated corpora, level `max` (see the README for the full
table). These are regression signals, not claims about real-world compression.

| Corpus | Ratio | Compress | Decompress |
|---|---|---|---|
| runs | 0.030 | 3 MB/s | 552 MB/s |
| source-like | 0.155 | 7 MB/s | 125 MB/s |
| text | 0.222 | 4 MB/s | 91 MB/s |
| incompressible | 1.000 | 235 MB/s | 606 MB/s |

Two readings matter more than the numbers:

* **Incompressible input costs almost nothing** (235 MB/s) because it is detected
  and stored. This is the most valuable thing the engine does; a compressor that
  burns CPU to make a file bigger is worse than useless.
* **Compression is 10–50x slower than decompression**, and that asymmetry is by
  design. Encoding may search; decoding may not.

The obvious next optimisation is parallel frame encoding, which the chunker and
the `INDEPENDENT` frame flag are already structured for, but which is not yet
wired up. Profiling is required before anything else: it is not known where the
time actually goes, and guessing has already produced one wrong answer in this
project (the `copy_within` suggestion).
