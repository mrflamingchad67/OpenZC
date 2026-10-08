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

## Transforms, and the heuristic that gates them

A transform rearranges a chunk so the LZ stage sees more redundancy. Only `delta`
exists — a byte-wise difference against the previous byte, applied before the
pipeline and inverted after, with the first byte differenced against zero so the
inverse needs no carried state.

It was implemented, tested, and unreachable: `transform_for` returned `None`
unconditionally, so the adaptive path could never select it. It is now wired up,
behind a measurement rather than a guess.

`DeltaTransform::worth_trying` measures average absolute byte difference over an
8 KiB sample. The obvious version of that statistic compares *adjacent* bytes, and
the obvious version of the obvious version uses one threshold for everything. Both
were wrong, and the measurements said so.

Adjacent drift alone, every number measured on the statistical pipeline at 512 KiB:

| data | adjacent drift | offered? | none → delta | outcome |
|---|---|---|---|---|
| bytes stepping by three | 5.9 | yes | 6 859 → 4 183 | **wins 1.64x** |
| random ACGT | 7.7 | yes | 168 514 → 236 234 | **loses 0.71x** |
| u64 counter | 31.9 | no | 133 263 → 206 756 | loses 0.64x |
| u16 counter | **120.3** | no | 433 550 → 9 475 | **wins 45.8x** |
| ordinary text | 36.2 | no | 4 259 → 4 271 | tie |
| uniform random | 86.6 | no | 534 326 → 534 324 | tie |

The last three rows are the interesting ones, and the `u16` row is the whole story.

**The threshold is set by a loss, not a win.** A four-symbol alphabet has low drift
by construction — any two of four symbols are usually close — yet delta makes it
0.71x worse there, because it destroys the short repeats LZ was already exploiting.
Low drift is necessary but not sufficient. The threshold is 8.0 because at 7.7 that
case sits just above it; the original 24.0 let it through.

**Adjacent drift is systematically wrong for interleaved numeric fields**, which is
the actual use case. A 16-bit counter stored little-endian is the byte stream
`lo0 hi0 lo1 hi1 ...`. Every *adjacent* pair straddles a field boundary and jumps,
so drift reads **120.3** and the transform is refused — while delta compresses that
data **45.8x better**. This is the largest ratio win delta has on any input, and
the heuristic rejected it, because it inspects adjacent bytes and the varying bytes
are not adjacent.

## Strides, and the one shape that still misfires

The bytes that carry the signal are *two apart*, so the fix is to measure drift at
several strides and offer delta if any of them is low. Stride 1 stays the primary
test and the others are purely additive, so nothing offered before stops being
offered.

| corpus | adjacent | stride | stride drift | none → delta | outcome |
|---|---|---|---|---|---|
| u16 pairs | 120.3 | 2 | 1.0 | 433 550 → 9 475 | **wins 45.8x** |
| u24 triples | 83.3 | 3 | 0.6 | 435 542 → 357 899 | wins 1.22x |
| u32 quads | 63.8 | 4 | 0.5 | 353 497 → 465 796 | loses 0.76x |
| xyz u16 records | 119.4 | 6 | 1.0 | 439 992 → 14 736 | **wins 29.9x** |
| u64 octets | 31.9 | 8 | 0.2 | 133 263 → 206 756 | loses 0.64x |
| random u16 pairs | 84.2 | — | ~85 | 534 350 → 534 350 | tie |
| uniform random | 86.6 | — | ~86 | 534 326 → 534 324 | tie |
| ordinary text | 36.2 | — | 26–37 | 4 259 → 4 271 | tie |

The last three rows keep this honest: noise does not produce a low stride drift at
any width, and text does not either, so the signal stays off on exactly the data
where it would be pure cost.

**The stride set is `{2, 3, 4, 6, 8}`, and each entry is there because it was
measured.** A power-of-two set alone misses `u24` (stride 3) and three-`u16` record
layouts (stride 6) — both ordinary shapes, and 29.9x on the latter. Scanning every
stride instead was measured and rejected twice over. It costs 2.3x more: 24.7 us
for this five-stride set against 56 us for a contiguous `2..=16` and 116 us for
`2..=32`. And at `K = 64` ordinary text registers low drift at stride 40, which is
not a measurement bug but what happens when you take a minimum over 63 noisy
statistics: with enough candidates, some dip under any fixed threshold. A blind
scan does not fail by being too narrow, it fails by being too permissive, and it
fails silently.

**The signal is deliberately loose, and the reason is a property of the planner.**
It also fires on `u32` and `u64`, where delta loses. That costs nothing in ratio: the
planner encodes every candidate and keeps the smallest, so an extra candidate can
only find something smaller or equal. It costs one wasted encode. Paying that on
`u32`/`u64` to catch 45.8x on `u16` and 29.9x on records is the right way round,
because a missed candidate costs ratio permanently and a wasted one costs CPU once.
This asymmetry — not any property of the drift statistic — is what makes it safe to
add candidates without predicting the winner, and it is why the RLE pruning problem
described later is hard but this was not.

**The one shape that still misfires is very short-period data.** `"hello world "
repeats every 12 bytes, so an 8 KiB sample contains only 12 distinct byte pairs per
stride; at stride 6 those average drift 7.00, just under the 8.0 threshold, and delta
gets offered. It loses and the planner discards it, so the cost is one wasted
encode.

The tempting fix is to lower the stride threshold to ~4, which does exclude this
corpus, and it is the wrong fix. Genuine cases sit at 0.2–1.0 and the periodic
four-symbol case at 2.92, so the only evidence for a threshold anywhere in the gap
is this one synthetic sample. A threshold chosen to exclude a single 12-byte-period
string is overfitting to that string. Realistic text does not have the problem: a
45-byte phrase cycle measures 24–37 at every stride, and CSV rows measure 12–19.
It is pinned by `short_period_text_can_false_positive_and_that_is_accepted`, which
asserts both the misfire and that it stays cheap.

Analysis cost, 8 KiB sample, best of 500: **4.1-24.7 us**, against 4.2 us for the
single adjacent pass this replaces, so roughly 6x on a check that is itself ~0.001% of
an encode. End-to-end at 512 KiB and level `default`, text encodes in 5 ms and uniform
random in 2 ms, both unchanged in ratio, so the overhead does not show up where it
would matter.

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

## Candidate pruning: what the measurements allowed

The planner runs candidates in order and keeps the smallest. The obvious
optimisation is to skip candidates that cannot win, so this section records what
was measured before anything was changed.

`Planner::candidates_for` and `Planner::explain_candidates` exist so that
*selection* can be read without running an encode. Which pipeline wins is a ratio
question settled by measurement; which candidates are even considered is a policy
question, and a policy nobody can read is a policy nobody can review.

### What each candidate costs

Per-candidate encode time on 2 MiB, best of three, and the size each produces:

| corpus | candidates | total | winner |
|---|---|---|---|
| random | 1 | 0–1 ms | `store` |
| numeric | 5 | 21–26 ms | `statistical/delta` |
| text | 3 | 40 ms | `statistical/none` |
| runs | 6 | 29 ms | `rle/none` |
| wide-text | 3 | 74 ms | `statistical/none` |
| near-random-acgt | 6 | 130 ms | `statistical/none` |
| **runs @ max** | 6 | **1 286 ms** | **`rle/none` (1 ms of it)** |
| **interleaved @ max** | 3 | **1 027 ms** | `statistical/none` |

The last two rows are the interesting ones, and they point in opposite directions.

### The one change that was safe

`rle` appeared **twice** in the candidate list on run-heavy data — added by the
registry loop and again by the explicit `Repetitive` push — so the same pipeline
was encoded twice, byte for byte, for no benefit. Seven candidates where six were
correct.

Deduplicating is provably output-neutral: a repeat encodes to the same payload and
cannot change which is smallest. Measured on `runs`, 7 → 6 candidates with every
ratio byte-identical (0.0302 at every level).

### The pruning that was not safe, and why

The tempting rules, and what the measurements say about each:

| rule | `runs` | `near-random-acgt` |
|---|---|---|
| "skip the LZ pipelines when RLE is offered" | correct (RLE 63 095 vs 76 238) | **wrong by 3.3x** (RLE 2 261 658 vs 676 298) |
| "trust the `Repetitive` class" | correct | **wrong** — ACGT is classified `LowEntropy` |

RLE is *offered* on both corpora and wins exactly one. So neither rule is safe, and
any threshold that separated them would be a guess fitted to two data points. This
is pinned by `rle_is_offered_on_data_where_it_can_lose_badly`, which exists so that
nobody adds such a rule without measuring first.

The same applies to dropping `lz-fast` whenever `statistical` is available:
statistical beat it in 7 of 7 corpora here, but that is seven synthetic inputs, and
`lz-fast` is what serves chunks too small for statistical's tables to amortise. A
rule that looks free on this sample is a rule with an unmeasured tail.

### The conclusion

**Candidate pruning is close to exhausted as a safe optimisation.** What remains
costs about 1 200 ms on `runs` at level `max` — and none of it can be removed
without a rule that is wrong on a corpus where RLE or LZ genuinely wins.

The remaining waste is real but it is a *prediction* problem, not a bookkeeping
one: skipping work safely needs a signal that says "this will lose", and the
signals available today do not separate the two cases. The honest options are to
build that signal (work in the analysis stage) or to accept the cost.

What was already in place and needed no change: the incompressible path is optimal.
Uniform random data gets exactly one candidate and encodes in under 10 ms, because
`looks_incompressible` short-circuits before any pipeline is built.

## Rejected and deferred

* **Dictionary priming.** Specified in the format (`dict_id`, pipeline 5) but not
  emitted. The field and its validation rules are in place so that adding it is not
  a breaking change, but shipping a half-used field would be worse than shipping
  it whole later.
* **Stride-aware transform selection.** Done. See "Strides, and the one shape
  that still misfires" above. The short version: drift is now measured at several
  strides, which recovers the 45.8x `u16` win and the 29.9x three-`u16`-record
  win that byte drift refused.

  What was *not* done is the part that would have looked cleverer: separating
  the layouts the new signal detects and loses on. It detects `u32` and `u64`
  interleaving too, where delta loses. That is safe rather than right, and the
  reason it is safe is worth stating plainly, because it is the load-bearing
  argument for the whole design: **adding a candidate to the planner cannot make
  the ratio worse.** The planner encodes every candidate and keeps the smallest,
  so a new candidate can only find something smaller or equal. A missed candidate
  costs ratio permanently; a wasted one costs one encode. That asymmetry is why
  the signal is deliberately loose instead of tuned tight.
* **A signal that predicts the RLE winner.** ~1 200 ms is spent per 2 MiB of
  run-heavy data at level `max` on candidates that provably lose, and none of it can
  be removed safely: RLE wins on one corpus and loses by 3.3x on another that looks
  identical to every cheap signal. Skipping that work needs a *prediction*, and the
  analysis stage does not have one. See "Candidate pruning" above.
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

### Parallel encoding, and what it actually bought

Frame encoding is now batched and parallel. Chunks are read sequentially, planned
across a thread pool, and the resulting frames written in input order — so the
output is byte-identical to a serial run, which a test pins.

Measured on 12 cores, 2 MiB of text:

| Level | Serial | Parallel | Speedup |
|---|---|---|---|
| fast | 43.7 MB/s | 61.8 MB/s | 1.42x |
| default | 35.1 MB/s | 57.2 MB/s | 1.63x |
| high | 16.6 MB/s | 16.8 MB/s | 1.01x |
| max | 4.0 MB/s | 4.0 MB/s | 1.01x |

Three things are worth recording, because none of them is what the change was
expected to deliver.

**The gain is capped well below the core count**, and 1.55x on 12 cores is
consistent with Amdahl's law rather than a bug: the planner also allocates and
hashes, and reading the next chunk is serial I/O that cannot overlap because there
is one reader.

**It does nothing at all on a single-chunk input.** The batch is sized in *bytes*
(`BATCH_BYTES / chunk_size`) so that peak memory stays bounded — a fixed chunk
*count* would hold 32 MiB at the default 4 MiB chunk size, which would quietly
contradict the bounded-memory claim. The consequence is that a 2 MiB input is one
chunk, so the batch holds one chunk, so there is nothing to parallelise. The
measured win needs multi-chunk input, which is what the `threads` benchmark
section uses.

**High levels gain nothing.** At `high` and `max` the speedup is 1.01x — inside
the noise. A single chunk takes long enough that scheduling is negligible against
it, and there is no second chunk in a 2 MiB corpus to overlap with anyway. So
parallelism helps where compression is cheap and is simply inert where it is
expensive, which is the useful direction: `Threads::Serial` exists for callers who
want to measure single-thread behaviour, and the run-to-run spread (the `high`
row measured 0.96x on one run and 1.01x on the next) is the size of the noise
floor here.

### Why there is no `pprof`

`pprof` needs signal-based sampling, which means Unix `nix` APIs and
`libc::pthread_t`. Neither exists on Windows, where this project is developed. A
profiler that cannot be run on the development machine does not get run, and an
unprofiled optimisation is a guess.

`benches/profile.rs` replaces it with portable Rust and reports flat self time per
section. The trade is real: there is no call tree, only which operations dominate.
That is enough to answer "where is the time going", which is the question that has
to be answered first. If a call tree is ever needed, install `pprof` on a Unix
host and profile there.

### Profiling before optimising, still

This project has now made three wrong guesses about where time goes: `copy_within`
(via a Clippy suggestion, which silently corrupted data), and two rounds of
assuming the statistical pipeline was the bottleneck when it is not — it is
*incompressible input detection* that makes the common case fast. The harness
exists so the fourth guess is cheaper than the third.
