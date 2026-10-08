# OpenZC

A modular, lossless, streamable compression engine in Rust.

OpenZC compresses data by **measuring several compression strategies against each
chunk of input and keeping whichever one actually produced the smallest output**.
No chunk is ever allowed to grow. The design goal is not "beat X on a benchmark"
but "never be the wrong tool for the data" — text, source code, run-heavy binary
data, and already-compressed data each get the pipeline that suits them, chosen by
measurement rather than by a heuristic.

Losslessness is a hard contract, not a goal: `decompress(compress(x)) == x` for
every input, verified by hash on every test and by the `openzc test` command.

## Status

**Early. Usable, tested, and honest about its limits.**

Working today:

* Five pipelines — `store`, `rle`, `lz-fast`, `statistical`, `dictionary`
  (dictionary priming is specified but not yet emitted)
* Two transforms — `none`, `delta`
* A versioned, fully specified container format with independent frame decoding
* Streaming compression and decompression with bounded memory
* BLAKE3 integrity per frame and over the whole stream
* A command line tool and a benchmark harness
* 220 unit tests; a green `fmt` / `check` / `test` / `clippy` gate

Not done yet, and listed here so you are not misled:

* No directory/archive mode — the CLI compresses one file at a time, so a folder
  of small files pays the container overhead repeatedly and can come out *larger*
* No `cargo-fuzz` targets yet. The decoder is written to return errors rather than
  panic and `tests/robustness.rs` checks that on truncated, bit-flipped and
  rewritten-field input on stable Rust, but it has not been fuzzed properly
* No published benchmarks against other compressors. The numbers below are from
  this project's own harness on generated data and are **not** a comparison
* Dictionary priming and transform selection are specified but never emitted
* The `MIT`/`Apache-2.0` licence text files are not in the repository yet

## Install

Requires Rust 1.85 or newer.

```sh
git clone https://github.com/mrflamingchad67/OpenZC
cd OpenZC
cargo build --release
```

## Use it from the command line

```sh
openzc compress   -l high report.txt        # -> report.txt.ozc
openzc info       report.txt.ozc            # header and per-frame detail
openzc decompress report.txt.ozc            # -> report.txt
openzc test       report.txt                # round trip, verified in memory
```

| Command | Does |
|---|---|
| `compress` | Compress files into `.ozc` streams |
| `decompress` | Decompress streams back to the original files |
| `info` | Report what a stream contains, without decoding it |
| `test` | Compress and decompress in memory, verifying the result |

Options: `-l/--level` (`none`, `fast`, `default`, `high`, `max`, or `0`–`9`),
`-s/--strategy` (`adaptive`, `fixed`, `never`), `-c/--checksum` (`full`, `chunk`,
`none`), `-t/--threads` (`0` auto, `1` serial), `-k/--chunk-size`, `-w/--window`,
`-m/--max-input`, `-o/--output`, `-f/--force`, `-q/--quiet`, `--`.

Exit codes: `0` success, `1` the operation failed, `2` bad arguments, `3` a
self-test mismatch.

## Use it as a library

```rust
use openzc::{Config, Level, OpenZc};

// In memory.
let config = Config::new(Level::High);
let openzc = OpenZc::new(config.clone())?;

let (compressed, stats) = openzc::compress_slice(data, &config)?;
let (restored, verified) = openzc::decompress_slice(&compressed, &config)?;
assert_eq!(restored, data, "lossless round trip");
assert!(verified.verified, "hashes checked");

// Or streaming, without holding the input in memory.
let stats = openzc.compress(File::open("in.bin")?, File::create("out.ozc")?)?;
```

The engine never panics on malformed input; it returns a typed `Error` that
distinguishes wrong magic, unsupported version, truncation, structural corruption
and checksum mismatch, because those have different remedies for whoever has to
repair the file.

## How it works

Input is split into bounded chunks. For each chunk the engine computes cheap
statistics (Shannon entropy, byte distribution, run lengths, trigram reuse), uses
them to discard candidates that cannot possibly help, runs the survivors, and keeps
the smallest result. `store` is always measured as a floor, which is what makes
"a chunk never grows" a guarantee rather than a hope.

The four active pipelines:

| Pipeline | Good at | Decode cost |
|---|---|---|
| `store` | incompressible data, and as the floor | memcpy |
| `rle` | long runs; no window, so trivially parallel | very low |
| `lz-fast` | general structured data | low |
| `statistical` | text and source, where modelling pays | moderate |

`statistical` is the one that adapts within a chunk: it runs LZ77, then codes the
resulting tokens against an order-1 context model with a range coder. It is the
only pipeline that beats `lz-fast` on ratio, and it is gated to chunks large
enough to amortise its model tables.

The container is a 28-byte header, a sequence of independently decodable frames,
and a mandatory end marker carrying a whole-content hash. Because each frame can
be decoded alone, frames decode in parallel, and damage is localised.

## Measured results

From `cargo run --release -p openzc-bench`, 512 KiB of generated data per corpus.
These are **this project's own numbers on synthetic input** — useful as a
regression signal, not as a claim about real-world compression. No baseline
compressor has been measured yet.

| Corpus | Level | Ratio | Compress | Decompress |
|---|---|---|---|---|
| text | max | 0.222 | 4 MB/s | 91 MB/s |
| source-like | max | 0.155 | 7 MB/s | 125 MB/s |
| runs | max | 0.030 | 3 MB/s | 552 MB/s |
| incompressible | max | 1.000 | 235 MB/s | 606 MB/s |

Two results are worth reading twice:

* **`incompressible` does not shrink** (1.000x) and costs almost no time. Detecting
  that and storing instead is the single most valuable thing the engine does, and
  it is what stops a compressor from burning CPU to make a file bigger.
* **`runs` compresses 33x** and decodes at over 500 MB/s, because `rle` has no
  match window at all.

Frame encoding is parallel: chunks are planned across a thread pool and written in
order, so the output is byte-identical to a serial run. Measured at 1.55x on a
12-core machine for cheap levels, with no gain on a single-chunk input (the batch
is sized in bytes to keep memory bounded, so one chunk means one batch of one).
`Threads::Serial` disables it for single-threaded measurement.

Real files behave very differently from generated ones. A 4 KB PDF in this
repository's test data compressed 24%, and a GIMP splash-screen PNG compressed
0.04% — because both are *already* compressed internally. A general-purpose
lossless codec cannot re-compress a well-compressed PNG; that needs a lossy codec,
which would break the lossless contract.

## Development

The project holds itself to a four-command gate, all of which must pass with no
warnings:

```sh
cargo fmt --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
```

Benchmarks and corpus:

```sh
cargo run --release -p openzc-bench              # generated corpora, all levels
cargo run --release -p openzc-bench -- file.txt  # a real file
```

## Documentation

* [`docs/FORMAT.md`](docs/FORMAT.md) — the normative format specification, written
  so a decoder can be implemented from it without linking this crate
* `crates/openzc/src/format/` — the reference implementation of that spec
* `docs/DESIGN.md` — architecture and rationale (not yet written)

## Licence

`MIT OR Apache-2.0`, at your option. The licence text files are not yet in the
repository; see the status note above.

[`constriction`]: https://crates.io/crates/constriction
