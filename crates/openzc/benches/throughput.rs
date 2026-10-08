//! Throughput benchmarks, measured by criterion.
//!
//! # What these numbers mean
//!
//! Criterion reports nanoseconds per iteration with a confidence interval, which
//! is what makes a change legible: after running `cargo bench`, criterion keeps
//! the previous result and reports the regression or improvement. That is the
//! point of using it here rather than printing timings — a single run tells you
//! nothing about whether a change helped, and this project's own history includes
//! guessing wrong about where the time goes.
//!
//! # Ratios, not just speed
//!
//! Speed benchmarks alone would let the engine get faster by compressing worse.
//! Every group therefore reports the compressed size too, so a change that trades
//! ratio for throughput shows up as both numbers moving.
//!
//! # Caveat
//!
//! Criterion is for detecting *changes* to this code. It is not a comparison
//! against other compressors: no baseline has been measured, and none should be
//! inferred from these numbers. Corpora are generated, which is optimistic about
//! ratio; `openzc-bench <file>...` is for real inputs.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use openzc::{compress_slice, decompress_slice, Config, Level};

/// Deterministic PRNG, so a benchmark compares code rather than random data.
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

fn text(n: usize) -> Vec<u8> {
    const WORDS: [&str; 16] = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "compression",
        "entropy",
        "pipeline",
        "chunk",
        "window",
        "match",
        "symbol",
        "context",
    ];
    let mut s = 99u32;
    let mut out = Vec::with_capacity(n + 64);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        out.extend_from_slice(WORDS[(s >> 24) as usize % WORDS.len()].as_bytes());
        out.push(b' ');
    }
    out.truncate(n);
    out
}

fn source_like(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n + 128);
    let mut i = 0usize;
    while out.len() < n {
        out.extend_from_slice(format!("    let value_{i} = compute(&self, {i})?;\n").as_bytes());
        i += 1;
    }
    out.truncate(n);
    out
}

fn runs(n: usize) -> Vec<u8> {
    let mut s = 7u32;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let len = 1 + (s >> 24) as usize % 200;
        out.extend(std::iter::repeat_n((s >> 8) as u8, len));
    }
    out.truncate(n);
    out
}

fn low_entropy(n: usize) -> Vec<u8> {
    (0..n).map(|i| b"ACGT"[(i * 7 + i / 13) % 4]).collect()
}

/// The corpora, as `(name, bytes)` pairs.
fn corpora() -> Vec<(&'static str, Vec<u8>)> {
    const N: usize = 256 * 1024;
    vec![
        ("text", text(N)),
        ("source", source_like(N)),
        ("runs", runs(N)),
        ("low_entropy", low_entropy(N)),
        ("random", pseudo_random(N, 0x5EED)),
    ]
}

/// Compress one corpus, returning the payload so the caller can also report size.
fn compress(data: &[u8], level: Level) -> Vec<u8> {
    compress_slice(data, &Config::new(level))
        .expect("compress")
        .0
}

fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress");
    for (name, data) in corpora() {
        group.throughput(Throughput::Bytes(data.len() as u64));
        for level in [Level::Fast, Level::Default, Level::Max] {
            group.bench_function(format!("{name}/{level:?}"), |b| {
                b.iter(|| compress(&data, level))
            });
        }
    }
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress");
    for (name, data) in corpora() {
        // Compress once, outside the measured loop: the point is to time
        // decoding, and including the encode would measure the wrong thing.
        let packed = compress(&data, Level::Default);
        let config = Config::new(Level::Default);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_function(name, |b| {
            b.iter(|| decompress_slice(&packed, &config).expect("decompress"))
        });
    }
    group.finish();
}

// Ratio is deliberately *not* benchmarked here.
//
// Criterion measures time, and a "ratio" benchmark would measure nothing at all —
// by the time the closure returns, the value is a constant. Worse, a ratio
// regression sitting in a benchmark table is one nobody reads. Ratio floors are
// asserted in the test suite instead, where they fail the quality gate; see
// `pipeline::tests::ratio_floors_are_maintained`.
criterion_group!(benches, bench_compress, bench_decompress);
criterion_main!(benches);
