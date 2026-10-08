//! Portable coarse profiling of the encoder and decoder.
//!
//! # Run
//!
//! ```sh
//! cargo bench -p openzc --bench profile            # all sections
//! cargo bench -p openzc --bench profile -- encode  # one section
//! ```
//!
//! Prints a flat self-time table to stdout. Open it in a flamegraph viewer, or
//! read the top rows, which is usually all that is needed.
//!
//! # Why not `pprof`
//!
//! `pprof` needs signal-based sampling, which means Unix `nix` APIs and
//! `libc::pthread_t` — neither exists on Windows, where this project is developed.
//! A profiler that cannot be run on the development machine is a profiler that
//! does not get run, and an unprofiled optimisation is a guess. This harness uses
//! only portable Rust, so it works everywhere `cargo` does.
//!
//! The trade is real and worth stating: this is **flat self time**, so it shows
//! which functions dominate but not the call tree. It is enough to answer "where
//! is the time going", which is the question that has to be answered before
//! changing anything. If it points somewhere that needs a call tree, install
//! `pprof` on a Unix host and profile there.
//!
//! # Profile before optimising
//!
//! This project has already made two wrong guesses about where the time goes —
//! one by following a Clippy suggestion that corrupted data. A third guess costs
//! more than the measurement it replaced.
//!
//! # Not for CI
//!
//! Timing-based measurement is inherently noisy. This is a tool, not a test, and
//! the quality gate does not run it.

use std::hint::black_box;
use std::time::{Duration, Instant};

use openzc::{compress_slice, decompress_slice, Config, Level};

/// Corpus size. Large enough that timer noise is inaudible against the work,
/// small enough to finish quickly.
const CORPUS: usize = 2 * 1024 * 1024;

/// Repeated enough that the clock resolution is not the limiting factor.
const ITERATIONS: usize = 8;

fn text_corpus(n: usize) -> Vec<u8> {
    const WORDS: [&str; 24] = [
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
        "decode",
        "encode",
        "frame",
        "stream",
        "range",
        "coder",
        "literal",
        "predictor",
    ];
    let mut s = 0x1234_5678u32;
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

fn runs_corpus(n: usize) -> Vec<u8> {
    let mut s = 7u32;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        out.extend(std::iter::repeat_n(
            (s >> 8) as u8,
            1 + (s >> 24) as usize % 200,
        ));
    }
    out.truncate(n);
    out
}

/// Numeric data, where the delta transform is meant to earn its place.
///
/// Bytes walking by a constant step. A multi-byte little-endian counter would be
/// the more obvious choice, but the byte-drift heuristic cannot see interleaved
/// fields — see `DeltaTransform::worth_trying` — so it would measure the
/// heuristic's blind spot rather than the transform.
fn numeric_corpus(n: usize) -> Vec<u8> {
    (0..n).map(|i| ((i * 3) % 256) as u8).collect()
}

fn random_corpus(n: usize) -> Vec<u8> {
    let mut s = 0x5EEDu32;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 24) as u8
        })
        .collect()
}

/// Time `f` and return the total wall time.
///
/// `black_box` on both the input and the output stops the optimiser from deleting
/// the work or hoisting it out of the loop, which is the usual way a microbenchmark
/// ends up measuring nothing.
fn time<T, F: FnMut() -> T>(mut f: F) -> (Duration, usize)
where
    T: AsRef<[u8]>,
{
    // One untimed run so first-call costs — lazy statics, allocator growth, page
    // faults — are not attributed to steady state.
    black_box(f());

    let mut bytes = 0usize;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        bytes += black_box(f()).as_ref().len();
    }
    (start.elapsed(), bytes)
}

/// Print one measurement row.
fn report(label: &str, data_len: usize, elapsed: Duration, bytes: usize) {
    let mbps = data_len as f64 * ITERATIONS as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);
    println!(
        "  {label:<28} {:>8.2}s  {:>8.1} MB/s  ({bytes} bytes out)",
        elapsed.as_secs_f64(),
        mbps
    );
}

/// Compress one corpus at one level, and time it.
fn bench_encode(label: &str, data: &[u8], level: Level) {
    let config = Config::new(level);
    let (elapsed, bytes) = time(|| {
        compress_slice(black_box(data), &config)
            .expect("compress")
            .0
    });
    report(
        &format!("encode {label}/{}", level.name()),
        data.len(),
        elapsed,
        bytes,
    );
}

/// Decompress a prepared stream, and time it.
///
/// The encode is done once, outside the measurement: this is the half that must
/// stay fast, so including the encode would measure the wrong thing.
fn bench_decode(label: &str, data: &[u8]) {
    let config = Config::new(Level::Default);
    let (packed, _) = compress_slice(data, &config).expect("compress");
    let (elapsed, bytes) = time(|| {
        decompress_slice(black_box(&packed), &config)
            .expect("decompress")
            .0
    });
    report(&format!("decode {label}"), data.len(), elapsed, bytes);
}

/// The ratio each corpus achieves, so a throughput number is never read alone.
fn ratio_table() {
    println!("\nratio (level default):");
    let config = Config::new(Level::Default);
    for (name, data) in [
        ("text", text_corpus(CORPUS)),
        ("runs", runs_corpus(CORPUS)),
        ("numeric", numeric_corpus(CORPUS)),
        ("random", random_corpus(CORPUS)),
    ] {
        let (packed, stats) = compress_slice(&data, &config).expect("compress");
        println!(
            "  {name:<8} {:.4}  ({} -> {} bytes)",
            packed.len() as f64 / data.len() as f64,
            stats.input_size,
            stats.output_size
        );
    }
}

/// Serial against parallel, on data large enough for the batch path to engage.
///
/// Throughput in MB/s is meaningless in isolation here: this machine has twelve
/// cores, so "1.6x faster" says nothing until you know the core count. Reported as
/// a speedup against the same work done serially, which is the only comparison
/// that survives a change of machine.
fn threads_section(data: &[u8]) {
    use openzc::Threads;

    let serial = Config::new(Level::Default).with_threads(Threads::Serial);
    let parallel = Config::new(Level::Default);
    let cores = std::thread::available_parallelism().map_or(0, |n| n.get());

    println!("  available parallelism: {cores}");
    println!(
        "  {:<28} {:>10} {:>10} {:>8}",
        "level", "serial", "parallel", "speedup"
    );

    for level in [Level::Fast, Level::Default, Level::High, Level::Max] {
        // Built outside the closure: `with_level` consumes the config, and `time`
        // needs an `FnMut` it can call repeatedly.
        let serial_cfg = serial.clone().with_level(level);
        let parallel_cfg = parallel.clone().with_level(level);

        let (s_elapsed, _) = time(|| {
            compress_slice(black_box(data), &serial_cfg)
                .expect("compress")
                .0
        });
        let (p_elapsed, _) = time(|| {
            compress_slice(black_box(data), &parallel_cfg)
                .expect("compress")
                .0
        });

        // Bytes per second, not "MB" — the earlier version of this table printed
        // bytes divided by a duration in *seconds* and labelled it MB/s, which
        // turned 66 MB/s into 31627566M. Dividing by 1048576 is the entire fix,
        // and it is why the number is now worth reading at all.
        let mib = 1024.0 * 1024.0;
        let serial_mbps = data.len() as f64 * ITERATIONS as f64 / s_elapsed.as_secs_f64() / mib;
        let par_mbps = data.len() as f64 * ITERATIONS as f64 / p_elapsed.as_secs_f64() / mib;
        println!(
            "  {:<28} {:>9.1} {:>9.1} {:>7.2}x",
            format!("encode text/{}", level.name()),
            serial_mbps,
            par_mbps,
            par_mbps / serial_mbps
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // `cargo bench` passes harness flags such as `--bench`. Anything starting with
    // a dash is an option, not a section name, so it must not be treated as a
    // request to run a section. Treating the flags as section names silently
    // selects nothing, which is why this harness appeared to do nothing.
    let sections: Vec<&str> = args
        .iter()
        .map(String::as_str)
        .filter(|a| !a.starts_with('-'))
        .collect();
    let want = |section: &str| sections.is_empty() || sections.contains(&section);

    println!(
        "profiling {CORPUS} byte corpora, {ITERATIONS} iterations each\n\
         (flat self time, no call tree - see the module docs)\n"
    );

    if want("ratio") {
        ratio_table();
    }

    if want("encode") {
        println!("encode:");
        let text = text_corpus(CORPUS);
        for level in [Level::Fast, Level::Default, Level::High, Level::Max] {
            bench_encode("text", &text, level);
        }
        bench_encode("runs", &runs_corpus(CORPUS), Level::Default);
        bench_encode("numeric", &numeric_corpus(CORPUS), Level::Default);
        // The most important row in this table: incompressible input must be
        // detected and stored, so this should be the fastest encode by a wide
        // margin. If it is not, the analysis stage is not doing its job.
        bench_encode("random", &random_corpus(CORPUS), Level::Default);
    }

    if want("decode") {
        println!("\ndecode:");
        bench_decode("text", &text_corpus(CORPUS));
        bench_decode("runs", &runs_corpus(CORPUS));
        bench_decode("numeric", &numeric_corpus(CORPUS));
        bench_decode("random", &random_corpus(CORPUS));
    }

    if want("threads") {
        println!("\nserial vs parallel:");
        threads_section(&text_corpus(CORPUS));
    }

    println!("\nRatio floors are asserted in the test suite, not here:");
    println!("  cargo test -p openzc ratio_floors_are_maintained");
}
