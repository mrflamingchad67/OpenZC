//! OpenZC benchmark harness.
//!
//! # What this measures, and what it does not
//!
//! Every figure printed here is measured on data this run generated or read
//! from disk, and every measurement is paired with a round-trip check — a ratio
//! without a verified decode is not a result, it is a guess. Nothing here is
//! extrapolated, and no baseline is claimed without running it.
//!
//! The generated corpora are synthetic and chosen to exercise specific shapes.
//! They are a smoke test for regressions, **not** a substitute for real files:
//! ratio on generated data is optimistic, and `bench <file>...` should be used
//! for anything worth quoting.
//!
//! # Usage
//!
//! ```text
//! openzc-bench                      # generated corpora, all levels
//! openzc-bench <file>...            # real files
//! openzc-bench --level high <file>  # one level
//! ```

use std::time::Instant;

use openzc::{compress_slice, decompress_slice, CompressStats, Config, Level, PipelineId};

/// One measurement of a corpus at one level.
struct Row {
    input: usize,
    stats: CompressStats,
    decode_secs: f64,
    verified: bool,
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut levels = Level::ALL.to_vec();
    if let Some(pos) = args.iter().position(|a| a == "--level") {
        let name = args.remove(pos + 1);
        args.remove(pos);
        match Level::parse(&name) {
            Ok(l) => levels = vec![l],
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        }
    }

    let mut corpora: Vec<(String, Vec<u8>)> = Vec::new();
    if args.is_empty() {
        corpora.extend(generated());
    } else {
        for path in &args {
            match std::fs::read(path) {
                Ok(bytes) => corpora.push((path.clone(), bytes)),
                Err(e) => eprintln!("skipping {path}: {e}"),
            }
        }
    }

    if corpora.is_empty() {
        eprintln!("no input");
        std::process::exit(2);
    }

    for (name, data) in &corpora {
        println!("\n{name}  ({} bytes)", data.len());
        println!(
            "  {:<8} {:>10} {:>8} {:>9} {:>10} {:>10} pipelines",
            "level", "output", "ratio", "bits/byte", "compress", "decompress"
        );
        for level in &levels {
            let row = measure(data, *level);
            let mbps = mb_per_second(row.input, row.stats.elapsed.as_secs_f64());
            let dmbps = mb_per_second(row.input, row.decode_secs);
            // An empty input has no meaningful ratio, and dividing by 1 would
            // print a large nonsense number rather than admitting that.
            let (ratio, bpb) = if row.input == 0 {
                ("n/a".to_string(), "n/a".to_string())
            } else {
                let r = row.stats.output_size as f64 / row.input as f64;
                (format!("{r:.3}x"), format!("{:.3}", r * 8.0))
            };
            println!(
                "  {:<8} {:>10} {:>7} {:>9} {:>9} {:>10} {}",
                level.name(),
                row.stats.output_size,
                ratio,
                bpb,
                format!("{mbps:.0} MB/s"),
                format!("{dmbps:.0} MB/s"),
                pipeline_summary(&row.stats),
            );
            if !row.verified {
                eprintln!("  !! round trip did NOT match at level {}", level.name());
                std::process::exit(1);
            }
        }
    }
}

/// Compress, then decompress and verify, timing each half.
fn measure(data: &[u8], level: Level) -> Row {
    let config = Config::new(level);
    let (compressed, stats) = compress_slice(data, &config).expect("compress");

    let start = Instant::now();
    let decoded = decompress_slice(&compressed, &config).expect("decompress");
    let decode_secs = start.elapsed().as_secs_f64();

    Row {
        input: data.len(),
        verified: decoded.0 == data,
        stats,
        decode_secs,
    }
}

/// Throughput in MB/s, guarding against a zero-length input dividing by zero.
fn mb_per_second(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return f64::INFINITY;
    }
    bytes as f64 / secs / (1024.0 * 1024.0)
}

fn pipeline_summary(stats: &CompressStats) -> String {
    if stats.pipelines.is_empty() {
        return "none".to_string();
    }
    stats
        .pipelines
        .iter()
        .map(|(id, count)| format!("{}x{}", short_name(*id), count))
        .collect::<Vec<_>>()
        .join(" ")
}

fn short_name(id: PipelineId) -> &'static str {
    match id {
        PipelineId::Store => "store",
        PipelineId::LzFast => "lz",
        PipelineId::Statistical => "stat",
        PipelineId::Rle => "rle",
        PipelineId::Dictionary => "dict",
        // Reserved ids cannot appear in a stream this build wrote, but a
        // corrupt one could still reach here via `info`.
        _ => "other",
    }
}

/// Synthetic corpora, one per data shape the pipelines claim to handle.
fn generated() -> Vec<(String, Vec<u8>)> {
    vec![
        ("text".to_string(), text(512 * 1024)),
        ("source-like".to_string(), source_like(512 * 1024)),
        ("runs".to_string(), runs(512 * 1024)),
        ("low-entropy".to_string(), low_entropy(512 * 1024)),
        (
            "incompressible".to_string(),
            pseudo_random(512 * 1024, 0x5EED),
        ),
        ("empty".to_string(), Vec::new()),
        ("tiny".to_string(), b"hi".to_vec()),
    ]
}

/// Deterministic PRNG, so a run is reproducible without an RNG dependency.
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
        // `repeat_n` is the by-value form of `repeat`; it needs no `Clone`.
        out.extend(std::iter::repeat_n((s >> 8) as u8, len));
    }
    out.truncate(n);
    out
}

fn low_entropy(n: usize) -> Vec<u8> {
    (0..n).map(|i| b"ACGT"[(i * 7 + i / 13) % 4]).collect()
}
