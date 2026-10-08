//! Baseline record of candidate selection, before any pruning is added.
//!
//! This is a measurement harness, not a test of desired behaviour: it prints what
//! the planner considers and what wins, so the effect of a pruning change can be
//! read rather than guessed. Run it before and after, and diff the output.
//!
//! `cargo test -p openzc --test candidate_probe -- --nocapture --ignored`

use std::time::Instant;

use openzc::pipeline::Planner;
use openzc::{compress_slice, Config, Level};

fn lcg(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

const N: usize = 2 * 1024 * 1024;

fn corpora() -> Vec<(&'static str, Vec<u8>)> {
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
    let mut text = Vec::with_capacity(N);
    while text.len() < N {
        text.extend_from_slice(WORDS[(lcg(&mut s) >> 24) as usize % WORDS.len()].as_bytes());
        text.push(b' ');
    }
    text.truncate(N);

    let mut runs = Vec::with_capacity(N);
    let mut r = 7u32;
    while runs.len() < N {
        r = lcg(&mut r);
        runs.extend(std::iter::repeat_n(
            (r >> 8) as u8,
            1 + (r >> 24) as usize % 200,
        ));
    }
    runs.truncate(N);

    let mut seed = 0x5EEDu32;
    let random: Vec<u8> = (0..N)
        .map(|_| {
            seed = lcg(&mut seed);
            (seed >> 24) as u8
        })
        .collect();

    vec![
        ("text", text),
        ("runs", runs),
        ("random", random),
        // Delta-eligible: the transform is expected to be added.
        ("numeric", (0..N).map(|i| ((i * 3) % 256) as u8).collect()),
        // Interleaved numeric: delta's best case, currently refused by the
        // byte-drift heuristic. Kept here because a pruning change must not make
        // this worse.
        (
            "interleaved",
            (0..N)
                .flat_map(|i| [((i / 256) % 256) as u8, (i % 256) as u8])
                .collect(),
        ),
        // Wide-alphabet text: high drift, so no transform, and the case where
        // statistical is most likely to be attempted without a chance of winning.
        ("wide-text", {
            let mut s2 = 13u32;
            (0..N)
                .map(|_| {
                    let v = lcg(&mut s2) >> 24;
                    let r = lcg(&mut s2);
                    match v % 6 {
                        0 => b'a' + (r % 26) as u8,
                        1 => b'A' + (r % 26) as u8,
                        2 => b' ' + (r % 2) as u8,
                        3 => b',',
                        4 => b'.',
                        _ => b'\n',
                    }
                })
                .collect()
        }),
        // Already-compressed-like: high entropy, low structure.
        ("near-random-acgt", {
            let mut s3 = 77u32;
            (0..N)
                .map(|_| b"ACGT"[(lcg(&mut s3) >> 24) as usize % 4])
                .collect()
        }),
    ]
}

#[test]
#[ignore = "diagnostic harness; run explicitly with --ignored --nocapture"]
fn record_candidate_selection() {
    use openzc::analyze::Analysis;
    use openzc::codec::EncodeContext;

    println!();
    println!("candidate selection baseline");
    println!(
        "{:<17} {:>4} {:>7} {:>8} {:>9} {:<24}  each candidate: size",
        "corpus", "cand", "level", "ratio", "encode", "winner"
    );
    let rule: String = "=".repeat(150);
    println!("{rule}");

    for (name, data) in corpora() {
        for level in [Level::Fast, Level::Default, Level::Max] {
            let cfg = Config::new(level);
            let planner = Planner::new(&cfg);
            let cands = planner.candidates_for(&data).expect("candidates");

            let start = Instant::now();
            let (_, stats) = compress_slice(&data, &cfg).expect("compress");
            let elapsed = start.elapsed();

            // Encode every candidate individually. This is what separates
            // "expensive work that pays" from "expensive work that does not" --
            // without the per-candidate sizes there is no way to know.
            let registry = openzc::codec::Registry::new();
            let mut sizes: Vec<String> = Vec::new();
            let mut best: Option<(usize, String)> = None;

            for (pipeline, transform) in &cands {
                // The planner applies the transform before handing bytes to the
                // codec. This probe must do the same, or every `delta` row below
                // is really measuring `none` and the whole table is a lie.
                let transformed =
                    openzc::transform::apply_forward(*transform, &data).expect("transform");
                let encoded = registry.get(*pipeline).unwrap().encode(&EncodeContext {
                    input: &transformed,
                    analysis: &Analysis::of(&transformed).expect("analysis"),
                    level,
                    dictionary: None,
                    independent: true,
                });
                match encoded {
                    Ok(payload) => {
                        let size = payload.len();
                        if best.as_ref().is_none_or(|(b, _)| size < *b) {
                            best =
                                Some((size, format!("{}/{}", pipeline.name(), transform.name())));
                        }
                        sizes.push(format!("{}/{}={}", pipeline.name(), transform.name(), size));
                    }
                    Err(e) => {
                        sizes.push(format!("{}/{}:ERR({e})", pipeline.name(), transform.name()))
                    }
                }
            }

            let (winner_size, winner) = best.expect("store always succeeds");
            println!(
                "{:<17} {:>4} {:>7} {:>8.4} {:>7.0}ms  {:<24}  {}",
                name,
                cands.len(),
                level.name(),
                stats.output_size as f64 / stats.input_size as f64,
                elapsed.as_secs_f64() * 1000.0,
                format!("{winner} ({winner_size})"),
                sizes.join("  ")
            );
        }
        println!();
    }
}

/// Per-candidate encode cost, so "which work is wasted" is a timing question
/// rather than an assumption.
#[test]
#[ignore = "diagnostic harness; run explicitly with --ignored --nocapture"]
fn per_candidate_cost() {
    use openzc::analyze::Analysis;
    use openzc::codec::EncodeContext;
    use std::hint::black_box;
    use std::time::Instant;

    println!();
    println!("per-candidate encode cost (ms, best of 3, 2 MiB input)");
    let rule: String = "=".repeat(130);
    println!("{rule}");

    for (name, data) in corpora() {
        for level in [Level::Default, Level::Max] {
            let cfg = Config::new(level);
            let planner = Planner::new(&cfg);
            let cands = planner.candidates_for(&data).expect("candidates");
            let registry = openzc::codec::Registry::new();

            let mut rows = Vec::new();
            let mut total = 0.0f64;

            for (pipeline, transform) in &cands {
                let transformed =
                    openzc::transform::apply_forward(*transform, &data).expect("transform");
                let analysis = Analysis::of(&transformed).expect("analysis");

                let mut best = f64::MAX;
                for _ in 0..3 {
                    let start = Instant::now();
                    let r = registry.get(*pipeline).unwrap().encode(&EncodeContext {
                        input: &transformed,
                        analysis: &analysis,
                        level,
                        dictionary: None,
                        independent: true,
                    });
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    black_box(&r);
                    if ms < best {
                        best = ms;
                    }
                }
                total += best;
                rows.push(format!(
                    "{}/{}={:.0}ms",
                    pipeline.name(),
                    transform.name(),
                    best
                ));
            }

            println!(
                "{:<17} {:>7}  total {:.0}ms   {}",
                name,
                level.name(),
                total,
                rows.join("  ")
            );
        }
        println!();
    }
}
