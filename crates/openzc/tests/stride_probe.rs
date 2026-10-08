//! Measurement: does stride-aware detection predict the delta winner, and what
//! does it cost?
//!
//! Run with `cargo test -p openzc --test stride_probe -- --ignored --nocapture`.
//!
//! The question is not "does delta ever help on interleaved data" — that was
//! already known — but whether *per-stride* drift predicts it, whether the signal
//! stays quiet on noise, and what the extra passes cost.

use openzc::codec::statistical::StatisticalCodec;
use openzc::codec::{Codec, EncodeContext};
use openzc::transform::detected_stride;
use openzc::{Level, TransformId};

const N: usize = 512 * 1024;

/// Mean absolute difference between bytes `stride` apart.
fn stride_drift(data: &[u8], stride: usize) -> f64 {
    let n = data.len().min(8192);
    if stride == 0 || n <= stride {
        return f64::MAX;
    }
    let mut total = 0u64;
    let mut count = 0u64;
    for i in stride..n {
        total += u64::from(data[i].abs_diff(data[i - stride]));
        count += 1;
    }
    if count == 0 {
        f64::MAX
    } else {
        total as f64 / count as f64
    }
}

fn encode_statistical(data: &[u8], transform: TransformId) -> usize {
    let t = openzc::transform::apply_forward(transform, data).expect("transform");
    let a = openzc::analyse(&t).expect("analysis");
    StatisticalCodec::new()
        .encode(&EncodeContext {
            input: &t,
            analysis: &a,
            level: Level::Default,
            dictionary: None,
            independent: true,
        })
        .map(|p| p.len())
        .unwrap_or(usize::MAX)
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    let mut out: Vec<(&'static str, Vec<u8>)> = Vec::with_capacity(12);

    out.push((
        "u16 pairs",
        (0..(N / 2))
            .flat_map(|i| ((i % 65536) as u16).to_le_bytes())
            .collect(),
    ));
    out.push((
        "u32 quads",
        (0..(N / 4))
            .flat_map(|i| (i as u32).to_le_bytes())
            .collect(),
    ));
    out.push((
        "u64 octets",
        (0..(N / 8))
            .flat_map(|i| (i as u64).to_le_bytes())
            .collect(),
    ));
    out.push((
        "f64 octets",
        (0..(N / 8))
            .flat_map(|i| ((i as f64) * 0.25).to_le_bytes())
            .collect(),
    ));

    // Interleaved but with no stride structure: the noise control.
    let mut r = Lcg(12345);
    out.push((
        "random u16 pairs",
        (0..(N / 2))
            .flat_map(|_| (r.next() as u16).to_le_bytes())
            .collect(),
    ));

    // Strides that are not powers of two: a 24-bit field, and three interleaved
    // 16-bit fields (a common record layout: x, y, z). These test whether the
    // chosen stride set earns its keep or is just the power-of-two cases.
    out.push((
        "u24 triples",
        (0..(N / 3))
            .flat_map(|i| {
                let v = i as u32;
                [
                    (v & 0xFF) as u8,
                    ((v >> 8) & 0xFF) as u8,
                    ((v >> 16) & 0xFF) as u8,
                ]
            })
            .collect(),
    ));
    out.push((
        "xyz u16 stride6",
        (0..(N / 6))
            .flat_map(|i| {
                let v = i as u32;
                (0..3u32).flat_map(move |f| ((v + f * 1000) as u16).to_le_bytes())
            })
            .collect(),
    ));

    // The corpora that must keep their current behaviour.
    out.push((
        "numeric step-3",
        (0..N).map(|i| ((i * 3) % 256) as u8).collect(),
    ));
    out.push((
        "low entropy periodic",
        (0..N).map(|i| b"ACGT"[(i * 7 + i / 13) % 4]).collect(),
    ));
    let mut r2 = Lcg(77);
    out.push((
        "random acgt",
        (0..N)
            .map(|_| b"ACGT"[(r2.next() >> 24) as usize % 4])
            .collect(),
    ));

    const WORDS: [&str; 8] = [
        "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ",
    ];
    let mut text = Vec::with_capacity(N);
    let mut i = 0;
    while text.len() < N {
        text.extend_from_slice(WORDS[i % WORDS.len()].as_bytes());
        i += 1;
    }
    text.truncate(N);
    out.push(("text", text));

    let mut runs = Vec::with_capacity(N);
    let mut r3 = Lcg(7);
    while runs.len() < N {
        let v = r3.next();
        runs.extend(std::iter::repeat_n(
            (v >> 8) as u8,
            1 + (v >> 24) as usize % 200,
        ));
    }
    runs.truncate(N);
    out.push(("runs", runs));

    let mut r4 = Lcg(0x5EED);
    out.push((
        "uniform random",
        (0..N).map(|_| (r4.next() >> 24) as u8).collect(),
    ));

    out
}

#[test]
#[ignore = "diagnostic harness; run explicitly with --ignored --nocapture"]
fn stride_detection_probe() {
    println!();
    println!("stride drift vs delta outcome");
    println!(
        "{:<22} {:>7} {:>6} {:>6} {:>6} {:>6} {:>7} {:>10} {:>10} {:>8}  delta wins",
        "corpus", "adj", "s2", "s3", "s4", "s8", "stride", "none", "delta", "gain"
    );
    let rule: String = "=".repeat(136);
    println!("{rule}");

    for (name, data) in corpus() {
        let adj = stride_drift(&data, 1);
        let s2 = stride_drift(&data, 2);
        let s3 = stride_drift(&data, 3);
        let s4 = stride_drift(&data, 4);
        let s8 = stride_drift(&data, 8);
        let detected = detected_stride(&data)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "-".to_string());

        let none = encode_statistical(&data, TransformId::None);
        let delta = encode_statistical(&data, TransformId::Delta);
        let gain = none as f64 / delta as f64;

        println!(
            "{name:<22} {adj:>7.2} {s2:>6.2} {s3:>6.2} {s4:>6.2} {s8:>6.2} {detected:>7} {none:>10} {delta:>10} {gain:>7.2}x  {}",
            // `gain` above is none/delta, so the win test is the other ratio.
            // A tie band is deliberately tight: random data varies by a handful
            // of bytes between runs and none of that is compression.
            match delta as f64 / none as f64 {
                r if r < 0.99 => "YES",
                r if r > 1.01 => "no",
                _ => "tie",
            }
        );
    }
}

/// End-to-end cost and result, which is what the detector has to justify itself on.
///
/// Reports the whole `compress_slice` call, because the analysis cost only
/// matters next to what the extra candidates buy.
#[test]
#[ignore = "diagnostic harness; run explicitly with --ignored --nocapture"]
fn end_to_end_probe() {
    use std::hint::black_box;
    use std::time::Instant;

    use openzc::{compress_slice, decompress_slice, Config};

    println!();
    println!("end to end, 512 KiB, level default, best of 3");
    println!(
        "{:<22} {:>9} {:>10} {:>9} {:>10} {:>9}",
        "corpus", "ratio", "bytes", "encode", "decode", "verified"
    );
    let rule: String = "=".repeat(80);
    println!("{rule}");

    let config = Config::new(Level::Default);
    for (name, data) in corpus() {
        let mut best_enc = f64::MAX;
        let mut packed = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let (p, _) = compress_slice(black_box(&data), &config).expect("compress");
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if ms < best_enc {
                best_enc = ms;
            }
            packed = p;
        }

        let mut best_dec = f64::MAX;
        let mut verified = false;
        for _ in 0..3 {
            let start = Instant::now();
            let (out, stats) = decompress_slice(black_box(&packed), &config).expect("decompress");
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if ms < best_dec {
                best_dec = ms;
            }
            verified = stats.verified && out == data;
        }

        println!(
            "{name:<22} {:>9.4} {:>10} {:>7.0}ms {:>8.0}ms {:>9}",
            packed.len() as f64 / data.len() as f64,
            packed.len(),
            best_enc,
            best_dec,
            if verified { "yes" } else { "NO" }
        );
    }
}

/// Cost of the analysis, since a detector that costs more than it saves is worse
/// than no detector.
#[test]
#[ignore = "diagnostic harness; run explicitly with --ignored --nocapture"]
fn analysis_cost_probe() {
    use std::hint::black_box;
    use std::time::Instant;

    use openzc::transform::{DeltaTransform, Transform};

    println!();
    println!("transform-analysis cost, 8 KiB sample, best of 500");
    println!("{:<22} {:>12}", "corpus", "per call");

    for (name, data) in corpus() {
        let mut best = f64::MAX;
        for _ in 0..500 {
            let start = Instant::now();
            black_box(DeltaTransform.worth_trying(&data));
            let ns = start.elapsed().as_secs_f64() * 1e9;
            if ns < best {
                best = ns;
            }
        }
        println!("{name:<22} {best:>9.0} ns");
    }
}
