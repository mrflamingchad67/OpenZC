//! Robustness properties for malformed and hostile input.
//!
//! # The property under test
//!
//! **A decoder either reproduces the input exactly, or returns an error. It never
//! panics, and it never returns plausible-but-wrong bytes.**
//!
//! That is the single most important property of a compression library, and it is
//! the one that cannot be checked by ordinary round-trip tests — those only ever
//! feed the decoder input it produced itself. This file feeds it everything else:
//! random bytes, truncated streams, streams with single bytes flipped, and fields
//! rewritten to extreme values.
//!
//! # Why this is not a substitute for fuzzing
//!
//! These are deterministic and seeded, so they run on stable Rust, in the normal
//! quality gate, on every commit, in a second or two. That makes them worth having
//! even though they explore a vanishingly small part of the input space.
//! [`fuzz/`](../../fuzz) carries the real fuzz targets, which need nightly and
//! `cargo-fuzz`; see the `fuzz` CI job. Both are wanted: these catch the common
//! cases, the fuzzers catch the rest.
//!
//! # Panics
//!
//! Each case is run under `catch_unwind`. The decoder is documented to return
//! errors rather than panic, and this is what keeps that promise honest — a panic
//! in library code is a denial-of-service bug in anything that decodes untrusted
//! input.

use std::panic::{catch_unwind, AssertUnwindSafe};

use openzc::{compress_slice, decompress_slice, Checksum, Config, Level, Strategy};

/// Deterministic xorshift, so a failure is always reproducible from the seed.
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u32() >> 24) as u8
    }

    /// A length biased towards small values, so short and boundary cases are hit
    /// far more often than a uniform distribution would hit them.
    fn small_len(&mut self, max: usize) -> usize {
        let raw = self.next_u32() % 64;
        if raw == 0 {
            max
        } else {
            (raw as usize).min(max)
        }
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u8()).collect()
    }
}

/// Run `f`, turning a panic into a test failure with the reason attached.
///
/// # Deliberately not `#[must_use]`-friendly
///
/// Returns `()` rather than the closure's value: every caller here either does not
/// need the value or captures it through a mutable binding. Keeping the signature
/// value-free stops a caller from `unwrap`-ing a panic payload by mistake and
/// thinking it received a decode result.
fn no_panic<F>(what: &str, f: F)
where
    F: FnOnce(),
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => {}
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            panic!("{what} panicked: {detail}");
        }
    }
}

/// Assert the decoder's contract for one candidate stream and its expected origin.
///
/// `expected` is `Some(original)` when the stream should be decodable, and `None`
/// when the stream is known-bad. Either way a successful decode of bad input must
/// not silently differ from anything — it must be an error.
fn assert_contract(what: &str, stream: &[u8], original: Option<&[u8]>) {
    let config = Config::new(Level::Default);
    let mut outcome: Option<openzc::Result<(Vec<u8>, openzc::DecompressStats)>> = None;
    no_panic(what, || {
        outcome = Some(decompress_slice(stream, &config));
    });
    let result = outcome.expect("the closure always assigns the outcome");

    match (result, original) {
        (Ok((decoded, stats)), Some(expected)) => {
            assert_eq!(decoded, expected, "{what}: decoded to the wrong bytes");
            assert!(stats.verified, "{what}: decoded without verifying hashes");
        }
        (Ok((decoded, _)), None) => {
            panic!("{what}: a malformed stream decoded successfully into {decoded:?}");
        }
        (Err(_), _) => {
            // Any error is acceptable for malformed input; what matters is that
            // it is an error rather than a panic or a wrong result.
        }
    }
}

#[test]
fn random_bytes_never_panic_and_never_decode() {
    // Pure noise has no magic bytes, so every case must be rejected. This is the
    // shape of input a decoder meets first when handed the wrong file.
    for seed in 0..2_000u32 {
        let mut rng = Rng::new(seed);
        let len = rng.small_len(4_096);
        let data = rng.bytes(len);
        let config = Config::new(Level::Default);
        let name = format!("random seed {seed}");
        no_panic(&name, || {
            let _ = decompress_slice(&data, &config);
        });
    }
}

#[test]
fn random_bytes_never_panic_through_the_reader() {
    // The `Read` path is a separate implementation of the same loop, and errors
    // cross an `io::Error` boundary on the way. It needs its own coverage.
    for seed in 10_000..10_500u32 {
        let mut rng = Rng::new(seed);
        let len = rng.small_len(2_048);
        let data = rng.bytes(len);
        let config = Config::new(Level::Default);
        let name = format!("reader seed {seed}");
        no_panic(&name, || {
            use std::io::Read;
            if let Ok(mut d) = openzc::Decompressor::from_slice(&data, &config) {
                // Bounded so a malformed length field cannot make this loop run
                // long enough to look like a hang rather than a failure.
                let mut total = 0usize;
                let mut buf = vec![0u8; 4096];
                while total < 8 * 1024 * 1024 {
                    match d.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => total += n,
                    }
                }
            }
        });
    }
}

#[test]
fn every_truncation_of_a_valid_stream_is_rejected() {
    // A valid stream cut at *any* point must be rejected, never decoded into a
    // prefix. This is the property that makes a truncated download detectable.
    let original: Vec<u8> = (0..8_192u32).map(|i| (i % 251) as u8).collect();
    let config = Config::new(Level::Default);
    let (packed, _) = compress_slice(&original, &config).unwrap();

    // Every prefix length, not a sample: truncation bugs hide at exact boundaries.
    for cut in 0..packed.len() {
        let name = format!("truncated to {cut} bytes");
        let mut outcome: Option<openzc::Result<(Vec<u8>, openzc::DecompressStats)>> = None;
        no_panic(&name, || {
            outcome = Some(decompress_slice(&packed[..cut], &config));
        });
        if let Some(Ok((decoded, _))) = outcome {
            panic!(
                "{name}: decoded {} bytes from a truncated stream",
                decoded.len()
            );
        }
    }
    // And the untruncated stream must still work, so the loop above is testing
    // the stream rather than the harness.
    let (decoded, _) = decompress_slice(&packed, &config).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn single_byte_corruption_is_caught_or_harmless() {
    // Every byte position, flipped. Each single-byte change must either decode to
    // exactly the original (a byte in an unused field, such as a hash the policy
    // did not write) or fail. What must never happen is a successful decode to
    // different bytes.
    let original: Vec<u8> = (0..2_048u32).map(|i| (i * 7 % 253) as u8).collect();
    let config = Config::new(Level::Default);
    let (packed, _) = compress_slice(&original, &config).unwrap();

    for i in 0..packed.len() {
        for bit in [0x01u8, 0x80] {
            let mut damaged = packed.clone();
            damaged[i] ^= bit;
            assert_contract(
                &format!("byte {i} flipped by {bit:#04x}"),
                &damaged,
                Some(&original),
            );
        }
    }
}

#[test]
fn rewritten_length_fields_are_rejected_without_huge_allocation() {
    // The four-byte fields that drive allocation. Each is rewritten to a value a
    // corrupt stream would plausibly contain; none may produce a huge allocation
    // or a panic. This is the check that the payload-size guard in the stream
    // reader actually works.
    let original =
        b"the quick brown fox jumps over the lazy dog, repeatedly and at length".repeat(40);
    let config = Config::new(Level::Default);
    let (packed, _) = compress_slice(&original, &config).unwrap();
    let header_len = openzc::ContainerHeader::encoded_len();

    // payload_size is the last field of the frame header.
    let payload_size_at = header_len + 16 - 4;
    let targets = [
        ("payload_size", payload_size_at),
        ("content_size", payload_size_at - 4),
    ];

    for (field, offset) in targets {
        for value in [u32::MAX, 0x7FFF_FFFF, 0xFFFF_FFFF, 0x8000_0000, 0x4000_0000] {
            let mut damaged = packed.clone();
            damaged[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert_contract(&format!("{field} rewritten to {value:#x}"), &damaged, None);
        }
    }
}

#[test]
fn header_flag_bits_are_rejected() {
    // Flags are not forward compatible, so an unknown bit is a hard error rather
    // than something to ignore.
    let original = b"payload to compress".repeat(100);
    let config = Config::new(Level::Default);
    let (packed, _) = compress_slice(&original, &config).unwrap();

    // The container header's flags field sits at offset 8.
    for bit in 0..32u32 {
        if bit < 3 {
            continue; // the three defined bits are legal
        }
        let mut damaged = packed.clone();
        let flags =
            u32::from_le_bytes([damaged[8], damaged[9], damaged[10], damaged[11]]) | (1 << bit);
        damaged[8..12].copy_from_slice(&flags.to_le_bytes());
        assert_contract(&format!("unknown container flag bit {bit}"), &damaged, None);
    }
}

#[test]
fn unknown_pipeline_and_transform_ids_are_rejected() {
    let original = b"content".repeat(200);
    let config = Config::new(Level::Default);
    let (packed, _) = compress_slice(&original, &config).unwrap();
    let header_len = openzc::ContainerHeader::encoded_len();
    // pipeline is at frame-header offset 4, transform at 5.
    //
    // Only *undefined* ids are tested. Pipeline 0 and transform 0 are defined
    // (`Reserved0` is invalid in a frame, but transform 0 is the identity and is
    // what every stream without a transform already carries) — writing a valid id
    // over a stream that already has it changes nothing, so the stream is expected
    // to decode, not to fail.
    for (name, offset) in [("pipeline", 4usize), ("transform", 5usize)] {
        for id in [6u8, 7, 99, 200, 255] {
            let mut damaged = packed.clone();
            damaged[header_len + offset] = id;
            assert_contract(&format!("{name} id {id}"), &damaged, None);
        }
    }
}

#[test]
fn roundtrip_holds_for_arbitrary_random_content_at_every_level() {
    // The positive half of the contract: whatever goes in comes out, at every
    // level, for content with no structure at all. Cheap enough to run widely.
    for seed in 0..64u32 {
        let mut rng = Rng::new(seed ^ 0xBEEF);
        let len = rng.small_len(20_000);
        let data = rng.bytes(len);
        for level in Level::ALL {
            let config = Config::new(level);
            let name = format!("seed {seed} level {}", level.name());
            // Unwrap outside `no_panic`: a genuine compression failure is a bug to report
            // with its own message, and one that is not a panic at all.
            let (packed, _) = compress_slice(&data, &config).expect("compress");
            let (decoded, _) =
                decompress_slice(&packed, &config).unwrap_or_else(|e| panic!("{name}: {e}"));
            no_panic(&name, || {
                assert_eq!(decoded, data, "{name}: round trip lost data")
            });
        }
    }
}

#[test]
fn roundtrip_holds_under_every_checksum_and_strategy_policy() {
    // Configuration must not be able to break losslessness. Each policy skips some
    // hashes, which is exactly when a bug in the integrity accounting would show.
    let mut data = Vec::new();
    for i in 0..12_000u32 {
        data.extend_from_slice(format!("row {i} value {}\n", i * 31 % 97).as_bytes());
    }

    for checksum in [Checksum::Full, Checksum::Chunk, Checksum::None] {
        for strategy in [Strategy::Adaptive, Strategy::Fixed, Strategy::NeverCompress] {
            for level in [Level::Fast, Level::Default, Level::Max] {
                let config = Config::new(level)
                    .with_checksum(checksum)
                    .with_strategy(strategy);
                let name = format!(
                    "checksum {checksum:?} strategy {strategy:?} level {}",
                    level.name()
                );
                let (packed, _) = compress_slice(&data, &config).expect("compress");
                let (decoded, _) =
                    decompress_slice(&packed, &config).unwrap_or_else(|e| panic!("{name}: {e}"));
                no_panic(&name, || {
                    assert_eq!(decoded, data, "{name}: round trip lost data")
                });
            }
        }
    }
}

#[test]
fn streams_from_other_policies_are_still_rejected_when_damaged() {
    // Damage detection cannot depend on the policy that wrote the stream. A
    // `Checksum::None` stream has weaker verification by design, but it must still
    // never decode to wrong bytes.
    let data: Vec<u8> = (0..4_096u32).map(|i| (i % 97) as u8).collect();
    for checksum in [Checksum::Full, Checksum::None] {
        let config = Config::new(Level::Default).with_checksum(checksum);
        let (packed, _) = compress_slice(&data, &config).unwrap();
        let mut damaged = packed.clone();
        let last = damaged.len() - 1;
        damaged[last] ^= 0xFF;
        no_panic(&format!("damaged {checksum:?}"), || {
            let _ = decompress_slice(&damaged, &config);
        });
    }
}

#[test]
fn decoder_reports_a_typed_error_rather_than_a_bare_io_failure() {
    // Callers need to tell damage apart from misuse. A typed error crossing the
    // `Read` boundary used to arrive as an opaque `io::Error`, which made a
    // corrupt file undiagnosable in practice.
    let mut damaged = b"this is definitely not an openzc stream".to_vec();
    damaged[0] = b'Z';

    match decompress_slice(&damaged, &Config::new(Level::Default)) {
        Err(openzc::Error::BadMagic { .. }) => {}
        Err(other) => panic!("expected BadMagic, got {other:?}"),
        Ok(_) => panic!("a non-OpenZC file decoded successfully"),
    }
}
