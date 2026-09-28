//! BLAKE3 content integrity.
//!
//! Integrity is enforced at two layers:
//!
//! * **per chunk** — every frame carries a BLAKE3 hash of the bytes it decodes
//!   to. Damage is localised, and chunks can be verified in parallel.
//! * **whole stream** — the end marker carries a BLAKE3 hash of all decoded
//!   content, which additionally catches a frame being dropped, reordered, or
//!   duplicated.
//!
//! The chunk hash is what makes a damaged stream *fail cleanly*. Structural
//! validation bounds every read, but a bit flip inside a legal-looking length
//! field would still decode to the wrong bytes; the hash is what makes that
//! detectable rather than silent.

use crate::bytes::ByteWriter;
use crate::error::Result;
use crate::format::HASH_LEN;

/// Length of an OpenZC content hash in bytes.
pub const HASH_BYTES: usize = HASH_LEN;

#[cfg(feature = "checksum")]
mod imp {
    use super::HASH_BYTES;
    use blake3::Hasher;

    /// Incremental BLAKE3 hasher.
    #[derive(Debug, Clone)]
    pub struct ContentHasher {
        inner: Hasher,
    }

    impl ContentHasher {
        pub fn new() -> Self {
            Self {
                inner: Hasher::new(),
            }
        }

        pub fn update(&mut self, data: &[u8]) {
            self.inner.update(data);
        }

        pub fn finalize(&self) -> [u8; HASH_BYTES] {
            // `blake3::Hash` derefs to `[u8; 32]`, so copy the array out.
            self.inner.finalize().into()
        }
    }

    impl Default for ContentHasher {
        fn default() -> Self {
            Self::new()
        }
    }

    /// One-shot hash.
    pub fn hash(data: &[u8]) -> [u8; HASH_BYTES] {
        blake3::hash(data).into()
    }

    /// True when the `checksum` feature is active.
    pub const ENABLED: bool = true;
}

#[cfg(not(feature = "checksum"))]
mod imp {
    use super::HASH_BYTES;

    /// Placeholder hasher used when the `checksum` feature is disabled.
    ///
    /// It performs no hashing; callers gate every call on [`ENABLED`], so this
    /// type is never asked for a real digest.
    #[derive(Debug, Clone, Default)]
    pub struct ContentHasher {
        _private: (),
    }

    impl ContentHasher {
        pub fn new() -> Self {
            Self { _private: () }
        }

        pub fn update(&mut self, _data: &[u8]) {}

        pub fn finalize(&self) -> [u8; HASH_BYTES] {
            [0u8; HASH_BYTES]
        }
    }

    /// One-shot hash. Returns zeros when the feature is disabled.
    pub fn hash(_data: &[u8]) -> [u8; HASH_BYTES] {
        [0u8; HASH_BYTES]
    }

    /// False when integrity checking is compiled out.
    pub const ENABLED: bool = false;
}

pub use imp::{hash, ContentHasher};

/// Whether content hashing is compiled in.
///
/// When the `checksum` feature is off the engine still writes the frame and end
/// markers, so the format is unchanged; only the hashes are absent, and
/// [`ContentHasher`] becomes a no-op that callers gate on.
pub const ENABLED: bool = imp::ENABLED;

/// Serialise a hash into a 32-byte buffer.
pub fn hash_to_bytes(h: &[u8; HASH_BYTES]) -> [u8; HASH_BYTES] {
    *h
}

/// Read a hash from a buffer.
pub fn bytes_to_hash(b: &[u8]) -> Result<[u8; HASH_BYTES]> {
    if b.len() < HASH_BYTES {
        return Err(crate::error::Error::UnexpectedEof {
            needed: HASH_BYTES,
            available: b.len(),
        });
    }
    let mut out = [0u8; HASH_BYTES];
    out.copy_from_slice(&b[..HASH_BYTES]);
    Ok(out)
}

/// Append a hash to a writer.
pub fn write_hash(w: &mut ByteWriter, h: &[u8; HASH_BYTES]) {
    w.bytes(h);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hasher_is_incremental() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let one_shot = hash(&data);
        let mut h = ContentHasher::new();
        for chunk in data.chunks(97) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), one_shot);
    }

    #[test]
    fn hash_roundtrips_through_bytes() {
        let h = hash(b"openzc");
        assert_eq!(hash_to_bytes(&h), h);
        assert_eq!(bytes_to_hash(&h).unwrap(), h);
    }

    #[test]
    fn short_hash_buffer_rejected() {
        assert!(bytes_to_hash(&[0u8; 31]).is_err());
        assert!(bytes_to_hash(&[0u8; 32]).is_ok());
    }

    #[test]
    fn distinct_inputs_differ() {
        assert_ne!(hash(b"a"), hash(b"b"));
        // A single-bit change must change the digest.
        assert_ne!(hash(&[0u8]), hash(&[1u8]));
    }
}
