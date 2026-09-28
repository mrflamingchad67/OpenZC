//! Compression pipelines: one encoder/decoder per [`PipelineId`].
//!
//! Each pipeline owns a complete, self-contained way to turn a chunk into bytes
//! and back. Adding a pipeline means adding a module here and a variant in
//! [`PipelineId`] — no changes to the format, the stream layer, or the CLI,
//! which is what keeps the architecture replaceable.
//!
//! # The pipelines
//!
//! | Id | Name | What it does | Decode cost |
//! |---|---|---|---|
//! | 1 | [`Store`](store::StoreCodec) | copies bytes | memcpy |
//! | 2 | [`LzFast`](lz::LzFastCodec) | LZ77, byte-oriented tokens | very low |
//! | 3 | [`Statistical`](statistical::StatisticalCodec) | transform + LZ + context model + range coder | moderate |
//! | 4 | [`Rle`](rle::RleCodec) | run-length only | very low |
//! | 5 | [`Dictionary`](store::StoreCodec) | a chunk that defines a dictionary | memcpy |

pub mod lz;
pub mod rle;
pub mod statistical;
pub mod store;

use crate::analyze::Analysis;
use crate::config::{Config, Level};
use crate::error::Result;
use crate::format::{FrameHeader, PipelineId, TransformId};
use crate::matchfinder::DictionaryIndex;
use std::collections::HashMap;

/// What a codec needs in order to encode one chunk.
#[derive(Debug)]
pub struct EncodeContext<'a> {
    /// The chunk bytes, pre-transform.
    pub input: &'a [u8],
    /// Statistics from [`crate::analyze`], used to prune candidates.
    pub analysis: &'a Analysis,
    /// The configured effort level.
    ///
    /// Pipelines use this to size their search — which match finder, how deep to
    /// walk a chain, how greedily to parse. It is carried here rather than read
    /// from a global so that a codec's cost is a function of its inputs and
    /// nothing else, which is what keeps the level knob honest.
    pub level: Level,
    /// Index of a pre-trained dictionary, if the config supplied one.
    pub dictionary: Option<&'a DictionaryIndex>,
    /// True when the frame may only reference its own output.
    pub independent: bool,
}

/// What a codec needs in order to decode one frame.
#[derive(Debug)]
pub struct DecodeContext<'a> {
    /// The frame payload.
    pub payload: &'a [u8],
    /// Number of bytes the frame must decode to. Always known, which is what
    /// lets a bounded decoder reject a corrupt stream instead of overrunning.
    pub content_size: usize,
    /// Transform to invert after the main decode.
    pub transform: TransformId,
    /// A dictionary the frame references, if any.
    pub dictionary: Option<&'a [u8]>,
    /// True when the frame may only reference its own output.
    pub independent: bool,
}

/// A compression pipeline.
///
/// The two halves are deliberately separate: encoding is allowed to be slow,
/// expensive, and adaptive, while decoding must be a direct, allocation-light
/// inverse with no search.
pub trait Codec: Send + Sync {
    /// The id written into the frame header.
    fn id(&self) -> PipelineId;

    /// Short name for `info` output and diagnostics.
    fn name(&self) -> &'static str;

    /// Whether this codec is worth attempting for a chunk, judged from cheap
    /// statistics. Must not do real compression work.
    fn candidate(&self, analysis: &Analysis, level: Level) -> bool;

    /// Compress a chunk.
    fn encode(&self, ctx: &EncodeContext<'_>) -> Result<Vec<u8>>;

    /// Decompress a frame into `out`, which is pre-sized to `content_size`.
    ///
    /// Implementations must write exactly `content_size` bytes and must not
    /// read outside `payload`.
    fn decode(&self, ctx: &DecodeContext<'_>, out: &mut [u8]) -> Result<()>;
}

/// Registry of the pipelines this build implements.
pub struct Registry {
    codecs: Vec<Box<dyn Codec>>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field(
                "pipelines",
                &self.codecs.iter().map(|c| c.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// Build the registry containing every implemented pipeline.
    pub fn new() -> Self {
        Self {
            codecs: vec![
                Box::new(store::StoreCodec::dictionary()),
                Box::new(store::StoreCodec::new()),
                Box::new(rle::RleCodec::new()),
                Box::new(lz::LzFastCodec::new()),
                Box::new(statistical::StatisticalCodec::new()),
            ],
        }
    }

    /// All registered pipelines, in ascending id order.
    pub fn all(&self) -> impl Iterator<Item = &dyn Codec> {
        self.codecs.iter().map(|c| c.as_ref())
    }

    /// Look up a pipeline by id.
    pub fn get(&self, id: PipelineId) -> Result<&dyn Codec> {
        self.codecs
            .iter()
            .find(|c| c.id() == id)
            .map(|c| c.as_ref())
            .ok_or(crate::error::Error::UnknownPipeline(u32::from(id as u8)))
    }

    /// Encode `input` with `pipeline`, reporting the transform used.
    pub fn encode(
        &self,
        pipeline: PipelineId,
        transform: TransformId,
        ctx: &EncodeContext<'_>,
    ) -> Result<Vec<u8>> {
        let codec = self.get(pipeline)?;
        let _ = transform;
        codec.encode(ctx)
    }
}

/// A decoded dictionary, keyed by the 1-based `dict_id` in a frame header.
#[derive(Debug, Default)]
pub struct DictionaryTable {
    entries: HashMap<u8, Vec<u8>>,
}

impl DictionaryTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a dictionary under `id`.
    pub fn insert(&mut self, id: u8, bytes: Vec<u8>) {
        self.entries.insert(id, bytes);
    }

    /// Look up a dictionary.
    pub fn get(&self, id: u8) -> Result<&[u8]> {
        self.entries
            .get(&id)
            .map(Vec::as_slice)
            .ok_or(crate::error::Error::UnknownDictionary(u32::from(id)))
    }

    /// Number of registered dictionaries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no dictionaries are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Frame header for a stored frame. Helper shared by several codecs.
#[allow(dead_code)]
pub(crate) fn frame_header(
    pipeline: PipelineId,
    transform: TransformId,
    content_size: usize,
    payload_size: usize,
    independent: bool,
    dict_id: u8,
) -> FrameHeader {
    FrameHeader {
        pipeline,
        transform,
        flags: crate::format::FrameFlags::from_bits_unchecked(if independent {
            crate::format::FrameFlags::INDEPENDENT
        } else {
            0
        }),
        dict_id,
        content_size: content_size as u32,
        payload_size: payload_size as u32,
    }
}

/// Build the match finder appropriate for a level.
///
/// The level is the only thing that decides how hard the search works: a small
/// table with no chain for cheap levels, a deeper chain for the expensive ones.
/// Keeping the mapping here means every pipeline gets the same effort curve, so
/// a level one pipeline quietly ignored cannot make it differ from the others.
pub(crate) fn finder_for(
    level: Level,
    data_len: usize,
    window: usize,
) -> Box<dyn crate::matchfinder::MatchFinder> {
    match level {
        Level::None | Level::Fast => Box::new(crate::matchfinder::HashTable3::new(15)),
        Level::Default => Box::new(crate::matchfinder::HashTable3::new(16)),
        Level::High => Box::new(crate::matchfinder::HashChain4::new(
            16, data_len, window, 32,
        )),
        Level::Max => Box::new(crate::matchfinder::HashChain4::new(
            18, data_len, window, 256,
        )),
    }
}

/// A match finder plus a match dictionary, i.e. everything the LZ stage needs
/// to look for back-references.
pub struct MatchContext {
    pub finder: Box<dyn crate::matchfinder::MatchFinder>,
    pub dictionary: Option<DictionaryIndex>,
}

impl std::fmt::Debug for MatchContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The trait objects carry no useful state for diagnostics, so only the
        // shape is reported.
        f.debug_struct("MatchContext")
            .field("finder", &"<dyn MatchFinder>")
            .field(
                "dictionary",
                &self.dictionary.as_ref().map(DictionaryIndex::len),
            )
            .finish()
    }
}

impl MatchContext {
    /// Build a match context for a level and optional dictionary bytes.
    pub fn build(level: Level, data_len: usize, window: usize, dict: Option<&[u8]>) -> Self {
        Self {
            finder: finder_for(level, data_len, window),
            dictionary: dict.map(DictionaryIndex::build),
        }
    }
}

/// Config accessor helper: the highest window a level will use.
#[allow(dead_code)]
pub(crate) fn max_window(config: &Config) -> usize {
    config.window().window_size as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_contains_every_declared_pipeline() {
        let r = Registry::new();
        for id in [
            PipelineId::Store,
            PipelineId::LzFast,
            PipelineId::Statistical,
            PipelineId::Rle,
            PipelineId::Dictionary,
        ] {
            assert!(r.get(id).is_ok(), "missing pipeline {}", id.name());
        }
    }

    #[test]
    fn registry_ids_match_declared_ids() {
        let r = Registry::new();
        let ids: Vec<u8> = r.all().map(|c| c.id() as u8).collect();
        for id in PipelineId::all() {
            assert!(ids.contains(&(*id as u8)), "{} not registered", id.name());
        }
        assert_eq!(
            ids.len(),
            PipelineId::all().len(),
            "duplicate registrations"
        );
    }

    #[test]
    fn unknown_dictionary_lookup_errors() {
        let t = DictionaryTable::new();
        assert!(t.is_empty());
        assert!(t.get(1).is_err());
    }

    #[test]
    fn dictionary_table_roundtrip() {
        let mut t = DictionaryTable::new();
        t.insert(1, b"hello".to_vec());
        t.insert(2, b"world".to_vec());
        assert_eq!(t.len(), 2);
        assert_eq!(t.get(1).unwrap(), b"hello");
        assert_eq!(t.get(2).unwrap(), b"world");
        assert!(t.get(3).is_err());
    }

    #[test]
    fn match_context_builds_with_and_without_dictionary() {
        let c = MatchContext::build(Level::Default, 1024, 65536, None);
        assert!(c.dictionary.is_none());
        let c = MatchContext::build(Level::High, 1024, 65536, Some(b"prefix data here"));
        assert!(c.dictionary.is_some());
    }
}
