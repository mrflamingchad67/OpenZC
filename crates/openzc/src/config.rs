//! Compression configuration: levels, strategy overrides, and integrity policy.

use std::time::Duration;

/// Number of threads to use for parallel work.
///
/// `Auto` defers to the Rayon global pool. `Serial` disables parallelism
/// entirely, which is useful for benchmarks that must isolate single-thread
/// behaviour and for embedded deployments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Threads {
    /// Use the Rayon global pool (one thread per available core).
    #[default]
    Auto,
    /// Force a specific worker count.
    Fixed(usize),
    /// Disable parallel chunk processing.
    Serial,
}

impl Threads {
    /// Resolve to a concrete worker count, or `None` for "use the global pool".
    pub fn resolve(self) -> Option<usize> {
        match self {
            Threads::Auto | Threads::Fixed(0) => None,
            Threads::Fixed(n) => Some(n),
            Threads::Serial => Some(1),
        }
    }
}

/// Compression level, trading speed against ratio.
///
/// Levels are deliberately coarse. Within a level the engine still adapts per
/// chunk, so adding a level should be a visible cost change rather than a
/// hidden parameter tweak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Level {
    /// No compression attempt; frames are stored.
    None = 0,
    /// Fast byte-oriented LZ. Decompression is the priority.
    Fast = 1,
    /// Default: LZ with adaptive context modelling and entropy coding.
    Default = 3,
    /// Stronger match finding, larger effective window.
    High = 6,
    /// Maximum effort. Slow, best ratio.
    Max = 9,
}

impl Level {
    /// All levels, ascending. Used by the test-suite to sweep every level.
    pub const ALL: [Level; 5] = [
        Level::None,
        Level::Fast,
        Level::Default,
        Level::High,
        Level::Max,
    ];

    /// Parse a level from its numeric name, e.g. `"6"`.
    pub fn from_num(n: u8) -> crate::error::Result<Self> {
        Ok(match n {
            0 => Level::None,
            1 => Level::Fast,
            2..=3 => Level::Default,
            4..=6 => Level::High,
            7..=9 => Level::Max,
            other => {
                return Err(crate::error::Error::Config(format!(
                    "compression level {other} is out of range 0..=9"
                )))
            }
        })
    }

    /// Nominal numeric level, for display and for `--level`.
    pub fn num(self) -> u8 {
        self as u8
    }

    /// Short name, for `--level`, `info` output, and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Level::None => "none",
            Level::Fast => "fast",
            Level::Default => "default",
            Level::High => "high",
            Level::Max => "max",
        }
    }

    /// Parse a level from either its name or its number.
    ///
    /// Accepting both is what lets a CLI say `--level high` while a script can
    /// pass `--level 6`; a typo is an error rather than a silent default, since
    /// silently compressing at the wrong level is worse than refusing.
    pub fn parse(s: &str) -> crate::error::Result<Self> {
        let lower = s.trim().to_ascii_lowercase();
        for level in Level::ALL {
            if level.name() == lower {
                return Ok(level);
            }
        }
        match lower.parse::<u8>() {
            Ok(n) => Level::from_num(n),
            Err(_) => Err(crate::error::Error::Config(format!(
                "unknown level {s:?}; expected one of {} or 0-9",
                Level::ALL.map(Level::name).join(", ")
            ))),
        }
    }
}

/// How the engine should choose a pipeline for each chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Strategy {
    /// Analyse each chunk and try candidates, keeping the smallest result.
    ///
    /// Cheap analysis prunes obviously-useless candidates first, so the
    /// expensive path is not run when it cannot help.
    #[default]
    Adaptive,
    /// Skip analysis; use the single pipeline implied by the level.
    Fixed,
    /// Never compress. Every chunk is stored.
    NeverCompress,
}

/// Integrity verification policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Checksum {
    /// Hash every chunk *and* the whole stream. Strongest, costs a full pass.
    #[default]
    Full,
    /// Hash each chunk only. Catches damage where it occurs and allows
    /// chunk-parallel verification.
    Chunk,
    /// No hashes. Structural validation only; fastest, least safe.
    None,
}

impl Checksum {
    /// Whether per-chunk content hashes are written.
    pub fn chunk_hashes(self) -> bool {
        !matches!(self, Checksum::None)
    }

    /// Whether a whole-stream content hash is written.
    pub fn content_hash(self) -> bool {
        matches!(self, Checksum::Full)
    }
}

/// Knobs that bound memory use.
///
/// Memory consumption is a function of these values, not of the input size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowConfig {
    /// Uncompressed bytes per frame. Larger improves ratio across chunk
    /// boundaries; smaller bounds memory and raises parallelism.
    pub chunk_size: u32,
    /// Maximum LZ match distance. Must be `<= chunk_size`.
    pub window_size: u32,
}

impl WindowConfig {
    /// Smallest legal chunk, used for very small inputs and tests.
    pub const MIN_CHUNK: u32 = 1 << 10;
    /// Default 1 MiB chunk / 64 KiB window.
    pub const DEFAULT: WindowConfig = WindowConfig {
        chunk_size: 1 << 20,
        window_size: 1 << 16,
    };
    /// Large default for high levels: 4 MiB chunk, 1 MiB window.
    pub const HIGH: WindowConfig = WindowConfig {
        chunk_size: 4 << 20,
        window_size: 1 << 20,
    };

    fn validate(self) -> crate::error::Result<()> {
        if self.chunk_size == 0 {
            return Err(crate::error::Error::Config("chunk_size must be > 0".into()));
        }
        if self.window_size == 0 {
            return Err(crate::error::Error::Config(
                "window_size must be > 0".into(),
            ));
        }
        if self.window_size > self.chunk_size {
            return Err(crate::error::Error::Config(format!(
                "window_size ({}) must not exceed chunk_size ({})",
                self.window_size, self.chunk_size
            )));
        }
        if self.chunk_size > WindowConfig::MAX_CHUNK {
            return Err(crate::error::Error::Config(format!(
                "chunk_size ({}) exceeds the format maximum ({})",
                self.chunk_size,
                WindowConfig::MAX_CHUNK
            )));
        }
        Ok(())
    }

    /// Hard ceiling on chunk size.
    ///
    /// A frame records its uncompressed length in a `u32`, so the format allows
    /// just under 4 GiB. The engine caps far below that on purpose: a decoder
    /// must hold a whole chunk before it can write any of it, and the encoder
    /// holds the chunk, its analysed statistics, its parsed token list, and the
    /// candidate payloads at once. One gibibyte is the point where that working
    /// set stops being a reasonable thing to promise, and it makes "bounded
    /// memory" mean something a caller can rely on.
    pub const MAX_CHUNK: u32 = 1 << 30;

    /// Clamp both fields to values a given level permits, preserving validity.
    pub fn clamped(self, max_chunk: u32, max_window: u32) -> Self {
        let chunk_size = self.chunk_size.clamp(WindowConfig::MIN_CHUNK, max_chunk);
        let window_size = self.window_size.clamp(1, chunk_size).min(max_window);
        Self {
            chunk_size,
            window_size,
        }
    }
}

/// Full configuration for compression and decompression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    level: Level,
    strategy: Strategy,
    window: WindowConfig,
    threads: Threads,
    checksum: Checksum,
    /// Reject the whole operation if any chunk fails to shrink below its
    /// original size. Useful for archive formats that need a size guarantee.
    pub strict_size: bool,
    /// Optional upper bound on total input bytes, as a guard against
    /// decompression bombs. `None` means unlimited.
    pub max_input_size: Option<u64>,
    /// Cap on frames per stream; guards against a hostile frame count.
    pub max_frames: Option<u64>,
    /// Wall-clock budget for compression. `None` means unlimited.
    pub max_compress_time: Option<Duration>,
    /// Pre-trained dictionary bytes used to prime matching.
    pub dictionary: Option<Vec<u8>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            level: Level::Default,
            strategy: Strategy::Adaptive,
            window: WindowConfig::DEFAULT,
            threads: Threads::Auto,
            checksum: Checksum::Full,
            strict_size: false,
            max_input_size: None,
            max_frames: None,
            max_compress_time: None,
            dictionary: None,
        }
    }
}

impl Config {
    /// Config for a given level, with all other settings at their defaults.
    pub fn new(level: Level) -> Self {
        let window = match level {
            Level::None => WindowConfig::DEFAULT,
            Level::Fast => WindowConfig::DEFAULT,
            Level::Default => WindowConfig::DEFAULT,
            Level::High | Level::Max => WindowConfig::HIGH,
        };
        Self {
            level,
            window,
            ..Self::default()
        }
    }

    /// Builder: choose the level.
    pub fn with_level(mut self, level: Level) -> Self {
        let window = match level {
            Level::None | Level::Fast | Level::Default => WindowConfig::DEFAULT,
            Level::High | Level::Max => WindowConfig::HIGH,
        };
        self.level = level;
        self.window = window;
        self
    }

    /// Builder: override the strategy.
    pub fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Builder: override chunk and window sizes.
    pub fn with_window(mut self, window: WindowConfig) -> Self {
        self.window = window;
        self
    }

    /// Builder: set thread policy.
    pub fn with_threads(mut self, threads: Threads) -> Self {
        self.threads = threads;
        self
    }

    /// Builder: set the integrity policy.
    pub fn with_checksum(mut self, checksum: Checksum) -> Self {
        self.checksum = checksum;
        self
    }

    /// Builder: attach a pre-trained dictionary.
    pub fn with_dictionary(mut self, dict: Vec<u8>) -> Self {
        self.dictionary = Some(dict);
        self
    }

    /// Builder: cap compression wall-clock time.
    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.max_compress_time = Some(budget);
        self
    }

    /// Builder: reject the whole operation if a chunk fails to shrink.
    pub fn with_strict_size(mut self, strict: bool) -> Self {
        self.strict_size = strict;
        self
    }

    /// Builder: cap total input bytes, as a decompression-bomb guard.
    pub fn with_max_input_size(mut self, max: u64) -> Self {
        self.max_input_size = Some(max);
        self
    }

    /// Builder: cap the frame count, as a decompression-bomb guard.
    pub fn with_max_frames(mut self, max: u64) -> Self {
        self.max_frames = Some(max);
        self
    }

    pub fn level(&self) -> Level {
        self.level
    }

    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    pub fn window(&self) -> WindowConfig {
        self.window
    }

    pub fn threads(&self) -> Threads {
        self.threads
    }

    pub fn checksum(&self) -> Checksum {
        self.checksum
    }

    pub fn dictionary(&self) -> Option<&[u8]> {
        self.dictionary.as_deref()
    }

    /// Validate the configuration, returning a normalised copy.
    ///
    /// Clamping is applied here rather than at each use site so that the
    /// container header always advertises sizes the engine will actually use.
    pub fn validated(&self) -> crate::error::Result<Self> {
        self.window.validate()?;
        if let Some(max) = self.max_input_size {
            if max == 0 {
                return Err(crate::error::Error::Config(
                    "max_input_size must be > 0 when set".into(),
                ));
            }
        }
        if let Some(frames) = self.max_frames {
            if frames == 0 {
                return Err(crate::error::Error::Config(
                    "max_frames must be > 0 when set".into(),
                ));
            }
        }
        if let Some(d) = &self.dictionary {
            if d.is_empty() {
                return Err(crate::error::Error::Config(
                    "dictionary must not be empty".into(),
                ));
            }
            if d.len() > WindowConfig::MAX_CHUNK as usize {
                return Err(crate::error::Error::Config(format!(
                    "dictionary of {} bytes exceeds the format maximum",
                    d.len()
                )));
            }
        }
        let mut out = self.clone();
        if self.level == Level::None || self.strategy == Strategy::NeverCompress {
            out.strategy = Strategy::NeverCompress;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let c = Config::default().validated().unwrap();
        assert_eq!(c.level(), Level::Default);
        assert_eq!(c.strategy(), Strategy::Adaptive);
        assert_eq!(c.checksum(), Checksum::Full);
    }

    #[test]
    fn level_none_forces_store() {
        let c = Config::new(Level::None).validated().unwrap();
        assert_eq!(c.strategy(), Strategy::NeverCompress);
    }

    #[test]
    fn window_must_fit_in_chunk() {
        let cfg = Config::new(Level::Default).with_window(WindowConfig {
            chunk_size: 1024,
            window_size: 2048,
        });
        assert!(cfg.validated().is_err());
    }

    #[test]
    fn zero_window_rejected() {
        let cfg = Config::new(Level::Default).with_window(WindowConfig {
            chunk_size: 0,
            window_size: 0,
        });
        assert!(cfg.validated().is_err());
    }

    #[test]
    fn high_levels_use_large_window() {
        assert_eq!(Config::new(Level::High).window(), WindowConfig::HIGH);
        assert_eq!(Config::new(Level::Max).window(), WindowConfig::HIGH);
    }

    #[test]
    fn clamp_keeps_window_under_chunk() {
        let w = WindowConfig {
            chunk_size: 4096,
            window_size: 1 << 20,
        };
        let c = w.clamped(1 << 16, 1 << 16);
        assert_eq!(c.chunk_size, 4096);
        assert!(c.window_size <= c.chunk_size);
    }

    #[test]
    fn level_parsing_covers_range() {
        for n in 0..=9u8 {
            assert!(Level::from_num(n).is_ok(), "level {n}");
        }
        assert!(Level::from_num(10).is_err());
    }

    #[test]
    fn empty_dictionary_rejected() {
        let cfg = Config::new(Level::Default).with_dictionary(Vec::new());
        assert!(cfg.validated().is_err());
    }

    #[test]
    fn threads_resolve() {
        assert_eq!(Threads::Auto.resolve(), None);
        assert_eq!(Threads::Fixed(0).resolve(), None);
        assert_eq!(Threads::Fixed(4).resolve(), Some(4));
        assert_eq!(Threads::Serial.resolve(), Some(1));
    }
}
