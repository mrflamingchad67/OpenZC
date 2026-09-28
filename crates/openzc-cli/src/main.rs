//! `openzc` — command line front-end for the OpenZC engine.
//!
//! # Why the CLI is this thin
//!
//! Every decision the engine can make is made by the library, and this binary
//! only turns arguments into a [`Config`] and reports what came back. There is no
//! compression logic here, which is deliberate: a CLI that "optimises" a stream on
//! the way past would produce files the library cannot guarantee, and the whole
//! value of the format is that a decoder needs nothing but the bytes.
//!
//! # Exit codes
//!
//! They are stable and meaningful, so scripts can branch on them:
//!
//! | Code | Meaning |
//! |---|---|
//! | 0 | success |
//! | 1 | the operation failed (corrupt input, I/O error, bad arguments) |
//! | 2 | the arguments themselves were wrong — nothing was attempted |
//! | 3 | a self-test found a mismatch |

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use openzc::{
    analyse, compress_slice, decompress_slice, predict_pipeline, Checksum, CompressStats, Config,
    ContainerFlags, ContainerHeader, Error, Level, OpenZc, PipelineId, Strategy, Threads,
    TransformId, WindowConfig,
};

/// Frames listed by `info` before it stops and says so.
///
/// A stream may legitimately hold far more than this; the cap exists so a
/// corrupt length field cannot make `info` print forever.
const MAX_FRAMES_REPORTED: usize = 4096;

/// Bad arguments: nothing was attempted.
const EXIT_USAGE: u8 = 2;
/// A self-test found a mismatch.
const EXIT_VERIFY: u8 = 3;

const USAGE: &str = "\
openzc - lossless adaptive compression

USAGE:
    openzc <COMMAND> [OPTIONS] <FILE>...

COMMANDS:
    compress     Compress files into .ozc streams
    decompress   Decompress .ozc streams back to the original files
    info         Report what a stream contains, without decoding it
    test         Compress and decompress in memory, verifying the result
    help         Show this message

COMMON OPTIONS:
    -l, --level <LEVEL>      none, fast, default, high, max, or 0-9  [default: default]
    -s, --strategy <MODE>    adaptive, fixed, never                 [default: adaptive]
    -c, --checksum <MODE>    full, chunk, none                       [default: full]
    -t, --threads <N>        worker threads; 0 = auto, 1 = serial   [default: 0]
    -f, --force              overwrite an existing output file
    -q, --quiet              suppress the summary line
    -h, --help               show this message

COMPRESS OPTIONS:
    -k, --chunk-size <N>     uncompressed bytes per frame
    -w, --window <N>         maximum match distance
    -m, --max-input <N>      refuse inputs larger than this

OUTPUT:
    Without -o, each input FILE becomes FILE.ozc / FILE (with .ozc stripped).
    A FILE of - means standard input; standard output is used when there is
    exactly one input and it is -, or when -o - is given.
";

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(Failure { code, message }) => {
            eprintln!("openzc: {message}");
            ExitCode::from(code)
        }
    }
}

/// A failure with the exit code it should produce.
struct Failure {
    code: u8,
    message: String,
}

impl Failure {
    fn new(code: u8, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// A usage error: the arguments were wrong, so nothing was attempted.
fn usage(message: impl Into<String>) -> Failure {
    Failure::new(EXIT_USAGE, message)
}

fn run() -> Result<u8, Failure> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print!("{USAGE}");
        return Ok(EXIT_USAGE);
    }

    let command = args.remove(0);
    if matches!(command.as_str(), "-h" | "--help" | "help") {
        print!("{USAGE}");
        return Ok(0);
    }
    if let Some(version) = command.strip_prefix("--version") {
        if version.is_empty() {
            println!("openzc {}", env!("CARGO_PKG_VERSION"));
            return Ok(0);
        }
    }

    let opts = Options::parse(&mut args)?;
    match command.as_str() {
        "compress" => compress(&opts),
        "decompress" => decompress(&opts),
        "info" => info(&opts),
        "test" => verify(&opts),
        other => Err(usage(format!(
            "unknown command {other:?}; try `openzc help`"
        ))),
    }
}

/// Parsed command line.
struct Options {
    config: Config,
    output: Option<PathBuf>,
    inputs: Vec<PathBuf>,
    force: bool,
    quiet: bool,
}

impl Options {
    fn parse(args: &mut Vec<String>) -> Result<Self, Failure> {
        let mut level = None;
        let mut strategy = None;
        let mut checksum = None;
        let mut threads = None;
        // Annotated because the `-k`/`-w` branches read the previous value while
        // also establishing the type, and inference cannot see that far ahead.
        let mut window: Option<WindowConfig> = None;
        let mut output = None;
        let mut max_input = None;
        let mut force = false;
        let mut quiet = false;
        let mut inputs: Vec<PathBuf> = Vec::new();
        // Everything after `--` is a filename, so a file called `-l` still works.
        let mut literal = false;

        while let Some(arg) = args.first().cloned() {
            args.remove(0);
            if literal {
                inputs.push(PathBuf::from(arg));
                continue;
            }
            match arg.as_str() {
                "--" => literal = true,
                "-h" | "--help" => {
                    print!("{USAGE}");
                    std::process::exit(0);
                }
                "-f" | "--force" => force = true,
                "-q" | "--quiet" => quiet = true,
                "-o" | "--output" => output = Some(PathBuf::from(take_value(args, "-o")?)),
                "-l" | "--level" => {
                    let v = take_value(args, "-l")?;
                    level = Some(Level::parse(&v).map_err(|e| usage(e.to_string()))?);
                }
                "-s" | "--strategy" => {
                    strategy = Some(match take_value(args, "-s")?.as_str() {
                        "adaptive" => Strategy::Adaptive,
                        "fixed" => Strategy::Fixed,
                        "never" | "store" => Strategy::NeverCompress,
                        other => {
                            return Err(usage(format!(
                                "unknown strategy {other:?}; expected adaptive, fixed or never"
                            )))
                        }
                    });
                }
                "-c" | "--checksum" => {
                    checksum = Some(match take_value(args, "-c")?.as_str() {
                        "full" => Checksum::Full,
                        "chunk" => Checksum::Chunk,
                        "none" => Checksum::None,
                        other => {
                            return Err(usage(format!(
                                "unknown checksum mode {other:?}; expected full, chunk or none"
                            )))
                        }
                    });
                }
                "-t" | "--threads" => {
                    let v = take_value(args, "-t")?;
                    let n: usize = v
                        .parse()
                        .map_err(|_| usage(format!("--threads expects a number, got {v:?}")))?;
                    threads = Some(if n == 0 {
                        Threads::Auto
                    } else {
                        Threads::Fixed(n)
                    });
                }
                "-k" | "--chunk-size" => {
                    let n = parse_size(&take_value(args, "-k")?, "--chunk-size")?;
                    let current = window.map_or(WindowConfig::DEFAULT.chunk_size, |w| w.chunk_size);
                    window = Some(WindowConfig {
                        chunk_size: n.min(u64::from(u32::MAX)) as u32,
                        // A window larger than the chunk would be a lie: a match
                        // can never reach further back than the chunk itself.
                        window_size: (u64::from(current)).min(n).min(u64::from(u32::MAX)) as u32,
                    });
                }
                "-w" | "--window" => {
                    let n = parse_size(&take_value(args, "-w")?, "--window")?;
                    let current = window.map_or(WindowConfig::DEFAULT.chunk_size, |w| w.chunk_size);
                    window = Some(WindowConfig {
                        chunk_size: (u64::from(current)).max(n).min(u64::from(u32::MAX)) as u32,
                        window_size: n.min(u64::from(u32::MAX)) as u32,
                    });
                }
                "-m" | "--max-input" => {
                    max_input = Some(parse_size(&take_value(args, "-m")?, "--max-input")?);
                }
                _ if arg.starts_with('-') && arg != "-" => {
                    return Err(usage(format!("unknown option {arg:?}; try `openzc help`")));
                }
                _ => inputs.push(PathBuf::from(arg)),
            }
        }

        if inputs.is_empty() {
            return Err(usage("no input files; try `openzc help`"));
        }

        let mut config = Config::new(level.unwrap_or(Level::Default));
        if let Some(s) = strategy {
            config = config.with_strategy(s);
        }
        if let Some(c) = checksum {
            config = config.with_checksum(c);
        }
        if let Some(t) = threads {
            config = config.with_threads(t);
        }
        if let Some(w) = window {
            config = config.with_window(w);
        }
        if let Some(m) = max_input {
            config = config.with_max_input_size(m);
        }

        Ok(Self {
            config,
            output,
            inputs,
            force,
            quiet,
        })
    }

    /// True when output should go to standard output.
    fn to_stdout(&self) -> bool {
        match &self.output {
            Some(p) => p.as_os_str() == "-",
            None => self.inputs.len() == 1 && self.inputs[0].as_os_str() == "-",
        }
    }
}

/// Consume the value belonging to `flag`.
fn take_value(args: &mut Vec<String>, flag: &str) -> Result<String, Failure> {
    if args.is_empty() {
        return Err(usage(format!("{flag} needs a value")));
    }
    Ok(args.remove(0))
}

/// Parse a byte count, allowing `K`/`M`/`G` suffixes.
fn parse_size(value: &str, flag: &str) -> Result<u64, Failure> {
    let (digits, scale) = match value.as_bytes().last() {
        Some(b'k' | b'K') => (&value[..value.len() - 1], 1024u64),
        Some(b'm' | b'M') => (&value[..value.len() - 1], 1024 * 1024),
        Some(b'g' | b'G') => (&value[..value.len() - 1], 1024 * 1024 * 1024),
        _ => (value, 1),
    };
    let n: u64 = digits
        .parse()
        .map_err(|_| usage(format!("{flag} expects a byte count, got {value:?}")))?;
    n.checked_mul(scale)
        .ok_or_else(|| usage(format!("{flag} value {value:?} is too large")))
}

/// Report a library error, keeping its classification for the exit code.
fn describe(context: &str, e: &Error) -> Failure {
    Failure::new(1, format!("{context}: {e}"))
}

fn compress(opts: &Options) -> Result<u8, Failure> {
    let engine = OpenZc::new(opts.config.clone()).map_err(|e| describe("configuration", &e))?;
    if opts.to_stdout() {
        let input = read_input(&opts.inputs[0])?;
        let mut out = BufWriter::new(io::stdout().lock());
        let stats = engine
            .compress(BufReader::new(&input[..]), &mut out)
            .map_err(|e| describe("compressing stdin", &e))?;
        out.flush()
            .map_err(|e| describe("writing stdout", &Error::from(e)))?;
        report(&stats, opts);
        return Ok(0);
    }

    for input in &opts.inputs {
        let output = match &opts.output {
            Some(o) if opts.inputs.len() == 1 => o.clone(),
            _ => with_suffix(input, "ozc"),
        };
        let stats = compress_one(&engine, input, &output, opts)?;
        report(&stats, opts);
    }
    Ok(0)
}

fn compress_one(
    engine: &OpenZc,
    input: &Path,
    output: &Path,
    opts: &Options,
) -> Result<CompressStats, Failure> {
    let src = File::open(input)
        .map_err(|e| describe(&format!("opening {}", input.display()), &e.into()))?;
    let mut dst = create_output(output, opts.force)?;
    let stats = engine
        .compress(BufReader::new(src), &mut dst)
        .map_err(|e| describe(&format!("compressing {}", input.display()), &e))?;
    // Only now that the stream is complete is the file safe to publish.
    drop(dst);
    Ok(stats)
}

fn decompress(opts: &Options) -> Result<u8, Failure> {
    if opts.to_stdout() {
        let input = read_input(&opts.inputs[0])?;
        let (data, _) = decompress_slice(&input, &opts.config)
            .map_err(|e| describe("decompressing stdin", &e))?;
        io::stdout()
            .lock()
            .write_all(&data)
            .map_err(|e| describe("writing stdout", &e.into()))?;
        return Ok(0);
    }

    for input in &opts.inputs {
        let output = match &opts.output {
            Some(o) if opts.inputs.len() == 1 => o.clone(),
            _ => input.with_extension(""),
        };
        let packed = std::fs::read(input)
            .map_err(|e| describe(&format!("reading {}", input.display()), &e.into()))?;
        let (data, stats) = decompress_slice(&packed, &opts.config)
            .map_err(|e| describe(&format!("decompressing {}", input.display()), &e))?;
        create_output(&output, opts.force)?
            .write_all(&data)
            .map_err(|e| describe(&format!("writing {}", output.display()), &e.into()))?;
        if !opts.quiet {
            println!(
                "{} -> {}  {} bytes in {} frames{}",
                input.display(),
                output.display(),
                stats.output_size,
                stats.frames,
                if stats.verified { ", verified" } else { "" }
            );
        }
    }
    Ok(0)
}

fn info(opts: &Options) -> Result<u8, Failure> {
    for input in &opts.inputs {
        let data = if input.as_os_str() == "-" {
            read_input(input)?
        } else {
            std::fs::read(input)
                .map_err(|e| describe(&format!("reading {}", input.display()), &e.into()))?
        };
        print_info(input, &data, &opts.config)?;
    }
    Ok(0)
}

fn print_info(path: &Path, data: &[u8], config: &Config) -> Result<(), Failure> {
    let header_len = ContainerHeader::encoded_len();
    let header = ContainerHeader::decode(&data[..header_len.min(data.len())])
        .map_err(|e| describe(&format!("reading the header of {}", path.display()), &e))?;

    let version = openzc::FORMAT_VERSION;
    println!("file           {}", path.display());
    println!("format         {}.{}", version.0, version.1);
    println!("header         {} bytes", header_len);
    println!("chunk size     {}", header.chunk_size);
    println!("window         {}", header.window_size);
    println!(
        "content size   {}",
        header
            .content_size
            .map_or_else(|| "unset".to_string(), |v| v.to_string())
    );
    println!(
        "content hash   {}",
        yes_no(header.flags.contains(ContainerFlags::CONTENT_CHECKSUM))
    );
    println!(
        "frame hash     {}",
        yes_no(header.flags.contains(ContainerFlags::CHUNK_CHECKSUM))
    );

    // Frame detail is the interesting part of `info` and needs only the frame
    // headers, never a decode. A truncated file still reports whatever it has.
    let frames = list_frames(data, config);
    println!("frames         {}", frames.len());
    for (i, f) in frames.iter().enumerate() {
        println!(
            "  [{i}] {:<11} transform {:<7} content {:>10}  payload {:>10}",
            f.0.name(),
            f.1.name(),
            f.2,
            f.3
        );
    }
    Ok(())
}

/// Pipeline, transform, content size and payload size of each frame.
fn list_frames(data: &[u8], config: &Config) -> Vec<(PipelineId, TransformId, u32, u32)> {
    let Ok(cfg) = config.validated() else {
        return Vec::new();
    };
    let Ok(mut reader) =
        openzc::stream::StreamReader::new(openzc::stream::SliceReader::new(data), &cfg)
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    // Bounded, so a corrupt stream cannot make this loop forever.
    for _ in 0..MAX_FRAMES_REPORTED {
        match reader.next_frame() {
            Ok(Some(raw)) => out.push((
                raw.header.pipeline,
                raw.header.transform,
                raw.header.content_size,
                raw.header.payload_size,
            )),
            _ => break,
        }
    }
    out
}

fn verify(opts: &Options) -> Result<u8, Failure> {
    for input in &opts.inputs {
        let data = if input.as_os_str() == "-" {
            read_input(input)?
        } else {
            std::fs::read(input)
                .map_err(|e| describe(&format!("reading {}", input.display()), &e.into()))?
        };
        let (packed, stats) = compress_slice(&data, &opts.config)
            .map_err(|e| describe(&format!("compressing {}", input.display()), &e))?;
        let (back, decoded) = decompress_slice(&packed, &opts.config)
            .map_err(|e| describe(&format!("decompressing {}", input.display()), &e))?;

        if back != data {
            return Err(Failure::new(
                EXIT_VERIFY,
                format!(
                    "{}: round trip did not reproduce the input",
                    input.display()
                ),
            ));
        }
        if !decoded.verified {
            return Err(Failure::new(
                EXIT_VERIFY,
                format!(
                    "{}: decoded without reporting verification",
                    input.display()
                ),
            ));
        }

        let a = analyse(&data).map_err(|e| describe("analysing", &e))?;
        let predicted = predict_pipeline(&data, &opts.config)
            .map_err(|e| describe("predicting a pipeline", &e))?;
        println!(
            "{}  {} -> {} bytes  ratio {:.3}  {} frame(s)  analysed as {} (entropy {:.2} bits/byte)  picked {}",
            input.display(),
            stats.input_size,
            stats.output_size,
            stats.output_size as f64 / stats.input_size.max(1) as f64,
            stats.frames,
            a.kind.name(),
            a.entropy,
            predicted.name()
        );
        for (id, count) in &stats.pipelines {
            println!("  {} x{count}", id.name());
        }
    }
    Ok(0)
}

fn report(stats: &CompressStats, opts: &Options) {
    if opts.quiet {
        return;
    }
    let ratio = stats.output_size as f64 / stats.input_size.max(1) as f64;
    let secs = stats.elapsed.as_secs_f64();
    // A small file can compress in microseconds, where MB/s is noise. Saying so is
    // more honest than printing a large number with four decimal places.
    let speed = if secs >= 0.01 {
        format!(
            "{:.0} MB/s",
            stats.input_size as f64 / secs / (1024.0 * 1024.0)
        )
    } else {
        "too fast to time".to_string()
    };
    println!(
        "{} -> {} bytes  ratio {:.3}  {} frame(s)  {speed}",
        stats.input_size, stats.output_size, ratio, stats.frames
    );
}

fn yes_no(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}

fn read_input(path: &Path) -> Result<Vec<u8>, Failure> {
    let mut buf = Vec::new();
    BufReader::new(io::stdin().lock())
        .read_to_end(&mut buf)
        .map_err(|e| describe("reading stdin", &e.into()))?;
    let _ = path;
    Ok(buf)
}

/// `foo.txt` + `ozc` = `foo.txt.ozc`.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".");
    name.push(suffix);
    PathBuf::from(name)
}

/// Create an output file, refusing to clobber unless asked.
fn create_output(path: &Path, force: bool) -> Result<BufWriter<File>, Failure> {
    if !force && path.exists() {
        return Err(Failure::new(
            1,
            format!(
                "{} already exists; pass --force to overwrite",
                path.display()
            ),
        ));
    }
    File::create(path)
        .map(BufWriter::new)
        .map_err(|e| describe(&format!("creating {}", path.display()), &Error::from(e)))
}
