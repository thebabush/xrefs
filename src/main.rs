use anyhow::Result;
use clap::{Parser, ValueEnum};
use std::io::Write as _;
use std::ops::ControlFlow;
use std::path::PathBuf;
use xr::output::{
    truncate_middle, ContextLine, CsvPrinter, JsonlPrinter, Printer, TextPrinter, XrefRecord,
};
use xr::rust::StringBlobIndex;
use xr::va::VaRange;
use xr::xref::{Confidence, XrefKind};
use xr::{Depth, LoadedBinary, PassConfig, Va, XrefPass};

/// Capacity for the stdout BufWriter (4 MiB).
///
/// Each batch is formatted sequentially then written in a single `write_all` call,
/// amortising syscall overhead on high-throughput output.
const STDOUT_BUF_CAPACITY: usize = 4 * 1024 * 1024;

#[derive(Parser)]
#[command(name = "xr", about = "Fast multi-level xref extraction")]
struct Cli {
    /// Binary to analyze (ELF, single-arch Mach-O, PE, or raw).
    /// Fat (universal) Mach-O binaries are not supported — extract the
    /// desired slice first with `lipo -extract <arch> <input> -output <output>`.
    binary: PathBuf,

    /// Override the load base VA for PIE ELF binaries (hex or decimal).
    /// By default PIE ELFs (ET_DYN with first PT_LOAD at 0) are rebased to
    /// 0x400000. Use this to match a specific runtime load address or to
    /// reproduce IDA's layout for a different base.
    /// Ignored for non-PIE ELF, Mach-O, and PE.
    #[arg(long, value_parser = Va::parse)]
    base: Option<Va>,

    /// Analysis depth
    #[arg(short, long, default_value = "paired")]
    depth: Depth,

    /// Number of worker threads (0 = all CPUs)
    #[arg(short = 'j', long, default_value = "0")]
    workers: usize,

    /// Output format (text, jsonl, csv)
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    /// Minimum target VA for emitted xrefs. Xrefs whose 'to' is below this
    /// are silently dropped. Default: auto-detect from binary (binary.min_va()).
    /// Set to 0 to disable filtering.
    #[arg(long, value_parser = Va::parse)]
    min_ref_va: Option<Va>,

    /// Filter output to xrefs of this kind.
    /// When omitted, all kinds are shown.
    #[arg(short = 'k', long)]
    kind: Option<KindFilter>,

    /// Instructions of disasm context BEFORE each xref site (like grep -B).
    /// When non-zero, enables context display for that xref.
    #[arg(short = 'B', long = "before-context", default_value = "0")]
    before: usize,

    /// Instructions of disasm context AFTER each xref site (like grep -A).
    /// When non-zero, enables context display for that xref.
    #[arg(short = 'A', long = "after-context", default_value = "0")]
    after: usize,

    /// Cap output at N xrefs (0 = unlimited).
    #[arg(long, default_value = "0")]
    limit: usize,

    /// Restrict scanning to `from` addresses >= this VA (hex or decimal).
    #[arg(long, value_parser = Va::parse)]
    start: Option<Va>,

    /// Restrict scanning to `from` addresses < this VA (hex or decimal).
    #[arg(long, value_parser = Va::parse)]
    end: Option<Va>,

    /// Retain only xrefs whose `to` address >= this VA (hex or decimal).
    #[arg(long, value_parser = Va::parse)]
    ref_start: Option<Va>,

    /// Retain only xrefs whose `to` address < this VA (hex or decimal).
    #[arg(long, value_parser = Va::parse)]
    ref_end: Option<Va>,

    /// ARM32 decode mode override.
    /// auto  = per-section heuristic (mapping symbols → section-name fallback).
    /// thumb = force all executable ARM32 segments to Thumb-2.
    /// arm   = force all executable ARM32 segments to classic ARM32 (A32).
    /// Ignored for non-ARM32 binaries.
    #[arg(long, default_value = "auto")]
    arm32_mode: Arm32Mode,

    /// Enable Rust-specific analysis: extract string literals from binaries.
    /// Scans .rodata for UTF-8 blobs and resolves (ptr, len) pairs to strings.
    #[arg(long)]
    rust: bool,

    /// Minimum size (bytes) for UTF-8 blobs when using --rust (default: 16).
    /// Rust concatenates string literals into large pools; this filters noise.
    #[arg(long, default_value = "16")]
    rust_min_blob: usize,

    /// Max display width for extracted Rust strings (0 = no limit).
    /// Longer strings are truncated with a middle ellipsis.
    #[arg(long, default_value = "100")]
    rust_string_max: usize,

    /// Do not print symbol names next to xref addresses.
    /// By default, names defined at exactly the `from`/`to` VA are shown
    /// (text: `<a|b>` after the address; jsonl: `from_names`/`to_names`;
    /// csv: trailing `from_names`/`to_names` columns).
    #[arg(long)]
    no_names: bool,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Jsonl,
    Csv,
}

/// ARM32 decode-mode override for `--arm32-mode`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Arm32Mode {
    /// Per-section heuristic: mapping symbols when present, otherwise
    /// `.plt` → ARM32, everything else → Thumb.
    Auto,
    /// Force all executable ARM32 segments to Thumb-2 mode.
    Thumb,
    /// Force all executable ARM32 segments to classic ARM32 (A32) mode.
    Arm,
}

/// Scored xref kind filter — the five canonical categories that IDA reports.
#[derive(Clone, Copy, ValueEnum)]
enum KindFilter {
    Call,
    Jump,
    #[value(name = "data_read")]
    DataRead,
    #[value(name = "data_write")]
    DataWrite,
    #[value(name = "data_ptr")]
    DataPtr,
}

impl KindFilter {
    fn to_scored_kind(self) -> XrefKind {
        match self {
            Self::Call => XrefKind::Call,
            Self::Jump => XrefKind::Jump,
            Self::DataRead => XrefKind::DataRead,
            Self::DataWrite => XrefKind::DataWrite,
            Self::DataPtr => XrefKind::DataPointer,
        }
    }
}


fn main() -> Result<()> {
    let cli = Cli::parse();
    let depth = cli.depth;

    eprintln!("loading {}...", cli.binary.display());
    let mut binary = LoadedBinary::load_with_base(&cli.binary, cli.base.map(Va::raw))?;

    // Apply --arm32-mode override before any scanning.
    if binary.arch == xr::Arch::Arm32 && cli.arm32_mode != Arm32Mode::Auto {
        let forced = match cli.arm32_mode {
            Arm32Mode::Thumb => xr::DecodeMode::Thumb,
            Arm32Mode::Arm   => xr::DecodeMode::Arm32,
            Arm32Mode::Auto  => unreachable!(),
        };
        for seg in &mut binary.segments {
            if let xr::SegmentArch::Arm32(ref mut a) = seg.arch {
                // Override both the default and clear any symbol-derived
                // switching points — the user explicitly chose a single mode.
                a.default_mode = forced;
                a.switches.clear();
            }
        }
    }

    eprintln!(
        "arch={:?}  segments={}  entry_points={}",
        binary.arch,
        binary.segments.len(),
        binary.entry_points.len()
    );

    // Build Rust string blob index if --rust is enabled
    let blob_index = if cli.rust {
        eprintln!(
            "scanning for Rust string blobs (min_len={})...",
            cli.rust_min_blob
        );
        let idx = StringBlobIndex::build(&binary, cli.rust_min_blob);
        eprintln!(
            "found {} blobs, {} total bytes",
            idx.len(),
            idx.total_bytes()
        );
        Some(idx)
    } else {
        None
    };

    let min_ref_va = Some(cli.min_ref_va.unwrap_or_else(|| binary.min_va()));

    let from_range = VaRange::from_bounds(cli.start, cli.end);
    let to_range = VaRange::from_bounds(cli.ref_start, cli.ref_end);

    let config = PassConfig {
        depth,
        workers: cli.workers,
        min_ref_va,
        from_range,
        to_range,
        ..Default::default()
    };
    eprintln!(
        "running xref pass (depth={depth:?}, workers={})...",
        config.workers
    );

    // ── Streaming xref pass ───────────────────────────────────────────────────

    // Context is enabled whenever -A or -B is non-zero (like grep).
    let want_context = cli.before > 0 || cli.after > 0;
    if want_context && binary.arch == xr::Arch::Arm32 {
        eprintln!(
            "warning: disassembly context (-A/-B) is not yet implemented for \
             ARM32/Thumb; context lines will be omitted"
        );
    }

    // Skip lookups entirely when disabled or when the binary has no names.
    let use_names = !cli.no_names && !binary.names.is_empty();

    let kind_filter = cli.kind;
    let limit = cli.limit;
    let mut emitted = 0usize;

    let printer: Box<dyn Printer> = match cli.format {
        OutputFormat::Text => Box::new(TextPrinter),
        OutputFormat::Jsonl => Box::new(JsonlPrinter),
        OutputFormat::Csv => Box::new(CsvPrinter),
    };

    // Single BufWriter — batches are pre-formatted in parallel then written
    // here in one write_all call per batch.
    let mut stdout = std::io::BufWriter::with_capacity(STDOUT_BUF_CAPACITY, std::io::stdout());

    let hdr = printer.header_bytes();
    if !hdr.is_empty() {
        stdout.write_all(&hdr)?;
    }

    let result = XrefPass::new(&binary, config).run(|batch| {
        if limit > 0 && emitted >= limit {
            return ControlFlow::Break(());
        }

        // Compute how many xrefs from this batch we actually need before
        // building context (disasm is expensive — don't render what we'll discard).
        let remaining = if limit > 0 {
            limit - emitted
        } else {
            usize::MAX
        };

        // Process in sub-chunks so output flushes incrementally.
        // Without chunking, the entire shard batch (potentially millions of
        // xrefs) folds into a blob before the first write — causing a hang
        // proportional to limit. CHUNK controls latency vs parallelism tradeoff.
        const CHUNK: usize = 8192;

        let format_chunk = |chunk: &[&xr::xref::Xref]| -> Vec<u8> {
            // Sequential formatting: each record is ~80 bytes of output, far too
            // small to benefit from rayon's fork/join.  The old par_iter path
            // spent ~18% of total runtime in rayon wait_until_cold / cthread_yield
            // contention — more than the formatting itself.
            let mut buf = Vec::with_capacity(chunk.len() * 80);
            for x in chunk {
                let context = if want_context {
                    let lines = xr::disasm::context(
                        binary.arch,
                        &binary.segments,
                        x.from,
                        cli.before,
                        cli.after,
                    );
                    Some(if lines.is_empty() {
                        binary
                            .segments
                            .iter()
                            .find(|s| s.contains(x.from))
                            .map(|seg| {
                                let data = seg.data();
                                let off = (x.from - seg.va) as usize;
                                let len = 8.min(data.len().saturating_sub(off));
                                vec![ContextLine::data(x.from, &data[off..off + len])]
                            })
                            .unwrap_or_default()
                    } else {
                        lines.iter().map(ContextLine::from_disasm).collect()
                    })
                } else {
                    None
                };

                // Extract Rust string if this is a byte-scanned data_ptr into a string blob.
                // Only ByteScan xrefs have `from` pointing at an actual {ptr, len} data
                // slot; instruction-derived DataPointers (ADRP, LEA) have `from` at an
                // instruction VA where `from + ptr_size` reads code bytes, not a length.
                let rust_string = if let Some(ref blobs) = blob_index {
                    if x.kind.scored_kind() == XrefKind::DataPointer
                        && x.confidence == Confidence::ByteScan
                    {
                        blobs.extract_rust_string(&binary, x.from, x.to)
                            .map(|s| truncate_middle(&s, cli.rust_string_max))
                    } else {
                        None
                    }
                } else {
                    None
                };

                let (from_names, to_names): (&[Box<str>], &[Box<str>]) = if use_names {
                    (binary.names.names_at(x.from), binary.names.names_at(x.to))
                } else {
                    (&[], &[])
                };

                let record = XrefRecord {
                    from: x.from,
                    to: x.to,
                    kind: x.kind,
                    confidence: x.confidence,
                    context,
                    rust_string,
                    from_names,
                    to_names,
                };
                printer.write_record(&record, &mut buf);
            }
            buf
        };

        let candidates: Vec<_> = batch
            .iter()
            .filter(|x| kind_filter.is_none_or(|k| x.kind.scored_kind() == k.to_scored_kind()))
            .take(remaining)
            .collect();

        for chunk in candidates.chunks(CHUNK) {
            let blob = format_chunk(chunk);
            if stdout.write_all(&blob).is_err() {
                return ControlFlow::Break(());
            }
            emitted += chunk.len();
        }

        if limit > 0 && emitted >= limit {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });

    let ftr = printer.footer_bytes();
    if !ftr.is_empty() {
        stdout.write_all(&ftr)?;
    }
    stdout.flush()?;

    result.print_summary();

    Ok(())
}
