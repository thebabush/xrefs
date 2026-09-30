use crate::arch::arm32;
use crate::arch::arm64;
use crate::arch::x86_64;
use crate::arch::{self, ScanRegion, SegmentDataIndex, SegmentIndex};
use crate::loader::{Arch, DecodeMode, LoadedBinary, Segment, SegmentArch};
use crate::shard::split_range;
use crate::va::{Va, VaRange};
use crate::xref::{Confidence, Xref};
use rayon::prelude::*;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

// Capacity of the bounded channel between the drain relay thread and the output
// thread.  Large enough that the output thread is rarely starved (scan batches
// arrive in bursts), but small enough that we don't buffer all 171M xrefs in
// memory if output falls behind.

/// Default shard boundary overlap in bytes.
///
/// 64 bytes = 16 ARM64 instructions of lookahead.  This ensures instruction
/// pairs (e.g. ADRP+LDR) straddling a shard boundary are still matched.
const DEFAULT_BOUNDARY_OVERLAP: u64 = 64;

/// Which analysis depth to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
pub enum Depth {
    /// Depth 0: byte scan of data sections only (pointer-sized values).
    #[value(alias = "scan")]
    ByteScan = 0,
    /// Depth 1: linear disasm, immediate targets + RIP-relative / direct branches.
    Linear = 1,
    /// Depth 2: ADRP pairing (ARM64) or register prop (x86-64).
    Paired = 2,
}

pub struct PassConfig {
    pub depth: Depth,
    /// Number of parallel workers. 0 = use all logical CPUs.
    pub workers: usize,
    /// Bytes of lookahead overlap per shard boundary (for cross-shard pairs).
    pub boundary_overlap: u64,
    /// Drop any xref whose `to` address is below this threshold.
    /// `None` = no filtering (default).
    /// Set to `Some(binary.min_va())` for PIE ELF to eliminate tiny-immediate FPs
    /// (e.g. `CMP rax, 8` generating a spurious data_ptr xref to VA 0x8).
    ///
    /// Note: for non-PIE binaries `min_va()` returns the lowest segment VA
    /// (which may be 0x0 for firmware or flat images), causing this filter to
    /// become a no-op — no xrefs are incorrectly dropped in that case.
    pub min_ref_va: Option<Va>,
    /// Restrict scanning to xrefs whose `from` address falls in `[start, end)`.
    /// Applied at shard-generation time — segments outside the range are skipped
    /// entirely, so no decode work is wasted.
    /// `None` = no restriction (scan everything).
    pub from_range: Option<VaRange>,
    /// Retain only xrefs whose `to` address falls in `[start, end)`.
    /// Applied as a post-filter after scanning.
    /// `None` = no restriction.
    pub to_range: Option<VaRange>,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            depth: Depth::Paired,
            workers: 0,
            boundary_overlap: DEFAULT_BOUNDARY_OVERLAP,
            min_ref_va: None,
            from_range: None,
            to_range: None,
        }
    }
}

/// Breakdown of emitted xrefs by confidence level.
///
/// Backed by an array indexed by [`Confidence`] discriminant (`repr(u8)`),
/// so new variants are handled automatically without updating a match arm.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceCounts {
    counts: [usize; Confidence::COUNT],
}

impl Default for ConfidenceCounts {
    fn default() -> Self {
        Self {
            counts: [0; Confidence::COUNT],
        }
    }
}

impl ConfidenceCounts {
    fn add(&mut self, c: Confidence) {
        self.counts[c as usize] += 1;
    }

    /// Get the count for a specific confidence level.
    pub fn get(&self, c: Confidence) -> usize {
        self.counts[c as usize]
    }
}

pub struct PassResult {
    pub elapsed_ms: u64,
    pub segments_scanned: usize,
    pub bytes_scanned: u64,
    pub xref_count: usize,
    /// How many xrefs were emitted at each confidence level.
    pub confidence_counts: ConfidenceCounts,
}

impl PassResult {
    pub fn print_summary(&self) {
        let cc = &self.confidence_counts;
        eprintln!(
            "xrefs: {}  |  {:.1}s  |  {:.1} MB scanned  |  {} segments",
            self.xref_count,
            self.elapsed_ms as f64 / 1000.0,
            self.bytes_scanned as f64 / 1_048_576.0,
            self.segments_scanned,
        );
        eprintln!(
            "  byte-scan={} linear={} pair-resolved={} local-prop={} fn-flow={}",
            cc.get(Confidence::ByteScan),
            cc.get(Confidence::LinearImmediate),
            cc.get(Confidence::PairResolved),
            cc.get(Confidence::LocalProp),
            cc.get(Confidence::FunctionFlow),
        );
    }
}

/// Run the xref pass over a loaded binary.
pub struct XrefPass<'a> {
    binary: &'a LoadedBinary,
    config: PassConfig,
}

impl<'a> XrefPass<'a> {
    pub fn new(binary: &'a LoadedBinary, config: PassConfig) -> Self {
        Self { binary, config }
    }

    /// Run the xref pass, calling `on_batch` for each completed shard's xrefs.
    /// Batches are deduplicated (boundary overlap trimmed by ownership) and
    /// filtered by `min_ref_va` before being passed to the callback.
    /// `on_batch` is called from the main thread, serially, so it can do I/O
    /// without any locking.
    ///
    /// `on_batch` returns `ControlFlow::Continue(())` to keep going or
    /// `ControlFlow::Break(())` to stop early.  When it breaks, in-flight
    /// shards finish but no new shards start, so the scan halts promptly
    /// (within one shard's worth of work).
    #[must_use = "PassResult contains timing, byte counts, and xref statistics; \
                  bind the return value or discard explicitly with `let _ =`"]
    pub fn run<F>(self, mut on_batch: F) -> PassResult
    where
        F: FnMut(&[Xref]) -> ControlFlow<()> + Send,
    {
        let t0 = Instant::now();

        let n_workers = if self.config.workers == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            self.config.workers
        };

        let pool = match rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: failed to build rayon thread pool: {e}");
                return PassResult {
                    elapsed_ms: t0.elapsed().as_millis() as u64,
                    segments_scanned: 0,
                    bytes_scanned: 0,
                    xref_count: 0,
                    confidence_counts: ConfidenceCounts::default(),
                };
            }
        };

        let all_segs = &self.binary.segments;
        let arch = self.binary.arch;
        let depth = self.config.depth;
        let overlap = self.config.boundary_overlap;
        let min_ref_va = self.config.min_ref_va;
        let from_range = self.config.from_range;
        let to_range = self.config.to_range;

        // Build the segment indices once — both shared across all shards via reference.
        // SegmentIndex: flags-only O(log n) membership/exec checks.
        // SegmentDataIndex: also carries data slices for O(log n) pointer reads
        //   (used in byte_scan_pointers and ADRP/LDR pointer-follow in scan_adrp).
        let seg_idx = SegmentIndex::build(all_segs);
        let data_idx = SegmentDataIndex::build(all_segs);

        // Instruction alignment for shard boundary snapping.
        // ARM64: 4-byte fixed. x86: 1 (variable length).
        let insn_align: u64 = match arch {
            Arch::Arm64 | Arch::Arm32 => 4,
            _ => 1,
        };

        let ptr_size: usize = arch.pointer_size().unwrap_or(4);

        // Channel: workers send completed shard batches; main thread drains.
        let (tx, rx) = mpsc::channel::<Vec<Xref>>();

        // Build the full shard list upfront so we know which is the last shard
        // per segment (last shard owns its overlap region too).
        // For each shard, compute (seg, scan_start, scan_end, owned_end).
        // owned_end = next shard's start (or seg_end for the last shard).
        // scan_end includes the overlap lookahead; owned_end does not.
        // Using next shard's start directly handles overlap >= chunk correctly.
        let code_shards: Vec<CodeShard<'_>> = if depth == Depth::ByteScan {
            // ByteScan is data-only — no instruction decoding.
            vec![]
        } else {
            self.binary
                .code_segments()
                .flat_map(|seg| {
                    let seg_end = seg.va + seg.data().len() as u64;
                    // Clamp to from_range before sharding — segments with no overlap
                    // produce an empty shard list and are skipped entirely.
                    let (scan_start, scan_end) = match from_range {
                        Some(r) => (seg.va.max(r.start), seg_end.min(r.end)),
                        None => (seg.va, seg_end),
                    };
                    if scan_start >= scan_end {
                        return vec![];
                    }
                    let shards = split_range(
                        scan_start.raw(),
                        scan_end.raw(),
                        n_workers,
                        overlap,
                        insn_align,
                    );
                    let n = shards.len();
                    let starts: Vec<Va> = shards.iter().map(|(s, _)| Va::new(*s)).collect();
                    shards
                        .into_iter()
                        .enumerate()
                        .map(move |(i, (start, end))| {
                            let owned_end = if i + 1 < n { starts[i + 1] } else { scan_end };
                            CodeShard {
                                seg,
                                scan_start: Va::new(start),
                                scan_end: Va::new(end),
                                owned_end,
                            }
                        })
                        .collect()
                })
                .collect()
        };

        let data_shards: Vec<DataShard<'_>> = if depth >= Depth::ByteScan {
            self.binary
                .scannable_data_segments()
                .flat_map(|seg| {
                    let seg_end = seg.va + seg.data().len() as u64;
                    // Clamp to from_range (data pointers are emitted at their source VA).
                    let (scan_start, scan_end) = match from_range {
                        Some(r) => (seg.va.max(r.start), seg_end.min(r.end)),
                        None => (seg.va, seg_end),
                    };
                    if scan_start >= scan_end {
                        return vec![];
                    }
                    let shards = split_range(
                        scan_start.raw(),
                        scan_end.raw(),
                        n_workers,
                        0,
                        // Align shard boundaries to pointer size so that
                        // byte_scan_pointers (which steps by ptr_size from
                        // offset 0 within each shard) always lands on
                        // ptr_size-aligned absolute addresses. Without this,
                        // non-aligned shard starts cause misaligned scans
                        // that systematically skip aligned pointer slots.
                        ptr_size as u64,
                    );
                    shards
                        .into_iter()
                        .map(move |(start, end)| DataShard {
                            seg,
                            scan_start: Va::new(start),
                            scan_end: Va::new(end),
                        })
                        .collect()
                })
                .collect()
        } else {
            vec![]
        };

        let ctx = ScanCtx {
            seg_idx: &seg_idx,
            data_idx: &data_idx,
            got_slots: &self.binary.got_slots,
            pie_base: self.binary.pie_base,
        };

        // Count distinct segments that will be scanned — computed here, before the
        // shards are consumed by into_par_iter().  Uses segment VA as the identity
        // key (each segment has a unique start VA).
        let segments_scanned = {
            let mut vas: Vec<Va> = code_shards
                .iter()
                .map(|s| s.seg.va)
                .chain(data_shards.iter().map(|s| s.seg.va))
                .collect();
            vas.sort_unstable();
            vas.dedup();
            vas.len()
        };

        // Cancellation flag: set by the output thread when on_batch returns Break.
        // Workers check this before starting each shard — no new shards start once
        // set, but in-flight shards finish normally.
        // Wrapped in Arc so it can be shared between the output thread, the drain
        // relay thread, and the rayon workers.
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_workers = Arc::clone(&cancelled);

        // Run workers and drain concurrently inside a thread scope so the drain
        // thread can borrow non-'static locals (on_batch, stop) while the workers
        // run in the rayon pool. The scope blocks until both finish.
        // Second channel: drain relay → output thread.
        // Bounded to n_workers * 4: enough runway for the output thread to stay
        // busy across a burst of completed shards without buffering all results.
        let (out_tx, out_rx) = mpsc::sync_channel::<Vec<Xref>>(n_workers * 4);

        let (xref_count, confidence_counts) = std::thread::scope(|s| {
            // Output thread: owns on_batch — formats and writes results.
            // Runs concurrently with the scan workers on a dedicated OS thread.
            // on_batch formats sequentially (no par_iter) so there is no need to
            // route through pool.install — this eliminates contention between
            // scan workers and the output path that previously caused ~18% of
            // runtime to be spent in rayon wait_until_cold / cthread_yield.
            // Returns (xref_count, confidence_counts) after all batches processed.
            let output = s.spawn(|| {
                let mut xref_count = 0usize;
                let mut confidence_counts = ConfidenceCounts::default();
                for batch in out_rx {
                    for x in &batch {
                        confidence_counts.add(x.confidence);
                        xref_count += 1;
                    }
                    if on_batch(&batch).is_break() {
                        // Signal workers and drain to stop, then exit.  Dropping
                        // `out_rx` (on closure return) closes the output channel so
                        // any further `out_tx.send` calls in the drain thread return
                        // `Err` and are silently discarded.  This keeps xref_count
                        // accurate: only xrefs emitted before the Break are counted.
                        cancelled.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                (xref_count, confidence_counts)
            });

            // Drain thread: pure relay — receives scan batches and forwards to
            // the output thread without blocking on formatting or I/O.
            // This keeps the scan channel drained so workers never stall.
            let drain = s.spawn(|| {
                for batch in rx {
                    // Fast path: if the output thread already broke and closed
                    // `out_rx`, skip the send entirely rather than paying the
                    // cost of a failed channel operation.
                    if cancelled_workers.load(Ordering::Relaxed) {
                        continue;
                    }
                    // If the output thread exited early, `out_tx.send` returns
                    // `Err`; the `let _ =` discards it so the drain loop
                    // continues until `rx` is exhausted.
                    let _ = out_tx.send(batch);
                }
                // Drop out_tx so the output thread's channel closes (if it
                // hasn't already exited via break).
                drop(out_tx);
            });

            pool.install(|| {
                let tx = tx.clone();

                // Relocation-derived data pointers — emit as a single batch.
                // These come from ELF .rela.dyn / .rel.dyn (R_*_RELATIVE, R_*_64)
                // and PE .pdata (exception directory) entries.
                //
                // These are authoritative metadata, so they bypass min_ref_va
                // (which exists to suppress byte-scan noise in low VA ranges).
                // PE .pdata entries reference image_base which is below the
                // first section — min_ref_va would wrongly filter those out.
                if !self.binary.reloc_pointers.is_empty() {
                    let batch: Vec<Xref> = self
                        .binary
                        .reloc_pointers
                        .iter()
                        .filter(|rp| {
                            from_range.is_none_or(|r| r.contains(rp.from))
                                && to_range.is_none_or(|r| r.contains(rp.to))
                        })
                        .map(|rp| Xref {
                            from: rp.from,
                            to: rp.to,
                            kind: crate::xref::XrefKind::DataPointer,
                            confidence: Confidence::ByteScan,
                        })
                        .collect();
                    if !batch.is_empty() {
                        let _ = tx.send(batch);
                    }
                }

                // Code shards — owned_end is the next shard's start (or seg_end).
                // Xrefs with from >= owned_end are in the overlap lookahead and
                // will be emitted by the next shard instead.
                code_shards
                    .into_par_iter()
                    .for_each_with(tx.clone(), |tx, shard| {
                        if cancelled_workers.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut batch = scan_shard(
                            shard.seg,
                            shard.scan_start,
                            shard.scan_end,
                            arch,
                            depth,
                            &ctx,
                        );
                        batch.retain(|x| {
                            x.from < shard.owned_end
                                && min_ref_va.is_none_or(|m| x.to >= m)
                                && to_range.is_none_or(|r| r.contains(x.to))
                        });
                        if !batch.is_empty() {
                            let _ = tx.send(batch);
                        }
                    });

                // Data shards (no overlap, no ownership trimming needed)
                data_shards.into_par_iter().for_each_with(tx, |tx, shard| {
                    if cancelled_workers.load(Ordering::Relaxed) {
                        return;
                    }
                    let region = ScanRegion::new(shard.seg, shard.scan_start, shard.scan_end);
                    let mut batch = arch::byte_scan_pointers(&region, ctx.data_idx, ptr_size);
                    batch.retain(|x| {
                        min_ref_va.is_none_or(|m| x.to >= m)
                            && to_range.is_none_or(|r| r.contains(x.to))
                    });
                    if !batch.is_empty() {
                        let _ = tx.send(batch);
                    }
                });
            });

            // Drop scan tx so scan channel closes → drain thread exits → drops
            // out_tx → output channel closes → output thread exits.
            drop(tx);
            drain.join().expect("drain thread panicked");
            output.join().expect("output thread panicked")
        });

        // Bytes scanned: the portion of each segment that actually fell within
        // `from_range` (or the full segment when no range was active).
        let code_bytes: u64 = if depth == Depth::ByteScan {
            0
        } else {
            self.binary
                .code_segments()
                .map(|s| seg_bytes_in_range(s, from_range))
                .sum()
        };
        let data_bytes: u64 = self
            .binary
            .scannable_data_segments()
            .map(|s| seg_bytes_in_range(s, from_range))
            .sum();
        let bytes_scanned = code_bytes + data_bytes;
        // `segments_scanned` was computed above, before the shards were consumed.

        PassResult {
            elapsed_ms: t0.elapsed().as_millis() as u64,
            segments_scanned,
            bytes_scanned,
            xref_count,
            confidence_counts,
        }
    }
}

/// A code shard to be scanned by a worker thread.
///
/// `scan_end` includes the overlap lookahead; `owned_end` does not.
/// Xrefs with `from >= owned_end` are in the overlap and will be emitted by
/// the next shard instead.
struct CodeShard<'a> {
    seg: &'a Segment,
    scan_start: Va,
    scan_end: Va,
    /// First address NOT owned by this shard (next shard's start, or seg_end).
    owned_end: Va,
}

/// A data shard to be byte-scanned for pointer values.
struct DataShard<'a> {
    seg: &'a Segment,
    scan_start: Va,
    scan_end: Va,
}

/// Shared read-only context threaded into every shard scanner.
/// Bundles the two indices so `scan_shard` stays under the clippy
/// `too_many_arguments` limit (7).
struct ScanCtx<'a> {
    seg_idx: &'a SegmentIndex,
    data_idx: &'a SegmentDataIndex<'a>,
    /// Known GOT / IAT slot VAs (from `LoadedBinary::got_slots`).
    ///
    /// Used by the x86-64 scanner to restrict `CALL [RIP+disp32]` /
    /// `JMP [RIP+disp32]` xref emission to actual import-table slots.
    /// Populated for ELF (GLOB_DAT / JUMP_SLOT) and PE (IAT); empty for Mach-O.
    got_slots: &'a rustc_hash::FxHashSet<Va>,
    /// PIE rebase offset applied to this binary's segments.
    /// Zero for non-PIE binaries and position-dependent executables.
    /// Forwarded to architecture scanners that need it (e.g. ARM32 Thumb
    /// LDR-literal pointer chaining).
    pie_base: u64,
}

/// Bytes of `seg` that fall within `from_range`, or the full segment length
/// when no range is active.  Used to compute accurate `bytes_scanned` totals
/// when `--start`/`--end` restrict the scan to a sub-range of the binary.
#[inline]
fn seg_bytes_in_range(seg: &Segment, from_range: Option<VaRange>) -> u64 {
    let seg_end = seg.va + seg.data().len() as u64;
    match from_range {
        None => seg.data().len() as u64,
        Some(r) => {
            let start = seg.va.max(r.start);
            let end = Va::new(seg_end.raw().min(r.end.raw()));
            if start >= end {
                0
            } else {
                end - start
            }
        }
    }
}

fn scan_shard(
    seg: &Segment,
    start_va: Va,
    end_va: Va,
    arch: Arch,
    depth: Depth,
    ctx: &ScanCtx<'_>,
) -> Vec<Xref> {
    match (arch, depth) {
        (Arch::Arm64, Depth::Linear) => {
            let region = ScanRegion::new(seg, start_va, end_va);
            arm64::scan_linear(&region, ctx.seg_idx)
        }
        (Arch::Arm64, Depth::Paired) => {
            let region = ScanRegion::new(seg, start_va, end_va);
            arm64::scan_adrp(&region, ctx.seg_idx, ctx.data_idx)
        }
        (Arch::Arm32, Depth::Linear | Depth::Paired) => {
            // Delegate to the mode-switch-aware helper which splits the shard
            // at every ModeSwitch boundary, dispatching the correct ISA scanner
            // (A32 or Thumb-2) for each sub-range.
            scan_arm32_shard(seg, start_va, end_va, ctx)
        }
        (Arch::X86_64, Depth::Linear) => {
            let region = ScanRegion::new(seg, start_va, end_va);
            x86_64::scan_linear(&region, ctx.seg_idx, ctx.got_slots, ctx.data_idx)
        }
        (Arch::X86_64, Depth::Paired) => {
            let region = ScanRegion::new(seg, start_va, end_va);
            x86_64::scan_with_prop(&region, ctx.seg_idx, ctx.got_slots, ctx.data_idx)
        }
        // ByteScan never generates code shards — the shard vec is empty for
        // this depth, so these arms are unreachable in production.
        // Listing them explicitly (rather than using `_`) ensures the compiler
        // flags unhandled combinations when new Depth variants are added.
        (Arch::Arm64 | Arch::Arm32 | Arch::X86_64, Depth::ByteScan) => vec![],
        // x86 (32-bit) and Unknown: no instruction scanner implemented.
        (Arch::X86 | Arch::Unknown, Depth::Linear | Depth::Paired | Depth::ByteScan) => vec![],
    }
}

/// Scan an ARM32 shard with full mode-switch awareness.
///
/// Splits `[start_va, end_va)` at every [`ModeSwitch`] boundary and dispatches
/// [`arm32::scan_arm32`] or [`arm32::scan_thumb`] for each sub-range.  This
/// ensures that a shard containing both A32 and Thumb-2 code — e.g. a section
/// with a Thumb stub following an ARM32 function — is decoded correctly
/// throughout instead of being run through a single mismatched scanner.
///
/// When no switches fall inside the shard the common fast path is taken: one
/// scanner call over the full range.
fn scan_arm32_shard(seg: &Segment, start_va: Va, end_va: Va, ctx: &ScanCtx<'_>) -> Vec<Xref> {
    let arm32_seg = match &seg.arch {
        SegmentArch::Arm32(a) => a,
        // This branch is unreachable in production: scan_arm32_shard is only called
        // from scan_shard when arch == Arch::Arm32, and every Arm32 loader produces
        // SegmentArch::Arm32 segments.  Guard with debug_assert so misuse surfaces
        // immediately in tests without silently applying the wrong decoder.
        _ => {
            debug_assert!(
                false,
                "scan_arm32_shard: segment '{}' at {:#x} has {:?}, expected Arm32; \
                 skipping (no xrefs produced)",
                seg.name, seg.va, seg.arch,
            );
            return vec![];
        }
    };

    let initial_mode = arm32_seg.mode_at(start_va);

    // Switches strictly inside the shard: start_va < sw.va < end_va.
    // `mode_at(start_va)` already accounts for any switch *at* start_va.
    let lo = arm32_seg.switches.partition_point(|s| s.va <= start_va);
    let hi = arm32_seg.switches.partition_point(|s| s.va < end_va);
    let inner = &arm32_seg.switches[lo..hi];

    if inner.is_empty() {
        // Fast path: uniform ISA throughout this shard.
        let region = ScanRegion::new(seg, start_va, end_va);
        return match initial_mode {
            DecodeMode::Thumb => arm32::scan_thumb(&region, ctx.seg_idx, ctx.pie_base),
            DecodeMode::Arm32 => arm32::scan_arm32(&region, ctx.seg_idx),
        };
    }

    // General path: iterate over sub-ranges separated by ISA transitions.
    let mut xrefs = Vec::new();
    let mut cur_va = start_va;
    let mut cur_mode = initial_mode;

    for sw in inner {
        if sw.va > cur_va {
            let region = ScanRegion::new(seg, cur_va, sw.va);
            let batch = match cur_mode {
                DecodeMode::Thumb => arm32::scan_thumb(&region, ctx.seg_idx, ctx.pie_base),
                DecodeMode::Arm32 => arm32::scan_arm32(&region, ctx.seg_idx),
            };
            xrefs.extend(batch);
        }
        cur_va = sw.va;
        cur_mode = sw.mode;
    }

    // Final sub-range — extends to end_va (which includes the lookahead overlap).
    if cur_va < end_va {
        let region = ScanRegion::new(seg, cur_va, end_va);
        let batch = match cur_mode {
            DecodeMode::Thumb => arm32::scan_thumb(&region, ctx.seg_idx, ctx.pie_base),
            DecodeMode::Arm32 => arm32::scan_arm32(&region, ctx.seg_idx),
        };
        xrefs.extend(batch);
    }

    xrefs
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(unused_must_use)] // Tests discard PassResult — only the collected xrefs matter.
mod tests {
    use super::*;
    use crate::loader::{Arch, SegData, Segment};
    use crate::xref::XrefKind;
    use ahash::AHashSet;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Encode an ARM64 BL instruction: BL <target>.
    /// `pc` is the address of the instruction, `target` is the call target.
    /// Panics if target is out of the ±128 MiB BL range.
    fn arm64_bl(pc: u64, target: u64) -> u32 {
        let offset = (target as i64) - (pc as i64);
        assert!(offset % 4 == 0, "BL target must be 4-byte aligned");
        let imm26 = (offset >> 2) as i32;
        assert!(
            (-(1 << 25)..(1 << 25)).contains(&imm26),
            "BL offset out of range"
        );
        0x94000000u32 | (imm26 as u32 & 0x03ff_ffff)
    }

    /// Encode an ARM64 NOP.
    fn arm64_nop() -> u32 {
        0xd503201f
    }

    /// Build a synthetic code segment backed by a leaked Vec (so it's 'static).
    fn make_code_seg(va: u64, words: Vec<u32>) -> Segment {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let data: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        Segment {
            va: Va::new(va),
            // Safety: data is leaked, truly 'static.
            data: unsafe { SegData::new_for_test(data) },
            executable: true,
            readable: true,
            writable: false,
            byte_scannable: false,
            arch: SegmentArch::Generic,
            name: "test_code".to_string(),
        }
    }

    /// Build a synthetic data segment (pointer table) backed by leaked Vec.
    fn make_data_seg(va: u64, pointers: Vec<u64>) -> Segment {
        let bytes: Vec<u8> = pointers.iter().flat_map(|p| p.to_le_bytes()).collect();
        let data: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        Segment {
            va: Va::new(va),
            // Safety: data is leaked, truly 'static.
            data: unsafe { SegData::new_for_test(data) },
            executable: false,
            readable: true,
            writable: true,
            byte_scannable: true,
            arch: SegmentArch::Generic,
            name: "test_data".to_string(),
        }
    }

    /// Build a minimal LoadedBinary from a list of segments.
    fn make_binary(arch: Arch, segments: Vec<Segment>) -> LoadedBinary {
        LoadedBinary::from_segments(arch, segments)
    }

    /// Collect all xrefs from a streaming pass into a Vec.
    fn collect_xrefs(binary: &LoadedBinary, depth: Depth, workers: usize) -> Vec<Xref> {
        let mut all = Vec::new();
        XrefPass::new(
            binary,
            PassConfig {
                depth,
                workers,
                ..Default::default()
            },
        )
        .run(|batch| {
            all.extend_from_slice(batch);
            ControlFlow::Continue(())
        });
        all
    }

    /// Assert no duplicate (from, to, kind) triples exist in the xref list.
    fn assert_no_duplicates(xrefs: &[Xref]) {
        let mut seen: AHashSet<(Va, Va, u8)> = AHashSet::new();
        for x in xrefs {
            let key = (x.from, x.to, x.kind as u8);
            assert!(
                seen.insert(key),
                "duplicate xref: from={:#x} to={:#x} kind={:?}",
                x.from,
                x.to,
                x.kind
            );
        }
    }

    // ── Uniqueness tests ──────────────────────────────────────────────────────

    /// Basic: a handful of BL instructions, single worker.
    /// Baseline — no shard boundary interaction at all.
    #[test]
    fn test_no_duplicates_single_worker() {
        // Code at 0x1000: 8 BL instructions targeting 0x10000.
        let target_va = 0x10000u64;
        let base_va = 0x1000u64;
        let n = 8usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let xrefs = collect_xrefs(&binary, Depth::Linear, 1);
        assert_no_duplicates(&xrefs);
        assert_eq!(xrefs.len(), n, "expected {n} xrefs, got {}", xrefs.len());
    }

    /// Multiple workers on a segment just large enough to be sharded.
    /// The overlap window is 64 bytes = 16 ARM64 instructions.
    /// We place BL instructions in and around the shard boundary.
    #[test]
    fn test_no_duplicates_multi_worker_boundary() {
        // 512 instructions = 2048 bytes. With 4 workers each shard is ~512 bytes.
        // Boundary at ~0x200, overlap=64 bytes = 16 instructions before that.
        let base_va = 0x4000u64;
        let target_va = 0x100000u64;
        let n = 512usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        // Run with several worker counts to stress different shard splits.
        for workers in [2, 4, 8, 16] {
            let xrefs = collect_xrefs(&binary, Depth::Linear, workers);
            assert_no_duplicates(&xrefs);
            assert_eq!(
                xrefs.len(),
                n,
                "workers={workers}: expected {n} xrefs, got {}",
                xrefs.len()
            );
        }
    }

    /// Adversarial: BL instruction placed exactly at the overlap boundary VA.
    /// It must appear exactly once — owned by exactly one shard.
    #[test]
    fn test_no_duplicates_xref_at_exact_boundary() {
        // With overlap=64 and 2 workers over 256 instructions (1024 bytes):
        // shard 0: [base, base+512+64) — owns [base, base+512)
        // shard 1: [base+512, base+1024) — owns [base+512, base+1024)
        // Place a BL exactly at base+512 (the boundary) — must appear once.
        let base_va = 0x8000u64;
        let target_va = 0x200000u64;
        let overlap: u64 = 64; // matches PassConfig default
        let shard_boundary = base_va + 512; // 128 instructions in

        let n = 256usize;
        let words: Vec<u32> = (0..n)
            .map(|i| {
                let va = base_va + i as u64 * 4;
                if va == shard_boundary {
                    arm64_bl(va, target_va)
                } else {
                    arm64_nop()
                }
            })
            .collect();

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let config = PassConfig {
            depth: Depth::Linear,
            workers: 2,
            boundary_overlap: overlap,
            ..Default::default()
        };
        let mut xrefs = Vec::new();
        XrefPass::new(&binary, config).run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });

        assert_no_duplicates(&xrefs);
        assert_eq!(
            xrefs.len(),
            1,
            "expected exactly 1 xref at boundary, got {}",
            xrefs.len()
        );
        assert_eq!(xrefs[0].from, Va::new(shard_boundary));
    }

    /// Adversarial: xref at `owned_end - 4` (last owned instruction).
    /// Must be emitted, not suppressed.
    #[test]
    fn test_xref_at_last_owned_instruction_emitted() {
        let base_va = 0xc000u64;
        let target_va = 0x300000u64;
        let overlap: u64 = 64;
        // 2 workers, 256 instructions: shard 0 owns [base, base+512)
        // Last owned instruction of shard 0: base + 512 - 4 = base + 508
        let last_owned = base_va + 512 - 4;

        let n = 256usize;
        let words: Vec<u32> = (0..n)
            .map(|i| {
                let va = base_va + i as u64 * 4;
                if va == last_owned {
                    arm64_bl(va, target_va)
                } else {
                    arm64_nop()
                }
            })
            .collect();

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let config = PassConfig {
            depth: Depth::Linear,
            workers: 2,
            boundary_overlap: overlap,
            ..Default::default()
        };
        let mut xrefs = Vec::new();
        XrefPass::new(&binary, config).run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });

        assert_no_duplicates(&xrefs);
        assert_eq!(xrefs.len(), 1, "last owned instruction must be emitted");
        assert_eq!(xrefs[0].from, Va::new(last_owned));
    }

    /// Single-shard segment: segment smaller than the overlap window.
    /// The one shard is also the last — must emit all instructions including
    /// those in the overlap tail (there is no next shard to own them).
    #[test]
    fn test_single_shard_emits_all() {
        // 4 instructions = 16 bytes, well below the 64-byte overlap window.
        // With 1 worker this is definitively a single shard.
        let base_va = 0x10000u64;
        let target_va = 0x400000u64;
        let n = 4usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let xrefs = collect_xrefs(&binary, Depth::Linear, 1);
        assert_no_duplicates(&xrefs);
        assert_eq!(xrefs.len(), n, "single shard must emit all {n} xrefs");
    }

    /// Segment smaller than overlap with multiple workers: split_range will
    /// collapse to fewer shards than workers due to alignment snapping.
    /// Whatever shards are produced, no duplicates and no xrefs lost.
    #[test]
    fn test_tiny_segment_many_workers_no_duplicates() {
        let base_va = 0x20000u64;
        let target_va = 0x500000u64;
        // 16 instructions = 64 bytes, exactly equal to the default overlap.
        let n = 16usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        for workers in [1, 4, 8, 32] {
            let xrefs = collect_xrefs(&binary, Depth::Linear, workers);
            assert_no_duplicates(&xrefs);
            assert_eq!(
                xrefs.len(),
                n,
                "workers={workers}: tiny segment must emit all {n} xrefs"
            );
        }
    }

    /// Data segment pointer scan: zero overlap, no ownership trimming.
    /// Every pointer must appear exactly once regardless of worker count.
    #[test]
    fn test_no_duplicates_data_segment_pointers() {
        // Code segment must be above 0x0100_0000 — byte_scan_pointers filters
        // exec-segment targets below that threshold as ASCII-string false positives.
        let code_va = 0x0200_0000u64;
        let data_va = 0x0400_0000u64;
        let n = 128usize;

        // All pointers target the code segment — all should be emitted.
        let pointers: Vec<u64> = (0..n).map(|i| code_va + i as u64 * 4).collect();
        let code_words: Vec<u32> = (0..n).map(|_| arm64_nop()).collect();

        let code_seg = make_code_seg(code_va, code_words);
        let data_seg = make_data_seg(data_va, pointers);
        let binary = make_binary(Arch::Arm64, vec![code_seg, data_seg]);

        for workers in [1, 2, 4, 8] {
            let xrefs = collect_xrefs(&binary, Depth::ByteScan, workers);
            assert_no_duplicates(&xrefs);
            // Every pointer should land exactly once
            let data_ptr_count = xrefs
                .iter()
                .filter(|x| x.kind == XrefKind::DataPointer)
                .count();
            assert_eq!(
                data_ptr_count, n,
                "workers={workers}: expected {n} DataPointer xrefs, got {data_ptr_count}"
            );
        }
    }

    /// Stress: large segment, many workers, dense BL instructions everywhere.
    /// No duplicate should survive regardless of how shards are cut.
    #[test]
    fn test_no_duplicates_stress_large_segment() {
        let base_va = 0x40000u64;
        let target_va = 0x1000000u64;
        // 4096 instructions = 16 KiB
        let n = 4096usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        for workers in [1, 3, 7, 13, 32] {
            let xrefs = collect_xrefs(&binary, Depth::Linear, workers);
            assert_no_duplicates(&xrefs);
            assert_eq!(
                xrefs.len(),
                n,
                "workers={workers}: expected {n}, got {}",
                xrefs.len()
            );
        }
    }

    /// Coverage: xref exactly at the first instruction of the segment must appear.
    #[test]
    fn test_xref_at_segment_start_emitted() {
        let base_va = 0x50000u64;
        let target_va = 0x2000000u64;
        let n = 64usize;
        let mut words: Vec<u32> = vec![arm64_nop(); n];
        words[0] = arm64_bl(base_va, target_va); // first instruction

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        for workers in [1, 2, 4, 8] {
            let xrefs = collect_xrefs(&binary, Depth::Linear, workers);
            assert_no_duplicates(&xrefs);
            assert_eq!(
                xrefs.len(),
                1,
                "workers={workers}: segment-start xref missing"
            );
            assert_eq!(xrefs[0].from, Va::new(base_va));
        }
    }

    /// Coverage: xref at the very last instruction of the segment must appear.
    #[test]
    fn test_xref_at_segment_end_emitted() {
        let base_va = 0x60000u64;
        let target_va = 0x3000000u64;
        let n = 64usize;
        let mut words: Vec<u32> = vec![arm64_nop(); n];
        words[n - 1] = arm64_bl(base_va + (n as u64 - 1) * 4, target_va);

        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        for workers in [1, 2, 4, 8] {
            let xrefs = collect_xrefs(&binary, Depth::Linear, workers);
            assert_no_duplicates(&xrefs);
            assert_eq!(
                xrefs.len(),
                1,
                "workers={workers}: segment-end xref missing"
            );
        }
    }

    // ── from_range (--start / --end) ──────────────────────────────────────────

    /// from_range exactly covering the full segment: all xrefs emitted.
    #[test]
    fn test_from_range_full_coverage() {
        let base_va = 0x1000u64;
        let target_va = 0x100000u64;
        let n = 16usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();
        let seg_end = base_va + n as u64 * 4;
        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let mut xrefs = Vec::new();
        XrefPass::new(
            &binary,
            PassConfig {
                depth: Depth::Linear,
                workers: 1,
                from_range: Some(VaRange::new(Va::new(base_va), Va::new(seg_end))),
                ..Default::default()
            },
        )
        .run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });
        assert_eq!(xrefs.len(), n, "full from_range should emit all {n} xrefs");
    }

    /// from_range covering only the first half: only the first-half xrefs emitted.
    #[test]
    fn test_from_range_first_half() {
        let base_va = 0x2000u64;
        let target_va = 0x200000u64;
        let n = 32usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();
        let midpoint = base_va + (n / 2) as u64 * 4;
        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let mut xrefs = Vec::new();
        XrefPass::new(
            &binary,
            PassConfig {
                depth: Depth::Linear,
                workers: 1,
                from_range: Some(VaRange::new(Va::new(base_va), Va::new(midpoint))),
                ..Default::default()
            },
        )
        .run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });
        assert_eq!(xrefs.len(), n / 2, "first-half from_range");
        assert!(
            xrefs.iter().all(|x| x.from < Va::new(midpoint)),
            "all froms below midpoint"
        );
    }

    /// from_range covering only the second half.
    #[test]
    fn test_from_range_second_half() {
        let base_va = 0x3000u64;
        let target_va = 0x300000u64;
        let n = 32usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();
        let midpoint = base_va + (n / 2) as u64 * 4;
        let seg_end = base_va + n as u64 * 4;
        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let mut xrefs = Vec::new();
        XrefPass::new(
            &binary,
            PassConfig {
                depth: Depth::Linear,
                workers: 1,
                from_range: Some(VaRange::new(Va::new(midpoint), Va::new(seg_end))),
                ..Default::default()
            },
        )
        .run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });
        assert_eq!(xrefs.len(), n / 2, "second-half from_range");
        assert!(
            xrefs.iter().all(|x| x.from >= Va::new(midpoint)),
            "all froms at or above midpoint"
        );
    }

    /// from_range entirely outside the segment: zero xrefs emitted.
    #[test]
    fn test_from_range_no_overlap() {
        let base_va = 0x4000u64;
        let target_va = 0x400000u64;
        let n = 16usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();
        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let mut xrefs = Vec::new();
        XrefPass::new(
            &binary,
            PassConfig {
                depth: Depth::Linear,
                workers: 1,
                from_range: Some(VaRange::new(Va::new(0xffff_0000), Va::new(0xffff_1000))), // nowhere near base_va
                ..Default::default()
            },
        )
        .run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });
        assert!(
            xrefs.is_empty(),
            "from_range outside segment must yield no xrefs"
        );
    }

    /// from_range with multiple workers: range filtering + no duplicates.
    #[test]
    fn test_from_range_multi_worker() {
        let base_va = 0x5000u64;
        let target_va = 0x500000u64;
        let n = 64usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();
        // Scan only the middle quarter.
        let q1 = base_va + (n / 4) as u64 * 4;
        let q3 = base_va + (3 * n / 4) as u64 * 4;
        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        for workers in [2, 4, 8] {
            let mut xrefs = Vec::new();
            XrefPass::new(
                &binary,
                PassConfig {
                    depth: Depth::Linear,
                    workers,
                    from_range: Some(VaRange::new(Va::new(q1), Va::new(q3))),
                    ..Default::default()
                },
            )
            .run(|batch| {
                xrefs.extend_from_slice(batch);
                ControlFlow::Continue(())
            });
            assert_no_duplicates(&xrefs);
            assert_eq!(
                xrefs.len(),
                n / 2,
                "workers={workers}: middle-half from_range"
            );
            assert!(xrefs
                .iter()
                .all(|x| x.from >= Va::new(q1) && x.from < Va::new(q3)));
        }
    }

    // ── to_range (--ref-start / --ref-end) ────────────────────────────────────

    /// to_range exactly matching the target: all xrefs pass through.
    #[test]
    fn test_to_range_full_pass() {
        let base_va = 0x6000u64;
        let target_va = 0x600000u64;
        let n = 16usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();
        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let mut xrefs = Vec::new();
        XrefPass::new(
            &binary,
            PassConfig {
                depth: Depth::Linear,
                workers: 1,
                to_range: Some(VaRange::new(Va::new(target_va), Va::new(target_va + 4))),
                ..Default::default()
            },
        )
        .run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });
        assert_eq!(
            xrefs.len(),
            n,
            "to_range covering target should pass all xrefs"
        );
    }

    /// to_range that excludes the target: zero xrefs pass through.
    #[test]
    fn test_to_range_excludes_all() {
        let base_va = 0x7000u64;
        let target_va = 0x700000u64;
        let n = 16usize;
        let words: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_va + i as u64 * 4, target_va))
            .collect();
        let code_seg = make_code_seg(base_va, words);
        let tgt_seg = make_code_seg(target_va, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![code_seg, tgt_seg]);

        let mut xrefs = Vec::new();
        XrefPass::new(
            &binary,
            PassConfig {
                depth: Depth::Linear,
                workers: 1,
                to_range: Some(VaRange::new(Va::new(0x1000_0000), Va::new(0x2000_0000))), // far from target_va
                ..Default::default()
            },
        )
        .run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });
        assert!(
            xrefs.is_empty(),
            "to_range excluding target should yield no xrefs"
        );
    }

    /// from_range and to_range combined: intersection of both filters.
    #[test]
    fn test_from_range_and_to_range_combined() {
        // Two code segments, each with BL instructions pointing to different targets.
        let base_a = 0x8000u64;
        let base_b = 0x9000u64;
        let target_a = 0x800000u64;
        let target_b = 0x900000u64;
        let n = 8usize;

        let words_a: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_a + i as u64 * 4, target_a))
            .collect();
        let words_b: Vec<u32> = (0..n)
            .map(|i| arm64_bl(base_b + i as u64 * 4, target_b))
            .collect();

        let seg_a = make_code_seg(base_a, words_a);
        let seg_b = make_code_seg(base_b, words_b);
        let tgt_a = make_code_seg(target_a, vec![arm64_nop()]);
        let tgt_b = make_code_seg(target_b, vec![arm64_nop()]);
        let binary = make_binary(Arch::Arm64, vec![seg_a, seg_b, tgt_a, tgt_b]);

        // Restrict from to seg_a only, and to to target_a only.
        let mut xrefs = Vec::new();
        XrefPass::new(
            &binary,
            PassConfig {
                depth: Depth::Linear,
                workers: 1,
                from_range: Some(VaRange::new(
                    Va::new(base_a),
                    Va::new(base_a + n as u64 * 4),
                )),
                to_range: Some(VaRange::new(Va::new(target_a), Va::new(target_a + 4))),
                ..Default::default()
            },
        )
        .run(|batch| {
            xrefs.extend_from_slice(batch);
            ControlFlow::Continue(())
        });
        assert_eq!(xrefs.len(), n, "combined filter: only seg_a→target_a xrefs");
        assert!(xrefs
            .iter()
            .all(|x| x.from >= Va::new(base_a) && x.from < Va::new(base_a + n as u64 * 4)));
        assert!(xrefs.iter().all(|x| x.to == Va::new(target_a)));
    }

    /// Multiple segments: uniqueness must hold across all segments combined.
    #[test]
    fn test_no_duplicates_multiple_segments() {
        let target_va = 0x5000000u64;
        let mut segs = vec![make_code_seg(target_va, vec![arm64_nop()])];

        // Three separate code segments, each with dense BL instructions.
        for s in 0..3u64 {
            let base = 0x100000u64 + s * 0x10000;
            let n = 256usize;
            let words: Vec<u32> = (0..n)
                .map(|i| arm64_bl(base + i as u64 * 4, target_va))
                .collect();
            segs.push(make_code_seg(base, words));
        }

        let binary = make_binary(Arch::Arm64, segs);
        for workers in [1, 4, 8] {
            let xrefs = collect_xrefs(&binary, Depth::Linear, workers);
            assert_no_duplicates(&xrefs);
            // 3 segments × 256 BL instructions = 768 unique (from, to, kind) triples
            assert_eq!(xrefs.len(), 768, "workers={workers}: got {}", xrefs.len());
        }
    }
}
