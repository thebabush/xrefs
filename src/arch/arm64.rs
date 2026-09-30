//! ARM64 xref extraction — three depth levels.
//!
//! Depth 0 (byte scan): handled by arch::byte_scan_pointers — not here.
//!
//! Depth 1 (linear immediate): decode each 4-byte word as an ARM64 instruction,
//! emit xrefs only for instructions with immediate targets (BL, B, Bcc, CBZ, TBZ).
//! Misses ADRP+completing pairs — that's depth 2.
//!
//! Depth 2 (ADRP pairing): sliding window that tracks pending ADRP page values
//! per register. When a completing instruction (ADD/LDR/STR with matching reg)
//! follows, resolve the full address. Also handles BLR/BR where the target
//! register was set by an ADRP+ADD.
//!
//! Thumb / ARM32: implemented in `arch/arm32.rs`.

use super::{ScanRegion, SegmentDataIndex, SegmentIndex};
use crate::arch::arm64_decode::Arm64Insn;
use crate::va::Va;
use crate::xref::{Confidence, Xref, XrefKind};

// How many instructions back we look for an ADRP that feeds the current insn.
// 8 is generous — in practice ADRP+ADD are 1-3 instructions apart.
// Larger window catches more but also more false positives from dead registers.
const ADRP_WINDOW: usize = 8;

/// Maximum entries in a recovered jump table.
const MAX_JUMP_TABLE_ENTRIES: usize = 8192;

/// Maximum left-shift applied to table entry offsets.
///
/// The ADD before BR uses LSL/UXTB/SXTH with a shift amount.  In practice
/// this is 0 (direct byte offset) or 2 (instruction-aligned); values above 4
/// are implausible for jump tables and likely indicate a misidentified pattern.
const MAX_JUMP_TABLE_SHIFT: u32 = 4;

/// How many instructions backward from the BR we scan for the jump-table
/// pattern (ADR+LDRB/H+ADD, ADRP+ADD for table base, CMP for bound).
///
/// The core pattern (ADR+LDRB/H+ADD+BR) is 4 instructions, but the ADRP+ADD
/// that set the table base and the CMP that set the bound can be several
/// instructions earlier.
const JUMP_TABLE_LOOKBACK: usize = 12;

// ── Register newtype ─────────────────────────────────────────────────────────

/// An ARM64 general-purpose register index in 0..=30 (X0–X30).
///
/// Register 31 encodes either SP or ZR depending on context; we never track
/// it, so the constructor rejects it.  Using `Reg` instead of bare `usize`
/// eliminates every scattered `if rd < 31` guard — the check is done once at
/// construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Reg(u8);

impl Reg {
    /// Try to convert a raw 5-bit field (0–31) into a trackable register.
    /// Returns `None` for register 31 (SP/ZR).
    #[inline]
    fn new(raw: u8) -> Option<Self> {
        if raw < 31 {
            Some(Self(raw))
        } else {
            None
        }
    }

    /// Construct from an `Arm64Insn`'s Rd field.
    #[inline]
    fn from_rd(insn: Arm64Insn) -> Option<Self> {
        Self::new(insn.rd())
    }

    /// Construct from an `Arm64Insn`'s Rn field.
    #[inline]
    fn from_rn(insn: Arm64Insn) -> Option<Self> {
        Self::new(insn.rn())
    }

    /// Index into a 32-element register array.
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}

// ── Per-register scan state ──────────────────────────────────────────────────

/// How this register value was resolved — determines what xref kinds it can
/// produce downstream.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RegSource {
    /// Set directly by ADRP/ADR/ADD — the value is a code or data address
    /// that may be called or jumped to.
    Direct,
    /// Set by a LDR pointer-follow (chained resolution).  The value is a
    /// data pointer loaded from memory — NOT callable/jumpable.  BLR/BR
    /// must not resolve through a chain.
    Chain,
}

/// State tracked per register during the ADRP sliding-window scan.
///
/// Populated when an ADRP/ADR/ADD/LDR sets a register to a known address,
/// consumed by completing instructions (ADD, LDR, STR, BLR, BR).
#[derive(Clone, Copy)]
struct AdrpRegState {
    /// Instruction index when this value was set (for window distance check).
    insn_index: usize,
    /// VA of the ADRP (or ADR/LDR for chained entries) — xrefs are emitted
    /// with `from = origin_va` to match IDA's convention.
    origin_va: Va,
    /// The resolved page address (ADRP) or full address (ADR/ADD/LDR chain).
    value: Va,
    /// How this value was resolved — controls what downstream xrefs are valid.
    source: RegSource,
}

/// Exclusive upper bound extracted from `CMP Wn, #imm` (= SUBS WZR, Wn, #imm).
///
/// Stored as `imm + 1` so the value directly represents the number of valid
/// table indices `0..bound`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CmpBound(u32);

impl CmpBound {
    /// Number of table entries this bound allows (clamped to a global max).
    fn table_limit(self) -> usize {
        (self.0 as usize).min(MAX_JUMP_TABLE_ENTRIES)
    }
}

/// Per-register CMP state: the bound and when it was set.
#[derive(Clone, Copy)]
struct CmpRegState {
    insn_index: usize,
    bound: CmpBound,
}

/// Bundles the two per-register state arrays that `scan_adrp` maintains.
///
/// All accesses go through [`Reg`], so the `< 31` bound is enforced once at
/// construction rather than at every use site.
struct ScanState {
    adrp: [Option<AdrpRegState>; 32],
    cmp: [Option<CmpRegState>; 32],
}

impl ScanState {
    fn new() -> Self {
        Self {
            adrp: [None; 32],
            cmp: [None; 32],
        }
    }

    #[inline]
    fn get_adrp(&self, r: Reg) -> Option<AdrpRegState> {
        self.adrp[r.idx()]
    }
    #[inline]
    fn set_adrp(&mut self, r: Reg, st: AdrpRegState) {
        self.adrp[r.idx()] = Some(st);
        self.cmp[r.idx()] = None;
    }
    #[inline]
    fn clear(&mut self, r: Reg) {
        self.adrp[r.idx()] = None;
        self.cmp[r.idx()] = None;
    }
    #[inline]
    fn clear_adrp(&mut self, r: Reg) {
        self.adrp[r.idx()] = None;
    }
    #[inline]
    fn set_cmp(&mut self, r: Reg, insn_index: usize, bound: CmpBound) {
        self.cmp[r.idx()] = Some(CmpRegState { insn_index, bound });
    }
    /// Retrieve the CMP bound for `r`, but only if the CMP was within
    /// `window` instructions of `current_index`.
    #[inline]
    fn get_cmp(&self, r: Reg, current_index: usize, window: usize) -> Option<CmpBound> {
        self.cmp[r.idx()].and_then(|st| {
            if current_index - st.insn_index <= window {
                Some(st.bound)
            } else {
                None
            }
        })
    }
}

// ── Depth-1 scan ─────────────────────────────────────────────────────────────

/// Depth 1: linear scan, immediate targets only.
/// No register tracking. Fast, no false positives on immediate targets.
pub(crate) fn scan_linear(region: &ScanRegion, idx: &SegmentIndex) -> Vec<Xref> {
    let mut xrefs = Vec::new();
    let data = region.data;
    let base = region.base_va;

    // ARM64: all instructions are exactly 4 bytes, 4-byte aligned.
    // If base_va is not 4-byte aligned something is wrong with the loader.
    if !base.raw().is_multiple_of(4) {
        return xrefs;
    }

    let n = data.len() / 4;
    for i in 0..n {
        let offset = i * 4;
        let va = base + offset as u64;
        let word = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .expect("4-byte aligned word"),
        );
        let insn = Arm64Insn::decode(word);
        if let Some(xref) = immediate_xref(insn, va, idx) {
            xrefs.push(xref);
        }
    }
    xrefs
}

// ── Depth-2 scan ─────────────────────────────────────────────────────────────

/// Depth 2: ADRP pairing + all depth-1 refs.
/// Returns combined set — superset of depth 1.
pub(crate) fn scan_adrp(
    region: &ScanRegion,
    idx: &SegmentIndex,
    data_idx: &SegmentDataIndex,
) -> Vec<Xref> {
    let mut xrefs = Vec::new();
    let data = region.data;
    let base = region.base_va;

    if !base.raw().is_multiple_of(4) {
        return xrefs;
    }

    let n = data.len() / 4;
    let mut state = ScanState::new();

    for i in 0..n {
        let offset = i * 4;
        let va = base + offset as u64;
        let word = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .expect("4-byte aligned word"),
        );

        // Fast-path: most instructions are not in the tracked set (BL/B/ADRP/ADD/LDR/STR/…).
        // For those we only need to invalidate their destination register — skip full decode.
        if !Arm64Insn::is_tracked(word) {
            let raw_rd = (word & 0x1F) as u8;
            if let Some(rd) = Reg::new(raw_rd) {
                state.clear(rd);
            }
            // CMP Wn, #imm (= SUBS WZR, Wn, #imm): 0111_0001_00xx_xxxx_xxxx_xxnn_nnn1_1111
            // CMP Xn, #imm (= SUBS XZR, Xn, #imm): 1111_0001_00xx_xxxx_xxxx_xxnn_nnn1_1111
            // Both have Rd=11111 (XZR/WZR), so raw_rd==31 and we skip the invalidation above.
            // Mask 0x7F80_0000 covers bits[30:23]; match 0x7100_0000 requires
            // op=1(SUB), S=1, shift=00 — accepts both sf=0 (W) and sf=1 (X).
            if raw_rd == 31 && word & 0x7F80_0000 == 0x7100_0000 {
                let rn_raw = ((word >> 5) & 0x1F) as u8;
                let imm12 = (word >> 10) & 0xFFF;
                let shift = (word >> 22) & 1;
                if let Some(rn) = Reg::new(rn_raw) {
                    if shift == 0 {
                        state.set_cmp(rn, i, CmpBound(imm12 + 1));
                    }
                }
            }
            continue;
        }

        let insn = Arm64Insn::decode(word);

        // First: try immediate xref (depth 1 — included here too)
        if let Some(xref) = immediate_xref(insn, va, idx) {
            xrefs.push(xref);
        }

        // Second: ADRP tracking
        match insn {
            Arm64Insn::Adrp(_) => {
                if let Some(rd) = Reg::from_rd(insn) {
                    let page = insn.adrp_page(va.raw());
                    state.set_adrp(
                        rd,
                        AdrpRegState {
                            insn_index: i,
                            origin_va: va,
                            value: Va::new(page),
                            source: RegSource::Direct,
                        },
                    );
                }
            }

            Arm64Insn::Adr(_) => {
                // ADR Xn, #label — PC-relative precise address (not page-aligned).
                // No data_ptr is emitted (analysis: 3949 FPs, 0 TPs vs IDA).
                // We DO track the value in state so downstream stores/loads via
                // this register can be resolved (e.g. STLXR [x19] where ADR set x19).
                if let Some(rd) = Reg::from_rd(insn) {
                    let target = insn.adr_target(va.raw());
                    state.set_adrp(
                        rd,
                        AdrpRegState {
                            insn_index: i,
                            origin_va: va,
                            value: Va::new(target),
                            source: RegSource::Direct,
                        },
                    );
                }
            }

            Arm64Insn::AddImm(_) => {
                // ADD Xd, Xn, #imm  — completes ADRP+ADD pairs.
                // Emits data_ptr at ADRP VA only (not ADD VA — ~6981 FPs vs ~454 TPs).
                let rd = Reg::from_rd(insn);
                let rn = Reg::from_rn(insn);
                if let Some(rn) = rn {
                    if let Some(st) = state.get_adrp(rn) {
                        if i - st.insn_index <= ADRP_WINDOW {
                            let target = st.value + insn.add_imm();
                            if idx.contains(target) {
                                xrefs.push(Xref {
                                    from: st.origin_va,
                                    to: target,
                                    kind: XrefKind::DataPointer,
                                    confidence: Confidence::PairResolved,
                                });
                            }
                            if let Some(rd) = rd {
                                state.set_adrp(
                                    rd,
                                    AdrpRegState {
                                        insn_index: i,
                                        origin_va: st.origin_va,
                                        value: target,
                                        source: RegSource::Direct,
                                    },
                                );
                            }
                            if rd != Some(rn) {
                                state.clear_adrp(rn);
                            }
                        }
                    }
                }
            }

            Arm64Insn::Ldr(_) => {
                // LDR Xt/Wt/Ht/Bt, [Xm, #imm]  — completes ADRP+LDR pairs.
                // IMPORTANT: snapshot adrp_state[rn] BEFORE clearing adrp_state[rt],
                // because Rt == Rn is common (e.g. `ADRP X0, page; LDR X0, [X0, #off]`).

                let rt = Reg::from_rd(insn);
                let rn = Reg::from_rn(insn);
                let is_64bit = insn.ldr_str_size() == 3;

                let base_state = rn.and_then(|r| state.get_adrp(r));
                if let Some(rt) = rt {
                    state.clear_adrp(rt);
                }

                if let Some(st) = base_state {
                    if i - st.insn_index <= ADRP_WINDOW {
                        let addr = st.value + insn.ldr_str_offset();
                        let addr_is_exec = idx.is_exec(addr);

                        if idx.contains(addr) {
                            // data_ptr at ADRP VA
                            xrefs.push(Xref {
                                from: st.origin_va,
                                to: addr,
                                kind: XrefKind::DataPointer,
                                confidence: Confidence::PairResolved,
                            });
                            // data_read at LDR VA
                            xrefs.push(Xref {
                                from: va,
                                to: addr,
                                kind: XrefKind::DataRead,
                                confidence: Confidence::PairResolved,
                            });
                        }

                        // Pointer-follow: only for 64-bit loads from non-exec segments.
                        let stored_val: Option<u64> = if is_64bit && !addr_is_exec {
                            data_idx
                                .read_u64_at_nonexec(addr)
                                .filter(|&v| v != 0 && idx.contains(Va::new(v)))
                        } else {
                            None
                        };

                        if let Some(v) = stored_val {
                            xrefs.push(Xref {
                                from: va,
                                to: Va::new(v),
                                kind: XrefKind::DataPointer,
                                confidence: Confidence::PairResolved,
                            });
                            if let Some(rt) = rt {
                                state.set_adrp(
                                    rt,
                                    AdrpRegState {
                                        insn_index: i,
                                        origin_va: va,
                                        value: Va::new(v),
                                        source: RegSource::Chain,
                                    },
                                );
                            }
                        }
                    }
                }
            }

            Arm64Insn::Str(_) => {
                // STR Xn, [Xm, #imm]  — data write via ADRP-resolved address.
                if let Some(rn) = Reg::from_rn(insn) {
                    if let Some(st) = state.get_adrp(rn) {
                        if i - st.insn_index <= ADRP_WINDOW {
                            let addr = st.value + insn.ldr_str_offset();
                            let is_writable_data = idx.flags_at(addr).is_some_and(|f| {
                                f.contains(super::SegFlags::WRITE)
                                    && !f.contains(super::SegFlags::EXEC)
                            });
                            if is_writable_data {
                                xrefs.push(Xref {
                                    from: va,
                                    to: addr,
                                    kind: XrefKind::DataWrite,
                                    confidence: Confidence::PairResolved,
                                });
                            }
                        }
                    }
                }
            }

            Arm64Insn::Blr(_) => {
                // BLR Xn — indirect call via ADRP-resolved address (non-chain only).
                if let Some(rn) = Reg::from_rn(insn) {
                    if let Some(st) = state.get_adrp(rn) {
                        if st.source == RegSource::Direct
                            && i - st.insn_index <= ADRP_WINDOW
                            && idx.is_exec(st.value)
                        {
                            xrefs.push(Xref {
                                from: va,
                                to: st.value,
                                kind: XrefKind::Call,
                                confidence: Confidence::PairResolved,
                            });
                        }
                    }
                }
            }

            Arm64Insn::Br(_) => {
                // BR Xn — indirect jump via ADRP-resolved address (non-chain only).
                if let Some(rn) = Reg::from_rn(insn) {
                    if let Some(st) = state.get_adrp(rn) {
                        if st.source == RegSource::Direct
                            && i - st.insn_index <= ADRP_WINDOW
                            && idx.is_exec(st.value)
                        {
                            xrefs.push(Xref {
                                from: va,
                                to: st.value,
                                kind: XrefKind::Jump,
                                confidence: Confidence::PairResolved,
                            });
                        }
                    }
                }
                // Attempt jump table recovery by scanning backward for the
                // ADR+LDR[BH]+ADD pattern.
                let jt_ctx = JumpTableCtx {
                    data,
                    base,
                    state: &state,
                    data_idx,
                    seg_idx: idx,
                };
                recover_arm64_jump_table(va, i, &jt_ctx, &mut xrefs);
            }

            // Any instruction that writes to a register we're tracking
            // should invalidate that register's ADRP state.
            // We do a conservative invalidation: if the destination register
            // is in our table and this instruction isn't one we already handled,
            // clear it. ARM64 destination is bits[4:0] for most encodings.
            Arm64Insn::LdrLiteral(_)
            | Arm64Insn::Bl(_)
            | Arm64Insn::B(_)
            | Arm64Insn::BCond(_)
            | Arm64Insn::Cbz(_)
            | Arm64Insn::Cbnz(_)
            | Arm64Insn::Tbz(_)
            | Arm64Insn::Tbnz(_)
            | Arm64Insn::Other(_) => {
                if let Some(rd) = Reg::from_rd(insn) {
                    state.clear(rd);
                }
            }
        }
    }

    xrefs
}

// ── Jump table recovery ─────────────────────────────────────────────────────

/// Width of each entry in a recovered jump table.
#[derive(Clone, Copy, PartialEq, Eq)]
enum JumpTableEntrySize {
    /// LDRB — each table entry is a single unsigned byte.
    Byte,
    /// LDRH — each table entry is a little-endian unsigned halfword.
    Halfword,
}

impl JumpTableEntrySize {
    /// Size in bytes of a single table entry.
    fn bytes(self) -> u64 {
        match self {
            Self::Byte => 1,
            Self::Halfword => 2,
        }
    }
}

/// Information extracted from the ADD instruction that combines the loaded
/// table offset with the ADR target base.
#[derive(Clone, Copy)]
struct JumpTableAddInfo {
    /// Left-shift applied to the loaded offset (imm6 or imm3).
    shift: u32,
    /// Whether the offset is sign-extended before shifting.
    ///
    /// Determined by `option[2]` (bit 15) of an ADD (extended register):
    /// set for SXTB/SXTH/SXTW/SXTX, clear for UXTB/UXTH/UXTW/UXTX.
    sign_extend: bool,
    /// Rn of the ADD — the register holding the ADR/target-base value.
    /// Used to verify the ADR's Rd matches.
    base_reg: Reg,
}

impl JumpTableAddInfo {
    /// Decode from the instruction word at the expected ADD position (BR − 1).
    /// Returns `None` if the instruction is not a recognised ADD form, or if
    /// the shift amount exceeds [`MAX_JUMP_TABLE_SHIFT`].
    fn decode(word: u32) -> Option<Self> {
        // ADD (shifted register): 1000_1011_xx0x_xxxx_xxxx_xxxx_xxxx_xxxx
        // bit[21]=0 distinguishes from extended register form.
        if word & 0xFF20_0000 == 0x8B00_0000 {
            let shift = (word >> 10) & 0x3F;
            let base_reg = Reg::new(((word >> 5) & 0x1F) as u8)?;
            return (shift <= MAX_JUMP_TABLE_SHIFT).then_some(Self {
                shift,
                sign_extend: false,
                base_reg,
            });
        }
        // ADD (extended register): 1000_1011_001x_xxxx_xxxx_xxxx_xxxx_xxxx
        // bit[21]=1. option[2] (bit 15) selects signed extension
        // (SXTB/SXTH/SXTW/SXTX) vs unsigned (UXTB/UXTH/UXTW/UXTX).
        if word & 0xFFE0_0000 == 0x8B20_0000 {
            let shift = (word >> 10) & 7;
            let sign_extend = word & (1 << 15) != 0;
            let base_reg = Reg::new(((word >> 5) & 0x1F) as u8)?;
            return (shift <= MAX_JUMP_TABLE_SHIFT).then_some(Self {
                shift,
                sign_extend,
                base_reg,
            });
        }
        // Not a recognised ADD form (shifted or extended register).
        None
    }

    /// Read one table entry and apply sign-extension according to this info.
    ///
    /// `entry_sz` determines the width; the raw value is read from `data_idx`
    /// at `slot_va`.
    fn read_offset(
        self,
        entry_sz: JumpTableEntrySize,
        data_idx: &SegmentDataIndex,
        slot_va: Va,
    ) -> Option<i64> {
        match (entry_sz, self.sign_extend) {
            (JumpTableEntrySize::Byte, false) => data_idx.read_u8_at(slot_va).map(|v| v as i64),
            (JumpTableEntrySize::Byte, true) => {
                data_idx.read_u8_at(slot_va).map(|v| (v as i8) as i64)
            }
            (JumpTableEntrySize::Halfword, false) => {
                data_idx.read_u16_at(slot_va).map(|v| v as i64)
            }
            (JumpTableEntrySize::Halfword, true) => {
                data_idx.read_u16_at(slot_va).map(|v| (v as i16) as i64)
            }
        }
    }
}

/// Results of the backward scan from a BR instruction.
///
/// All non-Option fields must be present for jump table recovery to proceed.
struct JumpTablePattern {
    /// ADR target — the base address to which table offsets are added.
    target_base: Va,
    /// Start of the table data (from ADRP+ADD resolved in the backward scan).
    table_base: Va,
    /// Width of each table entry (byte or halfword).
    entry_size: JumpTableEntrySize,
    /// Register used as the switch index (from the LDR Rm field).
    index_reg: Reg,
    /// Shift / sign-extension from the ADD that combines offset with target base.
    add_info: JumpTableAddInfo,
    /// CMP bound found in the backward scan (may be None if CMP is too far away;
    /// caller falls back to forward-scan `ScanState::get_cmp`).
    cmp_bound: Option<CmpBound>,
}

/// Shared immutable context passed through the jump-table recovery helpers.
///
/// Groups the parameters that every sub-function needs, replacing the
/// long positional parameter lists.
struct JumpTableCtx<'a> {
    data: &'a [u8],
    base: Va,
    state: &'a ScanState,
    data_idx: &'a SegmentDataIndex<'a>,
    seg_idx: &'a SegmentIndex,
}

/// Attempt to recover an ARM64 switch-table at a `BR Xn` instruction.
///
/// The canonical compiler pattern (GCC/Clang) is:
///
/// ```text
///   CMP    wIdx, #bound                    ; range check (sets cmp_bound)
///   B.HI   default                         ; skip if out of range
///   ADRP   Xtbl, table_page                ; table address (via adrp_state)
///   ADD    Xtbl, Xtbl, #off                ; table_base
///   ADR    Xbase, after_br                  ; target base (usually BR+4)
///   LDRB   Woff, [Xtbl, wIdx, UXTW]        ; load u8 offset (or LDRH for u16)
///   ADD    Xtgt, Xbase, Xoff [, SXTB/SXTH] ; compute target
///   BR     Xtgt                             ; indirect jump
/// ```
///
/// We scan backward from the BR to find ADR (target_base), the load
/// instruction (LDRB/LDRH with a register index into a known table address),
/// and the CMP bound.  Table entries are unsigned byte or halfword offsets
/// (sometimes sign-extended in the ADD) that are added to the ADR target.
fn recover_arm64_jump_table(br_va: Va, br_index: usize, ctx: &JumpTableCtx, xrefs: &mut Vec<Xref>) {
    let Some(pattern) = scan_backward_for_pattern(br_index, ctx) else {
        return;
    };

    // Table size from CMP bound.  Prefer the backward-scan CMP (survives
    // register overwrites between CMP and BR) over forward-scan state.
    // The forward-scan fallback applies the same window check as ADRP state
    // to avoid picking up a stale CMP from hundreds of instructions ago.
    let fwd_cmp = ctx
        .state
        .get_cmp(pattern.index_reg, br_index, JUMP_TABLE_LOOKBACK);
    let Some(bound) = pattern.cmp_bound.or(fwd_cmp) else {
        return;
    };
    let limit = bound.table_limit();
    let add_info = pattern.add_info;

    for k in 0..limit {
        let slot_va = pattern.table_base + (k as u64) * pattern.entry_size.bytes();
        let Some(offset_val) = add_info.read_offset(pattern.entry_size, ctx.data_idx, slot_va)
        else {
            break;
        };

        let shifted = offset_val << add_info.shift;
        let target = Va::new((pattern.target_base.raw() as i64).wrapping_add(shifted) as u64);
        if !ctx.seg_idx.is_exec(target) {
            break;
        }
        xrefs.push(Xref {
            from: br_va,
            to: target,
            kind: XrefKind::Jump,
            confidence: Confidence::LocalProp,
        });
    }
}

/// Scan backward from `br_index` looking for the jump-table pattern.
///
/// Resolves the table base address directly from ADRP+ADD instructions in the
/// backward window rather than relying on forward-scan `ScanState`, which may
/// have been overwritten by later instructions (e.g. an ADR that reuses the
/// table base register).  CMP bounds are also extracted here as a fallback
/// when the forward scan clears `cmp_bound` due to register reuse in the LDRH.
///
/// Lookback is [`JUMP_TABLE_LOOKBACK`] instructions: the core pattern
/// (ADR+LDRB/H+ADD+BR) is 4 instructions, but the ADRP+ADD that set the
/// table base and the CMP that set the bound can be several instructions
/// earlier.
fn scan_backward_for_pattern(br_index: usize, ctx: &JumpTableCtx) -> Option<JumpTablePattern> {
    let lookback = br_index.min(JUMP_TABLE_LOOKBACK);

    let mut target_base: Option<Va> = None;
    // ADD (shifted/extended register) that combines the loaded offset with target_base.
    let mut add_info: Option<JumpTableAddInfo> = None;
    // LDRB/LDRH results: table register, entry size, index register.
    let mut table_reg: Option<Reg> = None;
    let mut entry_size: Option<JumpTableEntrySize> = None;
    let mut index_reg: Option<Reg> = None;
    // ADRP+ADD resolution for the table register (self-contained).
    let mut adrp_page: Option<(Reg, u64)> = None; // (rd, page)
    let mut add_imm_val: Option<(Reg, u64)> = None; // (rd, imm) from ADD Xd, Xd, #imm
                                                    // CMP bound found in the backward scan.
    let mut cmp_bound: Option<(Reg, CmpBound)> = None;

    for back in 1..=lookback {
        let j = br_index - back;
        let offset = j * 4;
        if offset + 4 > ctx.data.len() {
            break;
        }
        let w = u32::from_le_bytes(
            ctx.data[offset..offset + 4]
                .try_into()
                .expect("guarded by offset + 4 <= data.len()"),
        );
        let insn_va = ctx.base + offset as u64;

        // ADD (shifted or extended register) — offset-to-target computation.
        // Accept the first (closest to BR) match; there may be intervening
        // instructions (e.g. MOV) between this ADD and the BR.
        if add_info.is_none() {
            add_info = JumpTableAddInfo::decode(w);
        }

        // ADR Xd, label — target base for the jump table.
        // Only accept if Rd matches the ADD's base register (Rn), i.e. the
        // ADR result actually feeds the ADD that computes the jump target.
        // This prevents unrelated ADR instructions in cascaded switches or
        // multi-ADR functions from being misidentified as the target base.
        if (w & 0x9F00_0000) == 0x1000_0000 {
            let adr_rd = Reg::new((w & 0x1F) as u8);
            let feeds_add = add_info.is_some_and(|ai| adr_rd == Some(ai.base_reg));
            if target_base.is_none() && feeds_add {
                let immlo = (w >> 29) & 3;
                let immhi = (w >> 5) & 0x7_FFFF;
                let mut imm = ((immhi << 2) | immlo) as i64;
                if imm & (1 << 20) != 0 {
                    imm -= 1 << 21;
                }
                let adr_target = (insn_va.raw() as i64 + imm) as u64;
                target_base = Some(Va::new(adr_target));
            }
        }

        // LDRB (register): 0011_1000_011m_mmmm_xxxx_xxnn_nnnt_tttt
        // First-wins: in cascaded switches the backward window may span
        // two switch patterns; keep the load closest to the BR.
        if table_reg.is_none() && (w & 0xFFE0_0C00) == 0x3860_0800 {
            let rn = Reg::new(((w >> 5) & 0x1F) as u8);
            let rm = Reg::new(((w >> 16) & 0x1F) as u8);
            if let (Some(rn), Some(rm)) = (rn, rm) {
                table_reg = Some(rn);
                entry_size = Some(JumpTableEntrySize::Byte);
                index_reg = Some(rm);
            }
        }

        // LDRH (register): 0111_1000_011m_mmmm_xxxx_xxnn_nnnt_tttt
        if table_reg.is_none() && (w & 0xFFE0_0C00) == 0x7860_0800 {
            let rn = Reg::new(((w >> 5) & 0x1F) as u8);
            let rm = Reg::new(((w >> 16) & 0x1F) as u8);
            if let (Some(rn), Some(rm)) = (rn, rm) {
                table_reg = Some(rn);
                entry_size = Some(JumpTableEntrySize::Halfword);
                index_reg = Some(rm);
            }
        }

        // ADRP Xd, page — potential table page.
        // First-wins: keep the one closest to the BR.
        if adrp_page.is_none() && (w & 0x9F00_0000) == 0x9000_0000 {
            if let Some(rd) = Reg::new((w & 0x1F) as u8) {
                let immlo = (w >> 29) & 3;
                let immhi = (w >> 5) & 0x7_FFFF;
                let imm21 = (immhi << 2) | immlo;
                let sext = (imm21 as i32) << 11 >> 11;
                let page_pc = insn_va.raw() & !0xFFF;
                let page = page_pc.wrapping_add((sext as i64 * 4096) as u64);
                adrp_page = Some((rd, page));
            }
        }

        // ADD Xd, Xn, #imm (immediate) — potential table offset.
        // Encoding: sf|0|0|100010|sh|imm12|Rn|Rd
        // bits[30:29]=00 distinguishes ADD from SUB (bit30=1) and ADDS (bit29=1).
        // Mask 0x7F00_0000 covers bits[30:24]; match 0x1100_0000 for ADD imm.
        if (w & 0x7F00_0000) == 0x1100_0000 {
            let rd_raw = (w & 0x1F) as u8;
            let rn_raw = ((w >> 5) & 0x1F) as u8;
            // Only accept ADD Xd, Xd, #imm (same register, common in ADRP+ADD).
            // First-wins: keep the one closest to the BR.
            if rd_raw == rn_raw && add_imm_val.is_none() {
                if let Some(rd) = Reg::new(rd_raw) {
                    let imm12 = ((w >> 10) & 0xFFF) as u64;
                    let shift = (w >> 22) & 3;
                    let imm = if shift == 1 { imm12 << 12 } else { imm12 };
                    add_imm_val = Some((rd, imm));
                }
            }
        }

        // CMP Wn, #imm (= SUBS WZR, Wn, #imm) — switch bound
        // Mask 0x7F80_0000 covers bits[30:23]; match 0x7100_0000 for
        // op=1(SUB), S=1, shift=00.  Accepts both sf=0 (W) and sf=1 (X).
        if (w & 0x1F) == 31 && (w & 0x7F80_0000) == 0x7100_0000 {
            let rn_raw = ((w >> 5) & 0x1F) as u8;
            let imm12 = (w >> 10) & 0xFFF;
            let shift = (w >> 22) & 1;
            if shift == 0 {
                if let Some(rn) = Reg::new(rn_raw) {
                    // Keep the CMP closest to the BR (first found going backward).
                    if cmp_bound.is_none() {
                        cmp_bound = Some((rn, CmpBound(imm12 + 1)));
                    }
                }
            }
        }
    }

    // Resolve the table base from ADRP+ADD found in the backward scan.
    let tbl_reg = table_reg?;
    let table_base = resolve_table_base(tbl_reg, adrp_page, add_imm_val, ctx);

    // CMP bound: use backward-scan result if it matches the index register.
    let idx_reg = index_reg?;
    let local_cmp = cmp_bound.and_then(|(r, b)| if r == idx_reg { Some(b) } else { None });

    Some(JumpTablePattern {
        target_base: target_base?,
        table_base: table_base?,
        entry_size: entry_size?,
        index_reg: idx_reg,
        add_info: add_info?,
        cmp_bound: local_cmp,
    })
}

/// Resolve the table base address from the backward-scan ADRP+ADD results,
/// falling back to forward-scan `ScanState` if the backward scan didn't find
/// a matching ADRP (e.g. ADRP was more than [`JUMP_TABLE_LOOKBACK`]
/// instructions before the BR).
fn resolve_table_base(
    tbl_reg: Reg,
    adrp_page: Option<(Reg, u64)>,
    add_imm_val: Option<(Reg, u64)>,
    ctx: &JumpTableCtx,
) -> Option<Va> {
    // Try backward-scan ADRP matching the table register.
    if let Some((adrp_rd, page)) = adrp_page {
        if adrp_rd == tbl_reg {
            // If ADD (immediate) also matches, combine page + offset.
            if let Some((add_rd, imm)) = add_imm_val {
                if add_rd == tbl_reg {
                    return Some(Va::new(page + imm));
                }
            }
            // ADRP only — no ADD offset (or ADD was for a different register).
            return Some(Va::new(page));
        }
    }
    // Fallback: forward-scan state (works when ADRP+ADD wasn't overwritten).
    ctx.state.get_adrp(tbl_reg).map(|st| st.value)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Extract an immediate xref from instructions with direct targets.
#[inline]
fn immediate_xref(insn: Arm64Insn, va: Va, idx: &SegmentIndex) -> Option<Xref> {
    match insn {
        Arm64Insn::Bl(_) => {
            let target = Va::new(insn.imm26_target(va.raw()));
            // IDA records calls only to executable addresses. Suppress BL to non-exec
            // (e.g. BL to .rodata in dead code regions — IDA never records these).
            if idx.contains(target) && idx.is_exec(target) {
                Some(Xref {
                    from: va,
                    to: target,
                    kind: XrefKind::Call,
                    confidence: Confidence::LinearImmediate,
                })
            } else {
                None
            }
        }
        Arm64Insn::B(_) => {
            let target = Va::new(insn.imm26_target(va.raw()));
            if idx.contains(target) {
                Some(Xref {
                    from: va,
                    to: target,
                    kind: XrefKind::Jump,
                    confidence: Confidence::LinearImmediate,
                })
            } else {
                None
            }
        }
        Arm64Insn::BCond(_) => {
            let target = Va::new(insn.imm19_target(va.raw()));
            if idx.contains(target) {
                Some(Xref {
                    from: va,
                    to: target,
                    kind: XrefKind::CondJump,
                    confidence: Confidence::LinearImmediate,
                })
            } else {
                None
            }
        }
        Arm64Insn::Cbz(_) | Arm64Insn::Cbnz(_) => {
            // CBZ/CBNZ Xn, #label — label is second operand
            let target = Va::new(insn.cbz_target(va.raw()));
            if idx.contains(target) {
                Some(Xref {
                    from: va,
                    to: target,
                    kind: XrefKind::CondJump,
                    confidence: Confidence::LinearImmediate,
                })
            } else {
                None
            }
        }
        Arm64Insn::Tbz(_) | Arm64Insn::Tbnz(_) => {
            // TBZ/TBNZ Xn, #bit, #label — label is third operand
            let target = Va::new(insn.imm14_target(va.raw()));
            if idx.contains(target) {
                Some(Xref {
                    from: va,
                    to: target,
                    kind: XrefKind::CondJump,
                    confidence: Confidence::LinearImmediate,
                })
            } else {
                None
            }
        }
        Arm64Insn::LdrLiteral(_) => {
            // LDR Xt/Wt, label — PC-relative literal load.
            // IDA records data_read at this VA to the literal pool address.
            let target = Va::new(insn.ldr_literal_target(va.raw()));
            if idx.contains(target) {
                Some(Xref {
                    from: va,
                    to: target,
                    kind: XrefKind::DataRead,
                    confidence: Confidence::LinearImmediate,
                })
            } else {
                None
            }
        }
        Arm64Insn::Adr(_) => {
            // ADR Xn, #label — PC-relative, computes address (not page-aligned).
            //
            // IDA does NOT record ADR as data_ptr. Analysis of 3949 ADR-sourced
            // data_ptr emissions confirmed zero overlap with IDA ground truth —
            // IDA has no xrefs at any of those source addresses. Suppressing ADR
            // eliminates 3949 FPs with no TP loss.
            //
            // We do still track the value in ADRP state so downstream instructions
            // (e.g. STLXR w10, w30, [x19] where x19 was set by ADR) can resolve.
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{ScanRegion, SegmentDataIndex};
    use crate::loader::{SegData, Segment, SegmentArch};
    use crate::xref::{Confidence, XrefKind};

    /// Build a fake executable segment covering exactly the given bytes at `base_va`.
    fn fake_seg(base_va: u64, data: &'static [u8]) -> Segment {
        Segment {
            va: Va::new(base_va),
            data: unsafe { SegData::new_for_test(data) },
            executable: true,
            readable: true,
            writable: false,
            byte_scannable: true,
            arch: SegmentArch::Generic,
            name: "test".to_string(),
        }
    }

    /// Build a non-executable data segment at `base_va`.
    fn fake_data_seg(base_va: u64, data: &'static [u8]) -> Segment {
        Segment {
            va: Va::new(base_va),
            data: unsafe { SegData::new_for_test(data) },
            executable: false,
            readable: true,
            writable: false,
            byte_scannable: false,
            arch: SegmentArch::Generic,
            name: "rodata".to_string(),
        }
    }

    fn region_for<'a>(seg: &'a Segment) -> ScanRegion<'a> {
        ScanRegion::new(seg, seg.va, seg.va + seg.data().len() as u64)
    }

    // ── BL ────────────────────────────────────────────────────────────────────

    /// BL #0x1010 from PC 0x1000 (imm26=+4 words)
    /// Encoding: 0x94000004
    #[test]
    fn test_bl_call() {
        // Code lives at 0x1000 (4 bytes). Target 0x1010 must be in a segment too.
        // We fake two segments: code at 0x1000 and a target page at 0x1010.
        static CODE: [u8; 4] = [0x04, 0x00, 0x00, 0x94]; // BL +16
        static TARGET: [u8; 4] = [0x00, 0x00, 0x00, 0x00];
        let code_seg = fake_seg(0x1000, &CODE);
        let tgt_seg = fake_seg(0x1010, &TARGET);
        let segs = vec![fake_seg(0x1000, &CODE), tgt_seg];

        let region = region_for(&code_seg);
        let idx = SegmentIndex::build(&segs);
        let xrefs = scan_linear(&region, &idx);

        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1000));
        assert_eq!(xrefs[0].to, Va::new(0x1010));
        assert_eq!(xrefs[0].kind, XrefKind::Call);
        assert_eq!(xrefs[0].confidence, Confidence::LinearImmediate);
    }

    // ── B (unconditional jump) ────────────────────────────────────────────────

    /// B +8 from 0x1008 (imm26=+2 words) → target 0x1010
    /// Encoding: 0x14000002
    #[test]
    fn test_b_jump() {
        static CODE: [u8; 4] = [0x02, 0x00, 0x00, 0x14];
        static TARGET: [u8; 4] = [0x00, 0x00, 0x00, 0x00];
        let code_seg = fake_seg(0x1008, &CODE);
        let tgt_seg = fake_seg(0x1010, &TARGET);
        let segs = vec![fake_seg(0x1008, &CODE), tgt_seg];

        let idx = SegmentIndex::build(&segs);
        let xrefs = scan_linear(&region_for(&code_seg), &idx);
        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1008));
        assert_eq!(xrefs[0].to, Va::new(0x1010));
        assert_eq!(xrefs[0].kind, XrefKind::Jump);
    }

    // ── CBZ ───────────────────────────────────────────────────────────────────

    /// CBZ X1, -4 (1 insn back = 0x1004) from PC 0x1008.
    /// Encoding: 0xb4ffffe1
    #[test]
    fn test_cbz_cond_jump() {
        static CODE: [u8; 8] = [
            0x00, 0x00, 0x00, 0x00, // padding at 0x1004 (NOP-like, just zeroes)
            0xe1, 0xff, 0xff, 0xb4, // CBZ X1, -4  at 0x1008
        ];
        let seg = fake_seg(0x1004, &CODE);
        let segs = vec![fake_seg(0x1004, &CODE)];

        let idx = SegmentIndex::build(&segs);
        let xrefs = scan_linear(&region_for(&seg), &idx);
        // CBZ at 0x1008 → target 0x1004
        let cbz_xref = xrefs.iter().find(|x| x.from == Va::new(0x1008)).unwrap();
        assert_eq!(cbz_xref.to, Va::new(0x1004));
        assert_eq!(cbz_xref.kind, XrefKind::CondJump);
    }

    // ── ADRP + ADD pair ───────────────────────────────────────────────────────

    /// ADRP X0, #1 (page = 0x2000) at 0x1000
    /// ADD  X0, X0, #0x100        at 0x1004
    /// → PairResolved DataRead from 0x1000 to 0x2100
    #[test]
    fn test_adrp_add_pair() {
        static CODE: [u8; 8] = [
            0x00, 0x00, 0x00, 0xb0, // ADRP X0, #1   (page = base+0x1000 = 0x2000)
            0x00, 0x00, 0x04, 0x91, // ADD X0, X0, #0x100
        ];
        static TARGET_PAGE: [u8; 0x200] = [0u8; 0x200];
        let code_seg = fake_seg(0x1000, &CODE);
        let data_seg = fake_seg(0x2000, &TARGET_PAGE);
        let segs = vec![fake_seg(0x1000, &CODE), data_seg];

        let idx = SegmentIndex::build(&segs);
        let didx = SegmentDataIndex::build(&segs);
        let xrefs = scan_adrp(&region_for(&code_seg), &idx, &didx);
        let pair = xrefs
            .iter()
            .find(|x| x.confidence == Confidence::PairResolved)
            .expect("expected a PairResolved xref");
        assert_eq!(pair.from, Va::new(0x1000)); // ADRP va (IDA records at ADRP, not ADD)
        assert_eq!(pair.to, Va::new(0x2100)); // 0x2000 + 0x100
        assert_eq!(pair.kind, XrefKind::DataPointer);
    }

    // ── ADRP + LDR with Rt == Rn (regression test) ──────────────────────────

    /// ADRP X0, page → LDR X0, [X0, #8]  (Rt == Rn == 0)
    /// This is the most common ADRP+LDR pattern. The LDR handler must snapshot
    /// the base register's ADRP state BEFORE clearing the destination register,
    /// or the pair is silently dropped (regression: data_read F1 0.906 → 0.042).
    #[test]
    fn test_adrp_ldr_same_register() {
        // ADRP X0, #1  → page = 0x2000 (from pc 0x1000)
        // LDR  X0, [X0, #8]  → addr = 0x2000 + 8 = 0x2008
        //
        // ADRP X0, #1: encoded as 0xB000_0000 (immlo=1 → +1 page, Rd=0)
        // LDR  X0, [X0, #8]: unsigned offset, imm12 = 8/8 = 1
        //   word = 0xF940_0400 | Rn=0<<5 | Rt=0 = 0xF940_0400
        static CODE: [u8; 8] = [
            0x00, 0x00, 0x00, 0xb0, // ADRP X0, #1 page
            0x00, 0x04, 0x40, 0xf9, // LDR X0, [X0, #8]
        ];
        // Data segment at 0x2000 so target 0x2008 is valid.
        static DATA: [u8; 0x100] = [0u8; 0x100];
        let code_seg = fake_seg(0x1000, &CODE);
        let segs = vec![
            fake_seg(0x1000, &CODE),
            Segment {
                va: Va::new(0x2000),
                data: unsafe { SegData::new_for_test(&DATA) },
                executable: false,
                readable: true,
                writable: false,
                byte_scannable: false,
                arch: SegmentArch::Generic,
                name: "data".to_string(),
            },
        ];

        let idx = SegmentIndex::build(&segs);
        let didx = SegmentDataIndex::build(&segs);
        let xrefs = scan_adrp(&region_for(&code_seg), &idx, &didx);

        // Must emit data_read at LDR VA (0x1004) → 0x2008
        let dr = xrefs.iter().find(|x| x.kind == XrefKind::DataRead).expect(
            "ADRP+LDR with Rt==Rn must emit data_read (regression: state cleared before read)",
        );
        assert_eq!(dr.from, Va::new(0x1004));
        assert_eq!(dr.to, Va::new(0x2008));

        // Must also emit data_ptr at ADRP VA (0x1000) → 0x2008
        let dp = xrefs
            .iter()
            .find(|x| x.kind == XrefKind::DataPointer && x.from == Va::new(0x1000))
            .expect("ADRP+LDR must emit data_ptr at ADRP VA");
        assert_eq!(dp.to, Va::new(0x2008));
    }

    // ── Target outside all segments → no xref ────────────────────────────────

    /// BL to an address not in any segment should be suppressed.
    #[test]
    fn test_bl_out_of_range_filtered() {
        static CODE: [u8; 4] = [0x04, 0x00, 0x00, 0x94]; // BL +16 → 0x1010
        let code_seg = fake_seg(0x1000, &CODE);
        // No segment covers 0x1010
        let segs = vec![fake_seg(0x1000, &CODE)];

        let idx = SegmentIndex::build(&segs);
        let xrefs = scan_linear(&region_for(&code_seg), &idx);
        assert!(
            xrefs.is_empty(),
            "xref to unmapped target should be suppressed"
        );
    }

    // ── Jump table recovery ──────────────────────────────────────────────────

    /// Canonical byte-entry jump table:
    ///   CMP W2, #3 / B.HI / ADRP X1 / ADD X1,X1,#0x100 /
    ///   ADR X3, after_br / LDRB W4,[X1,W2,UXTW] / ADD X3,X3,X4,UXTB#2 / BR X3
    /// Table at 0x20100 with byte offsets [0, 1, 2, 3].
    /// target_base = 0x10020 (ADR target = BR + 4), shift = 2.
    /// Expected targets: 0x10020, 0x10024, 0x10028, 0x1002C.
    #[test]
    fn test_jump_table_byte_entries() {
        #[rustfmt::skip]
        static CODE: [u8; 48] = [
            0x5F, 0x0C, 0x00, 0x71, // 0x10000: CMP W2, #3
            0x08, 0x01, 0x00, 0x54, // 0x10004: B.HI +32
            0x81, 0x00, 0x00, 0x90, // 0x10008: ADRP X1, +16 pages (→ 0x20000)
            0x21, 0x00, 0x04, 0x91, // 0x1000C: ADD  X1, X1, #0x100
            0x83, 0x00, 0x00, 0x10, // 0x10010: ADR  X3, +0x10 (→ 0x10020)
            0x24, 0x48, 0x62, 0x38, // 0x10014: LDRB W4, [X1, W2, UXTW]
            0x63, 0x08, 0x24, 0x8B, // 0x10018: ADD  X3, X3, X4, UXTB #2
            0x60, 0x00, 0x1F, 0xD6, // 0x1001C: BR   X3
            0x1F, 0x20, 0x03, 0xD5, // 0x10020: NOP (target 0)
            0x1F, 0x20, 0x03, 0xD5, // 0x10024: NOP (target 1)
            0x1F, 0x20, 0x03, 0xD5, // 0x10028: NOP (target 2)
            0x1F, 0x20, 0x03, 0xD5, // 0x1002C: NOP (target 3)
        ];
        // Data segment: 4-byte table at offset 0x100.
        static TABLE_DATA: [u8; 0x200] = {
            let mut d = [0u8; 0x200];
            d[0x100] = 0; // entry 0: offset = 0 → 0 << 2 = 0
            d[0x101] = 1; // entry 1: offset = 1 → 1 << 2 = 4
            d[0x102] = 2; // entry 2: offset = 2 → 2 << 2 = 8
            d[0x103] = 3; // entry 3: offset = 3 → 3 << 2 = 12
            d
        };

        let code_seg = fake_seg(0x10000, &CODE);
        let segs = vec![
            fake_seg(0x10000, &CODE),
            fake_data_seg(0x20000, &TABLE_DATA),
        ];

        let idx = SegmentIndex::build(&segs);
        let didx = SegmentDataIndex::build(&segs);
        let xrefs = scan_adrp(&region_for(&code_seg), &idx, &didx);

        let jt_xrefs: Vec<&Xref> = xrefs
            .iter()
            .filter(|x| x.kind == XrefKind::Jump && x.from == Va::new(0x1001C))
            .collect();
        assert_eq!(
            jt_xrefs.len(),
            4,
            "expected 4 jump table targets, got {jt_xrefs:?}"
        );
        let targets: Vec<Va> = jt_xrefs.iter().map(|x| x.to).collect();
        assert!(targets.contains(&Va::new(0x10020)));
        assert!(targets.contains(&Va::new(0x10024)));
        assert!(targets.contains(&Va::new(0x10028)));
        assert!(targets.contains(&Va::new(0x1002C)));
        // All should have LocalProp confidence.
        assert!(jt_xrefs
            .iter()
            .all(|x| x.confidence == Confidence::LocalProp));
    }

    /// Jump table recovery should bail when no CMP bound is present.
    #[test]
    fn test_jump_table_no_cmp_bound_bails() {
        // Same pattern as above but without the CMP instruction.
        // Replace CMP with NOP; recovery should produce zero jump xrefs.
        #[rustfmt::skip]
        static CODE: [u8; 48] = [
            0x1F, 0x20, 0x03, 0xD5, // 0x10000: NOP (was CMP)
            0x08, 0x01, 0x00, 0x54, // 0x10004: B.HI +32
            0x81, 0x00, 0x00, 0x90, // 0x10008: ADRP X1, +16 pages
            0x21, 0x00, 0x04, 0x91, // 0x1000C: ADD  X1, X1, #0x100
            0x83, 0x00, 0x00, 0x10, // 0x10010: ADR  X3, +0x10
            0x24, 0x48, 0x62, 0x38, // 0x10014: LDRB W4, [X1, W2, UXTW]
            0x63, 0x08, 0x24, 0x8B, // 0x10018: ADD  X3, X3, X4, UXTB #2
            0x60, 0x00, 0x1F, 0xD6, // 0x1001C: BR   X3
            0x1F, 0x20, 0x03, 0xD5, // 0x10020: NOP
            0x1F, 0x20, 0x03, 0xD5, // 0x10024: NOP
            0x1F, 0x20, 0x03, 0xD5, // 0x10028: NOP
            0x1F, 0x20, 0x03, 0xD5, // 0x1002C: NOP
        ];
        static TABLE_DATA: [u8; 0x200] = {
            let mut d = [0u8; 0x200];
            d[0x100] = 0;
            d[0x101] = 1;
            d[0x102] = 2;
            d[0x103] = 3;
            d
        };

        let code_seg = fake_seg(0x10000, &CODE);
        let segs = vec![
            fake_seg(0x10000, &CODE),
            fake_data_seg(0x20000, &TABLE_DATA),
        ];

        let idx = SegmentIndex::build(&segs);
        let didx = SegmentDataIndex::build(&segs);
        let xrefs = scan_adrp(&region_for(&code_seg), &idx, &didx);

        let jt_xrefs: Vec<&Xref> = xrefs
            .iter()
            .filter(|x| x.kind == XrefKind::Jump && x.from == Va::new(0x1001C))
            .collect();
        assert!(
            jt_xrefs.is_empty(),
            "no jump table xrefs without CMP bound, got {jt_xrefs:?}"
        );
    }
}
