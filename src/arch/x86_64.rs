//! x86-64 xref extraction — two depth levels.
//!
//! Depth 1 (linear immediate): decode each instruction, emit xrefs for
//! direct CALL/JMP/Jcc with immediate targets, and RIP-relative LEA/MOV/etc.
//! RIP-relative is x86-64's equivalent of ARM64's ADRP — almost all data
//! references use `[rip + disp32]` and are resolved directly during decode
//! since we know the instruction address.
//!
//! Depth 2 (local prop): track register assignments with simple constant
//! values and resolve indirect CALL/JMP where the target is a known constant.
//! This catches `mov rax, imm64; call rax` patterns.

use super::{ScanRegion, SegmentDataIndex, SegmentIndex};
use crate::loader::GotSlots;
use crate::va::Va;
use crate::xref::{Confidence, Xref, XrefKind};
use iced_x86::{
    Code, Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, OpAccess,
    OpKind, Register,
};

/// Tracks a register participating in a jump table dispatch.
///
/// Populated by `update_jt_state` when recognising the MOVSXD+ADD pattern,
/// consumed by `recover_jump_table` on the JMP instruction.
#[derive(Clone, Copy)]
struct JumpTableInfo {
    /// VA of the first table entry (base_reg_val + displacement).
    table_start: Va,
    /// Value added to each i32 offset to compute the target (the LEA result).
    target_base: Va,
    /// Upper bound from CMP+JA (None = unknown, use fallback).
    max_entries: Option<u32>,
}
/// Returns true if the memory operand at `op_idx` in `insn` is written to (not just read).
/// Uses InstructionInfoFactory for accurate semantics (handles CMP, TEST, ADD [mem], etc.).
#[inline]
fn mem_op_is_write(
    insn: &Instruction,
    op_idx: u32,
    info_factory: &mut InstructionInfoFactory,
) -> bool {
    let info = info_factory.info(insn);
    let access = info.op_access(op_idx);
    matches!(
        access,
        OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

/// Whether the decode loop should track register state for resolving
/// indirect calls/jumps and jump tables.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PropMode {
    /// Depth 1: direct branches, RIP-relative, GOT-indirect, imm-as-address only.
    Off,
    /// Depth 2: adds MOV/LEA register propagation, indirect CALL/JMP
    /// resolution, and CMP+MOVSXD+ADD+JMP jump table recovery.
    On,
}

/// Depth 1: linear disassembly — direct branches, RIP-relative, and
/// immediate-as-address only. No register propagation, no jump tables.
pub(crate) fn scan_linear(
    region: &ScanRegion,
    idx: &SegmentIndex,
    got: GotSlots<'_>,
    data_idx: &SegmentDataIndex,
) -> Vec<Xref> {
    scan_core(region, idx, got, data_idx, PropMode::Off)
}

/// Depth 2: linear disassembly + register constant propagation + jump table
/// recovery. Catches `mov rax, imm64 / call rax` and CMP+LEA+MOVSXD+ADD+JMP.
pub(crate) fn scan_with_prop(
    region: &ScanRegion,
    idx: &SegmentIndex,
    got: GotSlots<'_>,
    data_idx: &SegmentDataIndex,
) -> Vec<Xref> {
    scan_core(region, idx, got, data_idx, PropMode::On)
}

/// Shared decode loop for both depth levels.
fn scan_core(
    region: &ScanRegion,
    idx: &SegmentIndex,
    got: GotSlots<'_>,
    data_idx: &SegmentDataIndex,
    prop: PropMode,
) -> Vec<Xref> {
    let mut xrefs = Vec::new();
    let data = region.data;
    let base = region.base_va.raw();

    let region_is_exec = idx.is_exec(region.base_va);

    // Register propagation state — only allocated/updated when `propagate` is true.
    let mut reg_vals: [Option<u64>; 16] = [None; 16];
    let mut jt_info: [Option<JumpTableInfo>; 16] = [None; 16];
    let mut cmp_bound: [Option<u32>; 16] = [None; 16];

    let mut decoder = Decoder::with_ip(64, data, base, DecoderOptions::NONE);
    let mut insn = Instruction::default();
    let mut info_factory = InstructionInfoFactory::new();

    while decoder.can_decode() {
        decoder.decode_out(&mut insn);
        if insn.is_invalid() {
            continue;
        }

        let va = insn.ip();

        // Direct branch/call xrefs — only from executable regions.
        if region_is_exec {
            emit_direct_branches(&insn, va, idx, got, &mut xrefs);
        }

        emit_rip_relative(&insn, va, idx, &mut info_factory, &mut xrefs);

        // Propagation-only: resolve indirect call/jmp from tracked register state.
        if prop == PropMode::On {
            match insn.code() {
                Code::Call_rm64 | Code::Jmp_rm64 if insn.op0_kind() == OpKind::Register => {
                    let reg = insn.op0_register();
                    if let Some(ri) = gpr_index(reg) {
                        if let Some(known) = reg_vals[ri] {
                            if idx.contains(Va::new(known)) {
                                let kind = if insn.code() == Code::Call_rm64 {
                                    XrefKind::Call
                                } else {
                                    XrefKind::Jump
                                };
                                xrefs.push(Xref {
                                    from: Va::new(va),
                                    to: Va::new(known),
                                    kind,
                                    confidence: Confidence::LocalProp,
                                });
                            }
                        } else if insn.code() == Code::Jmp_rm64 {
                            if let Some(jt) = jt_info[ri] {
                                recover_jump_table(Va::new(va), jt, data_idx, idx, &mut xrefs);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Immediate-as-pointer.
        if region_is_exec {
            if let Some(imm) = imm_as_address(&insn) {
                if idx.contains(Va::new(imm)) {
                    xrefs.push(Xref {
                        from: Va::new(va),
                        to: Va::new(imm),
                        kind: XrefKind::DataPointer,
                        confidence: Confidence::LinearImmediate,
                    });
                }
            }
        }

        // Propagation-only: update register / jump table / CMP tracking.
        if prop == PropMode::On {
            update_jt_state(&insn, &reg_vals, &cmp_bound, &mut jt_info);
            update_cmp_state(&insn, &mut cmp_bound);
            update_prop_state(&insn, &mut reg_vals);
        }
    }

    xrefs
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Emit xrefs for all direct branch/call instructions and GOT-indirect calls.
///
/// This is the single source-of-truth for branch emission, shared by
/// `scan_linear_with_index` and `scan_with_prop`.  Both depth levels apply
/// identical branch-detection logic; the only difference between them is that
/// `scan_with_prop` additionally tracks register constants and resolves
/// indirect `CALL reg` / `JMP reg` targets.
///
/// Handles:
///   CALL rel32           → `Call`  (FlowControl::Call)
///   JMP rel32            → `Jump`  (FlowControl::UnconditionalBranch)
///   Jcc rel32/rel8       → `CondJump` (FlowControl::ConditionalBranch)
///   XBEGIN rel32         → `CondJump` (FlowControl::XbeginXabortXend)
///   CALL [RIP+disp32] }  → `Call`/`Jump` to GOT slot VA
///   JMP  [RIP+disp32] }  (FlowControl::IndirectCall / IndirectBranch, FF 15 / FF 25)
#[inline]
fn emit_direct_branches(
    insn: &Instruction,
    va: u64,
    idx: &SegmentIndex,
    got: GotSlots<'_>,
    xrefs: &mut Vec<Xref>,
) {
    match insn.flow_control() {
        iced_x86::FlowControl::Call
        | iced_x86::FlowControl::UnconditionalBranch
        | iced_x86::FlowControl::ConditionalBranch
        // XBEGIN encodes a NearBranch relative offset (the TSX abort handler)
        // but has FlowControl::XbeginXabortXend. Treat it as a conditional
        // jump — execution either enters the transaction or jumps to the handler.
        | iced_x86::FlowControl::XbeginXabortXend => {
            if let Some(target) = direct_target(insn) {
                if idx.contains(Va::new(target)) {
                    let kind = match insn.flow_control() {
                        iced_x86::FlowControl::Call => XrefKind::Call,
                        iced_x86::FlowControl::UnconditionalBranch => XrefKind::Jump,
                        iced_x86::FlowControl::ConditionalBranch
                        | iced_x86::FlowControl::XbeginXabortXend => XrefKind::CondJump,
                        // All cases are listed in the outer match arm — unreachable.
                        other => unreachable!(
                            "direct_target returned Some for non-branch flow {:?}",
                            other
                        ),
                    };
                    xrefs.push(Xref {
                        from: Va::new(va),
                        to: Va::new(target),
                        kind,
                        confidence: Confidence::LinearImmediate,
                    });
                }
            }
        }
        // Indirect call/jump through GOT (FF 15 / FF 25): emit xref to the
        // GOT slot VA itself. The benchmark normalizes IDA's extern VAs back
        // to GOT slot VAs so both sides match on (from, got_slot_va).
        FlowControl::IndirectCall | FlowControl::IndirectBranch => {
            emit_got_indirect(insn, va, got, xrefs);
        }
        _ => {}
    }
}

/// Emit a Call or Jump xref for an indirect `CALL [RIP+disp32]` / `JMP [RIP+disp32]`
/// instruction (FF 15 / FF 25) when the target is a known GOT slot.
///
/// The target is the GOT slot VA itself (the address of the pointer the CPU
/// dereferences at runtime).  The benchmark normalizes IDA's synthetic extern
/// VAs back to GOT slot VAs so both sides agree on `(from, got_slot_va)`.
///
/// Only fires for slots in `got_slots` (populated from GLOB_DAT / JUMP_SLOT
/// relocations) to avoid emitting spurious Call xrefs for non-GOT RIP-relative
/// indirect calls (e.g., function pointer tables).
#[inline]
fn emit_got_indirect(insn: &Instruction, va: u64, got: GotSlots<'_>, xrefs: &mut Vec<Xref>) {
    // Only applies when the single memory operand uses RIP-relative addressing.
    if insn.op_count() == 1
        && insn.op0_kind() == OpKind::Memory
        && insn.memory_base() == Register::RIP
    {
        let got_slot_va = Va::new(insn.memory_displacement64());
        if got.slots.contains(&got_slot_va) {
            let kind = if insn.flow_control() == FlowControl::IndirectCall {
                XrefKind::Call
            } else if got.call_only {
                // PE import thunk (`jmp [rip+iat]`): IDA records a data read of
                // the slot (emitted by `emit_rip_relative`), not a jump.
                return;
            } else {
                XrefKind::Jump
            };
            xrefs.push(Xref {
                from: Va::new(va),
                to: got_slot_va,
                kind,
                confidence: Confidence::LinearImmediate,
            });
        }
    }
}

/// Emit a RIP-relative data xref if the instruction has a `[RIP + disp32]` operand.
///
/// IDA distinguishes:
///   LEA r64, [rip+disp]  →  data_ptr   (takes the address; dr_O)
///   MOV reg, [rip+disp]  →  data_read  (loads from address; dr_R)
///   MOV [rip+disp], reg  →  data_write (stores to address; dr_W)
///   CMP/TEST [rip+disp]  →  data_read
///
/// Non-LEA instructions pointing into an executable segment are suppressed
/// (IDA does not record these).
///
/// This is the single source-of-truth for RIP-relative xref emission,
/// shared by `scan_linear_with_index` and `scan_with_prop`.
#[inline]
fn emit_rip_relative(
    insn: &Instruction,
    va: u64,
    idx: &SegmentIndex,
    info_factory: &mut InstructionInfoFactory,
    xrefs: &mut Vec<Xref>,
) {
    for op_idx in 0..insn.op_count() {
        if insn.op_kind(op_idx) == OpKind::Memory && insn.memory_base() == Register::RIP {
            let target = insn.memory_displacement64();
            if idx.contains(Va::new(target)) {
                let target_is_exec = idx.is_exec(Va::new(target));
                let is_lea = matches!(
                    insn.code(),
                    Code::Lea_r16_m | Code::Lea_r32_m | Code::Lea_r64_m
                );
                let is_write = !is_lea && mem_op_is_write(insn, op_idx, info_factory);
                let kind = if is_lea {
                    XrefKind::DataPointer
                } else if is_write {
                    XrefKind::DataWrite
                } else {
                    XrefKind::DataRead
                };
                // For non-LEA: suppress if target is exec (IDA doesn't record these)
                if is_lea || !target_is_exec {
                    xrefs.push(Xref {
                        from: Va::new(va),
                        to: Va::new(target),
                        kind,
                        confidence: Confidence::LinearImmediate,
                    });
                }
            }
            break; // at most one memory operand per instruction
        }
    }
}

/// Maximum value for a sign-extended 32-bit immediate treated as an address.
///
/// Values at or above 4 GiB are sign-extension artefacts from `imm32to64`
/// (e.g. `0xFFFF_FFFF_FFFF_FF80` from `CMP rax, -128`), not valid addresses.
const IMM32_ADDR_MAX: u64 = 0x1_0000_0000;

/// Extract a 32-bit immediate operand as a potential address value, for
/// instruction types that IDA records as data_ptr when the immediate
/// resolves to a mapped address.
///
/// IDA records data_ptr for:
///   MOV r64, imm64 / MOV r/m64, imm32 / MOV r32, imm32
///   CMP r/m64, imm32 / CMP rAX, imm32  (0x81 /7 / 0x3d)
///   SUB r/m64, imm32 / SUB rAX, imm32  (0x81 /5 / 0x2d)
///
/// IDA does NOT record data_ptr for AND, OR, XOR, TEST, PUSH, ADD with imm32
/// (treated as bitmask/constant values, not address pointers).
///
/// Returns the address value if this instruction qualifies, None otherwise.
#[inline]
fn imm_as_address(insn: &Instruction) -> Option<u64> {
    match insn.code() {
        // MOV reg, imm64 — direct 64-bit pointer
        Code::Mov_r64_imm64 => {
            let v = insn.immediate64();
            (v > 0).then_some(v)
        }
        // 32-bit immediates: MOV, CMP, SUB — zero/sign-extended address values
        Code::Mov_rm64_imm32
        | Code::Mov_r32_imm32
        | Code::Cmp_rm64_imm32
        | Code::Cmp_rm64_imm8
        | Code::Cmp_RAX_imm32
        | Code::Cmp_EAX_imm32
        | Code::Sub_rm64_imm32
        | Code::Sub_rm64_imm8
        | Code::Sub_RAX_imm32
        | Code::Sub_EAX_imm32 => {
            let v = insn.immediate32to64() as u64;
            (v > 0 && v < IMM32_ADDR_MAX).then_some(v)
        }
        _ => None,
    }
}

/// Maximum number of jump table entries to read before giving up.
/// Real switch statements rarely exceed ~500 cases; 4096 is a generous
/// upper bound that prevents runaway reads into adjacent data.
const MAX_JUMP_TABLE_ENTRIES: usize = 4096;

/// Update jump table tracking state for the current instruction.
///
/// Recognises the compiler pattern:
///   CMP  ridx, IMM                   ; sets cmp_bound[ridx] = IMM+1
///   JA   default_label               ; unsigned comparison
///   LEA  rbase, [rip + table]        ; sets reg_vals[rbase] = table_va
///   MOVSXD roff, [rbase + ridx*4]    ; loads signed 32-bit offset from table
///   ADD  rtgt, rbase                 ; computes target = offset + base
///   JMP  rtgt                        ; indirect jump through table
///
/// `jt_info[i] = Some((table_start, target_base, max_entries))` means register `i`
/// currently participates in a jump table dispatch with:
///   table_start  = VA of the first table entry (for reading i32 offsets)
///   target_base  = value to add to each offset (the LEA result)
///   max_entries  = upper bound from CMP+JA (None = unknown, use fallback)
///
/// Must be called BEFORE `update_prop_state` / `update_cmp_state` so it reads
/// old `reg_vals` and `cmp_bound`.
fn update_jt_state(
    insn: &Instruction,
    reg_vals: &[Option<u64>; 16],
    cmp_bound: &[Option<u32>; 16],
    jt_info: &mut [Option<JumpTableInfo>; 16],
) {
    if insn.op_count() == 0 || insn.op0_kind() != OpKind::Register {
        return;
    }
    let dst = insn.op0_register();
    let Some(di) = gpr_index(dst) else { return };

    match insn.code() {
        // MOVSXD r64, [base + idx*4 + disp] — potential jump table load
        Code::Movsxd_r64_rm32 if insn.op1_kind() == OpKind::Memory => {
            if insn.memory_index() != Register::None && insn.memory_index_scale() == 4 {
                if let Some(bi) = gpr_index(insn.memory_base()) {
                    if let Some(base_val) = reg_vals[bi] {
                        let table_start =
                            Va::new(base_val.wrapping_add(insn.memory_displacement64()));
                        // Look up the CMP bound for the index register
                        let max_entries =
                            gpr_index(insn.memory_index()).and_then(|ii| cmp_bound[ii]);
                        jt_info[di] = Some(JumpTableInfo {
                            table_start,
                            target_base: Va::new(base_val),
                            max_entries,
                        });
                        return;
                    }
                }
            }
            jt_info[di] = None;
        }

        // ADD r64, r64 — propagate jt_info through the base+offset addition.
        // Verify that the other operand's reg_vals matches the target_base
        // to avoid false positives from unrelated ADDs.
        Code::Add_rm64_r64 | Code::Add_r64_rm64
            if insn.op0_kind() == OpKind::Register && insn.op1_kind() == OpKind::Register =>
        {
            let src = insn.op1_register();
            if let Some(si) = gpr_index(src) {
                if let Some(info) = jt_info[di] {
                    // dst had the table offset; verify src is the base register
                    if reg_vals[si] == Some(info.target_base.raw()) {
                        // jt_info[di] stays — dst is now offset + base = target
                    } else {
                        jt_info[di] = None;
                    }
                } else if let Some(info) = jt_info[si] {
                    // src had the table offset; verify dst is the base register
                    if reg_vals[di] == Some(info.target_base.raw()) {
                        jt_info[di] = Some(info);
                    } else {
                        jt_info[di] = None;
                    }
                } else {
                    jt_info[di] = None;
                }
            } else {
                jt_info[di] = None;
            }
        }

        // Everything else that writes to a register: clear jt_info
        _ => {
            jt_info[di] = None;
        }
    }
}

/// Track CMP immediate bounds per register and propagate through MOV.
///
/// When `CMP rN, imm` is seen, records `imm + 1` as the maximum number of
/// valid jump table entries (valid indices are 0..imm). Propagated through
/// register-to-register MOV (handles `mov eax, ebx` preserving the bound).
/// Cleared by any other register write.
fn update_cmp_state(insn: &Instruction, cmp_bound: &mut [Option<u32>; 16]) {
    // CMP rN, imm → set bound
    match insn.code() {
        Code::Cmp_rm64_imm32 | Code::Cmp_rm32_imm32 | Code::Cmp_RAX_imm32 | Code::Cmp_EAX_imm32 => {
            if insn.op0_kind() == OpKind::Register {
                if let Some(ri) = gpr_index(insn.op0_register()) {
                    let imm = insn.immediate32() as u64;
                    if imm < MAX_JUMP_TABLE_ENTRIES as u64 {
                        cmp_bound[ri] = Some((imm + 1) as u32);
                    } else {
                        cmp_bound[ri] = None;
                    }
                    return;
                }
            }
        }
        Code::Cmp_rm64_imm8 | Code::Cmp_rm32_imm8 if insn.op0_kind() == OpKind::Register => {
            if let Some(ri) = gpr_index(insn.op0_register()) {
                let imm = insn.immediate8() as u64;
                if imm < MAX_JUMP_TABLE_ENTRIES as u64 {
                    cmp_bound[ri] = Some((imm + 1) as u32);
                } else {
                    cmp_bound[ri] = None;
                }
                return;
            }
        }
        _ => {}
    }

    // Propagate through register-to-register MOV
    let is_mov_reg = matches!(
        insn.code(),
        Code::Mov_r32_rm32 | Code::Mov_rm32_r32 | Code::Mov_r64_rm64 | Code::Mov_rm64_r64
    );
    if is_mov_reg && insn.op0_kind() == OpKind::Register && insn.op1_kind() == OpKind::Register {
        if let (Some(di), Some(si)) = (
            gpr_index(insn.op0_register()),
            gpr_index(insn.op1_register()),
        ) {
            cmp_bound[di] = cmp_bound[si];
            return;
        }
    }

    // Any other write to a register: clear bound
    if insn.op_count() > 0 && insn.op0_kind() == OpKind::Register {
        if let Some(di) = gpr_index(insn.op0_register()) {
            cmp_bound[di] = None;
        }
    }
}

/// Read a jump table starting at `jt.table_start`, emitting Jump xrefs from
/// `jmp_va` to each resolved target.
///
/// Each entry is a signed 32-bit offset added to `jt.target_base` to compute
/// the jump target.
///
/// Requires a CMP bound (`jt.max_entries`) to limit the read.  Without one
/// the table size is unknown, and random data in dead-code zones would produce
/// many false-positive targets.  Tables without a bound are silently skipped.
///
/// Also stops on the first entry that doesn't resolve to an executable address.
fn recover_jump_table(
    jmp_va: Va,
    jt: JumpTableInfo,
    data_idx: &SegmentDataIndex,
    seg_idx: &SegmentIndex,
    xrefs: &mut Vec<Xref>,
) {
    let Some(max) = jt.max_entries else {
        return; // no CMP bound — skip to avoid FP explosion in dead zones
    };
    let limit = (max as usize).min(MAX_JUMP_TABLE_ENTRIES);
    for i in 0..limit {
        let slot_va = jt.table_start + (i as u64) * 4;
        let Some(offset) = data_idx.read_i32_at(slot_va) else {
            break;
        };
        let target = Va::new((jt.target_base.raw() as i64).wrapping_add(offset as i64) as u64);
        if !seg_idx.is_exec(target) {
            break;
        }
        xrefs.push(Xref {
            from: jmp_va,
            to: target,
            kind: XrefKind::Jump,
            confidence: Confidence::LocalProp,
        });
    }
}

/// Extract the direct branch/call target if the instruction has one.
fn direct_target(insn: &Instruction) -> Option<u64> {
    // iced-x86 provides NearBranch64 for direct calls/jumps
    match insn.op0_kind() {
        OpKind::NearBranch16 => Some(insn.near_branch16() as u64),
        OpKind::NearBranch32 => Some(insn.near_branch32() as u64),
        OpKind::NearBranch64 => Some(insn.near_branch64()),
        OpKind::FarBranch16 | OpKind::FarBranch32 => None, // segment-based, skip
        _ => None,
    }
}

/// Update register propagation state for MOV reg, imm64 and similar.
fn update_prop_state(insn: &Instruction, vals: &mut [Option<u64>; 16]) {
    // We only track: MOV r64, imm64
    // Everything else that writes to a register invalidates it.
    let op_count = insn.op_count();
    if op_count == 0 {
        return;
    }

    // Destination is op 0 for most x86-64 instructions
    if insn.op0_kind() != OpKind::Register {
        return;
    }
    let dst = insn.op0_register();
    let Some(dst_idx) = gpr_index(dst) else {
        return;
    };

    // MOV rN, imm — set the value
    match insn.code() {
        Code::Mov_r64_imm64 => {
            vals[dst_idx] = Some(insn.immediate64());
        }
        Code::Mov_rm64_imm32 => {
            vals[dst_idx] = Some(insn.immediate32to64() as u64);
        }
        Code::Mov_r32_imm32 => {
            // mov r32, imm32 zero-extends to 64 bits on x86-64
            vals[dst_idx] = Some(insn.immediate32() as u64);
        }
        Code::Mov_r16_imm16 => {
            // MOV r16, imm16 writes only the lower 16 bits of the register;
            // the upper 48 bits are preserved (unlike MOV r32 which zero-extends).
            // We cannot know the full 64-bit value, so invalidate.
            vals[dst_idx] = None;
        }
        // LEA r64, [rip+disp] — we know the value since RIP-relative is resolved
        Code::Lea_r64_m if insn.memory_base() == Register::RIP => {
            vals[dst_idx] = Some(insn.memory_displacement64());
        }
        // Any other write — invalidate
        _ => {
            vals[dst_idx] = None;
        }
    }
}

// ── GPR index newtype ─────────────────────────────────────────────────────────

/// An x86-64 general-purpose register index in 0..=15 (RAX–R15).
///
/// Wraps a `u8` so that array accesses through `GprIdx` are always in bounds
/// for the 16-element register tracking arrays.  Using `GprIdx` instead of
/// bare `usize` eliminates scattered raw-index arithmetic at every use site.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct GprIdx(u8);

impl GprIdx {
    /// Index into a 16-element register array.
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}

impl std::ops::Index<GprIdx> for [Option<u64>; 16] {
    type Output = Option<u64>;
    #[inline]
    fn index(&self, i: GprIdx) -> &Self::Output {
        &self[i.idx()]
    }
}
impl std::ops::IndexMut<GprIdx> for [Option<u64>; 16] {
    #[inline]
    fn index_mut(&mut self, i: GprIdx) -> &mut Self::Output {
        &mut self[i.idx()]
    }
}
impl std::ops::Index<GprIdx> for [Option<JumpTableInfo>; 16] {
    type Output = Option<JumpTableInfo>;
    #[inline]
    fn index(&self, i: GprIdx) -> &Self::Output {
        &self[i.idx()]
    }
}
impl std::ops::IndexMut<GprIdx> for [Option<JumpTableInfo>; 16] {
    #[inline]
    fn index_mut(&mut self, i: GprIdx) -> &mut Self::Output {
        &mut self[i.idx()]
    }
}
impl std::ops::Index<GprIdx> for [Option<u32>; 16] {
    type Output = Option<u32>;
    #[inline]
    fn index(&self, i: GprIdx) -> &Self::Output {
        &self[i.idx()]
    }
}
impl std::ops::IndexMut<GprIdx> for [Option<u32>; 16] {
    #[inline]
    fn index_mut(&mut self, i: GprIdx) -> &mut Self::Output {
        &mut self[i.idx()]
    }
}

/// Map a GPR (in any width) to a `GprIdx`.
/// Returns `None` for non-GPR registers (XMM, segment, etc.).
fn gpr_index(reg: Register) -> Option<GprIdx> {
    let i = match reg {
        Register::RAX | Register::EAX | Register::AX | Register::AL | Register::AH => 0,
        Register::RCX | Register::ECX | Register::CX | Register::CL | Register::CH => 1,
        Register::RDX | Register::EDX | Register::DX | Register::DL | Register::DH => 2,
        Register::RBX | Register::EBX | Register::BX | Register::BL | Register::BH => 3,
        Register::RSP | Register::ESP | Register::SP | Register::SPL => 4,
        Register::RBP | Register::EBP | Register::BP | Register::BPL => 5,
        Register::RSI | Register::ESI | Register::SI | Register::SIL => 6,
        Register::RDI | Register::EDI | Register::DI | Register::DIL => 7,
        Register::R8 | Register::R8D | Register::R8W | Register::R8L => 8,
        Register::R9 | Register::R9D | Register::R9W | Register::R9L => 9,
        Register::R10 | Register::R10D | Register::R10W | Register::R10L => 10,
        Register::R11 | Register::R11D | Register::R11W | Register::R11L => 11,
        Register::R12 | Register::R12D | Register::R12W | Register::R12L => 12,
        Register::R13 | Register::R13D | Register::R13W | Register::R13L => 13,
        Register::R14 | Register::R14D | Register::R14W | Register::R14L => 14,
        Register::R15 | Register::R15D | Register::R15W | Register::R15L => 15,
        _ => return None,
    };
    Some(GprIdx(i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::ScanRegion;
    use crate::loader::{SegData, Segment, SegmentArch};
    use crate::xref::{Confidence, XrefKind};
    use rustc_hash::FxHashSet;

    fn fake_seg(va: u64, data: &'static [u8]) -> Segment {
        Segment {
            va: Va::new(va),
            data: unsafe { SegData::new_for_test(data) },
            executable: true,
            readable: true,
            writable: false,
            byte_scannable: true,
            arch: SegmentArch::Generic,
            name: "test".to_string(),
        }
    }

    fn fake_data_seg(va: u64, data: &'static [u8]) -> Segment {
        Segment {
            va: Va::new(va),
            data: unsafe { SegData::new_for_test(data) },
            executable: false,
            readable: true,
            writable: false,
            byte_scannable: false,
            arch: SegmentArch::Generic,
            name: "test_data".to_string(),
        }
    }

    fn region_for<'a>(seg: &'a Segment) -> ScanRegion<'a> {
        ScanRegion::new(seg, seg.va, seg.va + seg.data().len() as u64)
    }

    // ── CALL rel32 ────────────────────────────────────────────────────────────

    /// E8 fb0f0000 = CALL 0x2000 from 0x1000 (next_ip=0x1005, rel=0xffb)
    #[test]
    fn test_call_rel32() {
        static CODE: [u8; 5] = [0xe8, 0xfb, 0x0f, 0x00, 0x00];
        static TARGET: [u8; 4] = [0x00; 4];
        let code = fake_seg(0x1000, &CODE);
        let tgt = fake_seg(0x2000, &TARGET);
        let segs = vec![fake_seg(0x1000, &CODE), tgt];

        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_linear(
            &region_for(&code),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );
        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1000));
        assert_eq!(xrefs[0].to, Va::new(0x2000));
        assert_eq!(xrefs[0].kind, XrefKind::Call);
        assert_eq!(xrefs[0].confidence, Confidence::LinearImmediate);
    }

    // ── CALL/JMP [RIP+disp32] through a GOT / IAT slot ────────────────────────

    /// `ff 15 / ff 25 fa0f0000` at 0x1000: slot = 0x1006 + 0xffa = 0x2000.
    fn got_indirect_kinds(opcode: u8, call_only: bool) -> Vec<XrefKind> {
        static SLOT: [u8; 8] = [0x00; 8];
        static CALL: [u8; 6] = [0xff, 0x15, 0xfa, 0x0f, 0x00, 0x00];
        static JMP: [u8; 6] = [0xff, 0x25, 0xfa, 0x0f, 0x00, 0x00];
        let code = fake_seg(0x1000, if opcode == 0x15 { &CALL } else { &JMP });
        let segs = vec![fake_seg(0x1000, &CALL), fake_data_seg(0x2000, &SLOT)];
        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let slots: FxHashSet<Va> = [Va::new(0x2000)].into_iter().collect();
        let got = GotSlots {
            slots: &slots,
            call_only,
        };
        scan_linear(&region_for(&code), &idx, got, &data_idx)
            .iter()
            .filter(|x| x.to == Va::new(0x2000))
            .map(|x| x.kind)
            .collect()
    }

    #[test]
    fn test_got_indirect_call_emits_call() {
        assert!(got_indirect_kinds(0x15, false).contains(&XrefKind::Call));
        assert!(got_indirect_kinds(0x15, true).contains(&XrefKind::Call));
    }

    #[test]
    fn test_got_indirect_jmp_is_jump_unless_call_only() {
        assert!(got_indirect_kinds(0x25, false).contains(&XrefKind::Jump));
        // PE import thunk: IDA records a data read, not a jump.
        assert!(!got_indirect_kinds(0x25, true).contains(&XrefKind::Jump));
    }

    // ── JMP rel32 ─────────────────────────────────────────────────────────────

    /// E9 f60f0000 = JMP 0x2000 from 0x1005 (next_ip=0x100a, rel=0xff6)
    #[test]
    fn test_jmp_rel32() {
        static CODE: [u8; 10] = [
            0x90, 0x90, 0x90, 0x90, 0x90, // 5 NOPs at 0x1000..0x1005
            0xe9, 0xf6, 0x0f, 0x00, 0x00, // JMP 0x2000 at 0x1005
        ];
        static TARGET: [u8; 4] = [0x00; 4];
        let code = fake_seg(0x1000, &CODE);
        let tgt = fake_seg(0x2000, &TARGET);
        let segs = vec![fake_seg(0x1000, &CODE), tgt];

        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_linear(
            &region_for(&code),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );
        let jmp = xrefs.iter().find(|x| x.kind == XrefKind::Jump).unwrap();
        assert_eq!(jmp.from, Va::new(0x1005));
        assert_eq!(jmp.to, Va::new(0x2000));
    }

    // ── JE rel32 (conditional) ────────────────────────────────────────────────

    /// 0F84 f00f0000 = JE 0x2000 from 0x100a (next_ip=0x1010)
    #[test]
    fn test_je_cond_jump() {
        static CODE: [u8; 16] = [
            0x90, 0x90, 0x90, 0x90, 0x90, // 5 NOPs 0x1000..0x1005
            0x90, 0x90, 0x90, 0x90, 0x90, // 5 NOPs 0x1005..0x100a
            0x0f, 0x84, 0xf0, 0x0f, 0x00, 0x00, // JE 0x2000 at 0x100a
        ];
        static TARGET: [u8; 4] = [0x00; 4];
        let code = fake_seg(0x1000, &CODE);
        let tgt = fake_seg(0x2000, &TARGET);
        let segs = vec![fake_seg(0x1000, &CODE), tgt];

        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_linear(
            &region_for(&code),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );
        let je = xrefs.iter().find(|x| x.kind == XrefKind::CondJump).unwrap();
        assert_eq!(je.from, Va::new(0x100a));
        assert_eq!(je.to, Va::new(0x2000));
    }

    // ── LEA rax, [rip + disp32] ───────────────────────────────────────────────

    /// 48 8D 05 f91f0000 = LEA rax, [rip+0x1ff9] from 0x1000 → 0x3000
    /// (next_ip = 0x1007, disp = 0x3000 - 0x1007 = 0x1ff9)
    #[test]
    fn test_lea_rip_relative() {
        static CODE: [u8; 7] = [0x48, 0x8d, 0x05, 0xf9, 0x1f, 0x00, 0x00];
        static TARGET: [u8; 8] = [0x00; 8];
        let code = fake_seg(0x1000, &CODE);
        let tgt = fake_seg(0x3000, &TARGET);
        let segs = vec![fake_seg(0x1000, &CODE), tgt];

        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_linear(
            &region_for(&code),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );
        // LEA = takes address → DataPointer (IDA dr_O), not DataRead
        let lea = xrefs
            .iter()
            .find(|x| x.kind == XrefKind::DataPointer)
            .unwrap();
        assert_eq!(lea.from, Va::new(0x1000));
        assert_eq!(lea.to, Va::new(0x3000));
        assert_eq!(lea.confidence, Confidence::LinearImmediate);
    }

    // ── MOV rax, imm64 / CALL rax → LocalProp ─────────────────────────────────

    /// 48 B8 <imm64> = MOV rax, 0x4000  (10 bytes at 0x1000)
    /// FF D0         = CALL rax          (2 bytes at 0x100a)
    #[test]
    fn test_mov_call_reg_prop() {
        static CODE: [u8; 12] = [
            0x48, 0xb8, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // MOV rax, 0x4000
            0xff, 0xd0, // CALL rax
        ];
        static TARGET: [u8; 4] = [0x00; 4];
        let code = fake_seg(0x1000, &CODE);
        let tgt = fake_seg(0x4000, &TARGET);
        let segs = vec![fake_seg(0x1000, &CODE), tgt];

        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_with_prop(
            &region_for(&code),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );
        let prop = xrefs
            .iter()
            .find(|x| x.confidence == Confidence::LocalProp)
            .expect("expected a LocalProp xref");
        assert_eq!(prop.from, Va::new(0x100a));
        assert_eq!(prop.to, Va::new(0x4000));
        assert_eq!(prop.kind, XrefKind::Call);
    }

    // ── Target out of range → filtered ────────────────────────────────────────

    #[test]
    fn test_call_out_of_range_filtered() {
        static CODE: [u8; 5] = [0xe8, 0xfb, 0x0f, 0x00, 0x00]; // CALL 0x2000
                                                               // No segment at 0x2000
        let code = fake_seg(0x1000, &CODE);
        let segs = vec![fake_seg(0x1000, &CODE)];
        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_linear(
            &region_for(&code),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );
        assert!(xrefs.is_empty());
    }

    // ── Jump table recovery ───────────────────────────────────────────────────

    /// CMP + JA + LEA + MOVSXD + ADD + JMP rax — full jump table pattern.
    /// Table at 0x3000 with two i32 offsets pointing to targets at 0x2000, 0x2004.
    #[test]
    fn test_jump_table_basic() {
        // Code at 0x1000:
        //   0x1000: 83 FA 01                 cmp edx, 1          (bound = 2 entries)
        //   0x1003: 77 11                    ja  0x1016          (skip if > 1)
        //   0x1005: 48 8D 0D F4 1F 00 00    lea rcx, [rip+0x1FF4]  → 0x3000
        //   0x100C: 48 63 04 91              movsxd rax, [rcx+rdx*4]
        //   0x1010: 48 01 C8                 add rax, rcx
        //   0x1013: FF E0                    jmp rax
        //   0x1015: CC                       int3 (padding)
        //   0x1016: CC                       int3 (JA target)
        static CODE: [u8; 23] = [
            0x83, 0xFA, 0x01, // cmp edx, 1
            0x77, 0x11, // ja +0x11 (→ 0x1016)
            0x48, 0x8D, 0x0D, 0xF4, 0x1F, 0x00, 0x00, // lea rcx, [rip+0x1FF4]
            0x48, 0x63, 0x04, 0x91, // movsxd rax, [rcx+rdx*4]
            0x48, 0x01, 0xC8, // add rax, rcx
            0xFF, 0xE0, // jmp rax
            0xCC, // int3
            0xCC, // int3 (JA target)
        ];
        // Table at 0x3000: two i32 offsets → targets at 0x2000, 0x2004
        //   entry 0: 0x2000 - 0x3000 = -0x1000 = 0xFFFFF000
        //   entry 1: 0x2004 - 0x3000 = -0x0FFC = 0xFFFFF004
        static TABLE: [u8; 8] = [
            0x00, 0xF0, 0xFF, 0xFF, // -0x1000
            0x04, 0xF0, 0xFF, 0xFF, // -0x0FFC
        ];
        // Target code segment at 0x2000 (executable)
        static TARGETS: [u8; 8] = [0xCC; 8]; // INT3 placeholders

        let code_seg = fake_seg(0x1000, &CODE);
        let segs = vec![
            fake_seg(0x1000, &CODE),
            fake_seg(0x2000, &TARGETS),
            fake_data_seg(0x3000, &TABLE),
        ];

        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_with_prop(
            &region_for(&code_seg),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );

        // Should have Jump xrefs from 0x1013 to 0x2000 and 0x2004
        let jumps: Vec<&Xref> = xrefs
            .iter()
            .filter(|x| x.kind == XrefKind::Jump && x.confidence == Confidence::LocalProp)
            .collect();
        assert_eq!(
            jumps.len(),
            2,
            "expected 2 jump table targets, got {}",
            jumps.len()
        );
        assert!(jumps
            .iter()
            .any(|x| x.from == Va::new(0x1013) && x.to == Va::new(0x2000)));
        assert!(jumps
            .iter()
            .any(|x| x.from == Va::new(0x1013) && x.to == Va::new(0x2004)));
    }

    /// Jump table with forward offsets (positive i32 values) and CMP bound.
    #[test]
    fn test_jump_table_forward_offsets() {
        // Table at 0x3000, targets at 0x4000 and 0x4100 (forward from table)
        //   entry 0: 0x4000 - 0x3000 = +0x1000
        //   entry 1: 0x4100 - 0x3000 = +0x1100
        static CODE: [u8; 23] = [
            0x83, 0xFA, 0x01, // cmp edx, 1
            0x77, 0x11, // ja +0x11
            0x48, 0x8D, 0x0D, 0xF4, 0x1F, 0x00, 0x00, // lea rcx, [rip+0x1FF4] → 0x3000
            0x48, 0x63, 0x04, 0x91, // movsxd rax, [rcx+rdx*4]
            0x48, 0x01, 0xC8, // add rax, rcx
            0xFF, 0xE0, // jmp rax
            0xCC, // int3
            0xCC, // int3
        ];
        static TABLE: [u8; 8] = [
            0x00, 0x10, 0x00, 0x00, // +0x1000
            0x00, 0x11, 0x00, 0x00, // +0x1100
        ];
        static TARGETS: [u8; 0x200] = [0xCC; 0x200];

        let code_seg = fake_seg(0x1000, &CODE);
        let segs = vec![
            fake_seg(0x1000, &CODE),
            fake_data_seg(0x3000, &TABLE),
            fake_seg(0x4000, &TARGETS),
        ];

        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_with_prop(
            &region_for(&code_seg),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );

        let jumps: Vec<&Xref> = xrefs
            .iter()
            .filter(|x| x.kind == XrefKind::Jump && x.confidence == Confidence::LocalProp)
            .collect();
        assert_eq!(jumps.len(), 2);
        assert!(jumps
            .iter()
            .any(|x| x.from == Va::new(0x1013) && x.to == Va::new(0x4000)));
        assert!(jumps
            .iter()
            .any(|x| x.from == Va::new(0x1013) && x.to == Va::new(0x4100)));
    }

    /// No jump xrefs when table entries don't resolve to executable addresses.
    #[test]
    fn test_jump_table_no_valid_targets() {
        static CODE: [u8; 16] = [
            0x48, 0x8D, 0x0D, 0xF9, 0x1F, 0x00, 0x00, 0x48, 0x63, 0x04, 0x91, 0x48, 0x01, 0xC8,
            0xFF, 0xE0,
        ];
        // Table offsets point to unmapped addresses (no target segment)
        static TABLE: [u8; 8] = [
            0x00, 0x00, 0x10, 0x00, // +0x100000 → 0x103000 (unmapped)
            0x00, 0x00, 0x20, 0x00, // +0x200000 → 0x203000 (unmapped)
        ];

        let code_seg = fake_seg(0x1000, &CODE);
        let segs = vec![fake_seg(0x1000, &CODE), fake_data_seg(0x3000, &TABLE)];

        let idx = SegmentIndex::build(&segs);
        let data_idx = SegmentDataIndex::build(&segs);
        let xrefs = scan_with_prop(
            &region_for(&code_seg),
            &idx,
            GotSlots {
                slots: &FxHashSet::default(),
                call_only: false,
            },
            &data_idx,
        );

        let jumps: Vec<&Xref> = xrefs
            .iter()
            .filter(|x| x.kind == XrefKind::Jump && x.confidence == Confidence::LocalProp)
            .collect();
        assert!(
            jumps.is_empty(),
            "should emit no jump table xrefs for unmapped targets"
        );
    }
}
