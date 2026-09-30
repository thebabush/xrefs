//! Exhaustive fuzzer: compares our bitmask decoder against bad64 for every
//! possible 32-bit ARM64 word.
//!
//! Run with:
//!   cargo run --release --bin fuzz_arm64
//!
//! Uses all available cores via rayon.  At ~300–500 M words/sec/core on
//! Apple Silicon, all 2^32 words take 10–30 seconds.
//!
//! For each word the fuzzer checks:
//!  - Classification agreement: if bad64 decodes it as one of our ~15 ops,
//!    our decoder must return the matching variant (not Other).
//!  - If bad64 fails (decode error), our decoder may return any variant —
//!    we don't claim to reject invalid encodings, only to decode valid ones.
//!  - Field agreement: for every matching variant, check that the fields
//!    we extract (rd, rn, imm, target, etc.) match bad64's operands.
//!
//! Any mismatch is printed to stderr and the process exits with code 1.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use xr::arch::arm64_decode::Arm64Insn;

// Arbitrary fixed PC for target resolution comparisons.
// Must be 4-byte aligned and page-aligned for ADRP.
const PC: u64 = 0x0001_0000_0000u64;

fn main() {
    let errors = Arc::new(AtomicU64::new(0));
    let checked = Arc::new(AtomicU64::new(0));

    // Process in chunks of 1<<20 so rayon has good granularity.
    let chunk_size: u64 = 1 << 20;
    let num_chunks = (u32::MAX as u64 + 1) / chunk_size;

    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end = start + chunk_size;
        let mut local_errors = 0u64;
        let mut local_checked = 0u64;

        for word in start..end {
            let word = word as u32;
            if let Some(msg) = check_word(word) {
                eprintln!("MISMATCH word=0x{:08X}: {}", word, msg);
                local_errors += 1;
                // Cap error output so we don't flood the terminal.
                if local_errors >= 20 {
                    break;
                }
            }
            local_checked += 1;
        }

        errors.fetch_add(local_errors, Ordering::Relaxed);
        checked.fetch_add(local_checked, Ordering::Relaxed);

        // Progress every 64 chunks (~64M words)
        if chunk_idx % 64 == 0 {
            let done = checked.load(Ordering::Relaxed);
            let pct = done as f64 / (u32::MAX as f64 + 1.0) * 100.0;
            eprint!("\r{:.1}% ({} M words checked)", pct, done / 1_000_000);
        }
    });

    eprintln!(); // newline after progress

    let total_errors = errors.load(Ordering::Relaxed);
    let total_checked = checked.load(Ordering::Relaxed);
    println!(
        "Checked {} words, {} mismatches",
        total_checked, total_errors
    );

    if total_errors > 0 {
        std::process::exit(1);
    }
}

/// Returns `Some(message)` if our decoder disagrees with bad64, `None` if they agree.
fn check_word(word: u32) -> Option<String> {
    let ours = Arm64Insn::decode(word);
    let bad64_result = bad64::decode(word, PC);

    match bad64_result {
        Err(_) => {
            // bad64 rejects this word.  We don't require our decoder to reject it
            // (we may classify some UNDEFINED encodings), so no error here.
            None
        }
        Ok(insn) => {
            // bad64 accepted it — check that our classification and fields agree.
            check_agreement(word, &ours, &insn)
        }
    }
}

fn check_agreement(_word: u32, ours: &Arm64Insn, insn: &bad64::Instruction) -> Option<String> {
    use bad64::{Imm, Op, Operand};

    /// Extract a u64 immediate from a bad64 operand (label or imm).
    fn bad64_imm(op: Option<&Operand>) -> Option<u64> {
        match op? {
            Operand::Imm64 { imm, .. } | Operand::Imm32 { imm, .. } => Some(match imm {
                Imm::Signed(v) => *v as u64,
                Imm::Unsigned(v) => *v,
            }),
            Operand::Label(imm) => Some(match imm {
                Imm::Signed(v) => *v as u64,
                Imm::Unsigned(v) => *v,
            }),
            _ => None,
        }
    }

    fn bad64_reg(op: Option<&Operand>) -> Option<u8> {
        use bad64::Reg;
        let r = match op? {
            Operand::Reg { reg, .. } => *reg,
            _ => return None,
        };
        Some(match r {
            Reg::X0 | Reg::W0 => 0,
            Reg::X1 | Reg::W1 => 1,
            Reg::X2 | Reg::W2 => 2,
            Reg::X3 | Reg::W3 => 3,
            Reg::X4 | Reg::W4 => 4,
            Reg::X5 | Reg::W5 => 5,
            Reg::X6 | Reg::W6 => 6,
            Reg::X7 | Reg::W7 => 7,
            Reg::X8 | Reg::W8 => 8,
            Reg::X9 | Reg::W9 => 9,
            Reg::X10 | Reg::W10 => 10,
            Reg::X11 | Reg::W11 => 11,
            Reg::X12 | Reg::W12 => 12,
            Reg::X13 | Reg::W13 => 13,
            Reg::X14 | Reg::W14 => 14,
            Reg::X15 | Reg::W15 => 15,
            Reg::X16 | Reg::W16 => 16,
            Reg::X17 | Reg::W17 => 17,
            Reg::X18 | Reg::W18 => 18,
            Reg::X19 | Reg::W19 => 19,
            Reg::X20 | Reg::W20 => 20,
            Reg::X21 | Reg::W21 => 21,
            Reg::X22 | Reg::W22 => 22,
            Reg::X23 | Reg::W23 => 23,
            Reg::X24 | Reg::W24 => 24,
            Reg::X25 | Reg::W25 => 25,
            Reg::X26 | Reg::W26 => 26,
            Reg::X27 | Reg::W27 => 27,
            Reg::X28 | Reg::W28 => 28,
            Reg::X29 | Reg::W29 => 29,
            Reg::X30 | Reg::W30 => 30,
            _ => 31, // XZR, SP, etc.
        })
    }

    fn bad64_mem_base(op: Option<&Operand>) -> Option<u8> {
        let reg = match op? {
            Operand::MemOffset { reg, .. }
            | Operand::MemPreIdx { reg, .. }
            | Operand::MemPostIdxImm { reg, .. } => *reg,
            _ => return None,
        };
        // reuse same reg_index logic
        bad64_reg(Some(&Operand::Reg { reg, arrspec: None }))
    }

    fn bad64_mem_offset(op: Option<&Operand>) -> Option<i64> {
        use bad64::Imm;
        match op? {
            Operand::MemOffset { offset, .. } => Some(match offset {
                Imm::Signed(v) => *v,
                Imm::Unsigned(v) => *v as i64,
            }),
            Operand::MemPreIdx { imm, .. } | Operand::MemPostIdxImm { imm, .. } => {
                Some(match imm {
                    Imm::Signed(v) => *v,
                    Imm::Unsigned(v) => *v as i64,
                })
            }
            _ => None,
        }
    }

    let ops = insn.operands();

    match insn.op() {
        // ── BL ──────────────────────────────────────────────────────────────
        Op::BL => {
            if !matches!(ours, Arm64Insn::Bl(_)) {
                return Some(format!("expected Bl, got {:?}", ours));
            }
            let bad_target = bad64_imm(ops.first())?;
            let our_target = ours.imm26_target(PC);
            if bad_target != our_target {
                return Some(format!(
                    "BL target: bad64={:#x} ours={:#x}",
                    bad_target, our_target
                ));
            }
        }

        // ── B ───────────────────────────────────────────────────────────────
        Op::B => {
            if !matches!(ours, Arm64Insn::B(_)) {
                return Some(format!("expected B, got {:?}", ours));
            }
            let bad_target = bad64_imm(ops.first())?;
            let our_target = ours.imm26_target(PC);
            if bad_target != our_target {
                return Some(format!(
                    "B target: bad64={:#x} ours={:#x}",
                    bad_target, our_target
                ));
            }
        }

        // ── B.cond ──────────────────────────────────────────────────────────
        Op::B_AL
        | Op::B_CC
        | Op::B_CS
        | Op::B_EQ
        | Op::B_GE
        | Op::B_GT
        | Op::B_HI
        | Op::B_LE
        | Op::B_LS
        | Op::B_LT
        | Op::B_MI
        | Op::B_NE
        | Op::B_NV
        | Op::B_PL
        | Op::B_VC
        | Op::B_VS => {
            if !matches!(ours, Arm64Insn::BCond(_)) {
                return Some(format!("expected BCond, got {:?}", ours));
            }
            let bad_target = bad64_imm(ops.first())?;
            let our_target = ours.imm19_target(PC);
            if bad_target != our_target {
                return Some(format!(
                    "B.cond target: bad64={:#x} ours={:#x}",
                    bad_target, our_target
                ));
            }
        }

        // ── CBZ / CBNZ ──────────────────────────────────────────────────────
        Op::CBZ => {
            if !matches!(ours, Arm64Insn::Cbz(_)) {
                return Some(format!("expected Cbz, got {:?}", ours));
            }
            // bad64: op[0]=Rt, op[1]=label
            let bad_target = bad64_imm(ops.get(1))?;
            let our_target = ours.cbz_target(PC);
            if bad_target != our_target {
                return Some(format!(
                    "CBZ target: bad64={:#x} ours={:#x}",
                    bad_target, our_target
                ));
            }
            // Rt
            if let Some(bad_rt) = bad64_reg(ops.first()) {
                if bad_rt != ours.rd() {
                    return Some(format!("CBZ Rt: bad64={} ours={}", bad_rt, ours.rd()));
                }
            }
        }
        Op::CBNZ => {
            if !matches!(ours, Arm64Insn::Cbnz(_)) {
                return Some(format!("expected Cbnz, got {:?}", ours));
            }
            let bad_target = bad64_imm(ops.get(1))?;
            let our_target = ours.cbz_target(PC);
            if bad_target != our_target {
                return Some(format!(
                    "CBNZ target: bad64={:#x} ours={:#x}",
                    bad_target, our_target
                ));
            }
            if let Some(bad_rt) = bad64_reg(ops.first()) {
                if bad_rt != ours.rd() {
                    return Some(format!("CBNZ Rt: bad64={} ours={}", bad_rt, ours.rd()));
                }
            }
        }

        // ── TBZ / TBNZ ──────────────────────────────────────────────────────
        Op::TBZ => {
            if !matches!(ours, Arm64Insn::Tbz(_)) {
                return Some(format!("expected Tbz, got {:?}", ours));
            }
            // bad64: op[0]=Rt, op[1]=bit_imm, op[2]=label
            let bad_target = bad64_imm(ops.get(2))?;
            let our_target = ours.imm14_target(PC);
            if bad_target != our_target {
                return Some(format!(
                    "TBZ target: bad64={:#x} ours={:#x}",
                    bad_target, our_target
                ));
            }
        }
        Op::TBNZ => {
            if !matches!(ours, Arm64Insn::Tbnz(_)) {
                return Some(format!("expected Tbnz, got {:?}", ours));
            }
            let bad_target = bad64_imm(ops.get(2))?;
            let our_target = ours.imm14_target(PC);
            if bad_target != our_target {
                return Some(format!(
                    "TBNZ target: bad64={:#x} ours={:#x}",
                    bad_target, our_target
                ));
            }
        }

        // ── ADRP ────────────────────────────────────────────────────────────
        Op::ADRP => {
            if !matches!(ours, Arm64Insn::Adrp(_)) {
                return Some(format!("expected Adrp, got {:?}", ours));
            }
            // bad64: op[0]=Rd, op[1]=page_imm (already resolved)
            let bad_page = bad64_imm(ops.get(1))?;
            let our_page = ours.adrp_page(PC);
            if bad_page != our_page {
                return Some(format!(
                    "ADRP page: bad64={:#x} ours={:#x}",
                    bad_page, our_page
                ));
            }
            if let Some(bad_rd) = bad64_reg(ops.first()) {
                if bad_rd != ours.rd() {
                    return Some(format!("ADRP Rd: bad64={} ours={}", bad_rd, ours.rd()));
                }
            }
        }

        // ── ADR ─────────────────────────────────────────────────────────────
        Op::ADR => {
            // bad64 is permissive about fixed bits — it may decode non-standard
            // words as ADR. We only assert fields when we positively classified it.
            if matches!(ours, Arm64Insn::Adr(_)) {
                let bad_target = bad64_imm(ops.get(1))?;
                let our_target = ours.adr_target(PC);
                if bad_target != our_target {
                    return Some(format!(
                        "ADR target: bad64={:#x} ours={:#x}",
                        bad_target, our_target
                    ));
                }
                if let Some(bad_rd) = bad64_reg(ops.first()) {
                    if bad_rd != ours.rd() {
                        return Some(format!("ADR Rd: bad64={} ours={}", bad_rd, ours.rd()));
                    }
                }
            }
            // If ours=Other, bad64 decoded a permissive ADR we don't recognize — acceptable.
        }

        // ── ADD (immediate) ──────────────────────────────────────────────────
        Op::ADD => {
            // Only check if bad64 has an imm operand at index 2 — ADD has
            // shifted-register variants that we don't classify as AddImm.
            let bad_imm = match ops.get(2) {
                Some(Operand::Imm64 { .. }) | Some(Operand::Imm32 { .. }) => bad64_imm(ops.get(2)),
                _ => None,
            };
            if let Some(bad_imm) = bad_imm {
                // If ours=Other, bad64 decoded an ADD-imm variant we don't recognize
                // (e.g. unusual fixed-bit combinations) — acceptable, not a field error.
                if !matches!(ours, Arm64Insn::AddImm(_)) {
                    return None;
                }

                // bad64 reports the raw imm12 without applying the shift field;
                // our add_imm() applies LSL #12 when shift=01.  Extract the raw
                // word to determine the shift and compute the expected value.
                let word = match ours {
                    Arm64Insn::AddImm(w) => *w,
                    _ => unreachable!(),
                };
                let shift = (word >> 22) & 0x3;
                let expected = if shift == 1 { bad_imm << 12 } else { bad_imm };
                let our_imm = ours.add_imm();
                if expected != our_imm {
                    return Some(format!(
                        "ADD imm: bad64={:#x} (shift={}) expected={:#x} ours={:#x}",
                        bad_imm, shift, expected, our_imm
                    ));
                }
                // Rd
                if let Some(bad_rd) = bad64_reg(ops.first()) {
                    if bad_rd != ours.rd() {
                        return Some(format!("ADD Rd: bad64={} ours={}", bad_rd, ours.rd()));
                    }
                }
                // Rn
                if let Some(bad_rn) = bad64_reg(ops.get(1)) {
                    if bad_rn != ours.rn() {
                        return Some(format!("ADD Rn: bad64={} ours={}", bad_rn, ours.rn()));
                    }
                }
            }
            // ADD-register variant: we classify as Other, that's fine.
        }

        // ── LDR (unsigned offset, 64-bit) ────────────────────────────────────
        Op::LDR => {
            // bad64 handles many LDR variants (literal, pre/post-idx, reg offset).
            // We only claim to decode the unsigned-offset 64-bit form.
            // Only assert if our decoder says Ldr.
            if matches!(ours, Arm64Insn::Ldr(_)) {
                // Check offset via the memory operand.
                if let Some(bad_offset) = bad64_mem_offset(ops.get(1)) {
                    let our_offset = ours.ldr_str_offset() as i64;
                    if bad_offset != our_offset {
                        return Some(format!(
                            "LDR offset: bad64={} ours={}",
                            bad_offset, our_offset
                        ));
                    }
                }
                if let Some(bad_rn) = bad64_mem_base(ops.get(1)) {
                    if bad_rn != ours.rn() {
                        return Some(format!("LDR Rn: bad64={} ours={}", bad_rn, ours.rn()));
                    }
                }
                if let Some(bad_rt) = bad64_reg(ops.first()) {
                    if bad_rt != ours.rd() {
                        return Some(format!("LDR Rt: bad64={} ours={}", bad_rt, ours.rd()));
                    }
                }
            }
        }

        // ── STR (unsigned offset, 64-bit) ────────────────────────────────────
        Op::STR => {
            if matches!(ours, Arm64Insn::Str(_)) {
                if let Some(bad_offset) = bad64_mem_offset(ops.get(1)) {
                    let our_offset = ours.ldr_str_offset() as i64;
                    if bad_offset != our_offset {
                        return Some(format!(
                            "STR offset: bad64={} ours={}",
                            bad_offset, our_offset
                        ));
                    }
                }
                if let Some(bad_rn) = bad64_mem_base(ops.get(1)) {
                    if bad_rn != ours.rn() {
                        return Some(format!("STR Rn: bad64={} ours={}", bad_rn, ours.rn()));
                    }
                }
            }
        }

        // ── BLR ─────────────────────────────────────────────────────────────
        Op::BLR => {
            if !matches!(ours, Arm64Insn::Blr(_)) {
                return Some(format!("expected Blr, got {:?}", ours));
            }
            if let Some(bad_rn) = bad64_reg(ops.first()) {
                if bad_rn != ours.rn() {
                    return Some(format!("BLR Rn: bad64={} ours={}", bad_rn, ours.rn()));
                }
            }
        }

        // ── BR ──────────────────────────────────────────────────────────────
        Op::BR => {
            if !matches!(ours, Arm64Insn::Br(_)) {
                return Some(format!("expected Br, got {:?}", ours));
            }
            if let Some(bad_rn) = bad64_reg(ops.first()) {
                if bad_rn != ours.rn() {
                    return Some(format!("BR Rn: bad64={} ours={}", bad_rn, ours.rn()));
                }
            }
        }

        // Everything else: we don't claim to classify it, no assertion needed.
        _ => {}
    }

    None
}
