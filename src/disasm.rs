//! Disassembly context helpers for the `xr` CLI.
//!
//! Given a virtual address and a loaded binary, produces a window of
//! disassembled instructions centred on that address — similar to
//! `grep -A / -B` for source code.

use crate::loader::{Arch, Segment};
use crate::va::Va;

/// A single disassembled instruction line.
#[derive(Debug)]
pub struct DisasmLine {
    /// Virtual address of this instruction.
    pub va: u64,
    /// Raw bytes.
    pub bytes: Vec<u8>,
    /// Formatted text (Intel syntax for x86, standard for ARM64).
    pub text: String,
    /// True when this is the "focus" instruction (the xref site).
    pub is_focus: bool,
}

/// Decoded instruction data before focus/context tagging.
/// Used internally to accumulate instructions before building the final
/// `DisasmLine` list.
struct DecodedInsn {
    va: u64,
    bytes: Vec<u8>,
    text: String,
}

/// Disassemble a window of instructions around `focus_va`.
///
/// Returns up to `before + 1 + after` lines centred on the instruction at
/// `focus_va`. Lines are in address order; the focus line has `is_focus=true`.
/// Returns an empty vec if `focus_va` is not in any segment or the arch is
/// unsupported.
pub fn context(
    arch: Arch,
    segments: &[Segment],
    focus_va: Va,
    before: usize,
    after: usize,
) -> Vec<DisasmLine> {
    let seg = match segments.iter().find(|s| s.contains(focus_va)) {
        Some(s) => s,
        None => return vec![],
    };

    // Only disassemble executable segments; return empty for data so callers
    // can fall back to a hex dump.
    if !seg.executable {
        return vec![];
    }

    let focus_raw = focus_va.raw();
    match arch {
        Arch::X86_64 | Arch::X86 => disasm_x86(arch, seg, focus_raw, before, after),
        Arch::Arm64 => disasm_arm64(seg, focus_raw, before, after),
        // ARM32 / Thumb disassembly is not yet implemented.
        // Callers fall back to a hex dump via `ContextLine::data` when this
        // returns an empty vec.
        _ => vec![],
    }
}

// ── x86 / x86-64 ─────────────────────────────────────────────────────────────

fn disasm_x86(
    arch: Arch,
    seg: &Segment,
    focus_va: u64,
    before: usize,
    after: usize,
) -> Vec<DisasmLine> {
    use iced_x86::{
        Decoder, DecoderOptions, Formatter, FormatterOutput, FormatterTextKind, Instruction,
        IntelFormatter,
    };

    struct Buf(String);
    impl FormatterOutput for Buf {
        fn write(&mut self, text: &str, _kind: FormatterTextKind) {
            self.0.push_str(text);
        }
    }

    let bitness: u32 = if arch == Arch::X86_64 { 64 } else { 32 };

    let make_fmt = || {
        let mut fmt = IntelFormatter::new();
        fmt.options_mut().set_uppercase_keywords(false);
        fmt.options_mut().set_space_after_operand_separator(true);
        fmt
    };

    let seg_offset = (focus_va - seg.va.raw()) as usize;

    // ── Step 1: anchor-decode focus + after context from focus_va ────────────
    let anchor_end_off = (seg_offset + (after + 2) * 15).min(seg.data().len());
    let anchor_data = &seg.data()[seg_offset..anchor_end_off];
    let mut anchor_dec = Decoder::with_ip(bitness, anchor_data, focus_va, DecoderOptions::NONE);
    let mut anchor_fmt = make_fmt();

    let mut anchor_insns: Vec<DecodedInsn> = Vec::new();
    while anchor_dec.can_decode() {
        let ip = anchor_dec.ip();
        if anchor_insns.len() > after + 1 {
            break;
        }
        let insn = anchor_dec.decode();
        let len = insn.len();
        let off = (ip - focus_va) as usize;
        let raw = anchor_data[off..off + len].to_vec();
        let mut buf = Buf(String::new());
        anchor_fmt.format(&insn, &mut buf);
        anchor_insns.push(DecodedInsn {
            va: ip,
            bytes: raw,
            text: buf.0,
        });
    }

    // Focus must be the first anchor instruction.
    if anchor_insns.is_empty() || anchor_insns[0].va != focus_va {
        return vec![];
    }

    // ── Step 2: reconstruct before-context via backward linear probing ───────
    //
    // For each of the `before` slots, find the instruction that ends exactly at
    // the current "cursor" (starts at cursor - len, has length len).
    //
    // We try each possible instruction length 1..=15. A length is valid if
    // a probe starting at (cursor - len) decodes ONE instruction of exactly
    // `len` bytes landing on `cursor`.
    //
    // Among valid lengths for a given slot, we prefer the one that is also
    // reachable from the farthest-back probe (maximizes context depth). We
    // implement this by trying lengths in order and keeping the first hit, which
    // corresponds to trying large lengths first (they reach farther back and
    // are more likely to be the true boundary in well-aligned code).
    //
    // To handle cases where multiple lengths are valid (e.g., a 1-byte NOP is
    // always valid), we choose the LARGEST valid length (the instruction that
    // covers the most bytes before the cursor), which corresponds to the unique
    // true instruction boundary in well-formed x86 code.

    const MAX_X86_INSN: usize = 15;

    let mut before_context: Vec<DecodedInsn> = Vec::with_capacity(before);
    let mut cursor_off = seg_offset; // current "right edge" we're scanning back from

    for _ in 0..before {
        if cursor_off == 0 {
            break;
        }

        // Try all lengths from MAX down to 1; keep the largest valid one.
        // Cache the decoded Instruction (Copy type) so the match arm below can
        // format it directly without a second decode pass.
        let mut best: Option<(usize, Instruction)> = None;
        let max_try = cursor_off.min(MAX_X86_INSN);

        for try_len in (1..=max_try).rev() {
            let probe_off = cursor_off - try_len;
            let probe_va = seg.va + probe_off as u64;
            // Decode exactly one instruction from probe_off.
            let end_off = (probe_off + MAX_X86_INSN).min(seg.data().len());
            let slice = &seg.data()[probe_off..end_off];
            let mut dec = Decoder::with_ip(bitness, slice, probe_va.raw(), DecoderOptions::NONE);
            if !dec.can_decode() {
                continue;
            }
            let insn = dec.decode();
            if insn.len() == try_len {
                // This length is consistent: a `try_len`-byte instruction at probe_off
                // ends exactly at cursor_off.
                best = Some((try_len, insn));
                break; // largest valid length found
            }
        }

        match best {
            None => break, // no valid instruction found — stop
            Some((len, insn)) => {
                let probe_off = cursor_off - len;
                let probe_va = seg.va + probe_off as u64;
                let raw = seg.data()[probe_off..probe_off + len].to_vec();
                let mut fmt = make_fmt();
                let mut buf = Buf(String::new());
                fmt.format(&insn, &mut buf);
                before_context.push(DecodedInsn {
                    va: probe_va.raw(),
                    bytes: raw,
                    text: buf.0,
                });
                cursor_off = probe_off;
            }
        }
    }

    // before_context is in reverse order (last decoded = earliest instruction).
    before_context.reverse();

    // ── Step 3: combine before + focus + after ────────────────────────────────
    let after_count = anchor_insns.len().min(after + 1);
    let mut result: Vec<DisasmLine> = Vec::with_capacity(before_context.len() + after_count);
    for d in before_context {
        result.push(DisasmLine {
            va: d.va,
            bytes: d.bytes,
            text: d.text,
            is_focus: false,
        });
    }
    // Drain anchor_insns: first element is focus, remainder is after-context.
    let mut anchor_iter = anchor_insns.into_iter().take(after_count);
    if let Some(focus) = anchor_iter.next() {
        result.push(DisasmLine {
            va: focus.va,
            bytes: focus.bytes,
            text: focus.text,
            is_focus: true,
        });
    }
    for d in anchor_iter {
        result.push(DisasmLine {
            va: d.va,
            bytes: d.bytes,
            text: d.text,
            is_focus: false,
        });
    }
    result
}

// ── AArch64 ──────────────────────────────────────────────────────────────────

fn disasm_arm64(seg: &Segment, focus_va: u64, before: usize, after: usize) -> Vec<DisasmLine> {
    // AArch64 has fixed 4-byte instructions — no alignment ambiguity.
    let seg_offset = (focus_va - seg.va.raw()) as usize;

    // Align start to 4-byte boundary.
    let scan_back_bytes = before * 4;
    let scan_start_off = seg_offset.saturating_sub(scan_back_bytes) & !3;
    let scan_start_va = seg.va + scan_start_off as u64;

    let total = before + 1 + after;
    let scan_end_off = (scan_start_off + total * 4).min(seg.data().len());
    let data = &seg.data()[scan_start_off..scan_end_off];

    let mut all: Vec<DecodedInsn> = Vec::new();
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        let va = (scan_start_va + (i * 4) as u64).raw();
        let word = u32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)"));
        let text = match bad64::decode(word, va) {
            Ok(insn) => insn.to_string(),
            Err(_) => format!(".word 0x{word:08x}"),
        };
        all.push(DecodedInsn {
            va,
            bytes: chunk.to_vec(),
            text,
        });
    }

    build_window(all, focus_va, before, after)
}

// ── Shared window builder ─────────────────────────────────────────────────────

fn build_window(
    all: Vec<DecodedInsn>,
    focus_va: u64,
    before: usize,
    after: usize,
) -> Vec<DisasmLine> {
    let focus_idx = match all.iter().position(|d| d.va == focus_va) {
        Some(i) => i,
        // focus_va not cleanly decoded (e.g. mid-instruction on x86) — no output.
        None => return vec![],
    };

    let start = focus_idx.saturating_sub(before);
    let end = (focus_idx + after + 1).min(all.len());

    all.into_iter()
        .skip(start)
        .take(end - start)
        .map(|d| DisasmLine {
            is_focus: d.va == focus_va,
            va: d.va,
            bytes: d.bytes,
            text: d.text,
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{Arch, SegData, Segment, SegmentArch};

    fn exec_seg(va: u64, data: &'static [u8]) -> Segment {
        Segment {
            va: crate::va::Va::new(va),
            data: unsafe { SegData::new_for_test(data) },
            executable: true,
            readable: true,
            writable: false,
            byte_scannable: false,
            arch: SegmentArch::Generic,
            name: "test".to_string(),
        }
    }

    fn vas(lines: &[DisasmLine]) -> Vec<u64> {
        lines.iter().map(|l| l.va).collect()
    }

    fn focus(lines: &[DisasmLine]) -> Option<&DisasmLine> {
        lines.iter().find(|l| l.is_focus)
    }

    // ── focus always present ───────────────────────────────────────────────────

    /// Even when before=0 and after=0, the focus instruction must be returned.
    #[test]
    fn test_focus_only_no_context() {
        static CODE: [u8; 5] = [0xe8, 0xfb, 0x0f, 0x00, 0x00];
        let seg = exec_seg(0x1000, &CODE);
        let lines = disasm_x86(Arch::X86_64, &seg, 0x1000, 0, 0);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].va, 0x1000);
        assert!(lines[0].is_focus);
    }

    // ── clean alignment — before context correct ───────────────────────────────

    /// Three NOPs followed by CALL rel32 at 0x1003. With before=3, all three
    /// NOPs should appear before the focus.
    #[test]
    fn test_clean_before_context() {
        static CODE: [u8; 8] = [0x90, 0x90, 0x90, 0xe8, 0xf8, 0x0f, 0x00, 0x00];
        let seg = exec_seg(0x1000, &CODE);
        let lines = disasm_x86(Arch::X86_64, &seg, 0x1003, 3, 0);
        let foc = focus(&lines).expect("focus must be present");
        assert_eq!(foc.va, 0x1003);
        let before: Vec<_> = lines.iter().filter(|l| l.va < 0x1003).collect();
        assert_eq!(
            before.len(),
            3,
            "expected 3 before-context lines, got {before:?}"
        );
        assert_eq!(vas(&lines), vec![0x1000, 0x1001, 0x1002, 0x1003]);
    }

    // ── focus at segment start — no before possible ────────────────────────────

    #[test]
    fn test_focus_at_segment_start() {
        static CODE: [u8; 5] = [0xe8, 0xf8, 0x0f, 0x00, 0x00];
        let seg = exec_seg(0x1000, &CODE);
        let lines = disasm_x86(Arch::X86_64, &seg, 0x1000, 3, 0);
        assert!(!lines.is_empty(), "must return at least the focus line");
        let foc = focus(&lines).expect("focus must be present");
        assert_eq!(foc.va, 0x1000);
        assert!(lines.iter().all(|l| l.va >= 0x1000));
    }

    // ── fewer before-context lines available than requested ────────────────────

    #[test]
    fn test_fewer_before_than_requested() {
        static CODE: [u8; 6] = [0x90, 0xe8, 0xf4, 0x0e, 0x00, 0x00];
        let seg = exec_seg(0x1000, &CODE);
        let lines = disasm_x86(Arch::X86_64, &seg, 0x1001, 5, 0);
        let foc = focus(&lines).expect("focus must be present");
        assert_eq!(foc.va, 0x1001);
        let before_count = lines.iter().filter(|l| l.va < 0x1001).count();
        assert_eq!(before_count, 1, "only 1 instruction precedes focus");
    }

    // ── after context correct ──────────────────────────────────────────────────

    #[test]
    fn test_after_context() {
        static CODE: [u8; 7] = [0xe8, 0x00, 0x00, 0x00, 0x00, 0x90, 0x90];
        let seg = exec_seg(0x1000, &CODE);
        let lines = disasm_x86(Arch::X86_64, &seg, 0x1000, 0, 2);
        assert_eq!(vas(&lines), vec![0x1000, 0x1005, 0x1006]);
        assert!(lines[0].is_focus);
    }

    // ── misaligned: best probe wins (most before-context lines) ───────────────

    /// Tests the core misalignment bug: the single-probe approach starts at
    /// `focus_va - scan_back` which may be mid-instruction, causing it to decode
    /// the instruction immediately before the focus incorrectly.
    ///
    /// Layout (segment base 0x5000):
    ///   offsets 0x00..0x31: 50 zero bytes (padding)
    ///   offset  0x32..0x3b: 48 B8 90×8   MOV rax, imm64  (10 bytes) ← correct before
    ///   offset  0x3c..0x40: E8 00 00 00 00  CALL rel32 ← focus (VA 0x503c)
    ///
    /// with before=1, scan_back = (1+2)*15 = 45.
    /// probe_off = 0x3c - 45 = 0x3c - 0x2d = 0x0f  (inside zero-padding)
    ///
    /// From probe at 0x0f, decoding 00 00 pairs:
    ///   0x0f → 0x11 → 0x13 → ... → 0x31 (17 pairs = 34 bytes → probe+34=0x31)
    ///   then at 0x31: 00 48 → ADD [rax+0], cl? no: `00 48 B8` = ADD [rax-0x48], cl (3 bytes)?
    ///   Actually: `00 /r` with ModRM=0x48 (mod=01, reg=1, rm=0) → 3-byte with disp8 → skips ahead
    ///   Either way, the stream will NOT land cleanly on the MOV at 0x32 in general.
    ///
    /// The correct probe starting at 0x32 decodes: MOV (10 bytes) → CALL at 0x3c.
    /// A multi-probe search trying offsets 0x3b, 0x3a, ..., 0x32 will find that
    /// the probe starting at 0x32 lands on focus_va=0x503c, and yields before=[MOV@0x5032].
    ///
    /// The current single-probe implementation returns whatever the drifted probe
    /// from 0x0f lands on — which may be a NOP or a wrong instruction at 0x503b,
    /// not the MOV at 0x5032.
    #[test]
    fn test_misaligned_best_probe_wins() {
        // 50 zero bytes of padding, then 10-byte MOV, then CALL focus
        let mut code = [0x00u8; 65];
        // MOV rax, imm64 at offset 0x32 (50 decimal)
        code[0x32] = 0x48;
        code[0x33] = 0xb8;
        code[0x34] = 0x90;
        code[0x35] = 0x90;
        code[0x36] = 0x90;
        code[0x37] = 0x90;
        code[0x38] = 0x90;
        code[0x39] = 0x90;
        code[0x3a] = 0x90;
        code[0x3b] = 0x90;
        // CALL rel32 at offset 0x3c (focus)
        code[0x3c] = 0xe8;
        code[0x3d] = 0x00;
        code[0x3e] = 0x00;
        code[0x3f] = 0x00;
        code[0x40] = 0x00;

        let data: &'static [u8] = Box::leak(code.into());
        let seg = exec_seg(0x5000, data);
        let focus_va = 0x503c_u64; // seg.va + 0x3c

        let lines = disasm_x86(Arch::X86_64, &seg, focus_va, 1, 0);
        let foc = focus(&lines).expect("focus must be present");
        assert_eq!(foc.va, focus_va);
        assert_eq!(foc.bytes[0], 0xe8, "focus must be the CALL");

        let before: Vec<_> = lines.iter().filter(|l| l.va < focus_va).collect();
        assert_eq!(
            before.len(),
            1,
            "expected exactly 1 before-context line, got {:?}",
            before.iter().map(|l| (l.va, &l.bytes)).collect::<Vec<_>>()
        );
        assert_eq!(
            before[0].va, 0x5032,
            "before line must be the MOV at 0x5032, not {:?}",
            before[0].va
        );
        assert_eq!(
            before[0].bytes.len(),
            10,
            "before line must be the full 10-byte MOV"
        );
    }

    // ── after context unaffected by misalignment ───────────────────────────────

    #[test]
    fn test_after_context_correct_despite_misalignment() {
        static CODE: [u8; 20] = [
            0x48, 0xb8, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0xe8, 0x00, 0x00, 0x00,
            0x00, // focus CALL
            0x90, // NOP after (0x500f)
            0x90, // NOP after (0x5010)
            0x90, 0x90, 0x90, // padding
        ];
        let seg = exec_seg(0x5000, &CODE);
        let lines = disasm_x86(Arch::X86_64, &seg, 0x500a, 0, 2);
        let foc = focus(&lines).expect("focus must be present");
        assert_eq!(foc.va, 0x500a);
        let after: Vec<_> = lines.iter().filter(|l| l.va > 0x500a).collect();
        assert_eq!(after.len(), 2);
        assert_eq!(after[0].va, 0x500f);
        assert_eq!(after[1].va, 0x5010);
    }

    // ── focus bytes are correct (anchor decode, not probe decode) ─────────────

    /// The MOV's imm bytes (0x90...) look like NOP opcodes to a misaligned probe.
    /// The CALL at 0x500a must report bytes = [E8 00 00 00 00].
    #[test]
    fn test_focus_bytes_are_anchor_decoded() {
        static CODE: [u8; 15] = [
            0x48, 0xb8, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0xe8, 0x00, 0x00, 0x00,
            0x00,
        ];
        let seg = exec_seg(0x5000, &CODE);
        let lines = disasm_x86(Arch::X86_64, &seg, 0x500a, 0, 0);
        let foc = focus(&lines).expect("focus must be present");
        assert_eq!(foc.va, 0x500a);
        assert_eq!(
            foc.bytes,
            vec![0xe8, 0x00, 0x00, 0x00, 0x00],
            "focus bytes must be the full 5-byte CALL encoding"
        );
    }
}
