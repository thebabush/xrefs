/// ARM32 / Thumb-2 / Data mode predictor.
///
/// Depth-6 decision tree trained on ~2 M balanced samples from 59 ARM32 ELF
/// shared libraries (Debian armhf + armel, Android NDK libamp).
/// Evaluated on libamp.so (mixed ARM32/NEON/Thumb): 97 % overall accuracy,
/// 98.8 % ARM32 recall, 98.1 % Thumb recall.
///
/// Input:  8-byte lookahead window starting at the current 4-byte-aligned
///         position.  Zero-padded at section end.
/// Output: [`ArmMode`]
///
/// # Key features
/// - `b[3]` — bits 31:24 of the current 4-byte word.
///   For ARM32 this is the condition code byte (0xE? = "always", 0xF? = VFP/NEON).
///   For Thumb this is the high byte of the second 16-bit halfword.
/// - `b[7]` — bits 31:24 of the *next* 4-byte word.
///   The crucial run-detection signal: ARM32 regions have consecutive high-MSB
///   words; an isolated high MSB is more likely a Thumb-2 second halfword.
///
/// # Usage
/// Apply at every 4-byte-aligned offset in an executable section.  Feed
/// predictions through a small hysteresis filter (require H ≥ 2 consecutive
/// identical predictions before committing to a mode change) to absorb the
/// single-word ambiguity at ARM32/Thumb boundaries.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmMode {
    Arm32,
    Thumb,
    Data,
}

/// Predict the instruction mode at `offset` within `section`.
///
/// `offset` must be 4-byte aligned.
/// `section` is the raw bytes of a single executable ELF section.
#[inline]
pub fn predict_mode(section: &[u8], offset: usize) -> ArmMode {
    use ArmMode::*;

    debug_assert!(offset.is_multiple_of(4), "offset must be 4-byte aligned");

    // Extract 8-byte lookahead window, zero-padded at end.
    let b = |i: usize| -> u8 { section.get(offset + i).copied().unwrap_or(0) };
    let (b0, b1, b2, b3) = (b(0), b(1), b(2), b(3));
    let (b4, b5, b6, b7) = (b(4), b(5), b(6), b(7));

    // b3 = MSB of current word  (ARM32 condition code / Thumb second-halfword high byte)
    // b7 = MSB of next word     (run confirmation)
    match b3 {
        // ── 0x00 ─────────────────────────────────────────────────────────────
        // ARM32 EQ-condition instructions (cond = 0x0) are vanishingly rare in
        // real code; the 0x00 MSB slot is almost entirely data or specific
        // Thumb-2 encodings where the second halfword's high byte is zero.
        0x00 => {
            if b1 <= 0xE7 {
                // First halfword is NOT a Thumb-2 32-bit opener (those need b1 >= 0xE8).
                // Almost entirely data; small Thumb escape for mid-range b2/b7 combos.
                if b2 <= 0x3F {
                    Data
                } else if b2 <= 0x97 {
                    if b7 > 0x4F && b0 <= 0x23 {
                        Thumb
                    } else {
                        Data
                    }
                } else if b7 > 0x06 && b1 > 0x1D {
                    Thumb
                } else {
                    Data
                }
            } else {
                // b1 in 0xE8..=0xFF — Thumb-2 32-bit first-halfword opener territory.
                if b7 == 0 {
                    // Next word also zero: narrow path for LDRD / load-multiple where
                    // the second halfword happens to start with 0x00.
                    if b1 == 0xE8 && b5 > 0xE7 && b4 > 0x81 {
                        Thumb
                    } else {
                        Data
                    }
                } else if b1 <= 0xF8 {
                    if (b5 > 0xE9 && b6 <= 0xFC) || (b1 == 0xF8 && b5 <= 0xE9) {
                        Thumb // Thumb-2 LDRD / LDM (94 %)
                    } else {
                        Data
                    }
                } else if b0 <= 0x0B {
                    Thumb
                } else {
                    Data
                }
            }
        }

        // ── 0x01..=0xDF ──────────────────────────────────────────────────────
        // MSB is below the ARM32 "always" (AL) condition range.
        // For ARM32: a conditional instruction (NE / CS / CC / MI / PL / ...).
        // For Thumb:  most Thumb-16 and Thumb-2 second-halfword high bytes.
        0x01..=0xDF => {
            if b1 <= 0x16 {
                // Bits 15:8 very low — the narrow slice where conditional ARM32
                // can be distinguished from Thumb by looking at the next word.
                if b7 <= 0xD9 {
                    // Next word is also not in the unconditional ARM32 range.
                    if b5 <= 0x17 {
                        if b7 > 0x09 {
                            Arm32
                        } else {
                            Data
                        }
                    } else if b0 > 0x11 {
                        Thumb
                    } else {
                        Data
                    }
                } else if b7 <= 0xEB {
                    // b7 in 0xDA..=0xEB: next word IS an unconditional ARM32 word.
                    // A conditional ARM32 instruction followed by an AL-condition one
                    // confirms we are in an ARM32 code run.
                    if b2 <= 0xD2 {
                        Arm32
                    } else {
                        Data
                    }
                } else if b2 == 0x00 {
                    Arm32
                } else {
                    Thumb
                }
            } else {
                // b1 in 0x17..=0xFF — bulk of the Thumb instruction space.
                if b2 <= 0xFE {
                    if b5 == 0x00 {
                        // Very specific ARM32 conditional encoding (88 %).
                        if b0 <= 0x10 {
                            Arm32
                        } else {
                            Data
                        }
                    } else {
                        Thumb
                    }
                } else {
                    // b2 == 0xFF: only a handful of ARM32 encodings reach here.
                    if b1 <= 0xF6 {
                        if b3 <= 0x1B {
                            Arm32
                        } else {
                            Thumb
                        }
                    } else {
                        Arm32 // b1 in 0xF7..=0xFF with b2=0xFF → ARM32 (86–100 %)
                    }
                }
            }
        }

        // ── 0xE0..=0xEB ──────────────────────────────────────────────────────
        // Classic ARM32 "always" (AL) condition code.
        // 0xE? MSB is the original E-flag heuristic; this is the core signal.
        // Condition 0xE (= 14) with instruction-class bits 27:24 in 0x0..=0xB.
        0xE0..=0xEB => {
            if b3 <= 0xE5 {
                // b3 in 0xE0..=0xE5 — data-processing, load/store, PUSH/POP (STMDB/LDMIA).
                // Very strong ARM32 signal; only bail if next word's MSB is unusually
                // high (> 0xF4), which suggests a Thumb-2 second halfword in b7.
                if b7 <= 0xF4 {
                    Arm32
                } else {
                    Thumb
                }
            } else {
                // b3 in 0xE6..=0xEB — branch (B/BL), load/store-register, coprocessor-adjacent.
                // Confirm with the next word: a run of 0xE? MSBs = ARM32.
                // Accept a zero next-word (end of section / before literal pool).
                if b7 == 0x00 || (0xE0..=0xEB).contains(&b7) {
                    Arm32
                } else {
                    Thumb
                }
            }
        }

        // ── 0xEC..=0xFE ──────────────────────────────────────────────────────
        // The NEON / VFP range — and the hardest disambiguation.
        //
        // ARM32: condition field = 0xF (bits 31:28 = 1111) marks unconditional
        //        VFP/NEON instructions.  Example encodings:
        //          0xF2430110  VADD.I32 q0, q1, q0
        //          0xF4624ADD  VLD2.32 {d18,d19}, [r13]
        //          0xF3C86A32  VEXT.8  q3, q4, q2, #2
        //        These appear in dense NEON kernels as long *runs* of 0xF? MSBs.
        //
        // Thumb: many Thumb-2 32-bit second halfwords have high bytes in 0xE8..=0xFF,
        //        but they appear in isolation — the next word's MSB is NOT high.
        //
        // Decision rule: check b7 (next word's MSB).
        //   Isolated high MSB (b7 < 0xE0) → Thumb second halfword.
        //   Run of high MSBs  (b7 >= 0xE0) → NEON/VFP ARM32 (for b3 in 0xEC..=0xF4).
        0xEC..=0xFE => {
            if b7 >= 0xE0 {
                // Next word also has a high MSB — characteristic of a NEON run.
                if b3 <= 0xF4 {
                    Arm32 // 0xEC..=0xF4 with high-MSB run = NEON/VFP (61 %)
                } else {
                    Thumb // 0xF5..=0xFE: this sub-range is overwhelmingly Thumb (99 %)
                }
            } else {
                // Isolated high MSB — no run → Thumb second halfword.
                // Rare exception: b6 == 0xFF triggers an ARM32 leaf (59 %, low conf).
                if b6 == 0xFF {
                    Arm32
                } else {
                    Thumb
                }
            }
        }

        // ── 0xFF ─────────────────────────────────────────────────────────────
        // No ARM32 condition code uses 0xFF.  This is padding, undefined-instruction
        // trap words (0xFFFFFFFF / 0xDEFExxxx), or literal pool constants.
        0xFF => {
            if b2 > 0xF6 {
                Data
            } else if b1 > 0x90 {
                Thumb
            } else {
                Data
            }
        }
    }
}

/// Hysteresis wrapper: only commit to a new mode after `h` consecutive
/// identical raw predictions.  Returns `None` until the first mode locks.
///
/// This absorbs the single-word ambiguity at ARM32/Thumb boundaries (where
/// the lookahead window straddles both modes) without introducing more than
/// `h` words of additional latency.
pub struct ModePredictor {
    current: Option<ArmMode>,
    pending: Option<ArmMode>,
    run: u8,
    h: u8,
}

impl ModePredictor {
    pub fn new(hysteresis: u8) -> Self {
        Self {
            current: None,
            pending: None,
            run: 0,
            h: hysteresis,
        }
    }

    /// Feed one raw prediction; returns the stable mode (if locked).
    pub fn push(&mut self, raw: ArmMode) -> Option<ArmMode> {
        match self.pending {
            Some(p) if p == raw => {
                self.run += 1;
                if self.run >= self.h {
                    self.current = Some(raw);
                    self.pending = None;
                    self.run = 0;
                }
            }
            _ => {
                self.pending = Some(raw);
                self.run = 1;
                if self.run >= self.h {
                    self.current = Some(raw);
                    self.pending = None;
                    self.run = 0;
                }
            }
        }
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm32_mov() {
        // E3A00001 = MOV r0, #1  (AL condition, data-processing immediate)
        // E3A01002 = MOV r1, #2
        // LE bytes: [0x01,0x00,0xA0,0xE3, 0x02,0x10,0xA0,0xE3]
        //           b3=0xE3 (0xE0..=0xEB arm32 range), b7=0xE3 → run → Arm32
        let code: &[u8] = &[0x01, 0x00, 0xA0, 0xE3, 0x02, 0x10, 0xA0, 0xE3];
        assert_eq!(predict_mode(code, 0), ArmMode::Arm32);
    }

    #[test]
    fn arm32_push_stmdb() {
        // E92D4810 = STMDB r13!, {r4,r11,lr}  (ARM32 PUSH)
        // E24DD010 = SUB   r13, r13, #16
        // LE bytes: [0x10,0x48,0x2D,0xE9, 0x10,0xD0,0x4D,0xE2]
        //           b3=0xE9 (0xE6..=0xEB), b7=0xE2 (in 0xE0..=0xEB) → run → Arm32
        let code: &[u8] = &[0x10, 0x48, 0x2D, 0xE9, 0x10, 0xD0, 0x4D, 0xE2];
        assert_eq!(predict_mode(code, 0), ArmMode::Arm32);
    }

    #[test]
    fn thumb16_push_nop() {
        // B510     = PUSH {r4, lr}      (Thumb-16)
        // 46C0     = MOV  r0, r0 (NOP)  (Thumb-16)
        // F000 F8XX = BL  target         (Thumb-2 32-bit, first HW = 0xF000)
        // LE layout at 4-byte offset:
        //   bytes 0..3: [0x10,0xB5, 0xC0,0x46]  b3=0x46 (0x01..=0xDF)
        //   bytes 4..7: [0x00,0xF0, 0xXX,0xFA]  b7=0xFA
        // b3 in 0x01..=0xDF, b1=0xB5 > 0x16, b2=0xC0 <= 0xFE, b5=0xF0 > 0 → Thumb
        let code: &[u8] = &[0x10, 0xB5, 0xC0, 0x46, 0x00, 0xF0, 0x10, 0xFA];
        assert_eq!(predict_mode(code, 0), ArmMode::Thumb);
    }

    #[test]
    fn thumb2_push_w() {
        // Thumb-2 PUSH.W {r4, r11, lr} = two halfwords: 0xE92D / 0x4810
        // LE storage: first HW [0x2D,0xE9], second HW [0x10,0x48]
        // → 4 bytes: [0x2D,0xE9,0x10,0x48]  b3=0x48
        // next word: Thumb-2 SUB.W sp,sp,#16 = 0xB082 → [0x82,0xB0,0xC0,0x46]
        // b3=0x48 in 0x01..=0xDF, b1=0xE9 > 0x16, b5=0xB0 > 0 → Thumb
        let code: &[u8] = &[0x2D, 0xE9, 0x10, 0x48, 0x82, 0xB0, 0xC0, 0x46];
        assert_eq!(predict_mode(code, 0), ArmMode::Thumb);
    }

    #[test]
    fn neon_vadd_run() {
        // F2430110 = VADD.I32 q0, q1, q0   (ARM32 NEON, unconditional cond=0xF)
        // F3C86A32 = VEXT.8   q3, q4, q2, #2
        // LE: [0x10,0x01,0x43,0xF2, 0x32,0x6A,0xC8,0xF3]
        //      b3=0xF2 (0xEC..=0xFE), b7=0xF3 (>= 0xE0) → NEON run, b3<=0xF4 → Arm32
        let code: &[u8] = &[0x10, 0x01, 0x43, 0xF2, 0x32, 0x6A, 0xC8, 0xF3];
        assert_eq!(predict_mode(code, 0), ArmMode::Arm32);
    }

    #[test]
    fn neon_isolated_looks_like_thumb() {
        // Single NEON word followed by a low-MSB Thumb word:
        // the model correctly treats the isolated 0xF? as a Thumb second halfword.
        // b3=0xF2, b7=0x46 (low, < 0xE0) → Thumb
        let code: &[u8] = &[0x10, 0x01, 0x43, 0xF2, 0x10, 0xB5, 0xC0, 0x46];
        assert_eq!(predict_mode(code, 0), ArmMode::Thumb);
    }

    #[test]
    fn all_zeros_is_data() {
        let code: &[u8] = &[0x00; 8];
        assert_eq!(predict_mode(code, 0), ArmMode::Data);
    }

    #[test]
    fn hysteresis_suppresses_single_blip() {
        let mut pred = ModePredictor::new(2);
        pred.push(ArmMode::Arm32);
        assert_eq!(pred.push(ArmMode::Arm32), Some(ArmMode::Arm32));
        // Single Thumb blip — should not switch
        pred.push(ArmMode::Thumb);
        assert_eq!(pred.push(ArmMode::Arm32), Some(ArmMode::Arm32));
        // Two consecutive Thumb — should switch
        pred.push(ArmMode::Thumb);
        assert_eq!(pred.push(ArmMode::Thumb), Some(ArmMode::Thumb));
    }
}
