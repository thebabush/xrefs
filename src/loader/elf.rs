use super::{alloc_bss, ParseResult, SegData, Segment, Symbol};
use crate::loader::{Arch, Arm32Segment, DecodeMode, ModeSwitch, RelocPointer, SegmentArch};
use crate::va::Va;
use anyhow::Result;
use rustc_hash::{FxHashMap, FxHashSet};

/// Vote over the first [`PROBE_WORDS`] 4-byte-aligned words of an executable
/// section to decide between ARM32 and Thumb mode.
///
/// ## Why voting works
///
/// ARM32 instructions encode a condition code in bits\[31:28\].  The "always"
/// condition (AL = `0xE`) covers virtually every unconditional instruction, so
/// a typical ARM32 function prologue has ≥ 70 % of its words with top nibble
/// `0xE`.  Thumb code read as 32-bit LE words has top nibble `0xE` only for
/// 16-bit `B T2` instructions sitting in the high halfword of an aligned pair —
/// empirically ≤ 6 % of all Thumb words in our test suite.
///
/// ## Why we cap at 16 words (64 bytes)
///
/// ARM32 PIC armel shared libraries embed literal-pool data *between* functions
/// — GOT offsets, AES S-boxes, lookup tables — all with uniformly distributed
/// nibbles.  Sampling beyond the first prologue dilutes the ARM32 signal until
/// it falls to Thumb-like levels:
///
/// | section                | n=16 | n=64 | n=256 |
/// |------------------------|------|------|-------|
/// | ssl-rand `.text` (A32) | 0.81 | 0.34 | 0.12  |
/// | armhf `.text`  (Thumb) | 0.06 | 0.17 | 0.12  |
///
/// At n=16, all A32 sections in our corpus score ≥ 0.73 and all Thumb sections
/// score ≤ 0.06 — a comfortable margin for a 50 % majority threshold.
///
/// Returns `Arm32` if strictly more than half of the sampled words have top
/// nibble `0xE`, and `Arm32` also when the evidence is ambiguous — ARM32 false
/// negatives (missed branches) are cheaper than Thumb false positives
/// (hundreds of spurious jumps from ARM32 words decoded as 16-bit halfwords).
const PROBE_WORDS: usize = 16;

fn probe_arm32_section_mode(file: &[u8], offset: usize, size: usize) -> DecodeMode {
    if size < 4 || offset + 4 > file.len() {
        return DecodeMode::Arm32;
    }
    let available = size.min(file.len().saturating_sub(offset));
    let n = (available / 4).min(PROBE_WORDS);

    let arm32_votes = (0..n)
        .filter(|&i| {
            let w =
                u32::from_le_bytes(file[offset + i * 4..offset + i * 4 + 4].try_into().unwrap());
            w >> 28 == 0xE
        })
        .count();

    // ARM32 is the default: decoding ARM32 as Thumb produces hundreds of
    // spurious branch xrefs (every 0xE... word's low halfword looks like B T2);
    // the reverse only misses some branches.  Require a clear Thumb majority
    // (more than half of sampled words with top nibble != 0xE) to override.
    if arm32_votes * 2 >= n {
        DecodeMode::Arm32
    } else {
        DecodeMode::Thumb
    }
}

/// Default PIE base for ET_DYN ELF binaries whose lowest PT_LOAD has `p_vaddr == 0`.
/// Matches the traditional Linux x86-64 / AArch64 PIE base and IDA default.
const DEFAULT_PIE_BASE: u64 = 0x0040_0000;

/// # Safety (internal)
///
/// `bytes` must remain valid for the lifetime of any `Segment` in the result.
/// Guaranteed by `LoadedBinary` field-ordering invariant.
pub(super) fn parse_elf(
    bytes: &[u8],
    elf: &goblin::elf::Elf,
    bss_bufs: &mut Vec<Box<[u8]>>,
    base_override: Option<u64>,
) -> Result<ParseResult> {
    let arch = match elf.header.e_machine {
        goblin::elf::header::EM_X86_64 => Arch::X86_64,
        goblin::elf::header::EM_AARCH64 => Arch::Arm64,
        goblin::elf::header::EM_386 => Arch::X86,
        goblin::elf::header::EM_ARM => Arch::Arm32,
        m => {
            eprintln!("warning: unknown ELF e_machine {m:#x}, treating as unknown");
            Arch::Unknown
        }
    };

    // Sections that should NOT be byte-scanned for pointers.
    // `.dynsym` byte-scan produces wrong xrefs: the byte scanner uses the
    // `st_value` field VA (+4 from entry start) as `from`, but IDA records
    // xrefs from the entry-start VA.  We enumerate `.dynsym` explicitly via
    // `build_elf_dynsym_pointers` which gets both the `from` and the
    // dynstr/st_value `to` targets right.
    const NO_SCAN_SECTIONS: &[&str] = &[".data.rel.ro", ".data.rel.ro.local", ".dynsym"];

    struct SectionInfo {
        va: u64,
        end: u64,
        file_offset: usize,
        file_size: usize,
        name: String,
        byte_scannable: bool,
        is_code: bool,
        /// Per-section decode mode — only meaningful for ARM32 ELF.
        /// Set by the sliding classifier; ignored for non-ARM32 segments.
        arm32_mode: DecodeMode,
        /// Intra-section ISA transitions from the classifier.
        /// Empty for non-ARM32 and for uniform sections.
        arm32_switches: Vec<(u64, DecodeMode)>,
    }
    let mut section_infos: Vec<SectionInfo> = Vec::new();
    for sh in &elf.section_headers {
        use goblin::elf::section_header::*;
        if sh.sh_type == SHT_NULL || sh.sh_type == SHT_NOBITS || sh.sh_addr == 0 || sh.sh_size == 0
        {
            continue;
        }
        if let Some(name) = elf.shdr_strtab.get_at(sh.sh_name) {
            // Use the ELF section flag SHF_EXECINSTR as the authoritative signal
            // for whether a section contains machine instructions.  A name-based
            // blacklist is fragile: Android binaries place .ARM.exidx, .dynsym,
            // .hash, .rel.dyn, and .rodata all inside the same R-X PT_LOAD, so
            // the old blacklist decoded symbol tables and relocation tables as
            // Thumb code, producing hundreds of thousands of false-positive jumps.
            let is_code = sh.sh_flags & (SHF_EXECINSTR as u64) != 0;
            section_infos.push(SectionInfo {
                va: sh.sh_addr,
                end: sh.sh_addr + sh.sh_size,
                file_offset: sh.sh_offset as usize,
                file_size: sh.sh_size as usize,
                name: name.to_string(),
                byte_scannable: !NO_SCAN_SECTIONS.contains(&name),
                is_code,
                arm32_mode: DecodeMode::Arm32, // filled in below; irrelevant for non-ARM32
                arm32_switches: vec![],        // filled in below for ARM32 sections
            });
        }
    }
    section_infos.sort_by_key(|s| s.va);

    // For ARM32 ELF, run the sliding classifier (H=2) over each code section to
    // produce a per-word mode sequence with intra-section transitions.
    // For all other architectures the fields are unused.
    if arch == Arch::Arm32 {
        for si in &mut section_infos {
            let switches = classify_section_mode(bytes, si.file_offset, si.file_size, si.va, 2);
            si.arm32_mode = switches[0].1;
            si.arm32_switches = switches.into_iter().skip(1).collect();
        }
    }

    use goblin::elf::header::ET_DYN;
    use goblin::elf::program_header::PT_LOAD;
    let pie_base: u64 = if elf.header.e_type == ET_DYN {
        let min_load_va = elf
            .program_headers
            .iter()
            .filter(|ph| ph.p_type == PT_LOAD)
            .map(|ph| ph.p_vaddr)
            .min()
            .unwrap_or(1);
        if min_load_va == 0 {
            base_override.unwrap_or(DEFAULT_PIE_BASE)
        } else {
            0
        }
    } else {
        0
    };

    if pie_base != 0 {
        for si in &mut section_infos {
            si.va += pie_base;
            si.end += pie_base;
            for (va, _) in &mut si.arm32_switches {
                *va += pie_base;
            }
        }
    }

    let mut segments = Vec::new();
    for ph in &elf.program_headers {
        use goblin::elf::program_header::*;
        if ph.p_type != PT_LOAD {
            continue;
        }
        let exec = ph.p_flags & PF_X != 0;
        let read = ph.p_flags & PF_R != 0;
        let write = ph.p_flags & PF_W != 0;
        let ph_va = ph.p_vaddr + pie_base;

        if exec && !section_infos.is_empty() {
            let ph_va_start = ph_va;
            let ph_va_end = ph_va + ph.p_memsz;
            let secs: Vec<&SectionInfo> = section_infos
                .iter()
                .filter(|s| s.va >= ph_va_start && s.end <= ph_va_end)
                .collect();
            if !secs.is_empty() {
                for sec in &secs {
                    if sec.file_offset + sec.file_size > bytes.len() {
                        eprintln!(
                            "warning: ELF section '{}' at offset {:#x}+{:#x} exceeds file size, skipping",
                            sec.name, sec.file_offset, sec.file_size
                        );
                        continue;
                    }
                    let data = &bytes[sec.file_offset..sec.file_offset + sec.file_size];
                    // Safety: `bytes` is the mmap kept alive by LoadedBinary.
                    segments.push(Segment {
                        va: Va::new(sec.va),
                        data: unsafe { SegData::new(data) },
                        executable: sec.is_code,
                        readable: read,
                        writable: write,
                        byte_scannable: sec.byte_scannable,
                        arch: if arch == Arch::Arm32 && sec.is_code {
                            let mut arm32 = Arm32Segment::uniform(sec.arm32_mode);
                            arm32.switches = sec
                                .arm32_switches
                                .iter()
                                .map(|&(va, mode)| ModeSwitch {
                                    va: Va::new(va),
                                    mode,
                                })
                                .collect();
                            SegmentArch::Arm32(arm32)
                        } else {
                            SegmentArch::Generic
                        },
                        name: sec.name.clone(),
                    });
                }
                let last_end = secs.iter().map(|s| s.end).max().unwrap_or(ph_va_end);
                if last_end < ph_va_end {
                    let bss_sz = (ph_va_end - last_end) as usize;
                    let bss_data = alloc_bss(bss_sz, bss_bufs);
                    segments.push(Segment {
                        va: Va::new(last_end),
                        data: bss_data,
                        executable: false,
                        readable: read,
                        writable: write,
                        byte_scannable: false,
                        arch: SegmentArch::Generic,
                        name: format!("BSS[{:#x}]", last_end),
                    });
                }
                continue;
            }
        }

        if ph.p_filesz > 0 {
            let offset = ph.p_offset as usize;
            let filesz = ph.p_filesz as usize;
            if offset + filesz <= bytes.len() {
                let data = &bytes[offset..offset + filesz];
                let ph_va_end = ph_va + ph.p_filesz;
                let byte_scannable = !section_infos
                    .iter()
                    .any(|s| !s.byte_scannable && s.va < ph_va_end && s.end > ph_va);
                // Safety: `bytes` is the mmap kept alive by LoadedBinary.
                segments.push(Segment {
                    va: Va::new(ph_va),
                    data: unsafe { SegData::new(data) },
                    executable: exec,
                    readable: read,
                    writable: write,
                    byte_scannable,
                    arch: if arch == Arch::Arm32 && exec {
                        // No covering section: classify this LOAD segment directly.
                        // ph_va is already rebased, so classifier VAs are correct.
                        let switches = classify_section_mode(bytes, offset, filesz, ph_va, 2);
                        let default_mode = switches[0].1;
                        let mut arm32 = Arm32Segment::uniform(default_mode);
                        arm32.switches = switches
                            .into_iter()
                            .skip(1)
                            .map(|(va, mode)| ModeSwitch {
                                va: Va::new(va),
                                mode,
                            })
                            .collect();
                        SegmentArch::Arm32(arm32)
                    } else {
                        SegmentArch::Generic
                    },
                    name: format!("LOAD[{:#x}]", ph_va),
                });
            }
        }

        if ph.p_memsz > ph.p_filesz {
            let bss_va = ph_va + ph.p_filesz;
            let bss_sz = (ph.p_memsz - ph.p_filesz) as usize;
            let bss_data = alloc_bss(bss_sz, bss_bufs);
            segments.push(Segment {
                va: Va::new(bss_va),
                data: bss_data,
                executable: false,
                readable: read,
                writable: write,
                byte_scannable: false,
                arch: SegmentArch::Generic,
                name: format!("BSS[{:#x}]", bss_va),
            });
        }
    }

    let entry_points = if elf.entry != 0 {
        vec![Va::new(elf.entry + pie_base)]
    } else {
        vec![]
    };
    let mut symbols = Vec::new();
    for sym in &elf.syms {
        if sym.st_value == 0 {
            continue;
        }
        if let Some(name) = elf.strtab.get_at(sym.st_name) {
            if !name.is_empty() {
                let va = Va::new((sym.st_value & !1) + pie_base);
                symbols.push(Symbol {
                    name: name.to_string(),
                    va,
                });
            }
        }
    }

    let got_slots = build_elf_got_slots(elf, pie_base);
    let slot_names = build_elf_got_slot_names(elf, pie_base);
    let stub_names = build_elf_plt_stub_names(elf, bytes, arch, pie_base, &slot_names);
    let extra_names = build_elf_extra_names(elf, pie_base, slot_names, stub_names);
    let mut reloc_pointers = build_elf_reloc_pointers(elf, bytes, pie_base, &segments);
    reloc_pointers.extend(build_elf_dynsym_pointers(elf, pie_base));

    Ok(ParseResult {
        arch,
        segments,
        entry_points,
        symbols,
        pie_base,
        got_slots,
        got_call_only: false,
        reloc_pointers,
        extra_names,
    })
}

/// Classify the ISA mode sequence of an ARM32 executable section using the
/// depth-6 decision tree with hysteresis `hyst` (use 2 for production).
///
/// Returns a non-empty list of `(va, mode)` transitions in address order.
/// The first entry's mode is the dominant ISA at the section start.
/// Subsequent entries mark confirmed mode switches within the section.
///
/// When the classifier never locks (section too short or ambiguous), falls
/// back to [`probe_arm32_section_mode`] so callers always get a result.
fn classify_section_mode(
    file: &[u8],
    offset: usize,
    size: usize,
    base_va: u64,
    hyst: u8,
) -> Vec<(u64, DecodeMode)> {
    use crate::arch::arm32_mode_classifier::{predict_mode, ArmMode, ModePredictor};

    if size < 4 || offset.saturating_add(size) > file.len() {
        return vec![(base_va, DecodeMode::Arm32)];
    }

    let data = &file[offset..offset + size];
    let mut pred = ModePredictor::new(hyst);
    let mut switches: Vec<(u64, DecodeMode)> = Vec::new();
    let mut current: Option<DecodeMode> = None;

    for word_idx in 0..(size / 4) {
        let off = word_idx * 4;
        let va = base_va + off as u64;

        let committed = pred.push(predict_mode(data, off));

        let mode = match committed {
            Some(ArmMode::Arm32) => DecodeMode::Arm32,
            Some(ArmMode::Thumb) => DecodeMode::Thumb,
            // Data predictions or no lock yet: hold current mode, no switch.
            Some(ArmMode::Data) | None => continue,
        };

        if current != Some(mode) {
            switches.push((va, mode));
            current = Some(mode);
        }
    }

    if switches.is_empty() {
        // Classifier never locked — fall back to the majority-vote probe.
        vec![(base_va, probe_arm32_section_mode(file, offset, size))]
    } else {
        switches
    }
}

/// Emit the three `data_ptr` xrefs IDA records for each `.dynsym` entry.
///
/// IDA parses `.dynsym` as a typed `Elf32_Sym` / `Elf64_Sym` array and records
/// `dr_O` (offset/pointer) xrefs from the **entry-start VA** — not from a
/// specific field offset — to three targets per defined symbol:
///
/// 1. `.dynstr` section base  — anchor ref that IDA records for every non-null entry
/// 2. `.dynstr + st_name`     — the symbol-name string VA
/// 3. `st_value & !1`         — the function/data address (defined symbols only)
///
/// Undefined symbols (st_shndx == 0) produce refs 1+2 only.
///
/// Why we can't rely on the byte scanner: in PIE ELF, `st_value` is stored
/// as a pre-rebase VMA (e.g. `0x0004b6d4`); after rebasing to `0x400000+`
/// the byte scanner rejects it as unmapped.  Explicit enumeration with
/// `+ pie_base` produces the correct rebased address.
///
/// `.dynsym` is listed in `NO_SCAN_SECTIONS` so the byte scanner never runs
/// on it (it would use `entry + field_offset` as `from`, not `entry_start`).
fn build_elf_dynsym_pointers(elf: &goblin::elf::Elf, pie_base: u64) -> Vec<RelocPointer> {
    use goblin::elf::section_header::SHT_DYNSYM;

    // Locate .dynsym section header — present even in stripped binaries.
    let dynsym_sh = match elf
        .section_headers
        .iter()
        .find(|sh| sh.sh_type == SHT_DYNSYM)
    {
        Some(sh) => sh,
        None => return vec![],
    };
    let dynsym_va = dynsym_sh.sh_addr;

    // .dynstr is identified by sh_link in the .dynsym section header.
    let dynstr_va = match elf.section_headers.get(dynsym_sh.sh_link as usize) {
        Some(sh) => sh.sh_addr,
        None => return vec![],
    };

    let entry_size: u64 = if elf.is_64 { 24 } else { 16 };

    let mut pointers = Vec::new();
    for (i, sym) in elf.dynsyms.iter().enumerate() {
        // Skip the mandatory null entry (index 0, all fields zero).
        if sym.st_name == 0 && sym.st_value == 0 && sym.st_shndx == 0 {
            continue;
        }

        let from = Va::new(dynsym_va + i as u64 * entry_size + pie_base);

        // 1. Ref to .dynstr base — IDA records this for every non-null entry.
        pointers.push(RelocPointer {
            from,
            to: Va::new(dynstr_va + pie_base),
        });

        // 2. Ref to the symbol-name string (skip when st_name=0 — same as base).
        if sym.st_name != 0 {
            pointers.push(RelocPointer {
                from,
                to: Va::new(dynstr_va + sym.st_name as u64 + pie_base),
            });
        }

        // 3. Ref to the symbol's value — only for DATA symbols (non-exec section).
        // IDA records dr_O to st_value only when the symbol lives in a non-exec
        // section (.data, .data.rel.ro, .bss, etc.).  For function symbols in
        // .text/.plt/.init (SHF_EXECINSTR), IDA does not emit a data_ptr xref.
        // SHN_UNDEF=0 and SHN_ABS/COMMON/XINDEX (0xff00+) are excluded.
        if sym.st_shndx != 0 && sym.st_shndx < 0xff00 && sym.st_value != 0 {
            let is_exec_section = elf
                .section_headers
                .get(sym.st_shndx)
                .is_some_and(|sh| sh.sh_flags & 0x4 != 0); // SHF_EXECINSTR = 0x4
            if !is_exec_section {
                // Clear Thumb interworking bit (ARM32 LSB=1 for Thumb entry points).
                pointers.push(RelocPointer {
                    from,
                    to: Va::new((sym.st_value & !1) + pie_base),
                });
            }
        }
    }
    pointers
}

/// Yield `(slot_va, sym_index)` for every GLOB_DAT / JUMP_SLOT relocation that
/// names a symbol.  Shared by the GOT-slot set and the GOT-slot name builder so
/// the reloc-type filter lives in one place.
fn got_relocs(
    relocs: impl Iterator<Item = goblin::elf::Reloc>,
    pie_base: u64,
) -> impl Iterator<Item = (Va, usize)> {
    const R_X86_64_GLOB_DAT: u32 = 6;
    const R_X86_64_JUMP_SLOT: u32 = 7;
    const R_AARCH64_GLOB_DAT: u32 = 1025;
    const R_AARCH64_JUMP_SLOT: u32 = 1026;
    const R_ARM_GLOB_DAT: u32 = 21;
    const R_ARM_JUMP_SLOT: u32 = 22;

    let is_got_reloc = |r_type: u32| {
        matches!(
            r_type,
            R_X86_64_GLOB_DAT
                | R_X86_64_JUMP_SLOT
                | R_AARCH64_GLOB_DAT
                | R_AARCH64_JUMP_SLOT
                | R_ARM_GLOB_DAT
                | R_ARM_JUMP_SLOT
        )
    };

    relocs
        .filter(move |rel| is_got_reloc(rel.r_type) && rel.r_sym != 0)
        .map(move |rel| (Va::new(rel.r_offset + pie_base), rel.r_sym))
}

fn all_dyn_relocs<'a>(elf: &'a goblin::elf::Elf) -> impl Iterator<Item = goblin::elf::Reloc> + 'a {
    elf.dynrelas
        .iter()
        .chain(elf.dynrels.iter())
        .chain(elf.pltrelocs.iter())
}

fn build_elf_got_slots(elf: &goblin::elf::Elf, pie_base: u64) -> FxHashSet<Va> {
    got_relocs(all_dyn_relocs(elf), pie_base)
        .map(|(slot, _)| slot)
        .collect()
}

/// Name every GOT slot after the dynamic symbol its relocation binds, so a
/// normalized `call [rip+got]` resolves to e.g. `printf`.  `name_of` maps a
/// `.dynsym` index to its raw name.
fn got_slot_names<'a>(
    slots: impl Iterator<Item = (Va, usize)>,
    name_of: impl Fn(usize) -> Option<&'a str>,
) -> Vec<Symbol> {
    slots
        .filter_map(|(va, idx)| {
            let name = name_of(idx).filter(|n| !n.is_empty())?;
            Some(Symbol {
                name: name.to_string(),
                va,
            })
        })
        .collect()
}

fn build_elf_got_slot_names(elf: &goblin::elf::Elf, pie_base: u64) -> Vec<Symbol> {
    got_slot_names(got_relocs(all_dyn_relocs(elf), pie_base), |idx| {
        let sym = elf.dynsyms.get(idx)?;
        elf.dynstrtab.get_at(sym.st_name)
    })
}

/// Names beyond `.symtab`: defined `.dynsym` symbols (the only names a stripped
/// binary has), GOT slots named after their imported symbol, and PLT stubs
/// named after the slot they jump through.
fn build_elf_extra_names(
    elf: &goblin::elf::Elf,
    pie_base: u64,
    slot_names: Vec<Symbol>,
    stub_names: Vec<Symbol>,
) -> Vec<Symbol> {
    use goblin::elf::section_header::SHN_UNDEF;

    let mut names = Vec::new();
    for sym in &elf.dynsyms {
        if sym.st_shndx == SHN_UNDEF as usize || sym.st_value == 0 {
            continue;
        }
        if let Some(name) = elf.dynstrtab.get_at(sym.st_name) {
            if !name.is_empty() {
                names.push(Symbol {
                    name: name.to_string(),
                    va: Va::new((sym.st_value & !1) + pie_base),
                });
            }
        }
    }

    names.extend(slot_names);
    names.extend(stub_names);
    names
}

/// Slot VA a x86-64 PLT entry jumps through: `[f3 0f 1e fa] [f2] ff 25 disp32`
/// (`jmp [rip+disp32]`, optionally after `endbr64` / `bnd`).  Anchoring at the
/// entry start skips PLT0, whose `ff 35` push comes first.
fn x86_64_plt_slot(entry: &[u8], entry_va: u64) -> Option<u64> {
    let mut i = 0;
    if entry.starts_with(&[0xf3, 0x0f, 0x1e, 0xfa]) {
        i += 4;
    }
    if entry.get(i) == Some(&0xf2) {
        i += 1;
    }
    if entry.get(i..i + 2)? != [0xff, 0x25] {
        return None;
    }
    let disp = i32::from_le_bytes(entry.get(i + 2..i + 6)?.try_into().ok()?);
    Some((entry_va + i as u64 + 6).wrapping_add(disp as i64 as u64))
}

/// Slot VA an AArch64 PLT entry loads: `adrp xN, page ; ldr xM, [xN, #imm]`
/// somewhere in the first three words (a leading `bti c` shifts them).
fn aarch64_plt_slot(entry: &[u8], entry_va: u64) -> Option<u64> {
    use crate::arch::arm64_decode::Arm64Insn;

    let words: Vec<u32> = entry
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    words.windows(2).enumerate().find_map(|(i, w)| {
        let (adrp, ldr) = (Arm64Insn::decode(w[0]), Arm64Insn::decode(w[1]));
        if !matches!((adrp, ldr), (Arm64Insn::Adrp(_), Arm64Insn::Ldr(_))) {
            return None;
        }
        // 64-bit load off the register the adrp just wrote.
        if ldr.ldr_str_size() != 3 || ldr.rn() != adrp.rd() {
            return None;
        }
        let pc = entry_va + i as u64 * 4;
        Some(adrp.adrp_page(pc) + ldr.ldr_str_offset())
    })
}

/// Slot VA a classic ARM32 PLT entry loads:
/// `add ip, pc, #imm ; add ip, ip, #imm ; ldr pc, [ip, #imm]!`.
fn arm32_plt_slot(entry: &[u8], entry_va: u64) -> Option<u64> {
    let w: Vec<u32> = entry
        .chunks_exact(4)
        .take(3)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let [a, b, c] = w[..] else { return None };
    // Shifter-operand immediate: imm8 rotated right by 2 * rot.
    let imm = |w: u32| (w & 0xff).rotate_right(((w >> 8) & 0xf) * 2) as u64;
    // add ip, pc, #imm / add ip, ip, #imm / ldr pc, [ip, #+imm12]!
    if a & 0xffff_f000 != 0xe28f_c000
        || b & 0xffff_f000 != 0xe28c_c000
        || c & 0xffff_f000 != 0xe5bc_f000
    {
        return None;
    }
    Some(entry_va + 8 + imm(a) + imm(b) + (c & 0xfff) as u64)
}

/// Name each PLT stub entry VA after the import whose GOT slot the stub jumps
/// through.  Stub bytes come from the `.plt` family of sections; the decoded
/// slot lives in the same rebased space as `slot_names`.
fn build_elf_plt_stub_names(
    elf: &goblin::elf::Elf,
    bytes: &[u8],
    arch: Arch,
    pie_base: u64,
    slot_names: &[Symbol],
) -> Vec<Symbol> {
    let decode: fn(&[u8], u64) -> Option<u64> = match arch {
        Arch::X86_64 => x86_64_plt_slot,
        Arch::Arm64 => aarch64_plt_slot,
        Arch::Arm32 => arm32_plt_slot,
        _ => return Vec::new(),
    };
    let by_slot: FxHashMap<Va, &str> = slot_names.iter().map(|s| (s.va, s.name.as_str())).collect();

    let mut stubs = Vec::new();
    for sh in &elf.section_headers {
        let Some(name) = elf.shdr_strtab.get_at(sh.sh_name) else {
            continue;
        };
        if !matches!(name, ".plt" | ".plt.sec" | ".plt.got" | ".iplt") || sh.sh_addr == 0 {
            continue;
        }
        let Some(data) = (sh.sh_offset as usize)
            .checked_add(sh.sh_size as usize)
            .and_then(|end| bytes.get(sh.sh_offset as usize..end))
        else {
            continue;
        };
        // (entry length, stride).  ARM32 is probed at every word: GNU ld packs
        // 12-byte entries after a 20-byte PLT0, lld uses 16-byte entries, and
        // the decoder anchors on the leading `add ip, pc` so either layout
        // resolves.  Older x86-64 `.plt.got` entries are 8 bytes.
        let (len, step) = match (arch, name) {
            (Arch::Arm32, _) => (12, 4),
            (Arch::X86_64, ".plt.got") => (8, 8),
            _ => (16, 16),
        };
        let mut off = 0;
        while off + len <= data.len() {
            let va = sh.sh_addr + pie_base + off as u64;
            let slot = decode(&data[off..off + len], va);
            if let Some(n) = slot.and_then(|s| by_slot.get(&Va::new(s))) {
                stubs.push(Symbol {
                    name: n.to_string(),
                    va: Va::new(va),
                });
            }
            off += step;
        }
    }
    stubs
}

fn build_elf_reloc_pointers(
    elf: &goblin::elf::Elf,
    bytes: &[u8],
    pie_base: u64,
    segments: &[Segment],
) -> Vec<RelocPointer> {
    use goblin::elf::program_header::PT_LOAD;
    use goblin::elf::section_header::SHN_UNDEF;

    // x86-64
    const R_X86_64_RELATIVE: u32 = 8;
    const R_X86_64_64: u32 = 1;
    // AArch64
    const R_AARCH64_RELATIVE: u32 = 1027;
    const R_AARCH64_ABS64: u32 = 257;
    // ARM32 — REL format: addend lives in the word at *r_offset in the file
    const R_ARM_RELATIVE: u32 = 23;
    const R_ARM_ABS32: u32 = 2;

    // Translate an ELF VMA to its raw file byte offset via PT_LOAD headers.
    // Used to read the implicit in-place addend for ARM32 REL relocations.
    let vma_to_file = |vma: u64| -> Option<usize> {
        elf.program_headers.iter().find_map(|ph| {
            if ph.p_type == PT_LOAD && vma >= ph.p_vaddr && vma < ph.p_vaddr + ph.p_filesz {
                Some((vma - ph.p_vaddr + ph.p_offset) as usize)
            } else {
                None
            }
        })
    };

    // Read a little-endian u32 from the file at the given ELF VMA.
    let read_u32_at = |vma: u64| -> Option<u32> {
        let off = vma_to_file(vma)?;
        bytes
            .get(off..off + 4)
            .and_then(|s| s.try_into().ok())
            .map(u32::from_le_bytes)
    };

    // Simple linear membership test over the (small) segment list.
    // Called once per relocation entry during load; O(n_segments) is fine.
    let is_mapped = |va: Va| segments.iter().any(|s| s.contains(va));
    let mut result = Vec::new();

    for rel in elf.dynrelas.iter().chain(elf.dynrels.iter()) {
        let from = rel.r_offset + pie_base;
        let r_type = rel.r_type;

        if r_type == R_X86_64_RELATIVE || r_type == R_AARCH64_RELATIVE {
            // RELA: explicit addend encodes the pre-link target.
            let target = Va::new((rel.r_addend.unwrap_or(0) as u64).wrapping_add(pie_base));
            if is_mapped(target) {
                result.push(RelocPointer {
                    from: Va::new(from),
                    to: target,
                });
            }
        } else if r_type == R_ARM_RELATIVE {
            // REL: the word at *place in the file IS the pre-link target VMA.
            if let Some(addend) = read_u32_at(rel.r_offset) {
                let target = Va::new((addend as u64).wrapping_add(pie_base));
                if is_mapped(target) {
                    result.push(RelocPointer {
                        from: Va::new(from),
                        to: target,
                    });
                }
            }
        } else if (r_type == R_X86_64_64 || r_type == R_AARCH64_ABS64) && rel.r_sym != 0 {
            // RELA with symbol: target = sym.st_value + addend.
            let sym = elf
                .dynsyms
                .get(rel.r_sym)
                .or_else(|| elf.syms.get(rel.r_sym));
            if let Some(sym) = sym {
                if sym.st_shndx != SHN_UNDEF as usize && sym.st_value != 0 {
                    let target = Va::new(
                        sym.st_value
                            .wrapping_add(pie_base)
                            .wrapping_add(rel.r_addend.unwrap_or(0) as u64),
                    );
                    if is_mapped(target) {
                        result.push(RelocPointer {
                            from: Va::new(from),
                            to: target,
                        });
                    }
                }
            }
        } else if r_type == R_ARM_ABS32 && rel.r_sym != 0 {
            // REL with symbol: target = sym.st_value + implicit_addend.
            let sym = elf
                .dynsyms
                .get(rel.r_sym)
                .or_else(|| elf.syms.get(rel.r_sym));
            if let Some(sym) = sym {
                if sym.st_shndx != SHN_UNDEF as usize && sym.st_value != 0 {
                    let implicit_addend = read_u32_at(rel.r_offset).unwrap_or(0);
                    let target = Va::new(
                        sym.st_value
                            .wrapping_add(pie_base)
                            .wrapping_add(implicit_addend as u64),
                    );
                    if is_mapped(target) {
                        result.push(RelocPointer {
                            from: Va::new(from),
                            to: target,
                        });
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use goblin::elf::Reloc;

    fn reloc(r_offset: u64, r_type: u32, r_sym: usize) -> Reloc {
        Reloc {
            r_offset,
            r_addend: None,
            r_sym,
            r_type,
        }
    }

    fn dynsym_name(idx: usize) -> Option<&'static str> {
        ["", "printf", "", "__libc_start_main"].get(idx).copied()
    }

    #[test]
    fn test_got_relocs_filters_type_and_sym() {
        let rels = [
            reloc(0x100, 7, 1),    // JUMP_SLOT
            reloc(0x108, 6, 3),    // GLOB_DAT
            reloc(0x110, 8, 1),    // RELATIVE: not a GOT reloc
            reloc(0x118, 6, 0),    // no symbol
            reloc(0x120, 1026, 1), // AArch64 JUMP_SLOT
        ];
        let got: Vec<_> = got_relocs(rels.iter().cloned(), 0x1000).collect();
        assert_eq!(
            got,
            [
                (Va::new(0x1100), 1),
                (Va::new(0x1108), 3),
                (Va::new(0x1120), 1)
            ]
        );
    }

    #[test]
    fn test_got_slot_names() {
        let rels = [
            reloc(0x100, 7, 1),
            reloc(0x108, 6, 3),
            reloc(0x110, 7, 2),
            reloc(0x118, 7, 9),
        ];
        let syms = got_slot_names(got_relocs(rels.iter().cloned(), 0), dynsym_name);
        let got: Vec<_> = syms.iter().map(|s| (s.name.as_str(), s.va)).collect();
        // idx 2 has an empty name and idx 9 is out of range: both skipped.
        assert_eq!(
            got,
            [
                ("printf", Va::new(0x100)),
                ("__libc_start_main", Va::new(0x108))
            ]
        );
    }

    // The decoder tests below use entries copied from real binaries.

    #[test]
    fn test_x86_64_plt_slot_lazy() {
        // hello-linux-gcc .plt entry at 0x6030: jmp [rip+0x50962]
        let entry = [
            0xff, 0x25, 0x62, 0x09, 0x05, 0x00, 0x68, 0x00, 0x00, 0x00, 0x00, 0xe9, 0xe0, 0xff,
            0xff, 0xff,
        ];
        assert_eq!(x86_64_plt_slot(&entry, 0x6030), Some(0x56998));
        // PLT0 starts with `push [rip+..]`, never a named slot.
        let plt0 = [
            0xff, 0x35, 0x62, 0x09, 0x05, 0x00, 0xff, 0x25, 0x64, 0x09, 0x05, 0x00, 0x0f, 0x1f,
            0x40, 0x00,
        ];
        assert_eq!(x86_64_plt_slot(&plt0, 0x6020), None);
    }

    #[test]
    fn test_x86_64_plt_slot_plt_got_and_ibt() {
        // libssl3-amd64 .plt.got entry at 0x218a0 (8 bytes)
        let got = [0xff, 0x25, 0xea, 0x36, 0x08, 0x00, 0x66, 0x90];
        assert_eq!(x86_64_plt_slot(&got, 0x218a0), Some(0xa4f90));
        // libpjsip .plt.sec entry at 0xfe70: endbr64 ; bnd jmp [rip+0x5119d]
        let sec = [
            0xf3, 0x0f, 0x1e, 0xfa, 0xf2, 0xff, 0x25, 0x9d, 0x11, 0x05, 0x00, 0x0f, 0x1f, 0x44,
            0x00, 0x00,
        ];
        assert_eq!(x86_64_plt_slot(&sec, 0xfe70), Some(0x61018));
    }

    #[test]
    fn test_aarch64_plt_slot() {
        // libssl3-arm64 .plt entry at 0x1f0c0:
        //   adrp x16, 0xae000 ; ldr x17, [x16, #0x968] ; add x16, x16, #0x968 ; br x17
        let words: [u32; 4] = [0xf000_0470, 0xf944_b611, 0x9125_a210, 0xd61f_0220];
        let entry: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(aarch64_plt_slot(&entry, 0x1f0c0), Some(0xae968));
        // A leading `bti c` shifts the pair by one word.
        let mut bti = 0xd503_245fu32.to_le_bytes().to_vec();
        bti.extend_from_slice(&entry[..12]);
        assert_eq!(aarch64_plt_slot(&bti, 0x1f0bc), Some(0xae968));
        // Not a stub.
        assert_eq!(
            aarch64_plt_slot(&[0x1f, 0x20, 0x03, 0xd5].repeat(4), 0),
            None
        );
    }

    #[test]
    fn test_arm32_plt_slot() {
        // libssl3-arm32 .plt entry at 0x7ded0:
        //   add ip, pc, #0 ; add ip, ip, #0x7000 ; ldr pc, [ip, #0xe4c]!
        let words: [u32; 3] = [0xe28f_c600, 0xe28c_ca07, 0xe5bc_fe4c];
        let entry: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(arm32_plt_slot(&entry, 0x7ded0), Some(0x85d24));
        // PLT0 begins `str lr, [sp, #-4]!`.
        let plt0 = [
            0x04, 0xe0, 0x2d, 0xe5, 0x00, 0x60, 0x8f, 0xe2, 0x07, 0xea, 0x8e, 0xe2,
        ];
        assert_eq!(arm32_plt_slot(&plt0, 0x7deb0), None);
    }
}
