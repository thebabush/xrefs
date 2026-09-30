use super::{alloc_bss, ParseResult, SegData, Segment, Symbol};
use crate::loader::{Arch, RelocPointer, SegmentArch};
use crate::va::Va;
use anyhow::Result;
use rustc_hash::FxHashSet;

/// # Safety (internal)
///
/// `bytes` must remain valid for the lifetime of any `Segment` in the result.
/// Guaranteed by `LoadedBinary` field-ordering invariant.
pub(super) fn parse_macho(
    bytes: &[u8],
    macho: &goblin::mach::MachO,
    bss_bufs: &mut Vec<Box<[u8]>>,
) -> Result<ParseResult> {
    use goblin::mach::constants::cputype::*;
    let arch = match macho.header.cputype() {
        CPU_TYPE_X86_64 => Arch::X86_64,
        CPU_TYPE_ARM64 => Arch::Arm64,
        CPU_TYPE_X86 => Arch::X86,
        CPU_TYPE_ARM => Arch::Arm32,
        m => {
            eprintln!("warning: unknown Mach-O cputype {m:#x}, treating as unknown");
            Arch::Unknown
        }
    };

    const NO_SCAN_SECTIONS: &[&str] = &[
        "__DATA_CONST,__got",
        "__DATA,__got",
        "__DATA_CONST,__auth_got",
        "__DATA,__la_symbol_ptr",
        "__DATA,__nl_symbol_ptr",
        "__DATA_CONST,__cfstring",
        "__DATA,__cfstring",
    ];

    let mut segments = Vec::new();
    for seg in macho.segments.iter() {
        let seg_name = seg.name().unwrap_or("?").to_string();
        for section in seg.sections().map_err(|e| anyhow::anyhow!("{e}"))? {
            let (sect, data) = section;
            let sect_name = sect.name().unwrap_or("?").to_string();
            let size = sect.size as usize;
            if size == 0 {
                continue;
            }
            let exec = seg.initprot & goblin::mach::constants::VM_PROT_EXECUTE != 0;
            let read = seg.initprot & goblin::mach::constants::VM_PROT_READ != 0;
            let write = seg.initprot & goblin::mach::constants::VM_PROT_WRITE != 0;
            let full_name = format!("{seg_name},{sect_name}");
            let byte_scannable = !exec && !NO_SCAN_SECTIONS.iter().any(|&n| n == full_name);

            let section_data = if data.is_empty() {
                alloc_bss(size, bss_bufs)
            } else {
                let file_offset = sect.offset as usize;
                if file_offset + size > bytes.len() {
                    continue;
                }
                // Safety: `bytes` is the mmap kept alive by LoadedBinary.
                unsafe { SegData::new(&bytes[file_offset..file_offset + size]) }
            };

            segments.push(Segment {
                va: Va::new(sect.addr),
                data: section_data,
                executable: exec,
                readable: read,
                writable: write,
                arch: SegmentArch::Generic,
                name: full_name,
                byte_scannable,
            });
        }
    }

    let entry_points = vec![Va::new(macho.entry)];
    let mut symbols = Vec::new();
    if let Some(syms) = macho.symbols.as_ref() {
        for (name, nlist) in syms.iter().flatten() {
            if !name.is_empty() && nlist.n_value != 0 {
                symbols.push(Symbol {
                    name: name.to_string(),
                    va: Va::new(nlist.n_value),
                });
            }
        }
    }

    let preferred_base = macho
        .segments
        .iter()
        .find(|s| s.name().is_ok_and(|n| n == "__TEXT"))
        .map_or(0, |s| s.vmaddr);
    let reloc_pointers = build_macho_fixup_pointers(bytes, macho, preferred_base, &segments);

    let extra_names = build_macho_extra_names(bytes, macho);

    Ok(ParseResult {
        arch,
        segments,
        entry_points,
        symbols,
        pie_base: 0,
        got_slots: FxHashSet::default(),
        got_call_only: false,
        reloc_pointers,
        extra_names,
    })
}

// ── Chained fixup pointer formats ─────────────────────────────────────────────
//
// Each format defines how to decode a 64-bit chained fixup entry in a Mach-O
// binary.  The outer parsing (LC_DYLD_CHAINED_FIXUPS → segments → pages →
// chain walking) is format-independent; only the per-entry decode differs.
//
// Bit layouts (from Apple's `mach-o/fixup-chains.h`):
//
//   Formats 2 & 6 (DYLD_CHAINED_PTR_64 / _OFFSET):
//     [63]    bind       [62:51] next (12 bits, stride 4)
//     [50:44] reserved   [43:36] high8   [35:0] target (36 bits)
//
//   Formats 1, 9, 12 (ARM64E family):
//     [63]    auth       [62]    bind    [61:51] next (11 bits, stride 8)
//     Non-auth rebase:   [50:43] high8   [42:0]  target (43 bits)
//     Auth rebase:       [50:49] key     [48] addrDiv  [47:32] diversity
//                        [31:0]  target (32 bits)

/// Recognized chained-fixup pointer formats.
#[derive(Clone, Copy)]
enum ChainedPtrFormat {
    /// Format 2: target is absolute vmaddr.  Stride 4.
    Ptr64,
    /// Format 6: target is offset from preferred base.  Stride 4.
    Ptr64Offset,
    /// Format 1: ARM64E, target is absolute vmaddr.  Stride 8.
    Arm64e,
    /// Format 9: ARM64E userland, target is offset from preferred base.  Stride 8.
    Arm64eUserland,
    /// Format 12: ARM64E userland 24-bit bind, target is offset.  Stride 8.
    Arm64eUserland24,
}

impl ChainedPtrFormat {
    /// Map the raw `pointer_format` field to a known format, or `None`.
    fn from_raw(raw: u16) -> Option<Self> {
        match raw {
            2 => Some(Self::Ptr64),
            6 => Some(Self::Ptr64Offset),
            1 => Some(Self::Arm64e),
            9 => Some(Self::Arm64eUserland),
            12 => Some(Self::Arm64eUserland24),
            _ => None,
        }
    }

    /// Byte stride per "next" unit (4 for generic 64-bit, 8 for ARM64E).
    fn stride(self) -> usize {
        match self {
            Self::Ptr64 | Self::Ptr64Offset => 4,
            Self::Arm64e | Self::Arm64eUserland | Self::Arm64eUserland24 => 8,
        }
    }

    /// Extract the `next` delta (in stride units) from a raw entry.
    /// Returns 0 when this is the last entry in the chain.
    fn next_delta(self, val: u64) -> usize {
        match self {
            // Formats 2/6: next = bits [62:51], 12 bits
            Self::Ptr64 | Self::Ptr64Offset => ((val >> 51) & 0xFFF) as usize,
            // ARM64E: next = bits [61:51], 11 bits
            Self::Arm64e | Self::Arm64eUserland | Self::Arm64eUserland24 => {
                ((val >> 51) & 0x7FF) as usize
            }
        }
    }

    /// Extract the import ordinal from a bind entry, or `None` for a rebase.
    ///
    /// Auth and non-auth binds share the ordinal position, so only the bind
    /// bit matters.
    fn decode_bind_ordinal(self, val: u64) -> Option<u32> {
        match self {
            // bind = bit 63; ordinal = bits [23:0]
            Self::Ptr64 | Self::Ptr64Offset => {
                ((val >> 63) & 1 != 0).then_some((val & 0xFF_FFFF) as u32)
            }
            // bind = bit 62; ordinal = bits [15:0]
            Self::Arm64e | Self::Arm64eUserland => {
                ((val >> 62) & 1 != 0).then_some((val & 0xFFFF) as u32)
            }
            // bind = bit 62; ordinal = bits [23:0]
            Self::Arm64eUserland24 => ((val >> 62) & 1 != 0).then_some((val & 0xFF_FFFF) as u32),
        }
    }

    /// Decode a rebase entry into a target VA, or `None` if this is a bind.
    fn decode_rebase(self, val: u64, preferred_base: u64) -> Option<Va> {
        match self {
            Self::Ptr64 => {
                // bind = bit 63
                if (val >> 63) & 1 != 0 {
                    return None;
                }
                let target = val & 0xF_FFFF_FFFF; // bits [35:0]
                let high8 = (val >> 36) & 0xFF;
                Some(Va::new((high8 << 56) | target))
            }
            Self::Ptr64Offset => {
                if (val >> 63) & 1 != 0 {
                    return None;
                }
                let target_off = val & 0xF_FFFF_FFFF;
                let high8 = (val >> 36) & 0xFF;
                Some(Va::new((high8 << 56) | (preferred_base + target_off)))
            }
            Self::Arm64e => {
                // auth = bit 63, bind = bit 62
                let auth = (val >> 63) & 1 != 0;
                let bind = (val >> 62) & 1 != 0;
                if bind {
                    return None;
                }
                if auth {
                    // Auth rebase: target = bits [31:0], absolute vmaddr (32-bit)
                    let target = val & 0xFFFF_FFFF;
                    Some(Va::new(target))
                } else {
                    // Rebase: target = bits [42:0], high8 = bits [50:43]
                    let target = val & 0x7FF_FFFF_FFFF; // 43 bits
                    let high8 = (val >> 43) & 0xFF;
                    Some(Va::new((high8 << 56) | target))
                }
            }
            Self::Arm64eUserland | Self::Arm64eUserland24 => {
                let auth = (val >> 63) & 1 != 0;
                let bind = (val >> 62) & 1 != 0;
                if bind {
                    return None;
                }
                if auth {
                    // Auth rebase: target = bits [31:0], offset from preferred base
                    let target_off = val & 0xFFFF_FFFF;
                    Some(Va::new(preferred_base + target_off))
                } else {
                    // Rebase: target = bits [42:0], high8 = bits [50:43]
                    let target_off = val & 0x7FF_FFFF_FFFF;
                    let high8 = (val >> 43) & 0xFF;
                    Some(Va::new((high8 << 56) | (preferred_base + target_off)))
                }
            }
        }
    }
}

/// Walk LC_DYLD_CHAINED_FIXUPS to extract rebase pointers.
///
/// Supported pointer formats:
///   - `DYLD_CHAINED_PTR_64` (2) — x86_64, target is absolute vmaddr
///   - `DYLD_CHAINED_PTR_64_OFFSET` (6) — arm64, target is offset from base
///   - `DYLD_CHAINED_PTR_ARM64E` (1) — arm64e, target is absolute vmaddr
///   - `DYLD_CHAINED_PTR_ARM64E_USERLAND` (9) — arm64e, target is offset
///   - `DYLD_CHAINED_PTR_ARM64E_USERLAND24` (12) — arm64e, 24-bit bind ordinal
///
/// Each rebase entry encodes a pointer-sized slot whose value (after relocation)
/// resolves to a virtual address.  These are emitted as `DataPointer` xrefs.
/// Bind entries (imports from other dylibs) are skipped here; see
/// `build_macho_bind_names` for those.
fn build_macho_fixup_pointers(
    bytes: &[u8],
    macho: &goblin::mach::MachO,
    preferred_base: u64,
    segments: &[Segment],
) -> Vec<RelocPointer> {
    let is_mapped = |va: Va| segments.iter().any(|s| s.contains(va));
    let mut result = Vec::new();
    walk_chained_fixups(bytes, macho, |fmt, val, slot_va| {
        if let Some(target_va) = fmt.decode_rebase(val, preferred_base) {
            if is_mapped(target_va) {
                result.push(RelocPointer {
                    from: slot_va,
                    to: target_va,
                });
            }
        }
    });
    result
}

/// File offset and contents of the `LC_DYLD_CHAINED_FIXUPS` blob (header,
/// starts, imports, strings), if present and within the file.
fn chained_fixups_blob<'a>(
    bytes: &'a [u8],
    macho: &goblin::mach::MachO,
) -> Option<(usize, &'a [u8])> {
    use goblin::mach::load_command::CommandVariant;

    let cmd = macho.load_commands.iter().find_map(|lc| {
        if let CommandVariant::DyldChainedFixups(ref cmd) = lc.command {
            Some(cmd)
        } else {
            None
        }
    })?;
    let data_off = cmd.dataoff as usize;
    let data_size = cmd.datasize as usize;
    if data_off + data_size > bytes.len() || data_size < 28 {
        return None;
    }
    Some((data_off, &bytes[data_off..data_off + data_size]))
}

/// Walk every chain in LC_DYLD_CHAINED_FIXUPS, calling `visit(format, raw_entry,
/// slot_va)` for each entry (rebase and bind alike).
fn walk_chained_fixups(
    bytes: &[u8],
    macho: &goblin::mach::MachO,
    mut visit: impl FnMut(ChainedPtrFormat, u64, Va),
) {
    // Build a map from segment index → segment vmaddr so we can convert
    // chain offsets (which are file offsets within a segment) to VAs.
    // The chained fixups header lists segments by index matching the Mach-O
    // LC_SEGMENT_64 order.  `segment_offset` in each starts-in-segment
    // record is the segment's file offset, so:
    //   slot_va = seg_vmaddr + (chain_off - segment_offset)
    let seg_vmaddrs: Vec<u64> = macho.segments.iter().map(|s| s.vmaddr).collect();

    let Some((data_off, _)) = chained_fixups_blob(bytes, macho) else {
        return;
    };

    let starts_offset = u32::from_le_bytes(
        bytes[data_off + 4..data_off + 8]
            .try_into()
            .expect("checked above"),
    ) as usize;

    let si_off = data_off + starts_offset;
    if si_off + 4 > bytes.len() {
        return;
    }
    let seg_count =
        u32::from_le_bytes(bytes[si_off..si_off + 4].try_into().expect("checked above")) as usize;

    for seg_idx in 0..seg_count {
        let off_off = si_off + 4 + seg_idx * 4;
        if off_off + 4 > bytes.len() {
            break;
        }
        let seg_info_off = u32::from_le_bytes(
            bytes[off_off..off_off + 4]
                .try_into()
                .expect("checked above"),
        ) as usize;
        if seg_info_off == 0 {
            continue;
        }

        let ss_off = si_off + seg_info_off;
        if ss_off + 22 > bytes.len() {
            continue;
        }

        let page_size =
            u16::from_le_bytes(bytes[ss_off + 4..ss_off + 6].try_into().expect("checked")) as usize;
        let pointer_format =
            u16::from_le_bytes(bytes[ss_off + 6..ss_off + 8].try_into().expect("checked"));
        let segment_offset =
            u64::from_le_bytes(bytes[ss_off + 8..ss_off + 16].try_into().expect("checked"));
        let page_count =
            u16::from_le_bytes(bytes[ss_off + 20..ss_off + 22].try_into().expect("checked"))
                as usize;

        let fmt = match ChainedPtrFormat::from_raw(pointer_format) {
            Some(f) => f,
            None => continue,
        };

        // Resolve this segment's vmaddr for file-offset → VA conversion.
        // `segment_offset` is a file offset; `seg_vmaddr` is where the
        // segment is mapped.  slot_va = seg_vmaddr + (chain_off - segment_offset).
        let seg_vmaddr = seg_vmaddrs.get(seg_idx).copied().unwrap_or(0);

        for p_idx in 0..page_count {
            let ps_off = ss_off + 22 + p_idx * 2;
            if ps_off + 2 > bytes.len() {
                break;
            }
            let page_start =
                u16::from_le_bytes(bytes[ps_off..ps_off + 2].try_into().expect("checked"));
            const DYLD_CHAINED_PTR_START_NONE: u16 = 0xFFFF;
            if page_start == DYLD_CHAINED_PTR_START_NONE {
                continue;
            }

            let mut chain_off = segment_offset as usize + p_idx * page_size + page_start as usize;

            loop {
                if chain_off + 8 > bytes.len() {
                    break;
                }
                let val = u64::from_le_bytes(
                    bytes[chain_off..chain_off + 8]
                        .try_into()
                        .expect("checked above"),
                );

                // Convert file offset to VA:
                // chain_off is a file offset within this segment;
                // segment_offset is the segment's file offset.
                let offset_in_seg = chain_off as u64 - segment_offset;
                visit(fmt, val, Va::new(seg_vmaddr + offset_in_seg));

                let next = fmt.next_delta(val);
                if next == 0 {
                    break;
                }
                chain_off += next * fmt.stride();
            }
        }
    }
}

// ── Stub names ────────────────────────────────────────────────────────────────

/// Bind slot names plus stub entry names.
fn build_macho_extra_names(bytes: &[u8], macho: &goblin::mach::MachO) -> Vec<Symbol> {
    let mut out = build_macho_bind_names(bytes, macho);
    out.extend(build_macho_stub_names(bytes, macho));
    out
}

const INDIRECT_SYMBOL_LOCAL: u32 = 0x8000_0000;
const INDIRECT_SYMBOL_ABS: u32 = 0x4000_0000;
const SECTION_TYPE_MASK: u32 = 0xff;
const S_SYMBOL_STUBS: u32 = 0x8;

/// Names for `__stubs` / `__auth_stubs` entries, from the indirect symbol
/// table.  Only S_SYMBOL_STUBS sections are used; pointer sections (`__got`,
/// `__la_symbol_ptr`) are already named by the bind opcodes / chained fixups.
///
/// goblin does not expose `reserved1` / `reserved2`, so the section headers
/// are read from the raw segment load commands.  Everything is bounds-checked.
fn build_macho_stub_names(bytes: &[u8], macho: &goblin::mach::MachO) -> Vec<Symbol> {
    use goblin::mach::load_command::CommandVariant;

    let mut symtab = None;
    let mut dysymtab = None;
    for lc in &macho.load_commands {
        match lc.command {
            CommandVariant::Symtab(ref c) => symtab = Some(*c),
            CommandVariant::Dysymtab(ref c) => dysymtab = Some(*c),
            _ => {}
        }
    }
    let (Some(symtab), Some(dysymtab)) = (symtab, dysymtab) else {
        return Vec::new();
    };

    let rd32 = |off: usize| -> Option<u32> {
        let b = bytes.get(off..off.checked_add(4)?)?;
        Some(u32::from_le_bytes(b.try_into().ok()?))
    };

    let ind_off = dysymtab.indirectsymoff as usize;
    let ind_len = (dysymtab.nindirectsyms as usize).min(bytes.len() / 4);
    let indirect: Vec<u32> = (0..ind_len)
        .map_while(|i| rd32(ind_off.checked_add(i * 4)?))
        .collect();

    let nlist_size = if macho.is_64 { 16 } else { 12 };
    let name_of = |idx: u32| -> Option<String> {
        if idx >= symtab.nsyms {
            return None;
        }
        let entry =
            (symtab.symoff as usize).checked_add((idx as usize).checked_mul(nlist_size)?)?;
        let strx = rd32(entry)? as usize;
        if strx >= symtab.strsize as usize {
            return None;
        }
        let start = (symtab.stroff as usize).checked_add(strx)?;
        let end = (symtab.stroff as usize).checked_add(symtab.strsize as usize)?;
        let tail = bytes.get(start..end.min(bytes.len()))?;
        let len = tail.iter().position(|&b| b == 0)?;
        std::str::from_utf8(&tail[..len]).ok().map(str::to_string)
    };

    let (seg_hdr, sect_size) = if macho.is_64 { (72, 80) } else { (56, 68) };
    let addr_size = if macho.is_64 { 8 } else { 4 };
    let mut out = Vec::new();
    for lc in &macho.load_commands {
        let nsects = match lc.command {
            CommandVariant::Segment32(ref s) => s.nsects,
            CommandVariant::Segment64(ref s) => s.nsects,
            _ => continue,
        };
        for i in 0..nsects as usize {
            let Some(sect) = lc
                .offset
                .checked_add(seg_hdr + i * sect_size)
                .and_then(|o| bytes.get(o..o.checked_add(sect_size)?))
            else {
                break;
            };
            let word = |o: usize| u32::from_le_bytes(sect[o..o + 4].try_into().unwrap());
            let (addr, size) = if macho.is_64 {
                (
                    u64::from_le_bytes(sect[32..40].try_into().unwrap()),
                    u64::from_le_bytes(sect[40..48].try_into().unwrap()),
                )
            } else {
                (word(32) as u64, word(36) as u64)
            };
            // after addr/size: offset, align, reloff, nreloc, flags, reserved1, reserved2
            let base = 32 + 2 * addr_size;
            let flags = word(base + 16);
            let (reserved1, reserved2) = (word(base + 20), word(base + 24));
            if flags & SECTION_TYPE_MASK != S_SYMBOL_STUBS || reserved2 == 0 {
                continue;
            }
            out.extend(stub_entry_names(
                addr,
                size / reserved2 as u64,
                reserved2 as u64,
                reserved1 as usize,
                &indirect,
                &name_of,
            ));
        }
    }
    out
}

/// Map `count` stub entries of `entry_size` bytes starting at `addr` to names:
/// entry `i` stands for `indirect[first + i]`, an index into the symtab.
/// LOCAL / ABS markers and out-of-range indices are skipped.
fn stub_entry_names(
    addr: u64,
    count: u64,
    entry_size: u64,
    first: usize,
    indirect: &[u32],
    name_of: &dyn Fn(u32) -> Option<String>,
) -> Vec<Symbol> {
    let mut out = Vec::new();
    for i in 0..count {
        let Some(&idx) = first.checked_add(i as usize).and_then(|k| indirect.get(k)) else {
            break;
        };
        if idx & (INDIRECT_SYMBOL_LOCAL | INDIRECT_SYMBOL_ABS) != 0 {
            continue;
        }
        let Some(name) = name_of(idx).filter(|n| !n.is_empty()) else {
            continue;
        };
        let Some(va) = i.checked_mul(entry_size).and_then(|o| addr.checked_add(o)) else {
            break;
        };
        out.push(Symbol {
            name,
            va: Va::new(va),
        });
    }
    out
}

// ── Bind slot names ───────────────────────────────────────────────────────────

/// Names for every pointer slot dyld binds to an imported symbol.
///
/// Covers chained fixups (bind entries resolved through the header's imports
/// table) and classic `LC_DYLD_INFO` bind / weak_bind / lazy_bind opcode
/// streams.  Names are raw (leading underscore kept); slot VAs are
/// segment-vmaddr based, like the rest of this loader.
fn build_macho_bind_names(bytes: &[u8], macho: &goblin::mach::MachO) -> Vec<Symbol> {
    use goblin::mach::load_command::CommandVariant;

    let mut out = Vec::new();

    if let Some((_, blob)) = chained_fixups_blob(bytes, macho) {
        let imports = parse_chained_imports(blob);
        walk_chained_fixups(bytes, macho, |fmt, val, slot_va| {
            let Some(ordinal) = fmt.decode_bind_ordinal(val) else {
                return;
            };
            if let Some(name) = imports.get(ordinal as usize).filter(|n| !n.is_empty()) {
                out.push(Symbol {
                    name: name.clone(),
                    va: slot_va,
                });
            }
        });
    }

    let dyld_info = macho.load_commands.iter().find_map(|lc| match lc.command {
        CommandVariant::DyldInfo(ref c) | CommandVariant::DyldInfoOnly(ref c) => Some(*c),
        _ => None,
    });
    if let Some(info) = dyld_info {
        let seg_vmaddrs: Vec<u64> = macho.segments.iter().map(|s| s.vmaddr).collect();
        let ptr_size = if macho.is_64 { 8 } else { 4 };
        let streams = [
            (info.bind_off, info.bind_size, false),
            (info.weak_bind_off, info.weak_bind_size, false),
            (info.lazy_bind_off, info.lazy_bind_size, true),
        ];
        for (off, size, lazy) in streams {
            let (off, size) = (off as usize, size as usize);
            if size == 0 || off.saturating_add(size) > bytes.len() {
                continue;
            }
            parse_bind_opcodes(
                &bytes[off..off + size],
                &seg_vmaddrs,
                ptr_size,
                lazy,
                &mut out,
            );
        }
    }

    out
}

/// Decode the imports table of a chained-fixups blob into symbol names,
/// indexed by import ordinal.  Malformed or truncated tables yield what could
/// be read.
///
/// Header (`dyld_chained_fixups_header`): `imports_offset` @8, `symbols_offset`
/// @12, `imports_count` @16, `imports_format` @20, all u32.  Formats:
///   1 `DYLD_CHAINED_IMPORT`          u32: lib:8 weak:1 name_offset:23
///   2 `DYLD_CHAINED_IMPORT_ADDEND`   the u32 above + i32 addend
///   3 `DYLD_CHAINED_IMPORT_ADDEND64` u64: lib:16 weak:1 reserved:15 name_offset:32
///                                    + i64 addend
fn parse_chained_imports(blob: &[u8]) -> Vec<String> {
    let rd32 = |off: usize| -> Option<u32> {
        let b = blob.get(off..off.checked_add(4)?)?;
        Some(u32::from_le_bytes(b.try_into().ok()?))
    };
    let (Some(imports_off), Some(symbols_off), Some(count), Some(format)) =
        (rd32(8), rd32(12), rd32(16), rd32(20))
    else {
        return Vec::new();
    };
    let (imports_off, symbols_off) = (imports_off as usize, symbols_off as usize);
    let entry_size = match format {
        1 => 4,
        2 => 8,
        3 => 16,
        _ => return Vec::new(),
    };

    let name_at = |name_off: usize| -> String {
        let Some(tail) = symbols_off
            .checked_add(name_off)
            .and_then(|o| blob.get(o..))
        else {
            return String::new();
        };
        let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
        String::from_utf8_lossy(&tail[..end]).into_owned()
    };

    let mut names = Vec::new();
    for i in 0..count as usize {
        let Some(off) = i
            .checked_mul(entry_size)
            .and_then(|o| o.checked_add(imports_off))
        else {
            break;
        };
        let name_off = match format {
            1 | 2 => rd32(off).map(|v| (v >> 9) as usize),
            _ => blob
                .get(off..off.saturating_add(8))
                .map(|b| (u64::from_le_bytes(b.try_into().expect("8 bytes")) >> 32) as usize),
        };
        let Some(name_off) = name_off else { break };
        names.push(name_at(name_off));
    }
    names
}

/// Read a ULEB128 at `*pos`, advancing it.  `None` on truncation.
fn read_uleb(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *data.get(*pos)?;
        *pos += 1;
        if shift < 64 {
            result |= u64::from(byte & 0x7F) << shift;
        }
        shift += 7;
        if byte & 0x80 == 0 {
            return Some(result);
        }
    }
}

/// Interpret one classic dyld bind opcode stream, pushing a named slot for
/// every `DO_BIND*`.  Stops (keeping what it has) on truncation, an
/// out-of-range segment index or a threaded-bind opcode.
fn parse_bind_opcodes(
    data: &[u8],
    seg_vmaddrs: &[u64],
    ptr_size: u64,
    lazy: bool,
    out: &mut Vec<Symbol>,
) {
    const BIND_OPCODE_MASK: u8 = 0xF0;
    const BIND_IMMEDIATE_MASK: u8 = 0x0F;

    let mut pos = 0usize;
    let mut name: &[u8] = &[];
    let mut seg_base: Option<u64> = None;
    let mut addr = 0u64;

    let mut emit = |name: &[u8], seg_base: Option<u64>, addr: u64| {
        if let (Some(base), false) = (seg_base, name.is_empty()) {
            out.push(Symbol {
                name: String::from_utf8_lossy(name).into_owned(),
                va: Va::new(base.wrapping_add(addr)),
            });
        }
    };

    while let Some(&op) = data.get(pos) {
        pos += 1;
        let imm = op & BIND_IMMEDIATE_MASK;
        match op & BIND_OPCODE_MASK {
            // DONE terminates the bind and weak streams; lazy entries are
            // separated by it.
            0x00 if lazy => {}
            0x00 => break,
            // SET_DYLIB_ORDINAL_IMM / SET_DYLIB_SPECIAL_IMM / SET_TYPE_IMM
            0x10 | 0x30 | 0x50 => {}
            // SET_DYLIB_ORDINAL_ULEB / SET_ADDEND_SLEB (sleb skips like uleb)
            0x20 | 0x60 => {
                if read_uleb(data, &mut pos).is_none() {
                    break;
                }
            }
            // SET_SYMBOL_TRAILING_FLAGS_IMM
            0x40 => {
                let Some(tail) = data.get(pos..) else { break };
                let Some(len) = tail.iter().position(|&b| b == 0) else {
                    break;
                };
                name = &tail[..len];
                pos += len + 1;
            }
            // SET_SEGMENT_AND_OFFSET_ULEB
            0x70 => {
                let Some(off) = read_uleb(data, &mut pos) else {
                    break;
                };
                let Some(&base) = seg_vmaddrs.get(imm as usize) else {
                    break;
                };
                seg_base = Some(base);
                addr = off;
            }
            // ADD_ADDR_ULEB
            0x80 => {
                let Some(delta) = read_uleb(data, &mut pos) else {
                    break;
                };
                addr = addr.wrapping_add(delta);
            }
            // DO_BIND
            0x90 => {
                emit(name, seg_base, addr);
                addr = addr.wrapping_add(ptr_size);
            }
            // DO_BIND_ADD_ADDR_ULEB
            0xA0 => {
                emit(name, seg_base, addr);
                let Some(delta) = read_uleb(data, &mut pos) else {
                    break;
                };
                addr = addr.wrapping_add(delta).wrapping_add(ptr_size);
            }
            // DO_BIND_ADD_ADDR_IMM_SCALED
            0xB0 => {
                emit(name, seg_base, addr);
                addr = addr
                    .wrapping_add(u64::from(imm) * ptr_size)
                    .wrapping_add(ptr_size);
            }
            // DO_BIND_ULEB_TIMES_SKIPPING_ULEB
            0xC0 => {
                let (Some(count), Some(skip)) =
                    (read_uleb(data, &mut pos), read_uleb(data, &mut pos))
                else {
                    break;
                };
                // Bound the loop by what the stream could plausibly describe.
                for _ in 0..count.min(1 << 20) {
                    emit(name, seg_base, addr);
                    addr = addr.wrapping_add(skip).wrapping_add(ptr_size);
                }
            }
            // THREADED and unknown opcodes: not decodable here.
            _ => break,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_entry_names_maps_and_skips() {
        let names = ["_a", "_b", "", "_d"];
        let name_of = |i: u32| names.get(i as usize).map(|s| s.to_string());
        // table: [pad, 1, LOCAL, ABS, 2 (empty name), 99 (oob), 3, 0]
        let ind = [
            7,
            1,
            INDIRECT_SYMBOL_LOCAL,
            INDIRECT_SYMBOL_ABS,
            2,
            99,
            3,
            0,
        ];
        let got = stub_entry_names(0x1000, 7, 12, 1, &ind, &name_of);
        let got: Vec<(String, u64)> = got.into_iter().map(|s| (s.name, s.va.raw())).collect();
        assert_eq!(
            got,
            vec![
                ("_b".to_string(), 0x1000),
                ("_d".to_string(), 0x1000 + 5 * 12),
                ("_a".to_string(), 0x1000 + 6 * 12),
            ]
        );
    }

    #[test]
    fn stub_entry_names_truncates_at_table_end() {
        let name_of = |i: u32| Some(format!("_s{i}"));
        let got = stub_entry_names(0x2000, 10, 6, 1, &[0, 1, 2], &name_of);
        assert_eq!(got.len(), 2);
        assert!(stub_entry_names(0x2000, 3, 6, usize::MAX, &[0], &name_of).is_empty());
    }

    const BASE: u64 = 0x1_0000_0000; // typical Mach-O __TEXT vmaddr

    // ── Format helpers ────────────────────────────────────────────────────

    /// Build a DYLD_CHAINED_PTR_64 (format 2) rebase entry.
    /// Layout: [63] bind=0  [62:51] next  [43:36] high8  [35:0] target (abs vmaddr)
    fn ptr64_rebase(target: u64, high8: u64, next: u64) -> u64 {
        (target & 0xF_FFFF_FFFF) | ((high8 & 0xFF) << 36) | ((next & 0xFFF) << 51)
    }

    /// Build a DYLD_CHAINED_PTR_64_OFFSET (format 6) rebase entry.
    /// Same bit layout as format 2, but target is an offset from preferred base.
    fn ptr64_offset_rebase(target_off: u64, high8: u64, next: u64) -> u64 {
        ptr64_rebase(target_off, high8, next) // same encoding
    }

    /// Build a DYLD_CHAINED_PTR_64 bind entry (bit 63 set).
    fn ptr64_bind(next: u64) -> u64 {
        (1u64 << 63) | ((next & 0xFFF) << 51)
    }

    /// ARM64E non-auth rebase: auth=0 bind=0.
    /// [42:0] target  [50:43] high8  [61:51] next
    fn arm64e_rebase(target: u64, high8: u64, next: u64) -> u64 {
        (target & 0x7FF_FFFF_FFFF) | ((high8 & 0xFF) << 43) | ((next & 0x7FF) << 51)
    }

    /// ARM64E auth rebase: auth=1 bind=0.
    /// [31:0] target  [47:32] diversity  [48] addrDiv  [50:49] key  [61:51] next
    fn arm64e_auth_rebase(target: u64, next: u64) -> u64 {
        (target & 0xFFFF_FFFF) | ((next & 0x7FF) << 51) | (1u64 << 63)
    }

    /// ARM64E bind: bind=1 (bit 62).
    fn arm64e_bind(next: u64) -> u64 {
        (1u64 << 62) | ((next & 0x7FF) << 51)
    }

    /// ARM64E auth bind: auth=1 bind=1.
    fn arm64e_auth_bind(next: u64) -> u64 {
        (1u64 << 63) | (1u64 << 62) | ((next & 0x7FF) << 51)
    }

    // ── Format 2: DYLD_CHAINED_PTR_64 ────────────────────────────────────

    #[test]
    fn test_ptr64_rebase_absolute() {
        let fmt = ChainedPtrFormat::Ptr64;
        let val = ptr64_rebase(0x1_0000_1000, 0, 5);
        let result = fmt.decode_rebase(val, BASE);
        assert_eq!(result, Some(Va::new(0x1_0000_1000)));
        assert_eq!(fmt.next_delta(val), 5);
        assert_eq!(fmt.stride(), 4);
    }

    #[test]
    fn test_ptr64_rebase_with_high8() {
        let fmt = ChainedPtrFormat::Ptr64;
        let val = ptr64_rebase(0x1_0000_1000, 0x80, 0);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result.raw() >> 56, 0x80);
        assert_eq!(result.raw() & 0x00FF_FFFF_FFFF_FFFF, 0x1_0000_1000);
    }

    #[test]
    fn test_ptr64_bind_skipped() {
        let fmt = ChainedPtrFormat::Ptr64;
        let val = ptr64_bind(3);
        assert_eq!(fmt.decode_rebase(val, BASE), None);
        assert_eq!(fmt.next_delta(val), 3);
    }

    // ── Format 6: DYLD_CHAINED_PTR_64_OFFSET ─────────────────────────────

    #[test]
    fn test_ptr64_offset_rebase() {
        let fmt = ChainedPtrFormat::Ptr64Offset;
        let offset = 0x1000u64;
        let val = ptr64_offset_rebase(offset, 0, 2);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(BASE + offset));
        assert_eq!(fmt.next_delta(val), 2);
    }

    // ── Format 1: DYLD_CHAINED_PTR_ARM64E ────────────────────────────────

    #[test]
    fn test_arm64e_rebase_absolute() {
        let fmt = ChainedPtrFormat::Arm64e;
        let target = 0x1_0000_2000u64;
        let val = arm64e_rebase(target, 0, 7);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(target));
        assert_eq!(fmt.next_delta(val), 7);
        assert_eq!(fmt.stride(), 8);
    }

    #[test]
    fn test_arm64e_rebase_with_high8() {
        let fmt = ChainedPtrFormat::Arm64e;
        let val = arm64e_rebase(0x1_0000_2000, 0xAB, 0);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result.raw() >> 56, 0xAB);
        assert_eq!(result.raw() & 0x00FF_FFFF_FFFF_FFFF, 0x1_0000_2000);
    }

    #[test]
    fn test_arm64e_auth_rebase_absolute() {
        let fmt = ChainedPtrFormat::Arm64e;
        // Auth rebase: 32-bit absolute target
        let val = arm64e_auth_rebase(0x1234_5678, 4);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(0x1234_5678));
        assert_eq!(fmt.next_delta(val), 4);
    }

    #[test]
    fn test_arm64e_bind_skipped() {
        let fmt = ChainedPtrFormat::Arm64e;
        assert_eq!(fmt.decode_rebase(arm64e_bind(1), BASE), None);
    }

    #[test]
    fn test_arm64e_auth_bind_skipped() {
        let fmt = ChainedPtrFormat::Arm64e;
        assert_eq!(fmt.decode_rebase(arm64e_auth_bind(2), BASE), None);
    }

    // ── Format 9: DYLD_CHAINED_PTR_ARM64E_USERLAND ───────────────────────

    #[test]
    fn test_arm64e_userland_rebase_offset() {
        let fmt = ChainedPtrFormat::Arm64eUserland;
        let offset = 0x5000u64;
        let val = arm64e_rebase(offset, 0, 3);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(BASE + offset));
    }

    #[test]
    fn test_arm64e_userland_auth_rebase_offset() {
        let fmt = ChainedPtrFormat::Arm64eUserland;
        let offset = 0xABCD_0000u64;
        let val = arm64e_auth_rebase(offset, 1);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(BASE + offset));
    }

    #[test]
    fn test_arm64e_userland_bind_skipped() {
        let fmt = ChainedPtrFormat::Arm64eUserland;
        assert_eq!(fmt.decode_rebase(arm64e_bind(0), BASE), None);
        assert_eq!(fmt.decode_rebase(arm64e_auth_bind(0), BASE), None);
    }

    // ── Format 12: DYLD_CHAINED_PTR_ARM64E_USERLAND24 ────────────────────

    #[test]
    fn test_arm64e_userland24_rebase_same_as_userland() {
        // Rebase encoding is identical to format 9; only bind ordinal width differs.
        let fmt = ChainedPtrFormat::Arm64eUserland24;
        let offset = 0x8000u64;
        let val = arm64e_rebase(offset, 0, 5);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(BASE + offset));
        assert_eq!(fmt.stride(), 8);
    }

    // ── from_raw ──────────────────────────────────────────────────────────

    #[test]
    fn test_from_raw_known_formats() {
        assert!(ChainedPtrFormat::from_raw(1).is_some());
        assert!(ChainedPtrFormat::from_raw(2).is_some());
        assert!(ChainedPtrFormat::from_raw(6).is_some());
        assert!(ChainedPtrFormat::from_raw(9).is_some());
        assert!(ChainedPtrFormat::from_raw(12).is_some());
    }

    #[test]
    fn test_from_raw_unknown_formats() {
        for f in [0, 3, 4, 5, 7, 8, 10, 11, 13, 100] {
            assert!(
                ChainedPtrFormat::from_raw(f).is_none(),
                "format {f} should be None"
            );
        }
    }

    // ── next_delta field width ────────────────────────────────────────────

    #[test]
    fn test_next_delta_max_ptr64() {
        let fmt = ChainedPtrFormat::Ptr64;
        // All 12 bits set in the next field
        let val = 0xFFF_u64 << 51;
        assert_eq!(fmt.next_delta(val), 0xFFF);
    }

    #[test]
    fn test_next_delta_max_arm64e() {
        let fmt = ChainedPtrFormat::Arm64e;
        // All 11 bits set in the next field, plus auth+bind bits above
        let val = (0x7FF_u64 << 51) | (1u64 << 63) | (1u64 << 62);
        assert_eq!(fmt.next_delta(val), 0x7FF);
    }

    // ── Bind ordinals ─────────────────────────────────────────────────────

    #[test]
    fn test_bind_ordinal_ptr64() {
        // bind bit 63, ordinal [23:0], addend [31:24] must not leak in
        let val = (1u64 << 63) | (0xAB << 24) | 0x12_3456;
        assert_eq!(
            ChainedPtrFormat::Ptr64.decode_bind_ordinal(val),
            Some(0x12_3456)
        );
        assert_eq!(
            ChainedPtrFormat::Ptr64Offset.decode_bind_ordinal(val),
            Some(0x12_3456)
        );
        assert_eq!(
            ChainedPtrFormat::Ptr64.decode_bind_ordinal(ptr64_rebase(BASE, 0, 1)),
            None
        );
    }

    #[test]
    fn test_bind_ordinal_arm64e_16bit() {
        // bind (bit 62), ordinal [15:0]; addend bits [50:32] ignored
        let val = (1u64 << 62) | (0x7 << 32) | 0xBEEF;
        assert_eq!(
            ChainedPtrFormat::Arm64e.decode_bind_ordinal(val),
            Some(0xBEEF)
        );
        assert_eq!(
            ChainedPtrFormat::Arm64eUserland.decode_bind_ordinal(val),
            Some(0xBEEF)
        );
        // auth bind shares the ordinal position
        let auth = val | (1u64 << 63) | (0x1234 << 32);
        assert_eq!(
            ChainedPtrFormat::Arm64e.decode_bind_ordinal(auth),
            Some(0xBEEF)
        );
        assert_eq!(
            ChainedPtrFormat::Arm64e.decode_bind_ordinal(arm64e_rebase(BASE, 0, 1)),
            None
        );
    }

    #[test]
    fn test_bind_ordinal_arm64e_userland24() {
        let val = (1u64 << 63) | (1u64 << 62) | 0xAB_CDEF;
        assert_eq!(
            ChainedPtrFormat::Arm64eUserland24.decode_bind_ordinal(val),
            Some(0xAB_CDEF)
        );
        assert_eq!(
            ChainedPtrFormat::Arm64eUserland24.decode_bind_ordinal(arm64e_auth_rebase(BASE, 0)),
            None
        );
    }

    // ── Chained imports table ─────────────────────────────────────────────

    /// Header + imports + strings, with `imports_offset` = 28.
    fn chained_blob(format: u32, imports: &[u8], count: u32, strings: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_le_bytes()); // fixups_version
        b.extend_from_slice(&0u32.to_le_bytes()); // starts_offset
        b.extend_from_slice(&28u32.to_le_bytes()); // imports_offset
        b.extend_from_slice(&(28 + imports.len() as u32).to_le_bytes()); // symbols_offset
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&format.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // symbols_format
        b.extend_from_slice(imports);
        b.extend_from_slice(strings);
        b
    }

    #[test]
    fn test_chained_imports_format1() {
        // lib ordinal 1 / name offset 1, lib 2 (weak) / name offset 8
        let mut imp = Vec::new();
        imp.extend_from_slice(&((1u32 << 9) | 1).to_le_bytes());
        imp.extend_from_slice(&((9u32 << 9) | (1 << 8) | 2).to_le_bytes());
        let blob = chained_blob(1, &imp, 2, b"\0_malloc\0_free\0");
        assert_eq!(parse_chained_imports(&blob), ["_malloc", "_free"]);
    }

    #[test]
    fn test_chained_imports_format2_addend() {
        let mut imp = Vec::new();
        imp.extend_from_slice(&((1u32 << 9) | 1).to_le_bytes());
        imp.extend_from_slice(&(-8i32).to_le_bytes());
        let blob = chained_blob(2, &imp, 1, b"\0_open\0");
        assert_eq!(parse_chained_imports(&blob), ["_open"]);
    }

    #[test]
    fn test_chained_imports_format3_addend64() {
        let mut imp = Vec::new();
        imp.extend_from_slice(&((1u64 << 32) | 3).to_le_bytes());
        imp.extend_from_slice(&0x10i64.to_le_bytes());
        let blob = chained_blob(3, &imp, 1, b"\0_read\0");
        assert_eq!(parse_chained_imports(&blob), ["_read"]);
    }

    #[test]
    fn test_chained_imports_malformed() {
        // Unknown format, truncated header, count past the table, name past the pool
        assert!(parse_chained_imports(&chained_blob(9, &[], 1, b"")).is_empty());
        assert!(parse_chained_imports(&[0u8; 10]).is_empty());
        let imp = ((1u32 << 9) | 1).to_le_bytes();
        let blob = chained_blob(1, &imp, 1000, b"\0_ok\0");
        // the table runs into the string pool, which is read as (garbage) entries
        assert_eq!(parse_chained_imports(&blob)[0], "_ok");
        let imp = (0xFF_FFFFu32 << 9).to_le_bytes();
        assert_eq!(
            parse_chained_imports(&chained_blob(1, &imp, 1, b"\0x\0")),
            [""]
        );
    }

    // ── Classic bind opcodes ──────────────────────────────────────────────

    fn bound(data: &[u8], lazy: bool) -> Vec<(u64, String)> {
        let mut out = Vec::new();
        parse_bind_opcodes(data, &[0x1000, 0x2000], 8, lazy, &mut out);
        out.into_iter().map(|s| (s.va.raw(), s.name)).collect()
    }

    #[test]
    fn test_bind_opcodes_do_bind_variants() {
        let mut d = vec![0x11, 0x40];
        d.extend_from_slice(b"_a\0");
        d.extend_from_slice(&[0x71, 0x10]); // segment 1, offset 0x10
        d.push(0x90); // DO_BIND @0x2010
        d.extend_from_slice(&[0xA0, 0x08]); // @0x2018, then +8+8
        d.push(0x40);
        d.extend_from_slice(b"_b\0");
        d.push(0xB1); // @0x2028, then +1*8+8
        d.extend_from_slice(&[0xC0, 0x02, 0x00]); // 2 times, skip 0 @0x2038, 0x2040
        d.push(0x00);
        assert_eq!(
            bound(&d, false),
            [
                (0x2010, "_a".into()),
                (0x2018, "_a".into()),
                (0x2028, "_b".into()),
                (0x2038, "_b".into()),
                (0x2040, "_b".into()),
            ]
        );
    }

    #[test]
    fn test_bind_opcodes_lazy_continues_past_done() {
        let mut d = Vec::new();
        for (off, name) in [(0x00u8, b"_x"), (0x08, b"_y")] {
            d.push(0x40);
            d.extend_from_slice(name);
            d.extend_from_slice(&[0x00, 0x71, off, 0x90, 0x00, 0x90, 0x00]);
        }
        // eager stream stops at first DONE, lazy one does not
        assert!(bound(&d, false).len() < bound(&d, true).len());
        assert!(bound(&d, true)
            .iter()
            .any(|(va, n)| *va == 0x2010 && n == "_y"));
    }

    #[test]
    fn test_bind_opcodes_malformed() {
        // bad segment index, truncated uleb, unterminated name, threaded opcode
        assert!(bound(&[0x40, b'a', 0, 0x7F, 0, 0x90], false).is_empty());
        assert!(bound(&[0x70, 0x80], false).is_empty());
        assert!(bound(&[0x40, b'a', b'b'], false).is_empty());
        assert!(bound(&[0xD0, 0x40, b'a', 0, 0x70, 0, 0x90], false).is_empty());
        // DO_BIND without a symbol name or segment records nothing
        assert!(bound(&[0x90], false).is_empty());
    }
}
