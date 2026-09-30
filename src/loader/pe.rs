use super::{alloc_bss, ParseResult, SegData, Segment, Symbol};
use crate::loader::{Arch, RelocPointer, SegmentArch};
use crate::va::Va;
use anyhow::Result;
use rustc_hash::FxHashSet;

/// # Safety (internal)
///
/// `bytes` must remain valid for the lifetime of any `Segment` in the result.
/// Guaranteed by `LoadedBinary` field-ordering invariant.
pub(super) fn parse_pe(
    bytes: &[u8],
    pe: &goblin::pe::PE,
    bss_bufs: &mut Vec<Box<[u8]>>,
) -> Result<ParseResult> {
    use goblin::pe::header::*;

    let arch = match pe.header.coff_header.machine {
        COFF_MACHINE_X86_64 => Arch::X86_64,
        COFF_MACHINE_X86 => Arch::X86,
        COFF_MACHINE_ARM64 => Arch::Arm64,
        COFF_MACHINE_ARMNT => Arch::Arm32,
        m => {
            eprintln!("warning: unknown PE machine {m:#x}, treating as unknown");
            Arch::Unknown
        }
    };

    let image_base = pe.image_base as u64;
    let mut segments = Vec::new();

    for section in &pe.sections {
        use goblin::pe::section_table::*;
        let raw_offset = section.pointer_to_raw_data as usize;
        let raw_size = section.size_of_raw_data as usize;
        let virt_size = section.virtual_size as usize;
        let chars = section.characteristics;
        let exec = chars & IMAGE_SCN_MEM_EXECUTE != 0;
        let read = chars & IMAGE_SCN_MEM_READ != 0;
        let write = chars & IMAGE_SCN_MEM_WRITE != 0;
        let va = Va::new(image_base + section.virtual_address as u64);
        let name = section.name().unwrap_or("?").to_string();

        let is_debug = name.starts_with(".debug_");
        if is_debug {
            if raw_size > 0 && raw_offset + raw_size <= bytes.len() {
                // Safety: `bytes` is the mmap kept alive by LoadedBinary.
                segments.push(Segment {
                    va,
                    data: unsafe { SegData::new(&bytes[raw_offset..raw_offset + raw_size]) },
                    executable: exec,
                    readable: read,
                    writable: write,
                    arch: SegmentArch::Generic,
                    name,
                    byte_scannable: false,
                });
            }
            continue;
        }

        if raw_size == 0 {
            if virt_size == 0 {
                continue;
            }
            let bss_data = alloc_bss(virt_size, bss_bufs);
            segments.push(Segment {
                va,
                data: bss_data,
                executable: exec,
                readable: read,
                writable: write,
                arch: SegmentArch::Generic,
                name,
                byte_scannable: false,
            });
            continue;
        }

        if raw_offset + raw_size > bytes.len() {
            eprintln!(
                "warning: PE section '{}' at offset {:#x}+{:#x} exceeds file size, skipping",
                name, raw_offset, raw_size
            );
            continue;
        }
        let data = &bytes[raw_offset..raw_offset + raw_size];

        if virt_size > raw_size {
            let bss_va = va + raw_size as u64;
            let bss_sz = virt_size - raw_size;
            let bss_data = alloc_bss(bss_sz, bss_bufs);
            segments.push(Segment {
                va: bss_va,
                data: bss_data,
                executable: exec,
                readable: read,
                writable: write,
                arch: SegmentArch::Generic,
                name: format!("{name}[bss]"),
                byte_scannable: false,
            });
        }

        // Safety: `bytes` is the mmap kept alive by LoadedBinary.
        segments.push(Segment {
            va,
            data: unsafe { SegData::new(data) },
            executable: exec,
            readable: read,
            writable: write,
            arch: SegmentArch::Generic,
            name,
            byte_scannable: true,
        });
    }

    let entry_points = if pe.entry != 0 {
        vec![Va::new(image_base + pe.entry as u64)]
    } else {
        vec![]
    };

    let mut symbols = Vec::new();
    for export in &pe.exports {
        if let Some(name) = export.name {
            symbols.push(Symbol {
                name: name.to_string(),
                va: Va::new(image_base + export.rva as u64),
            });
        }
    }

    let mut reloc_pointers = build_pe_reloc_pointers(bytes, pe, image_base, &segments);
    build_pe_pdata_xrefs(bytes, pe, image_base, &segments, &mut reloc_pointers);
    let got_slots = build_pe_iat_slots(pe, image_base);
    let extra_names = build_pe_extra_names(pe, image_base);

    Ok(ParseResult {
        arch,
        segments,
        entry_points,
        symbols,
        pie_base: 0,
        got_slots,
        got_call_only: true,
        reloc_pointers,
        extra_names,
    })
}

// ── .pdata / UNWIND_INFO ──────────────────────────────────────────────────────

fn build_pe_pdata_xrefs(
    bytes: &[u8],
    pe: &goblin::pe::PE,
    image_base: u64,
    segments: &[Segment],
    out: &mut Vec<RelocPointer>,
) {
    let oh = match pe.header.optional_header.as_ref() {
        Some(oh) => oh,
        None => return,
    };
    let dd = match oh.data_directories.get_exception_table() {
        Some(dd) => dd,
        None => return,
    };

    let sec_map = PeSectionMap::build(pe);
    let is_mapped = |va: Va| segments.iter().any(|s| s.contains(va));

    let dir_rva = dd.virtual_address;
    let dir_size = dd.size;
    let count = dir_size / 12;
    let base_va = Va::new(image_base);

    for i in 0..count {
        let entry_rva = dir_rva + i * 12;
        let file_off = match sec_map.rva_to_file_offset(entry_rva) {
            Some(off) => off,
            None => continue,
        };
        if file_off + 12 > bytes.len() {
            break;
        }

        let begin_rva = u32::from_le_bytes(
            bytes[file_off..file_off + 4]
                .try_into()
                .expect("guarded by file_off + 12 <= len"),
        );
        let end_rva = u32::from_le_bytes(
            bytes[file_off + 4..file_off + 8]
                .try_into()
                .expect("guarded by file_off + 12 <= len"),
        );
        let unwind_rva = u32::from_le_bytes(
            bytes[file_off + 8..file_off + 12]
                .try_into()
                .expect("guarded by file_off + 12 <= len"),
        );

        let from = Va::new(image_base + entry_rva as u64);
        out.push(RelocPointer { from, to: base_va });

        for rva in [begin_rva, end_rva, unwind_rva & !1u32] {
            if rva != 0 {
                let target = Va::new(image_base + rva as u64);
                if is_mapped(target) {
                    out.push(RelocPointer { from, to: target });
                }
            }
        }
    }

    build_pe_unwind_handler_xrefs(
        bytes, image_base, dir_rva, dir_size, &sec_map, &is_mapped, out,
    );
}

fn build_pe_unwind_handler_xrefs(
    bytes: &[u8],
    image_base: u64,
    pdata_rva: u32,
    pdata_size: u32,
    sec_map: &PeSectionMap,
    is_mapped: &impl Fn(Va) -> bool,
    out: &mut Vec<RelocPointer>,
) {
    const UNW_FLAG_EHANDLER: u8 = 0x01;
    const UNW_FLAG_UHANDLER: u8 = 0x02;

    let count = pdata_size / 12;
    let mut seen_unwind: rustc_hash::FxHashSet<u32> = Default::default();

    for i in 0..count {
        let entry_off = match sec_map.rva_to_file_offset(pdata_rva + i * 12) {
            Some(off) => off,
            None => continue,
        };
        if entry_off + 12 > bytes.len() {
            break;
        }
        let unwind_rva = u32::from_le_bytes(
            bytes[entry_off + 8..entry_off + 12]
                .try_into()
                .expect("guarded by entry_off + 12 <= len"),
        ) & !1u32;
        if unwind_rva == 0 || !seen_unwind.insert(unwind_rva) {
            continue;
        }

        let ui_off = match sec_map.rva_to_file_offset(unwind_rva) {
            Some(off) => off,
            None => continue,
        };
        if ui_off + 4 > bytes.len() {
            continue;
        }

        let flags = bytes[ui_off] >> 3;
        if flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) == 0 {
            continue;
        }

        let count_of_codes = bytes[ui_off + 2] as usize;
        let mut handler_file_off = ui_off + 4 + count_of_codes * 2;
        if !handler_file_off.is_multiple_of(4) {
            handler_file_off += 4 - (handler_file_off % 4);
        }
        if handler_file_off + 4 > bytes.len() {
            continue;
        }

        let handler_rva = u32::from_le_bytes(
            bytes[handler_file_off..handler_file_off + 4]
                .try_into()
                .expect("guarded by handler_file_off + 4 <= len"),
        );
        if handler_rva == 0 {
            continue;
        }
        let target = Va::new(image_base + handler_rva as u64);
        if !is_mapped(target) {
            continue;
        }

        let handler_field_rva = unwind_rva + (handler_file_off - ui_off) as u32;
        let from = Va::new(image_base + handler_field_rva as u64);
        out.push(RelocPointer { from, to: target });
    }
}

// ── IAT / base relocations ───────────────────────────────────────────────────

/// IAT slot VAs, one per import.  goblin's `Import.offset` is the slot RVA;
/// `Import.rva` is the hint/name entry (and 0 for imports by ordinal).
fn build_pe_iat_slots(pe: &goblin::pe::PE, image_base: u64) -> FxHashSet<Va> {
    let mut slots = FxHashSet::default();
    for import in &pe.imports {
        slots.insert(Va::new(image_base + import.offset as u64));
    }
    slots
}

/// Names beyond `symbols` (which already carries the named exports): IAT slots
/// named `Func@DLL` / `#ordinal@DLL` after the import they receive.
///
/// goblin's `Import.offset` is the IAT slot RVA (`rva` is the hint/name entry,
/// and 0 for imports by ordinal), so names are keyed on `offset`.
fn build_pe_extra_names(pe: &goblin::pe::PE, image_base: u64) -> Vec<Symbol> {
    pe.imports
        .iter()
        .map(|import| {
            let name = (import.rva != 0).then_some(import.name.as_ref());
            Symbol {
                name: pe_import_name(import.dll, name, import.ordinal),
                va: Va::new(image_base + import.offset as u64),
            }
        })
        .collect()
}

/// `Func@DLL` for a named import, `#<ordinal>@DLL` for one by ordinal.
fn pe_import_name(dll: &str, name: Option<&str>, ordinal: u16) -> String {
    match name {
        Some(n) => format!("{n}@{dll}"),
        None => format!("#{ordinal}@{dll}"),
    }
}

struct PeSectionEntry {
    rva: u32,
    end_rva: u32,
    raw_offset: usize,
    raw_size: usize,
}

struct PeSectionMap {
    entries: Vec<PeSectionEntry>,
}

impl PeSectionMap {
    fn build(pe: &goblin::pe::PE) -> Self {
        let mut entries: Vec<PeSectionEntry> = pe
            .sections
            .iter()
            .filter(|s| s.size_of_raw_data > 0)
            .map(|s| PeSectionEntry {
                rva: s.virtual_address,
                end_rva: s.virtual_address + s.virtual_size,
                raw_offset: s.pointer_to_raw_data as usize,
                raw_size: s.size_of_raw_data as usize,
            })
            .collect();
        entries.sort_unstable_by_key(|e| e.rva);
        Self { entries }
    }

    fn rva_to_file_offset(&self, rva: u32) -> Option<usize> {
        let idx = self.entries.partition_point(|e| e.rva <= rva);
        if idx == 0 {
            return None;
        }
        let e = &self.entries[idx - 1];
        if rva >= e.rva && rva < e.end_rva {
            let offset_in_sec = (rva - e.rva) as usize;
            if offset_in_sec < e.raw_size {
                return Some(e.raw_offset + offset_in_sec);
            }
        }
        None
    }

    fn read_u64_at_rva(&self, bytes: &[u8], rva: u32) -> Option<u64> {
        let off = self.rva_to_file_offset(rva)?;
        let slice: &[u8; 8] = bytes.get(off..off + 8)?.try_into().ok()?;
        Some(u64::from_le_bytes(*slice))
    }
}

fn build_pe_reloc_pointers(
    bytes: &[u8],
    pe: &goblin::pe::PE,
    image_base: u64,
    segments: &[Segment],
) -> Vec<RelocPointer> {
    let is_mapped = |va: Va| segments.iter().any(|s| s.contains(va));
    let sec_map = PeSectionMap::build(pe);

    let mut result = Vec::new();

    const IMAGE_REL_BASED_DIR64: u16 = 10;

    let reloc_dir = pe.header.optional_header.and_then(|oh| {
        let dd = oh.data_directories.get_base_relocation_table()?;
        Some((dd.virtual_address, dd.size))
    });

    if let Some((dir_rva, dir_size)) = reloc_dir {
        let reloc_size = dir_size as usize;

        if let Some(base_off) = sec_map.rva_to_file_offset(dir_rva) {
            let reloc_bytes = bytes.get(base_off..base_off + reloc_size).unwrap_or(&[]);
            let mut pos = 0usize;
            while pos + 8 <= reloc_bytes.len() {
                let page_rva =
                    u32::from_le_bytes(reloc_bytes[pos..pos + 4].try_into().expect("4-byte slice"));
                let block_size = u32::from_le_bytes(
                    reloc_bytes[pos + 4..pos + 8]
                        .try_into()
                        .expect("4-byte slice"),
                ) as usize;
                if block_size < 8 || pos + block_size > reloc_bytes.len() {
                    break;
                }
                let mut entry_pos = pos + 8;
                while entry_pos + 2 <= pos + block_size {
                    let entry = u16::from_le_bytes(
                        reloc_bytes[entry_pos..entry_pos + 2]
                            .try_into()
                            .expect("2-byte slice"),
                    );
                    let rel_type = entry >> 12;
                    let offset = entry & 0x0FFF;
                    if rel_type == IMAGE_REL_BASED_DIR64 {
                        let slot_rva = page_rva + offset as u32;
                        let slot_va = image_base + slot_rva as u64;
                        if let Some(ptr_val) = sec_map.read_u64_at_rva(bytes, slot_rva) {
                            if ptr_val != 0 && is_mapped(Va::new(ptr_val)) {
                                result.push(RelocPointer {
                                    from: Va::new(slot_va),
                                    to: Va::new(ptr_val),
                                });
                            }
                        }
                    }
                    entry_pos += 2;
                }
                pos += block_size;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::pe_import_name;

    #[test]
    fn import_by_name() {
        assert_eq!(
            pe_import_name("KERNEL32.dll", Some("CreateFileW"), 0x12c),
            "CreateFileW@KERNEL32.dll"
        );
    }

    #[test]
    fn import_by_ordinal() {
        assert_eq!(pe_import_name("WS2_32.dll", None, 17), "#17@WS2_32.dll");
    }

    #[test]
    fn dll_name_kept_verbatim() {
        assert_eq!(
            pe_import_name("api-ms-win-crt-heap-l1-1-0", Some("free"), 0),
            "free@api-ms-win-crt-heap-l1-1-0"
        );
    }
}
