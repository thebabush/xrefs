use super::{alloc_bss, ParseResult, SegData, Segment, Symbol};
use crate::loader::{Arch, SegmentArch};
use crate::va::Va;
use anyhow::Result;
use rustc_hash::FxHashSet;

/// Parse a COFF object file (`.o` / `.obj`).
///
/// COFF object files are unlinked and carry `VirtualAddress = 0` for every
/// section, so there is no inherent load address.  IDA Pro assigns VAs by
/// stacking sections in file order starting at VA 0:
///
/// ```text
/// section[0].va = 0
/// section[1].va = section[0].va + effective_size(section[0])
/// section[N].va = sum of effective_size(section[0..N])
/// ```
///
/// where `effective_size = max(SizeOfRawData, VirtualSize)`.  We apply the
/// same scheme so that addresses printed by xr match IDA's output.
///
/// # Safety (internal)
///
/// `bytes` must remain valid for the lifetime of any `Segment` in the result.
/// Guaranteed by the `LoadedBinary` field-ordering invariant in the caller.
pub(super) fn parse_coff(
    bytes: &[u8],
    coff: &goblin::pe::Coff,
    bss_bufs: &mut Vec<Box<[u8]>>,
) -> Result<ParseResult> {
    use goblin::pe::header::*;
    use goblin::pe::section_table::*;

    let arch = match coff.header.machine {
        COFF_MACHINE_X86_64 => Arch::X86_64,
        COFF_MACHINE_ARM64 => Arch::Arm64,
        COFF_MACHINE_X86 => Arch::X86,
        COFF_MACHINE_ARMNT => Arch::Arm32,
        m => {
            eprintln!("warning: unknown COFF machine {m:#x}, treating as unknown");
            Arch::Unknown
        }
    };

    // Pass 1 — compute the sequential VA for every section (including skipped
    // ones) so that symbol addresses derived from section_number + value are
    // consistent across the whole object.
    let section_vas = compute_section_vas(&coff.sections);

    // Pass 2 — build Segments for sections we actually want to scan.
    let mut segments = Vec::new();
    for (section, &va_base) in coff.sections.iter().zip(section_vas.iter()) {
        let chars = section.characteristics;

        // Skip linker-only / removed sections — never mapped.
        if chars & (IMAGE_SCN_LNK_INFO | IMAGE_SCN_LNK_REMOVE) != 0 {
            continue;
        }

        let name = section.name().unwrap_or("?").to_string();

        // Skip DWARF / CodeView debug sections by name.
        if name.starts_with(".debug_") || name.starts_with(".zdebug_") {
            continue;
        }

        let exec = chars & (IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_CNT_CODE) != 0;
        // Code sections are implicitly readable.
        let read = exec || chars & IMAGE_SCN_MEM_READ != 0;
        let write = chars & IMAGE_SCN_MEM_WRITE != 0;
        // Only non-executable, readable sections are byte-scanned for data pointers.
        let byte_scannable = !exec && read;

        let raw_offset = section.pointer_to_raw_data as usize;
        let raw_size = section.size_of_raw_data as usize;
        let virt_size = section.virtual_size as usize;
        let va = Va::new(va_base);

        // BSS-style: pointer_to_raw_data == 0 means no bytes on disk.
        // The in-memory size is max(size_of_raw_data, virtual_size).
        if raw_offset == 0 {
            let bss_size = raw_size.max(virt_size);
            if bss_size == 0 {
                continue;
            }
            let bss_data = alloc_bss(bss_size, bss_bufs);
            segments.push(Segment {
                va,
                data: bss_data,
                executable: exec,
                readable: read,
                writable: write,
                arch: SegmentArch::Generic,
                name,
                byte_scannable,
            });
            continue;
        }

        if raw_offset + raw_size > bytes.len() {
            eprintln!(
                "warning: COFF section '{name}' raw data [{raw_offset:#x}..{:#x}] \
                 exceeds file size, skipping",
                raw_offset + raw_size
            );
            continue;
        }

        // Safety: `bytes` is the mmap kept alive by `LoadedBinary::_mmap`.
        segments.push(Segment {
            va,
            data: unsafe { SegData::new(&bytes[raw_offset..raw_offset + raw_size]) },
            executable: exec,
            readable: read,
            writable: write,
            arch: SegmentArch::Generic,
            name,
            byte_scannable,
        });
    }

    let symbols = extract_symbols(coff, &section_vas);

    Ok(ParseResult {
        arch,
        segments,
        entry_points: vec![],
        symbols,
        pie_base: 0,
        got_slots: FxHashSet::default(),
        got_call_only: false,
        reloc_pointers: Vec::new(),
        extra_names: Vec::new(),
    })
}

/// Assign a sequential virtual address to every section.
///
/// Returns a `Vec<u64>` parallel to `sections`: `section_vas[i]` is the
/// VA at which `sections[i]` is loaded.
fn compute_section_vas(sections: &[goblin::pe::section_table::SectionTable]) -> Vec<u64> {
    let mut vas = Vec::with_capacity(sections.len());
    let mut cursor: u64 = 0;
    for section in sections {
        vas.push(cursor);
        let raw = section.size_of_raw_data as u64;
        let virt = section.virtual_size as u64;
        // Advance by the larger of the two sizes so we never overlap.
        cursor += raw.max(virt);
    }
    vas
}

/// Extract named symbols from the COFF symbol table.
///
/// Only external and static symbols defined in a concrete section
/// (`section_number > 0`) are included.
/// VA = `section_vas[section_number - 1]` + `symbol.value`.
fn extract_symbols(coff: &goblin::pe::Coff, section_vas: &[u64]) -> Vec<Symbol> {
    use goblin::pe::symbol::{IMAGE_SYM_CLASS_EXTERNAL, IMAGE_SYM_CLASS_STATIC};

    let sym_table = match coff.symbols.as_ref() {
        Some(t) => t,
        None => return vec![],
    };

    let mut symbols = Vec::new();

    for (_idx, _inline_name, sym) in sym_table.iter() {
        // section_number == 0  → undefined external reference.
        // section_number < 0   → special (IMAGE_SYM_ABSOLUTE / UNDEFINED / DEBUG).
        if sym.section_number <= 0 {
            continue;
        }
        // Only externally-visible (public) and file-scope (static) symbols.
        if sym.storage_class != IMAGE_SYM_CLASS_EXTERNAL
            && sym.storage_class != IMAGE_SYM_CLASS_STATIC
        {
            continue;
        }

        let section_idx = (sym.section_number - 1) as usize;
        let sec_va = match section_vas.get(section_idx) {
            Some(&v) => v,
            None => continue,
        };

        // VA = section sequential VA + offset within section.
        let va = Va::new(sec_va + sym.value as u64);

        // Resolve the symbol name (handles names > 8 bytes via string table).
        let name: String = if let Some(strtab) = coff.strings.as_ref() {
            match sym.name(strtab) {
                Ok(n) => n.to_string(),
                Err(_) => continue,
            }
        } else {
            match std::str::from_utf8(&sym.name) {
                Ok(n) => n.trim_end_matches('\0').to_string(),
                Err(_) => continue,
            }
        };

        if name.is_empty() {
            continue;
        }

        symbols.push(Symbol { name, va });
    }

    symbols
}
