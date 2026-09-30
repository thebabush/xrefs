use super::{ParseResult, SegData, Segment};
use crate::loader::{Arch, SegmentArch};
use crate::va::Va;
use anyhow::{anyhow, Result};
use dylex::DyldContext;
use rustc_hash::FxHashSet;
use std::path::Path;

/// Extended return type for dyld cache: includes the DyldContext so the caller
/// can store it and keep the mmaps alive for the lifetime of the segments.
pub(super) struct DyldParseResult {
    pub parsed: ParseResult,
    pub dyld_ctx: DyldContext,
}

pub(super) fn parse_dyld_cache(path: &Path) -> Result<DyldParseResult> {
    let ctx =
        DyldContext::open(path).map_err(|e| anyhow!("failed to open dyld shared cache: {e}"))?;

    let arch = match ctx.architecture() {
        a if a.starts_with("arm64") => Arch::Arm64,
        a if a.starts_with("x86_64") => Arch::X86_64,
        a if a.starts_with("i386") => Arch::X86,
        a => {
            eprintln!("warning: unknown dyld cache arch '{a}', treating as unknown");
            Arch::Unknown
        }
    };

    eprintln!(
        "dyld shared cache: arch={arch:?}  mappings={}  images={}  subcaches={}",
        ctx.mappings.len(),
        ctx.image_count(),
        ctx.subcaches.len(),
    );

    let mut segments = Vec::new();

    for mapping in &ctx.mappings {
        if mapping.size == 0 {
            continue;
        }

        // Zero-copy: slice directly into the DyldContext's mmap(s).
        // Safety: ctx is moved into LoadedBinary::_dyld_ctx and lives at least
        // as long as the segments Vec — field declaration order guarantees
        // _dyld_ctx is dropped after segments.
        let data = match ctx.data_at_addr(mapping.address, mapping.size as usize) {
            Ok(slice) => unsafe { SegData::new(slice) },
            Err(e) => {
                eprintln!(
                    "warning: skipping mapping {:#x}+{:#x}: {e}",
                    mapping.address, mapping.size
                );
                continue;
            }
        };

        segments.push(Segment {
            va: Va::new(mapping.address),
            data,
            executable: mapping.is_executable(),
            readable: mapping.is_readable(),
            writable: mapping.is_writable(),
            byte_scannable: mapping.is_readable() && !mapping.is_executable(),
            arch: SegmentArch::Generic,
            name: format!("DSC[{:#x}]", mapping.address),
        });
    }

    if segments.is_empty() {
        return Err(anyhow!("dyld shared cache: no usable mappings found"));
    }

    Ok(DyldParseResult {
        parsed: ParseResult {
            arch,
            segments,
            entry_points: vec![],
            symbols: vec![],
            pie_base: 0,
            got_slots: FxHashSet::default(),
            reloc_pointers: Vec::new(),
            extra_names: Vec::new(),
        },
        dyld_ctx: ctx,
    })
}
