# xr — Status & Architecture

## Goal

Build `xr`: a standalone Rust crate for ultra-fast, parallel cross-reference
extraction from stripped binaries (ELF, Mach-O, PE). Maximise F1 score across
all xref kinds.

---

## Current Scores (Paired depth)

Tested against ground truth on 32 binaries across ELF (x86-64,
AArch64, ARM32), Mach-O (ARM64), and PE (x86-64, ARM64).

| Category | n | F1 range | Notes |
|----------|----|----------|-------|
| ELF x86-64 | 8 | 0.867–0.969 | Low end: libharlem-shake.so (PLT call FNs). High: curl-amd64 (static, fully resolved) |
| ELF ARM64 | 5 | 0.840–0.952 | Low end: libziggy.so / libssl3 (unresolved ADRP pairs, jump table FNs) |
| Mach-O ARM64 | 1 | 0.982 | Fixup chain parsing recovers most data_ptr |
| PE x86-64 (MinGW/Rust) | 2 | 0.954–0.995 | .pdata + UNWIND_INFO very effective; win32kbase_rs.sys near-perfect |
| PE x86-64 (MSVC) | 8 | 0.561–0.918 | Low end: concrt140.dll (.pdata FPs). 32-bit RVA EH/RTTI invisible to 8-byte scanner |
| PE ARM64 (MSVC) | 2 | 0.623–0.752 | Limited by data_ptr gaps in ADRP-heavy code |
| COFF object (x86-64 / x86-32) | — | — | Format newly supported; no ground-truth benchmark yet |
| ARM32 ELF armel (A32) | 3 | 0.878–0.912 | call prec ≥0.999; data_ptr prec ≥0.982 |
| ARM32 ELF armhf (Thumb-2) | 3 | 0.783–0.878 | call prec ≥0.997; jump FPs reduced by classifier; data_ptr recall limited by dynsym gap |
| ARM32 ELF (Android, mixed) | 1 | 0.881 | was 0.495; classifier eliminated ~254k/260k jump FPs from intra-section ARM↔Thumb transitions |

Call precision is ≥0.948 on all tested binaries. Call recall and F1 vary:
near-perfect on static ELF (curl-amd64 F1=0.999) and armel (F1=0.998), lower
on binaries with many indirect/PLT calls (libharlem-shake.so F1=0.934 from PLT
stub VA mismatch; hello-linux-gcc F1=0.647 from unresolved indirect calls in a
large static binary).

---

## Architecture

### Binary

- **Languages**: Rust (core), Python (analysis scripts)
- **Parallelism**: Custom Rayon thread pool (`n_workers` threads); shard-per-segment dispatch
- **Zero-copy**: `memmap2` mmap, `SegData` newtype segment slices (hides `&'static` lifetime)
- **Depth levels**:
  - `ByteScan` (0): pointer-aligned 8-byte scan of data segments
  - `Linear` (1): sequential instruction decode of exec segments
  - `Paired` (2): ADRP+ADD/LDR pair resolution (ARM64), register const-prop (x86-64)

### Threading model

```
scan workers (rayon custom pool, n_workers threads)
    └─ tx (mpsc unbounded) ──► drain relay thread
                                    └─ out_tx (sync_channel, n_workers*4) ──► output thread
                                                                                  └─ pool.install(on_batch)
```

- **Scan workers**: `n_workers` rayon threads; each scans one shard, sends `Vec<Xref>` via `tx`
- **Drain relay**: pure channel relay; counts xrefs, forwards to output thread without blocking on I/O; sets stop flag on `Break`
- **Output thread**: owns `on_batch` — formats xrefs sequentially (removed `par_iter`; eliminated ~18% runtime spent in rayon contention), then `write_all` once per 8192-record chunk through a 4 MiB `BufWriter`
- **Bounded output channel** (`sync_channel(n_workers*4)`): applies backpressure to scan if output falls behind, bounding peak memory

### ARM64 hot-path decode

`scan_adrp` uses a two-level dispatch:
1. `Arm64Insn::is_tracked(word)` — cheap bitmask union of all tracked encoding families (BL/B/ADRP/ADD/LDR/STR/branches); ~60–70% of instructions return `false`
2. For untracked words: `rd = word & 0x1F`; invalidate `adrp_state[rd]`; `continue` — no enum allocation
3. Only tracked words go through full `Arm64Insn::decode`

### Segment model

Each binary is split into `Segment` structs with:
- `executable: bool` — whether to instruction-scan
- `byte_scannable: bool` — whether to byte-scan for pointers
- For ELF: exec PT_LOADs are split per-section so `.rodata`/`.eh_frame*` inside
  the exec PT_LOAD are `executable=false` (not instruction-scanned)
- `.data.rel.ro` / `.data.rel.ro.local` → `byte_scannable=false`
  (relocation tables produce ~5–29x FP:TP ratio without reloc context)
- PIE ELFs (ET_DYN with first PT_LOAD at p_vaddr==0) are rebased to `0x0040_0000`
  (the default load address used by common disassemblers)
- ARM32 ELF: a depth-6 decision tree (8-byte lookahead, H=2 hysteresis) classifies
  each 4-byte-aligned word as Arm32, Thumb, or Data.  Committed mode changes become
  `ModeSwitch` entries on the segment so `scan_arm32_shard` dispatches the correct
  ISA scanner at every intra-section boundary.  Trained on 59 ARM32 ELF libraries;
  97.9% arm32 recall / 97.8% thumb recall on libamp.so (held-out mixed corpus).
  Falls back to the majority-vote probe (first 16 words) if the classifier never
  locks (section too short / ambiguous).

### Xref kinds

| Kind | Source insns (ARM64) | Source insns (x86-64) |
|------|----------------------|-----------------------|
| Call | BL (exec target only), BLR (resolved) | CALL rel32, CALL r/m64 |
| Jump | B, B.cond, CBZ, CBNZ, TBZ, TBNZ, BR | Jcc, JMP rel, JMP r/m64 |
| DataRead | LDR/LDRB/LDRH + ADRP resolve | MOV [RIP+d], LEA reads |
| DataWrite | STR/STRB/STRH + ADRP resolve | MOV [RIP+d] writes |
| DataPointer | ADRP (emit at ADRP VA, not ADD VA) | LEA RIP+d, byte-scan, CMP/SUB/MOV imm32 |

### Type system

Strong typing throughout:
- `Va` newtype for virtual addresses (not raw `u64`)
- `Reg` newtype (0–30) for ARM64 registers, validated at construction
- `CmpBound`, `JumpTableEntrySize`, `JumpTableAddInfo`, `JumpTablePattern`, `JumpTableCtx` — ARM64 jump table recovery types
- `SegFlags` newtype for segment permission bitmasks
- `RelocPointer`, `Symbol` structs (not bare tuples)

### GOT-indirect call/jump resolution

xr emits `to=got_slot_va` (the real address the CPU dereferences) for
GOT-indirect calls/jumps. The benchmark normalizes extern-target xrefs back
to GOT slot VAs by decoding instruction bytes at each `from`.

On PE the IAT plays the GOT's role, but IDA records `jmp [rip+iat]` import
thunks as a `data_read` of the slot (not a jump), so only `call [rip+iat]`
is emitted as a `call` there (`LoadedBinary::got_call_only`). The slot VA
is goblin's `Import.offset` (`Import.rva` is the hint/name entry). Fixing
this added `call` xrefs only (e.g. sudo.exe +2064: call recall 0.635 → 0.905,
concrt140.dll 0.806 → 0.869, simple.exe 0.870 → 0.965); the kind-agnostic
(from, to) scores are unchanged because those pairs were already matched as
`data_read`.

### Relocation-derived data_ptr recovery

Relocation tables are parsed to extract authoritative pointer pairs:
- **ELF**: `.rela.dyn` / `.rel.dyn` — `R_*_RELATIVE`, `R_*_64` / `R_*_ABS64`
- **PE**: base relocation table (`IMAGE_REL_BASED_DIR64`), `.pdata` exception
  directory, UNWIND_INFO handler RVAs, IAT slots
- **Mach-O**: `LC_DYLD_CHAINED_FIXUPS` — formats 1, 2, 6, 9, 12 (including ARM64E)

These are emitted as `DataPointer` xrefs and bypass `min_ref_va` filtering
(authoritative metadata, not heuristic).

### Symbolic names

`NameTable` (`src/names.rs`) is a VA-sorted multimap built after loading from
`LoadedBinary.symbols` plus loader-specific extra names, and queried per xref
endpoint by exact VA. Printers emit `<a|b>` (text), `from_names`/`to_names`
(jsonl, csv). Disabled with `--no-names`.
- **ELF**: `.symtab`, defined `.dynsym`, GOT slots (GLOB_DAT/JUMP_SLOT), PLT
  stubs (x86-64, AArch64, ARM32)
- **PE**: named exports, IAT slots as `Func@DLL` / `#ordinal@DLL`
- **Mach-O**: `.symtab`, bind slots (chained fixups, dyld-info opcodes),
  `__stubs`/`__auth_stubs` via the indirect symbol table

Known gaps are listed in `docs/TODO.md`.

### Jump table recovery

**x86-64**: Recognises `CMP+JA+LEA+MOVSXD+ADD+JMP` pattern. Reads i32
offset tables from `.rodata`, computes targets, emits `Jump` xrefs. CMP
bound tracking per register limits table size precisely.

**ARM64**: Recognises `ADRP+ADD+CMP+LDRB/LDRH+ADD+BR` patterns with
backward scan from BR. Uses `Reg`-indexed `ScanState`, first-wins
semantics, `JUMP_TABLE_LOOKBACK` window, register chain verification.

---

## File Map

```
src/
  lib.rs                         ← public API re-exports
  main.rs                        ← CLI entry point, output formatting
  va.rs                          ← Va newtype (virtual address)
  xref.rs                        ← Xref, XrefKind, Confidence
  shard.rs                       ← split_range: parallel shard boundaries
  pass.rs                        ← XrefPass: orchestrates parallel scan
  disasm.rs                      ← disassembly context for -A/-B output
  names.rs                       ← NameTable: exact-VA symbol name lookup
  output.rs                      ← Printer trait, text/json/csv formatters
  loader/
    mod.rs                       ← Segment, LoadedBinary, shared types, dispatch
    elf.rs                       ← ELF parsing, GOT slots, reloc pointers
    macho.rs                     ← Mach-O parsing, LC_DYLD_CHAINED_FIXUPS
    pe.rs                        ← PE parsing, .pdata, IAT, base relocations
    coff.rs                      ← COFF object file parsing (sequential VA layout)
    dyld.rs                      ← dyld shared cache
  arch/
    mod.rs                       ← byte_scan_pointers, SegmentDataIndex
    arm32.rs                     ← Thumb-2 + ARM32 (A32) scanners
    arm64.rs                     ← ADRP pair scan, jump table recovery
    arm64_decode.rs              ← pure bitmask ARM64 decoder
    arm32_mode_classifier.rs     ← depth-6 decision tree + hysteresis for ARM32/Thumb classification
    x86_64.rs                    ← x86-64 scanner, jump table recovery
  bin/
    benchmark.rs                 ← benchmark vs ground truth
    fuzz_arm64.rs                ← ARM64 decoder fuzzer

scripts/
  ida_extract_xrefs_binary.py    ← ground-truth extraction script
  batch_extract_xrefs.sh         ← batch ground truth for all testcases
  score_all.sh                   ← run benchmark on all testcases
  eval.py                        ← quick eval without rebuild

testcases/                       ← test binaries + .xrefs.json (gitignored)
```

---

## Remaining Gaps & Root Causes

### ARM64 jump FNs (~495 on curl-aarch64)

Patterns without CMP bound in the backward scan window, or table base
register set outside the lookback window. Diminishing returns.

### ARM64 data_ptr FNs

- **ADD-VA mismatch**: ground truth records xref at ADD VA, xr at ADRP VA.
  Re-enabling ADD-VA gives +6496 TPs / +6981 FPs (net negative).
- **LDR through unresolved registers**: needs interprocedural data flow.
- **Byte-scan pointers to exec segment**: suppressed (10–14x FP:TP ratio).

### x86-64 jump FPs (~5807 on curl-amd64)

~4881 in a 153KB dead zone within `.text` where ground truth records only 13 xrefs.
FDE filtering would remove ~5780 FPs but add ~4946 FNs (net +0.002 F1).

### PE MSVC C++ EH/RTTI data_ptr FNs

MSVC exception handling and RTTI metadata stores references as 32-bit
image-relative RVAs (not 64-bit pointers), invisible to the 8-byte scanner.
Blind 32-bit RVA scanning has 14.5% precision. No tractable fix without
deep MSVC EH metadata parsing.

### PLT call resolution (x86-64 ELF)

`CALL rel32` through PLT stubs → ground truth records `to=extern_va`, xr records
`to=PLT_stub_va`. Causes ~711 call FNs on libharlem-shake.so.

### ARM32 residual jump FPs (~4k on libamp.so, ~2–6k on armhf)

The depth-6 classifier reduced jump FPs from ~260k to ~4k on libamp.so and
from ~10–15k to ~2–6k on armhf binaries.  Remaining FPs come from literal pool
words where the 8-byte lookahead window straddles a mode boundary and the
H=2 filter hasn't committed yet (4-byte latency at each transition).

### ARM32 data_ptr recall ~70–85%

The register-state scanner captures most intra-function LDR+ADD PC pairs.
Remaining FNs are primarily:

- **Cross-function GOT-pointer chains** (~15%): some binaries pass a GOT
  offset as a function argument; the `LDR` is in the caller and `ADD PC` in
  the callee.  Requires inter-procedural data-flow analysis.
- **`.dynsym` xrefs** (~13k per armhf binary): IDA records `data_ptr` from
  each dynamic symbol entry to the symbol's `st_value`.  These are ELF
  structure references, not code xrefs.  Low priority to replicate.
- **Thumb literal-pool absolute VAs** (<200 per binary): a tiny number of
  `.text` pool words that contain absolute pre-link VAs but are never used
  in a `LDR+ADD PC` pair (loaded directly into a register for use as a
  function pointer).

### data_write FNs

All register-based stores where the base register was set far earlier
(function arg or overwritten beyond the ADRP window). Requires
interprocedural data flow.

---

## What Was Tried & Outcome

### Fixes that worked

| Fix | Impact | Notes |
|-----|--------|-------|
| ARM32 section-probe mode detection | armel binaries 0.000→0.715–0.763 | Top-nibble probe replaces name heuristic; correctly classifies armel (A32) vs armhf (Thumb) |
| ARM32 depth-6 decision tree classifier (H=2) | libamp.so 0.495→0.881; armhf +0.05 | Sliding per-word ISA classifier with hysteresis replaces section-wide probe; eliminates ~254k/260k jump FPs on mixed Android binary |
| ARM32 R_ARM_RELATIVE reloc parsing | data_ptr 0 TPs → 2k–24k TPs | REL in-place addend; vma_to_file helper for PT_LOAD mapping |
| ARM32 LDR+ADD PC pair detection | armel data_ptr F1 0.189–0.292 → 0.268–0.361 | Adjacent `LDR Rd,[PC,#N]; ADD Rd,PC,Rd` pair; emits DataPointer from LDR, ADD, and pool |
| Thumb+A32 register-state LDR+ADD scanner | armhf 0.627–0.651 → 0.766–0.832; armel 0.715–0.763 → 0.875–0.918 | Non-adjacent pairs via ldr_st[16]; analogous to ARM64 ADRP scanner |
| 32-bit wrap fix for LDR+ADD resolver | armel data_ptr F1 0.268 → 0.774 (libssl3) | pool_word signed offset; u64 cast silently discarded ~87% of pairs with negative offsets |
| seg_data in ScanRegion | small improvement on multi-worker scans | read_pool now uses full segment data, not just shard slice |
| GOT slot VA approach | blackcat call F1 0.644→0.964 | Emit to=got_slot_va, normalize in benchmark |
| ELF reloc data_ptr | +24k TPs across all PIE ELFs | R_*_RELATIVE + R_*_64/ABS64 |
| Mach-O fixup chain parsing | hello.aarch64 F1 0.946→0.980 | Formats 2, 6, 1, 9, 12 |
| PE .pdata + UNWIND_INFO | win32kbase F1 0.894→0.995 | 4 xrefs per RUNTIME_FUNCTION |
| PE IAT slot population | PE indirect calls work | `got_slots` from PE import table |
| x86-64 jump table recovery | curl-amd64 jump FN 3448→403 | CMP+MOVSXD+ADD+JMP pattern |
| ARM64 jump table recovery | curl-aarch64 jump FN 3310→2815 | ADRP+ADD+CMP+LDR+ADD+BR pattern |
| Exec PT_LOAD section split | −1915 ARM64 jump FPs | .rodata in exec PT_LOAD → non-exec |
| Suppress ADD-VA data_ptr | ARM64 +0.015 | Emit at ADRP VA only |
| `.data.rel.ro` byte scan suppress | x86-64 +0.096 | 5:1 FP:TP ratio without reloc context |
| BLR/BR exec-target suppression | −77 call FP, −118 jump FP | Non-exec targets suppressed |
| Pure Rust ARM64 decoder | −28% CPU (was memset) | Replaced bad64 C FFI |
| is_tracked fast-path | −18% decode cost | Skip 65% of instructions |

### Fixes that were tried and reverted

| Fix | Why abandoned |
|-----|---------------|
| Re-enable ADD-VA data_ptr | +6496 TPs but +6981 FPs; net F1 +0.001 |
| .pdata xrefs with field-offset `from` | Ground truth uses entry start VA, not field offsets. 0 TP. |
| Blind 32-bit RVA scan of .rdata | 14.5% precision — too many random u32 matches |
| UNWIND_INFO scope table parsing | Layout varies by handler type. 2937 FP. Handler RVA alone is 100% precise. |
| FDE/`.eh_frame` coverage filter | −5780 FPs but +4946 FNs; net +0.002 F1 |
| Forward register tracker for data_write | F1 0.407 vs 0.541 — register reuse causes massive FPs |
| Extern VA replication algorithm | Binary-dependent layout; 5184 FP on blackcat.elf |
