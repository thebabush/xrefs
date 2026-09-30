# TODO

## Architecture & Design

- [ ] No GOT/IAT map for Mach-O binaries. `parse_macho` returns empty
  `got_slots`, so indirect `BLR Xn` / `CALL [RIP+got]` xrefs to extern
  symbols are never resolved for Mach-O. Needs `__got`/`__la_symbol_ptr`
  parsing. (Symbolic names for Mach-O bind slots and stubs are built
  separately and do work; see `build_macho_bind_names` /
  `build_macho_stub_names`.)
  (`src/loader/macho.rs`)

- [ ] `build_elf_got_slots` is missing x86 (32-bit) relocation types
  (`R_386_GLOB_DAT`=6, `R_386_JMP_SLOT`=7) — GOT-indirect calls on x86
  produce no extern-VA xrefs. ARM32 and x86-64/AArch64 are already handled.
  (`src/loader/elf.rs`)

- [ ] ARM32 armhf binaries have ~26k `data_ptr` FNs per binary (e.g.
  libcrypto-1.1.1n: 26075 FN, rec=0.573). The `.dynsym` hypothesis was
  investigated and disproved: emitting `(st_value_field_va → st_value+pie_base)`
  for every `.dynsym` entry produced ~4400 pure FPs with zero TP gain — IDA
  does not record these. Root cause of the FNs is still unknown; likely
  complex LDR+ADD PC chains or jump-table entries IDA resolves via CFG that
  xr's linear scanner misses.

- [ ] ARM32 classifier H=2 latency leaves a 4-byte blind spot at each
  ISA transition — the mode switch is recorded at word N+1 rather than N.
  Increasing to H=3 adds an 8-byte blind spot but may reduce residual FPs
  on armhf where short Thumb runs inside ARM32 sections cause rapid flipping.
  Worth a benchmark sweep over H=1..4 on the armhf corpus.
  (`src/arch/arm32_mode_classifier.rs`, `src/loader/elf.rs`)

- [ ] `src/arch/arm64.rs` is 1316 lines and growing. Consider splitting into
  `arm64_scan.rs` (linear + ADRP pair scanner) and `arm64_jump_table.rs`
  (jump table recovery) once a natural seam presents itself.
  (`src/arch/arm64.rs`)

## Symbolic names

- [ ] Untested or unsupported name sources: threaded binds (Mach-O); PE
  delay-load imports; PE32 and PE ordinal imports (untested on real
  binaries); GNU-ld 12-byte ARM32 PLT and Thumb PLT; IRELATIVE PLT stubs;
  ELFs with stripped section headers. dyld shared caches only get the names
  their symbols provide.
  (`src/loader/elf.rs`, `src/loader/pe.rs`, `src/loader/macho.rs`)

- [ ] Optional: containing-symbol lookup (`name+0xoff`) for addresses inside
  a function. Names are currently exact-VA only.
  (`src/names.rs`)

## Performance

- [ ] `ContextLine` allocates a `String` for `hex` on every disassembly line
  via `bytes_to_hex`. Writing hex directly into the output buffer would
  avoid the intermediate allocation.
  (`src/output.rs`)

- [ ] `ContextLine::from_disasm` clones `line.text` into a new `String` on
  every context line. Since `DisasmLine` is consumed immediately after
  `from_disasm` is called, the field could be moved instead of cloned.
  (`src/output.rs`)

## Testing

- [ ] Test helpers in `pass.rs` and `arch/arm32.rs` use `Box::leak` to create
  `&'static [u8]` slices for synthetic segments. This leaks memory on
  every test invocation. Harmless for correctness but accumulates with
  `--test-threads=1` under sanitisers. A `ManuallyDrop`+destructor or
  a test-scoped arena would avoid the leaks.
  (`src/pass.rs`, `src/arch/arm32.rs`)

## Minor

- [ ] `benchmark.rs` uses `ahash` while the main crate uses `rustc_hash`.
  Inconsistent but harmless. Consider standardising on one.
  (`src/bin/benchmark.rs`)

- [ ] `benchmark.rs` `run_pass` discards the `PassResult` returned by
  `XrefPass::run` (assigned to `_result`). The `elapsed_ms` and
  `confidence_counts` fields could replace the manual `Instant` timing
  and provide a per-confidence breakdown in the benchmark output.
  (`src/bin/benchmark.rs`)
