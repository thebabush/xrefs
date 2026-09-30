# xr — fast binary cross-reference extractor

`xr` is a standalone Rust CLI that extracts cross-references — calls, jumps, and
data references — from stripped binaries (ELF, Mach-O, PE, and the Apple dyld
shared cache) in seconds. It emits `(from_va, to_va, kind)` tuples orders of
magnitude faster than loading the binary into a full disassembler like IDA,
Ghidra, or Binary Ninja, making it a fast building block for reverse-engineering
pipelines, binary diffing, and large-scale program analysis.

## Install

```sh
cargo install xrefs        # crate is `xrefs`, binary is `xr`
```

Or build from source with `cargo build --release` (binary at `target/release/xr`).

## Quick Start

```sh
# Analyse a binary at the recommended depth
xr /path/to/binary --depth paired

# Output as JSONL or CSV
xr /path/to/binary --depth paired --format jsonl
xr /path/to/binary --depth paired --format csv

# Filter to a specific xref kind
xr /path/to/binary --depth paired --kind call

# Show disasm context around each xref site (like grep -A/-B)
xr /path/to/binary --depth paired -A 3 -B 2
```

## Performance

206 million xrefs from a 4.6 GB **dyld shared cache** (3240 images) in under 6 seconds:

```
$ xr /System/Library/dyld/dyld_shared_cache_x86_64 > /dev/null
dyld shared cache: arch=X86_64  mappings=24  images=3240  subcaches=5
xrefs: 206432528  |  5.7s  |  4620.3 MB scanned  |  24 segments
```

## Example Output

Default text output:
```
$ xr binary --depth paired --limit 4
0x00000000006a268f -> 0x00000000006a2699  jump  [linear-immediate]
0x00000000006a26a1 -> 0x00000000006a25ea  call  [linear-immediate]
0x00000000006a26b7 -> 0x0000000000798345  call  [linear-immediate]
0x00000000006a26cf -> 0x00000000006a2368  data_ptr  [linear-immediate]
```

With disassembly context (`-B 2 -A 1`):
```
$ xr binary -k call --limit 2 -B 2 -A 1
0x00000000006a26d6 -> 0x00000000006a2393  call  [linear-immediate]
    0x00000000006a26ca  48 8d 4c 24 0c            lea rcx, [rsp+0Ch]
    0x00000000006a26cf  48 8d 15 92 fc ff ff      lea rdx, [6A2368h]
  > 0x00000000006a26d6  e8 b8 fc ff ff            call 00000000006A2393h
    0x00000000006a26db  48 83 c4 18               add rsp, 18h
```

With Rust string heuristics (`--rust`):
```
$ xr target/release/my_app --rust -k data_ptr
...
0x00000001002b0238 -> 0x0000000100243834  data_ptr  [byte-scan]  "STRIKETHROUGH"
0x00000001002b4b68 -> 0x00000001002744d2  data_ptr  [byte-scan]  "failed to write the buffered data"
0x00000001002b5de0 -> 0x000000010022e8b0  data_ptr  [byte-scan]  "src/arch/arm64.rs"
0x00000001002b5e10 -> 0x000000010022e8c2  data_ptr  [byte-scan]  "src/arch/x86_64.rs"
```

## Symbolic Names

By default each xref's `from` and `to` show the name(s) defined at exactly that
address. Pass `--no-names` to turn this off.

```
$ xr testcases/libssl3-amd64.so.3 -k call
0x000000000046f1a9 -> 0x000000000041f3c0 <ERR_new>  call  [linear-immediate]
0x0000000000421000 <gettimeofday> -> 0x00000000004a4b30 <gettimeofday>  jump  [linear-immediate]
```

JSONL adds `from_names` / `to_names` string arrays, omitted when empty:
```
{"from":4622960,"to":4322240,"kind":"call","confidence":"linear-immediate","to_names":["ERR_new"]}
```

CSV gets trailing `from_names,to_names` columns. Multiple names at one address
are joined with `|` (text: `<a|b>`).

- Names are raw: no demangling, Mach-O leading underscores are kept, versioned
  ELF dynsym names appear as stored, control characters are escaped in text
  output.
- Exact-VA lookup only: an address inside a function gets no name (no `+off`),
  and there is no `sub_XXXX` fallback.
- Sources: ELF `.symtab`, defined `.dynsym`, GOT slots (GLOB_DAT/JUMP_SLOT
  relocs) and PLT stubs (x86-64, AArch64, ARM32); PE named exports and IAT slots
  (`Func@DLL`, `#ordinal@DLL`); Mach-O `.symtab`, bind slots (chained fixups and
  dyld-info bind/lazy/weak opcodes) and `__stubs`/`__auth_stubs`. Stripped
  binaries only get import/dynamic names, not local functions.
- Not covered: threaded binds, PE delay-load imports, IRELATIVE PLT stubs,
  ELFs with stripped section headers, GNU-ld 12-byte ARM32 PLT and Thumb PLT.
  PE32 and PE ordinal imports are untested on real binaries. dyld shared caches
  get only the names their symbols provide.

## Analysis Depths

| Flag | Name | What it does |
|------|------|-------------|
| `--depth scan` | ByteScan | Pointer-sized byte scan of data sections |
| `--depth linear` | Linear | Linear disasm, immediate targets + RIP-relative |
| `--depth paired` | Paired | ADRP+ADD/LDR pairs (ARM64) or register prop (x86-64), **recommended** |

## Accuracy

See [docs/STATUS.md](docs/STATUS.md) for architecture details and known gaps.

## Supported Formats

- ELF (x86-64, AArch64), including PIE (ET_DYN)
- Single-arch Mach-O (x86-64, ARM64). Fat binaries require `lipo -extract` first
- PE / COFF (x86-64, ARM64)
- ELF ARM32 (Thumb-2 / A32), with per-section mode detection
- COFF object files / `.o` / `.obj` (x86-64, ARM64, x86-32, ARM32)
- Apple dyld shared cache
- Raw flat binary (treated as single executable segment)

COFF object files have no fixed load address; xr assigns VAs by stacking
sections sequentially from 0 (matching the convention used by common disassemblers).

x86-32 binaries are loaded but not scanned (architecture stub only).

## Options

```
USAGE:
    xr [OPTIONS] <BINARY>

OPTIONS:
    -d, --depth <DEPTH>         Analysis depth: scan | linear | paired [default: paired]
    -j, --workers <N>           Worker threads; 0 = all CPUs [default: 0]
    -f, --format <FORMAT>       Output format: text | jsonl | csv [default: text]
    -k, --kind <KIND>           Filter by kind: call | jump | data_read | data_write | data_ptr
        --base <VA>             Override PIE ELF load base (hex or decimal)
        --arm32-mode <MODE>     ARM32 decode mode: auto (default) | thumb | arm
        --min-ref-va <VA>       Drop xrefs whose 'to' VA is below this value
        --start <VA>            Scan only 'from' addresses >= VA
        --end <VA>              Scan only 'from' addresses < VA
        --ref-start <VA>        Retain only xrefs with 'to' >= VA
        --ref-end <VA>          Retain only xrefs with 'to' < VA
        --limit <N>             Cap output at N xrefs (0 = unlimited)
    -A, --after-context <N>     Show N instructions after each xref site
    -B, --before-context <N>    Show N instructions before each xref site
        --rust                  Extract Rust string literals from data_ptr xrefs
        --rust-min-blob <N>     Min UTF-8 blob size in bytes [default: 16]
        --rust-string-max <N>   Max display width for strings (0 = unlimited) [default: 100]
        --no-names              Do not print symbol names for xref addresses
```

## Benchmarking

The `benchmark` helper ships as an example (it is not installed by `cargo install`):

```sh
# Run against a ground-truth file
cargo run --release --example benchmark -- \
    --binary /path/to/binary \
    --ground-truth /path/to/binary.xrefs.json \
    --depth paired
```

## Claude Code Skill

This project includes a [Claude Code skill](https://docs.anthropic.com/en/docs/claude-code/skills)
in `.claude/skills/xrefs/` that teaches Claude how to use `xr` for binary
reverse-engineering workflows: finding callers/callees, data references,
pointer hunting, and more.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
