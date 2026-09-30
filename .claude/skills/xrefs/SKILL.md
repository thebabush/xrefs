---
name: xrefs
description: Find cross-references (callers, callees, data references) in native binaries using the `xr` tool. Use when reverse engineering ELF, Mach-O, PE, or dyld-cache binaries and you need to find what calls a function, what references a symbol, who reads/writes a data address, or what code a given function calls — without reaching for objdump/grep pipelines.
---

# xr — Binary Cross-Reference Tool

`xr` is a fast, multi-threaded cross-reference scanner for native binaries. It extracts caller→callee edges and data references directly from machine code — no debug info or symbols required. Where symbols or import tables exist, names are printed next to addresses.

Install with `cargo install --path .` from the repo root, or run directly via `cargo run --release --`.

**Reach for `xr` instead of `objdump | grep` when you need to:**
- Find all callers of a function at a known address
- Find all references to a data symbol or GOT entry
- Map the call graph around a specific address
- See what a function calls (its callees)
- Locate pointer references to a region in data sections

## Supported formats & architectures

| Format | Architectures (fully scanned) |
|--------|-------------------------------|
| ELF | x86-64, AArch64 |
| Mach-O (single-arch) | x86-64, ARM64 |
| PE / COFF | x86-64, ARM64 |
| Apple dyld shared cache | arm64, arm64e, x86_64, x86_64h |
| Raw flat binary | treated as single executable segment at VA 0 |

x86-32 and ARM32 are parsed and loaded but not yet scanned.
**Fat/universal Mach-O**: run `lipo -extract <arch> input output` first.

## Quick reference

```
xr [OPTIONS] <BINARY>

  -f text|jsonl|csv       Output format (default: text)
  -k call|jump|data_read|data_write|data_ptr
                          Filter to one xref kind
  -d byte-scan|linear|paired
                          Analysis depth (default: paired)
  -B N / -A N             Show N disasm lines before/after each xref site
  --ref-start VA          Keep only xrefs whose target >= VA
  --ref-end   VA          Keep only xrefs whose target <  VA
  --start VA / --end VA   Restrict scan to source addresses in [VA, VA)
  --base VA               Override PIE ELF load base (default 0x400000)
  --limit N               Cap output at N results
  -j N                    Worker threads (0 = all CPUs)
  --rust                  Extract Rust string literals from data_ptr xrefs
  --rust-min-blob N       Min UTF-8 blob size in bytes (default: 16)
  --rust-string-max N     Max display width for strings; 0 = no limit (default: 100)
  --no-names              Do not print symbol names (on by default)
```

VAs accept `0x`-prefixed hex or plain decimal.

## Xref kinds

| Kind | What it means |
|------|---------------|
| `call` | Direct or indirect call (CALL / BL / BLR) |
| `jump` | Unconditional or conditional branch (JMP / B / Jcc / CBZ…) |
| `data_read` | Load from a data address (LDR / MOV from [rip+disp]) |
| `data_write` | Store to a data address (STR / MOV to [rip+disp]) |
| `data_ptr` | Pointer in a data section pointing into code/data; ADRP+ADD pairs; LEA [rip+disp] |

## Common RE workflows

### 1. Find all callers of a function

```bash
# Who calls the function at 0x401234?
xr binary --ref-start 0x401234 --ref-end 0x401235 -k call

# Unique caller list
xr binary --ref-start 0x401234 --ref-end 0x401235 -k call -f csv \
  | awk -F, '{print $1}' | sort -u
```

### 2. Find all callees of a function

```bash
# What does the function at 0x401000–0x4012ff call?
xr binary --start 0x401000 --end 0x401300 -k call
```

### 3. Find all references to a data address

```bash
# What reads or writes the global at 0x408000?
xr binary --ref-start 0x408000 --ref-end 0x408008 -k data_read
xr binary --ref-start 0x408000 --ref-end 0x408008 -k data_write

# All reference kinds at once
xr binary --ref-start 0x408000 --ref-end 0x408008
```

### 4. Inspect call sites with disassembly context

```bash
# Like grep -B3 -A3 but for machine code
xr binary --ref-start 0x401234 --ref-end 0x401235 -k call -B 3 -A 3

# JSONL with embedded context for scripting
xr binary --ref-start 0x401234 --ref-end 0x401235 -k call -B 2 -A 2 -f jsonl
```

### 5. Scan only one function's range (avoid full-binary cost)

```bash
xr binary --start 0x401000 --end 0x402000
```

### 6. Pointer hunting in data sections

```bash
# Find all data pointers into the .text range 0x401000–0x410000
xr binary --ref-start 0x401000 --ref-end 0x410000 -k data_ptr
```

### 7. PIE binary with custom base

```bash
# Rebase a PIE ELF to the address ASLR placed it at runtime
xr binary --base 0x7f4000000000 \
  --ref-start 0x7f4000401234 --ref-end 0x7f4000401235
```

### 8. dyld shared cache

```bash
# Works directly on the cache — no extraction needed
xr /System/Library/dyld/dyld_shared_cache_arm64e \
  --ref-start 0x1a3b00000 --ref-end 0x1a3b00010 -k call -B 2
```

### 9. Extract Rust string literals from a stripped binary

```bash
# Show all data pointers that resolve to string literals
xr rust_binary --rust -k data_ptr

# Grep for a specific error message to find where it's referenced
xr rust_binary --rust -k data_ptr 2>/dev/null | grep "connection refused"

# Find all string refs in a function's address range
xr rust_binary --rust -k data_ptr --start 0x401000 --end 0x402000

# JSONL output with string field for scripting
xr rust_binary --rust -k data_ptr -f jsonl \
  | jq -r 'select(.string) | "\(.from) \(.string)"'

# Widen blobs for binaries with many small strings
xr rust_binary --rust --rust-min-blob 8 -k data_ptr

# Show full strings without truncation
xr rust_binary --rust --rust-string-max 0 -k data_ptr
```

Rust concatenates string literals into large blobs in `.rodata`. Each `&str` is
a `(ptr, len)` pair — `--rust` resolves these automatically. Especially useful
for stripped binaries where symbols are gone but panic messages, error strings,
and file paths survive — giving you landmarks to navigate the binary.

### 10. Quick call graph (JSONL + jq)

```bash
xr binary -k call -f jsonl \
  | jq -r '"\(.from) -> \(.to)"' \
  | sort -u | head -50
```

Note: `from` and `to` in JSONL output are decimal integers. To get hex:
```bash
xr binary -k call -f jsonl \
  | jq -r '"0x" + (.from | tostring | ltrimstr("0x")) + " -> 0x" + (.to | tostring | ltrimstr("0x"))'
```

Or use text format for human-readable hex output.

## Output formats

**text** (default):
```
0x0000000000401234 -> 0x0000000000403000  call  [linear-immediate]
```
With `-B`/`-A`, indented disasm lines follow each xref entry.
Names defined at exactly the `from`/`to` VA are shown as `<a|b>` after the address (omitted when there are none):
```
0x000000000046f1a9 -> 0x000000000041f3c0 <ERR_new>  call  [linear-immediate]
```
With `--rust`, resolved strings are appended inline:
```
0x00000001002b4b68 -> 0x00000001002744d2  data_ptr  [byte-scan]  "failed to write the buffered data"
```

**jsonl** (JSON Lines — one JSON object per line):
```
{"from":4198964,"to":4206592,"kind":"call","confidence":"linear-immediate"}
```
With `-B`/`-A`, each object gains a `"context"` array of `{va, hex, text, focus}`.
With `--rust`, matching records gain a `"string"` field.
Records gain `"from_names"` / `"to_names"` string arrays when names exist (omitted when empty):
```
{"from":4622960,"to":4322240,"kind":"call","confidence":"linear-immediate","to_names":["ERR_new"]}
```

**csv**: `from,to,kind,confidence,string,from_names,to_names` per line — no context even with `-B`/`-A`.
The `string` column is populated when `--rust` is active. `from_names`/`to_names` join multiple names with `|` and are empty when there is none. Columns 1-2 (`from`, `to`) are unchanged, so `awk -F,` on them still works.

**Names**: raw symbol names (no demangling; Mach-O keeps the leading `_`), exact-VA only — an address inside a function has no name, and there is no `sub_XXXX` fallback. Sources: ELF symtab/dynsym/GOT/PLT stubs, PE exports and IAT (`Func@DLL`), Mach-O symtab/binds/stubs. Stripped binaries get import/dynamic names only. Not covered: threaded binds, PE delay-load, IRELATIVE PLT stubs, dyld cache beyond its own symbols. Use `--no-names` for the old output (e.g. when parsing text lines with a strict regex) or to skip the lookups.

## Analysis depth (`-d`)

| Value | Finds | Speed |
|-------|-------|-------|
| `byte-scan` | Pointer-sized values in data sections | Fastest, most false positives |
| `linear` | Direct branches + RIP-relative / ADRP memory refs | Fast, ~85% recall |
| `paired` (default) | All of linear + ADRP pairs (ARM64) + register const-prop (x86-64) | Best quality |

## Confidence levels (in output)

| Value | Source |
|-------|--------|
| `linear-immediate` | Direct branch/call with immediate target |
| `pair-resolved` | ADRP+ADD/LDR pair (ARM64) or RIP-relative LEA/MOV (x86-64) |
| `local-prop` | Register constant propagation (`MOV rax, imm64; CALL rax`) |
| `byte-scan` | Pointer-sized value in a data section — highest false positive rate |

## Tips

- **Names in text output**: with names on, a text line can have `<...>` after either address. Use `--no-names` before parsing text with fixed-column tools, or prefer `-f jsonl`.
- **Start narrow**: pin a target with `--ref-start`/`--ref-end` rather than scanning everything and post-filtering.
- **Disasm context** (`-B`/`-A`) is anchor-decoded from the exact xref VA — no alignment drift. It replaces `objdump -d | sed -n '/addr/,+N p'` workflows.
- **Fat Mach-O**: if `xr` rejects the binary, run `lipo -extract arm64 binary thin` first.
- **ARM64 indirect calls** (BLR xN) resolved via ADRP+ADD pairing are emitted as `call` with `pair-resolved` confidence.
- **False positives**: `byte-scan` confidence entries are speculative. Filter them out with `-k call` or `-k data_read` when you want only decoded references.
- **No xrefs found? Widen the target range.** Binaries have strong address locality — related functions cluster together. If searching for refs to `0x401234` turns up nothing (e.g. the target is reached via an indirect call that wasn't resolved), try widening `--ref-end` to cover the whole function or the surrounding ~0x100–0x1000 byte neighborhood. Refs to nearby addresses are likely from the same callers.

  ```bash
  # Nothing found for exact address — try the whole function body
  xr binary --ref-start 0x401200 --ref-end 0x401400 -k call

  # Or scan neighbors: anything referencing the page around 0x401234
  xr binary --ref-start 0x401000 --ref-end 0x402000 -k call
  ```

  Similarly, if you find refs to a neighbor function `0x401300`, the caller likely also calls or is adjacent to your target — pivot to scanning that caller's range (`--start`/`--end`) to see what else it references.
- **Stripped Rust binary?** Use `--rust` to recover string literals. Panic messages, error strings, and source file paths survive stripping and give you function-level landmarks even without symbols. Combine with `--start`/`--end` to see what strings a specific function references.
