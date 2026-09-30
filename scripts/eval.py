#!/usr/bin/env python3
"""
Evaluate xr against IDA ground truth JSON.

Usage:
    python3 scripts/eval.py <binary> [--depth 2] [--kind call|jump|data_ptr|all]

Runs ./target/release/xr on the binary, compares against <binary>.xrefs.json,
and prints precision / recall / F1 per xref kind.
"""
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

BINARY_PATH = Path(__file__).parent.parent / "target" / "release" / "xr"
DEPTH_MAP = {0: "scan", 1: "linear", 2: "paired"}

def run_scanner(binary: Path, depth: int = 2) -> list[dict]:
    """Run scanner and return list of {from, to, kind} dicts via CSV output."""
    depth_str = DEPTH_MAP.get(depth, "paired")
    cmd = [str(BINARY_PATH), str(binary), "--depth", depth_str, "--format", "csv", "--no-names"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Scanner error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    xrefs = []
    lines = result.stdout.splitlines()
    if not lines:
        return xrefs
    # Skip CSV header
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            try:
                xrefs.append({"from": int(parts[0], 16), "to": int(parts[1], 16), "kind": parts[2]})
            except ValueError:
                pass
    return xrefs

def load_ground_truth(binary: Path, scanner_pie_base: int = 0x400000) -> list[dict]:
    """Load IDA ground-truth xrefs, rebasing PIE ELF VAs when necessary.

    IDA exports xrefs relative to its own image base.  When the ground-truth JSON
    was generated from a PIE ELF binary (ET_DYN) loaded by IDA at a non-zero base,
    the stored VAs already include IDA's base.  The xr scanner rebases the same
    binary to ``scanner_pie_base`` (default 0x400000, matching IDA's Linux default).

    Two JSON shapes are supported:
      - ``[{...}, ...]``                        — flat list (legacy / non-PIE)
      - ``{"image_base": <int>, "xrefs": [...]}`` — new format with explicit base

    When the explicit ``image_base`` field is present and non-zero the stored VAs are
    already rebased by IDA's base; we rebase them to ``scanner_pie_base`` so that
    from/to addresses align with what xr emits.

    For flat-list files the function assumes VAs are already in the scanner's
    address space (i.e. the ground-truth JSON was pre-rebased, as described in
    STATUS.md under "Ground truth cleanup").
    """
    gt_path = Path(str(binary) + ".xrefs.json")
    with open(gt_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        ida_base = int(raw.get("image_base", 0))
        xrefs = raw.get("xrefs", [])
    else:
        # Flat list — assume VAs are already in scanner address space.
        ida_base = 0
        xrefs = raw

    if ida_base != 0 and ida_base != scanner_pie_base:
        # Shift all from/to VAs from IDA's base to the scanner's PIE base.
        delta = scanner_pie_base - ida_base
        rebased = []
        addr_keys = (
            ("from", "from_va", "src"),
            ("to", "to_va", "dst", "target"),
        )
        for xref in xrefs:
            entry = dict(xref)
            for keys in addr_keys:
                for k in keys:
                    if k in entry and entry[k] is not None:
                        entry[k] = entry[k] + delta
                        break
            rebased.append(entry)
        return rebased

    return xrefs

def normalize_kind(kind: str) -> str:
    k = kind.lower()
    if k in ("call", "fl_cn", "fl_cf"):
        return "call"
    if k in ("jump", "cond_jump", "condjump", "fl_jn", "fl_jf"):
        return "jump"
    if k in ("data_ptr", "data_pointer", "dataptr", "dr_o"):
        return "data_ptr"
    if k in ("data_read", "dr_r"):
        return "data_read"
    if k in ("data_write", "dr_w"):
        return "data_write"
    return k

def compute_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("binary")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--kind", default="all", help="Filter: call, jump, data_ptr, all")
    p.add_argument(
        "--pie-base",
        type=lambda x: int(x, 0),
        default=0x400000,
        help="PIE ELF load base used by xr (default: 0x400000). "
             "Must match --base passed to xr. Used to rebase ground-truth VAs "
             "when the JSON contains an explicit image_base field.",
    )
    args = p.parse_args()

    binary = Path(args.binary)
    print(f"Running scanner on {binary.name} (depth={args.depth})...", flush=True)

    predicted = run_scanner(binary, args.depth)
    ground_truth = load_ground_truth(binary, scanner_pie_base=args.pie_base)

    # Build sets of (from, to, kind) tuples
    def to_key(x, is_gt=False):
        if is_gt:
            fr = x.get("from") or x.get("from_va") or x.get("src")
            to = x.get("to") or x.get("to_va") or x.get("dst") or x.get("target")
            kind = x.get("kind") or x.get("type") or x.get("xref_type", "")
        else:
            fr = x.get("from")
            to = x.get("to")
            kind = x.get("kind", "")
        if fr is None or to is None:
            return None
        return (fr, to, normalize_kind(str(kind)))

    pred_keys = set()
    for x in predicted:
        k = to_key(x)
        if k:
            pred_keys.add(k)

    gt_keys = set()
    for x in ground_truth:
        k = to_key(x, is_gt=True)
        if k:
            gt_keys.add(k)

    # Filter by kind if requested
    target_kinds = None
    if args.kind != "all":
        target_kinds = {args.kind}

    def filter_keys(keys):
        if target_kinds is None:
            return keys
        return {k for k in keys if k[2] in target_kinds}

    pred_f = filter_keys(pred_keys)
    gt_f   = filter_keys(gt_keys)

    tp = len(pred_f & gt_f)
    fp = len(pred_f - gt_f)
    fn = len(gt_f - pred_f)
    prec, rec, f1 = compute_f1(tp, fp, fn)

    print(f"\n── Overall ({args.kind}) ────────────────────────────────────────────────")
    print(f"  Predicted:    {len(pred_f):>8}")
    print(f"  Ground truth: {len(gt_f):>8}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    # Per-kind breakdown
    print(f"\n── Per-kind breakdown ──────────────────────────────────────────────────")
    all_kinds = sorted({k[2] for k in pred_keys | gt_keys})
    for kind in all_kinds:
        p_k = {k for k in pred_keys if k[2] == kind}
        g_k = {k for k in gt_keys  if k[2] == kind}
        tp_k = len(p_k & g_k)
        fp_k = len(p_k - g_k)
        fn_k = len(g_k - p_k)
        pr, re, f = compute_f1(tp_k, fp_k, fn_k)
        print(f"  {kind:12}  pred={len(p_k):6}  gt={len(g_k):6}  "
              f"TP={tp_k:6}  FP={fp_k:5}  FN={fn_k:5}  "
              f"P={pr:.3f}  R={re:.3f}  F1={f:.3f}")

    # Show sample FPs and FNs for call kind
    if args.kind in ("call", "all"):
        call_fps = [k for k in (pred_f - gt_f) if k[2] == "call"]
        call_fns = [k for k in (gt_f - pred_f) if k[2] == "call"]
        print(f"\n── Call FPs (predicted but not in GT): {len(call_fps)} ────────────────")
        for k in sorted(call_fps)[:20]:
            print(f"  from={k[0]:#x}  to={k[1]:#x}")
        print(f"\n── Call FNs (in GT but not predicted): {len(call_fns)} ────────────────")
        for k in sorted(call_fns)[:20]:
            print(f"  from={k[0]:#x}  to={k[1]:#x}")

if __name__ == "__main__":
    main()
