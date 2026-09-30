#!/usr/bin/env bash
#
# Run the benchmark scorer against every testcase that has ground truth.
# Prints a summary table at the end.
#
# Usage:
#   ./scripts/score_all.sh              # all testcases
#   ./scripts/score_all.sh --depth linear
#   ./scripts/score_all.sh testcases/concrt140.dll   # specific binary
#
set -euo pipefail
cd "$(dirname "$0")/.."

DEPTH="paired"
JOBS=4
TARGETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --depth)  DEPTH="$2"; shift 2 ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        *)        TARGETS+=("$1"); shift ;;
    esac
done

# Build once
cargo build --release --example benchmark 2>&1 | grep -v '^\s*$'

# If no explicit targets, find all with ground truth
if [ ${#TARGETS[@]} -eq 0 ]; then
    for gt in testcases/*.xrefs.json; do
        binary="${gt%.xrefs.json}"
        if [ -f "$binary" ]; then
            TARGETS+=("$binary")
        fi
    done
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "No testcases with ground truth found."
    exit 1
fi

# Collect results
declare -a NAMES=()
declare -a LINES=()

for binary in "${TARGETS[@]}"; do
    gt="${binary}.xrefs.json"
    if [ ! -f "$gt" ]; then
        echo "SKIP $(basename "$binary") — no ground truth"
        continue
    fi
    name=$(basename "$binary")
    # Extract the overall line and parse out just the numbers
    raw=$(cargo run --release --example benchmark -- \
        -b "$binary" -g "$gt" --depth "$DEPTH" --runs 1 -j "$JOBS" 2>&1 \
        | grep '  overall' || true)
    if [ -n "$raw" ]; then
        # Parse: "  overall  xr= 80898  ida= 86867  TP= 80003  FP=   895  FN=  6864  prec=0.989  rec=0.921  F1=0.954"
        read -r tp fp fn prec rec f1 <<< "$(echo "$raw" | \
            sed -E 's/.*TP= *([0-9]+).*FP= *([0-9]+).*FN= *([0-9]+).*prec=([0-9.]+).*rec=([0-9.]+).*F1=([0-9.]+).*/\1 \2 \3 \4 \5 \6/')"
        NAMES+=("$name")
        LINES+=("$(printf '%7s %7s %7s  %5s %5s %5s' "$tp" "$fp" "$fn" "$prec" "$rec" "$f1")")
    else
        NAMES+=("$name")
        LINES+=("FAILED")
    fi
done

echo
printf '%-38s %7s %7s %7s  %5s %5s %5s\n' "Binary" "TP" "FP" "FN" "Prec" "Rec" "F1"
printf '%-38s %7s %7s %7s  %5s %5s %5s\n' "------" "--" "--" "--" "----" "---" "--"
for i in "${!NAMES[@]}"; do
    printf '%-38s %s\n' "${NAMES[$i]}" "${LINES[$i]}"
done
