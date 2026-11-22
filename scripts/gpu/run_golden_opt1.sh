#!/bin/bash
# scripts/gpu/run_golden_opt1.sh <DATA_DIR>
# SPEC 13 step 6: golden check of mixed_sep_opt1 with the parameters of its base variant mixed_sep
# (golden shape, t = 2, p in 0 2 10, tile in 8 16 32). Appends 9 lines to results/gpu/golden/checksums_gpu.txt
# (lines of an earlier run of this script are replaced, the other lines are kept) and writes the C files to
# <DATA_DIR>/golden_out/gpu/mixed_sep_opt1_tile<n>_p<p>.f32 (SPEC 17.5).
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: run_golden_opt1.sh <DATA_DIR>" >&2
    exit 1
fi
DATA_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH="$REPO_DIR/gpu/bench_gpu"
if [ ! -x "$BENCH" ]; then
    echo "run_golden_opt1.sh: $BENCH not found; build it with make -C gpu" >&2
    exit 1
fi

GOLDEN_DIR="$REPO_DIR/results/gpu/golden"
C_DIR="$DATA_DIR/golden_out/gpu"
CHECKSUMS="$GOLDEN_DIR/checksums_gpu.txt"
CSV="$GOLDEN_DIR/golden_runs_opt1.csv"
if [ ! -f "$CHECKSUMS" ]; then
    echo "run_golden_opt1.sh: $CHECKSUMS not found; run run_golden.sh first" >&2
    exit 1
fi
mkdir -p "$C_DIR"
rm -f "$CSV"

bash "$SCRIPT_DIR/write_env.sh" golden_opt1

NEW="$GOLDEN_DIR/checksums_opt1_new.tmp"
: > "$NEW"
for p in 0 2 10; do
    for tile in 8 16 32; do
        out=$("$BENCH" --variant mixed_sep_opt1 --shape golden --M 256 --K 256 --N 256 --true-pct 2 \
            --iso-pct "$p" --tile "$tile" --data-dir "$DATA_DIR" --out-csv "$CSV" \
            --write-c "$C_DIR/mixed_sep_opt1_tile${tile}_p${p}.f32")
        hex=$(echo "$out" | sed -n 's/^C_CHECKSUM //p')
        if [ -z "$hex" ]; then
            echo "run_golden_opt1.sh: no C_CHECKSUM from mixed_sep_opt1 tile=$tile p=$p" >&2
            exit 1
        fi
        echo "mixed_sep_opt1 tile=$tile p=$p $hex" >> "$NEW"
    done
done

# grep -v exits 1 when it keeps no line; that case does not apply here, the file always has the base lines.
grep -v '^mixed_sep_opt1 ' "$CHECKSUMS" > "$CHECKSUMS.tmp"
cat "$NEW" >> "$CHECKSUMS.tmp"
mv "$CHECKSUMS.tmp" "$CHECKSUMS"
rm -f "$NEW"

echo "updated $CHECKSUMS"
