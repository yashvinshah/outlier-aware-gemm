#!/bin/bash
# scripts/gpu/run_golden.sh <DATA_DIR>
# SPEC 13 steps 1 and 3 for the GPU lane: golden shape, t = 2, p in {0, 2, 10}, tile in {8, 16, 32}.
# Writes results/gpu/golden/checksums_gpu.txt (one line per run) and the C files under
# <DATA_DIR>/golden_out/gpu/ (SPEC 17.5). Needs gen_data output for the golden shape in <DATA_DIR>
# and gpu/bench_gpu built. A run that fails stops the script.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: run_golden.sh <DATA_DIR>" >&2
    exit 1
fi
DATA_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH="$REPO_DIR/gpu/bench_gpu"
if [ ! -x "$BENCH" ]; then
    echo "run_golden.sh: $BENCH not found; build it with make -C gpu" >&2
    exit 1
fi

GOLDEN_DIR="$REPO_DIR/results/gpu/golden"
C_DIR="$DATA_DIR/golden_out/gpu"
CHECKSUMS="$GOLDEN_DIR/checksums_gpu.txt"
CSV="$GOLDEN_DIR/golden_runs.csv"
mkdir -p "$GOLDEN_DIR" "$C_DIR"
: > "$CHECKSUMS"
rm -f "$CSV"

bash "$SCRIPT_DIR/write_env.sh" golden

# run_one <variant> <tile> <p>; p is NA for int8_pertensor (SPEC 17.5).
run_one() {
    local variant="$1" tile="$2" p="$3"
    local cfile="$C_DIR/${variant}_tile${tile}_p${p}.f32"
    local out
    if [ "$p" = "NA" ]; then
        out=$("$BENCH" --variant "$variant" --shape golden --M 256 --K 256 --N 256 --true-pct 2 \
            --tile "$tile" --data-dir "$DATA_DIR" --out-csv "$CSV" --write-c "$cfile")
    else
        out=$("$BENCH" --variant "$variant" --shape golden --M 256 --K 256 --N 256 --true-pct 2 \
            --iso-pct "$p" --tile "$tile" --data-dir "$DATA_DIR" --out-csv "$CSV" --write-c "$cfile")
    fi
    local hex
    hex=$(echo "$out" | sed -n 's/^C_CHECKSUM //p')
    if [ -z "$hex" ]; then
        echo "run_golden.sh: no C_CHECKSUM from $variant tile=$tile p=$p" >&2
        exit 1
    fi
    echo "$variant tile=$tile p=$p $hex" >> "$CHECKSUMS"
}

for tile in 8 16 32; do
    run_one int8_pertensor "$tile" NA
done
for variant in mixed_sep mixed_fused; do
    for p in 0 2 10; do
        for tile in 8 16 32; do
            run_one "$variant" "$tile" "$p"
        done
    done
done

echo "wrote $CHECKSUMS"
