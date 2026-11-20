#!/bin/bash
# scripts/gpu/run_B_sweep.sh <DATA_DIR> [shape]
# COVERAGE section 2, GPU group B: mixed_sep and mixed_fused, 5 shapes x 3 t x 7 p, default tile 16.
# Writes results/gpu/B_sweep.csv. With no shape argument the CSV is rewritten from scratch; with a shape
# argument the rows are appended, so the caller owns the file (used only to split a slow job).
# REPO_DIR must be set in the environment (submit_all.sh passes it to sbatch).
#SBATCH -p rtx4060ti16g
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH -J gpuB_sweep
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] || [ -z "${REPO_DIR:-}" ]; then
    echo "usage: REPO_DIR=<repo> run_B_sweep.sh <DATA_DIR> [shape]" >&2
    exit 1
fi
DATA_DIR="$1"
ONLY_SHAPE="${2:-}"

BENCH="$REPO_DIR/gpu/bench_gpu"
if [ ! -x "$BENCH" ]; then
    echo "run_B_sweep.sh: $BENCH not found; build it with make -C gpu" >&2
    exit 1
fi

CSV="$REPO_DIR/results/gpu/B_sweep.csv"
TAG="B_sweep${ONLY_SHAPE:+_$ONLY_SHAPE}"
bash "$REPO_DIR/scripts/gpu/write_env.sh" "$TAG"
if [ -z "$ONLY_SHAPE" ]; then
    rm -f "$CSV"
fi

# dims <shape> sets M K N (SPEC 5).
dims() {
    case "$1" in
        sq1024)    M=1024; K=1024; N=1024 ;;
        sq2048)    M=2048; K=2048; N=2048 ;;
        sq4096)    M=4096; K=4096; N=4096 ;;
        skinny16)  M=16;   K=4096; N=4096 ;;
        skinny128) M=128;  K=4096; N=4096 ;;
        *) echo "run_B_sweep.sh: unknown shape $1" >&2; exit 1 ;;
    esac
}

SHAPES="sq1024 sq2048 sq4096 skinny16 skinny128"
if [ -n "$ONLY_SHAPE" ]; then
    SHAPES="$ONLY_SHAPE"
fi

for shape in $SHAPES; do
    dims "$shape"
    for t in 0.5 2 5; do
        for variant in mixed_sep mixed_fused; do
            for p in 0 0.5 1 2 5 10 20; do
                "$BENCH" --variant "$variant" --shape "$shape" --M "$M" --K "$K" --N "$N" --true-pct "$t" \
                    --iso-pct "$p" --data-dir "$DATA_DIR" --out-csv "$CSV"
            done
        done
    done
done

echo "wrote $CSV"
