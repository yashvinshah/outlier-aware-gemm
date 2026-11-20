#!/bin/bash
# scripts/gpu/run_C_tiles.sh <DATA_DIR> [shape]
# COVERAGE section 2, GPU group C: mixed_sep and mixed_fused, tile in 8 16 32, shapes sq2048 and sq4096,
# t = 2, p = 5. Writes results/gpu/C_tiles.csv. With no shape argument the CSV is rewritten from scratch;
# with a shape argument the rows are appended, so the caller owns the file (used only to split a slow job).
# REPO_DIR must be set in the environment (submit_all.sh passes it to sbatch).
#SBATCH -p rtx4060ti16g
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH -J gpuC_tiles
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] || [ -z "${REPO_DIR:-}" ]; then
    echo "usage: REPO_DIR=<repo> run_C_tiles.sh <DATA_DIR> [shape]" >&2
    exit 1
fi
DATA_DIR="$1"
ONLY_SHAPE="${2:-}"

BENCH="$REPO_DIR/gpu/bench_gpu"
if [ ! -x "$BENCH" ]; then
    echo "run_C_tiles.sh: $BENCH not found; build it with make -C gpu" >&2
    exit 1
fi

CSV="$REPO_DIR/results/gpu/C_tiles.csv"
TAG="C_tiles${ONLY_SHAPE:+_$ONLY_SHAPE}"
bash "$REPO_DIR/scripts/gpu/write_env.sh" "$TAG"
if [ -z "$ONLY_SHAPE" ]; then
    rm -f "$CSV"
fi

# dims <shape> sets M K N (SPEC 5); group C uses only these two shapes.
dims() {
    case "$1" in
        sq2048) M=2048; K=2048; N=2048 ;;
        sq4096) M=4096; K=4096; N=4096 ;;
        *) echo "run_C_tiles.sh: group C has no shape $1" >&2; exit 1 ;;
    esac
}

SHAPES="sq2048 sq4096"
if [ -n "$ONLY_SHAPE" ]; then
    SHAPES="$ONLY_SHAPE"
fi

for shape in $SHAPES; do
    dims "$shape"
    for variant in mixed_sep mixed_fused; do
        for tile in 8 16 32; do
            "$BENCH" --variant "$variant" --shape "$shape" --M "$M" --K "$K" --N "$N" --true-pct 2 \
                --iso-pct 5 --tile "$tile" --data-dir "$DATA_DIR" --out-csv "$CSV"
        done
    done
done

echo "wrote $CSV"
