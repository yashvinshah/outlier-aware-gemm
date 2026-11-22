#!/bin/bash
# scripts/gpu/run_E_opt.sh <DATA_DIR>
# COVERAGE section 2 and 5, GPU group E: mixed_sep_opt1, shapes sq2048 and sq4096, t = 2, p in 1 5 10, default
# tile 16 (6 rows). Writes results/gpu/E_opt.csv, rewritten from scratch on every run.
# REPO_DIR must be set in the environment.
#SBATCH -p rtx4060ti16g
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH -J gpuE_opt
set -euo pipefail

if [ "$#" -ne 1 ] || [ -z "${REPO_DIR:-}" ]; then
    echo "usage: REPO_DIR=<repo> run_E_opt.sh <DATA_DIR>" >&2
    exit 1
fi
DATA_DIR="$1"

BENCH="$REPO_DIR/gpu/bench_gpu"
if [ ! -x "$BENCH" ]; then
    echo "run_E_opt.sh: $BENCH not found; build it with make -C gpu" >&2
    exit 1
fi

CSV="$REPO_DIR/results/gpu/E_opt.csv"
bash "$REPO_DIR/scripts/gpu/write_env.sh" E_opt
rm -f "$CSV"

for shape in sq2048 sq4096; do
    case "$shape" in
        sq2048) dim=2048 ;;
        sq4096) dim=4096 ;;
    esac
    for p in 1 5 10; do
        "$BENCH" --variant mixed_sep_opt1 --shape "$shape" --M "$dim" --K "$dim" --N "$dim" --true-pct 2 \
            --iso-pct "$p" --data-dir "$DATA_DIR" --out-csv "$CSV"
    done
done

echo "wrote $CSV"
