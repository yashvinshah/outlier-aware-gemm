#!/bin/bash
# scripts/gpu/run_E_base.sh <DATA_DIR>
# Control for group E (not a COVERAGE group): the base variant mixed_sep on the six group E configurations
# (sq2048 and sq4096, t = 2, p in 1 5 10, default tile 16), run on the node that ran group E, so the before
# and after numbers can be compared on one node. Group B stays the official "before" source (COVERAGE section 5).
# Writes results/gpu/E_base.csv, rewritten from scratch on every run. Submit with --nodelist (see the commands
# given with this script); the node is recorded in results/gpu/env_E_base.txt.
# REPO_DIR must be set in the environment.
#SBATCH -p rtx4060ti16g
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH -J gpuE_base
set -euo pipefail

if [ "$#" -ne 1 ] || [ -z "${REPO_DIR:-}" ]; then
    echo "usage: REPO_DIR=<repo> run_E_base.sh <DATA_DIR>" >&2
    exit 1
fi
DATA_DIR="$1"

BENCH="$REPO_DIR/gpu/bench_gpu"
if [ ! -x "$BENCH" ]; then
    echo "run_E_base.sh: $BENCH not found; build it with make -C gpu" >&2
    exit 1
fi

CSV="$REPO_DIR/results/gpu/E_base.csv"
bash "$REPO_DIR/scripts/gpu/write_env.sh" E_base
rm -f "$CSV"

for shape in sq2048 sq4096; do
    case "$shape" in
        sq2048) dim=2048 ;;
        sq4096) dim=4096 ;;
    esac
    for p in 1 5 10; do
        "$BENCH" --variant mixed_sep --shape "$shape" --M "$dim" --K "$dim" --N "$dim" --true-pct 2 \
            --iso-pct "$p" --data-dir "$DATA_DIR" --out-csv "$CSV"
    done
done

echo "wrote $CSV"
