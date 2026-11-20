#!/bin/bash
# scripts/cpu/run_B_sweep.sh [shape ...]: COVERAGE group B, CPU lane.
# mixed_sep_avx2 and mixed_fused_avx2 on the five shapes, t in {0.5, 2, 5}, p in {0, 0.5, 1, 2, 5, 10, 20},
# at T threads. Appends to results/cpu/B_sweep.csv. With shape arguments only those run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SHAPES=$(pick_shapes "sq1024 sq2048 sq4096 skinny16 skinny128" "$@")
CSV=B_sweep.csv
require_fresh "$CSV" $SHAPES
start_job "$(env_tag B_sweep "$@")"

for shape in $SHAPES; do
    for t in 0.5 2 5; do
        for p in 0 0.5 1 2 5 10 20; do
            run_cfg "$CSV" mixed_sep_avx2 "$shape" "$t" "$p" "$T"
            run_cfg "$CSV" mixed_fused_avx2 "$shape" "$t" "$p" "$T"
        done
    done
done
