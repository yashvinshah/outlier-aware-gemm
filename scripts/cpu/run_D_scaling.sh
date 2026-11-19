#!/bin/bash
# scripts/cpu/run_D_scaling.sh [shape ...]: COVERAGE group D, CPU lane.
# fp32_omp, mixed_sep_avx2 and mixed_fused_avx2 on sq2048 and sq4096, t = 2, p = 5 (mixed only), for every
# thread count in the list L. Appends to results/cpu/D_scaling.csv. With shape arguments only those run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SHAPES=$(pick_shapes "sq2048 sq4096" "$@")
CSV=D_scaling.csv
require_fresh "$CSV" $SHAPES
start_job "$(env_tag D_scaling "$@")"

for shape in $SHAPES; do
    for threads in $(thread_list); do
        run_cfg "$CSV" fp32_omp "$shape" 2 NA "$threads"
        run_cfg "$CSV" mixed_sep_avx2 "$shape" 2 5 "$threads"
        run_cfg "$CSV" mixed_fused_avx2 "$shape" 2 5 "$threads"
    done
done
