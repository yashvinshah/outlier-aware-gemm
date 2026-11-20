#!/bin/bash
# scripts/cpu/run_smoke.sh [shape ...]: small job that exercises the Round 3 scripts' code paths before the
# full groups: every variant once, p = 0 (the k = 0 path) and p = 5, and the thread list L. Default
# shapes: golden and sq1024. The CSV goes to data/smoke/smoke.csv (git-ignored); the env file goes to
# results/cpu/env_smoke.txt like every other env file. The RUN lines give wall times for walltime estimates.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CSV_DIR="$DATA_DIR/smoke"
if [ "$#" -eq 0 ]; then
    set -- golden sq1024
fi
SHAPES=$(pick_shapes "golden sq1024 sq2048" "$@")
CSV=smoke.csv
require_fresh "$CSV" $SHAPES
start_job smoke

for shape in $SHAPES; do
    run_cfg "$CSV" fp32_serial "$shape" 2 NA 1
    run_cfg "$CSV" fp32_omp "$shape" 2 NA "$T"
    run_cfg "$CSV" int8_pertensor "$shape" 2 NA "$T"
    for p in 0 5; do
        run_cfg "$CSV" mixed_sep_avx2 "$shape" 2 "$p" "$T"
        run_cfg "$CSV" mixed_fused_avx2 "$shape" 2 "$p" "$T"
    done
    run_cfg "$CSV" mixed_sep_scalar "$shape" 2 5 1
    for threads in $(thread_list); do
        run_cfg "$CSV" mixed_sep_avx2 "$shape" 2 5 "$threads"
    done
done
echo "elapsed seconds: $SECONDS"
