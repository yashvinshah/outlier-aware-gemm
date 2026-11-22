#!/bin/bash
# scripts/cpu/run_C_ladder.sh [replicate number] [shape ...]: COVERAGE group C, CPU lane.
# With a replicate number N the rows go to results/cpu/rep/C_ladder_rep<N>.csv (see lib.sh).
# mixed_sep_scalar (1 thread), mixed_sep_avx2 (1 thread) and mixed_sep_avx2 (T threads) on sq1024 and
# sq2048, t = 2, p in {0, 5}. Appends to results/cpu/C_ladder.csv. With shape arguments only those run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SHAPES=$(pick_shapes "sq1024 sq2048" "$@")
CSV=$(csv_name C_ladder)
require_fresh "$CSV" $SHAPES
start_job "$(env_tag C_ladder "$@")"

for shape in $SHAPES; do
    for p in 0 5; do
        run_cfg "$CSV" mixed_sep_scalar "$shape" 2 "$p" 1
        run_cfg "$CSV" mixed_sep_avx2 "$shape" 2 "$p" 1
        run_cfg "$CSV" mixed_sep_avx2 "$shape" 2 "$p" "$T"
    done
done
