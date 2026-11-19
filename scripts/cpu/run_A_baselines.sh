#!/bin/bash
# scripts/cpu/run_A_baselines.sh [shape ...]: COVERAGE group A, CPU lane.
# fp32_serial and fp32_omp on the five shapes at t = 2, and int8_pertensor on the five shapes at
# t = 0.5, 2, 5. Appends to results/cpu/A_baselines.csv. With shape arguments only those run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SHAPES=$(pick_shapes "sq1024 sq2048 sq4096 skinny16 skinny128" "$@")
CSV=A_baselines.csv
require_fresh "$CSV" $SHAPES
start_job "$(env_tag A_baselines "$@")"

for shape in $SHAPES; do
    run_cfg "$CSV" fp32_serial "$shape" 2 NA 1
    run_cfg "$CSV" fp32_omp "$shape" 2 NA "$T"
    for t in 0.5 2 5; do
        run_cfg "$CSV" int8_pertensor "$shape" "$t" NA "$T"
    done
done
