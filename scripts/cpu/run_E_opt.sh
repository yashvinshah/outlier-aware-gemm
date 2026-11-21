#!/bin/bash
# scripts/cpu/run_E_opt.sh: COVERAGE group E, CPU lane.
# mixed_sep_avx2_opt1 on sq2048 and sq4096, t = 2, p in {1, 5, 10}, T threads (6 rows) -> E_opt.csv.
# In the same job, right after each opt1 run, the base variant mixed_sep_avx2 runs the same configuration
# -> E_base.csv, so the before/after comparison is made on the same node state. Group B rows are context only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SHAPES="sq2048 sq4096"
require_fresh E_opt.csv $SHAPES
require_fresh E_base.csv $SHAPES
start_job E_opt

for shape in $SHAPES; do
    for p in 1 5 10; do
        run_cfg E_opt.csv mixed_sep_avx2_opt1 "$shape" 2 "$p" "$T"
        run_cfg E_base.csv mixed_sep_avx2 "$shape" 2 "$p" "$T"
    done
done
