#!/bin/bash
# scripts/cpu/run_E_opt.sh [replicate number]: COVERAGE group E, CPU lane.
# mixed_sep_avx2_opt1 on sq2048 and sq4096, t = 2, p in {1, 5, 10}, T threads (6 rows) -> E_opt.csv.
# In the same job, right after each opt1 run, the base variant mixed_sep_avx2 runs the same configuration
# -> E_base.csv, so the before/after comparison is made on the same node state. Group B rows are context only.
# With a replicate number N the rows go to results/cpu/rep/E_opt_rep<N>.csv and E_base_rep<N>.csv instead,
# so a repeat job never touches the first run's files.
set -euo pipefail

REP="${1:-}"
if [ -n "$REP" ]; then
    case "$REP" in
        *[!0-9]*) echo "run_E_opt.sh: replicate number must be a whole number" >&2; exit 1 ;;
    esac
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    CSV_DIR="$REPO_ROOT/results/cpu/rep"
    OPT_CSV="E_opt_rep$REP.csv"
    BASE_CSV="E_base_rep$REP.csv"
    TAG="E_opt_rep$REP"
else
    OPT_CSV=E_opt.csv
    BASE_CSV=E_base.csv
    TAG=E_opt
fi
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SHAPES="sq2048 sq4096"
require_fresh "$OPT_CSV" $SHAPES
require_fresh "$BASE_CSV" $SHAPES
start_job "$TAG"

for shape in $SHAPES; do
    for p in 1 5 10; do
        run_cfg "$OPT_CSV" mixed_sep_avx2_opt1 "$shape" 2 "$p" "$T"
        run_cfg "$BASE_CSV" mixed_sep_avx2 "$shape" 2 "$p" "$T"
    done
done
