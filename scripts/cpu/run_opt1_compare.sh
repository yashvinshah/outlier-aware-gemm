#!/bin/bash
# scripts/cpu/run_opt1_compare.sh: Round 5 bit comparison of mixed_sep_avx2_opt1 against mixed_sep_avx2 on
# sq1024 and skinny16, t = 2, p = 5, T threads, using --write-c and common/compare_c (SPEC 13 step 5).
# The C files and the run CSV stay in the git-ignored data/opt1_check/. Needs common/compare_c built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

COMPARE="$REPO_DIR/common/compare_c"
OUT_DIR="$DATA_DIR/opt1_check"
CSV="$OUT_DIR/opt1_check_runs.csv"

if [ ! -x "$COMPARE" ]; then
    echo "run_opt1_compare.sh: $COMPARE not found; build it with make -C common" >&2
    exit 1
fi
start_job opt1_compare
mkdir -p "$OUT_DIR"
rm -f "$CSV"

status=0
for shape in sq1024 skinny16; do
    shape_dims "$shape"
    for variant in mixed_sep_avx2 mixed_sep_avx2_opt1; do
        "$BENCH" --variant "$variant" --shape "$shape" --M "$M" --K "$K" --N "$N" --true-pct 2 \
            --iso-pct 5 --threads "$T" --warmups 3 --reps 10 --data-dir "$DATA_DIR" \
            --out-csv "$CSV" --write-c "$OUT_DIR/${variant}_${shape}_p5.f32"
    done
    echo "=== compare_c $shape p=5: mixed_sep_avx2 vs mixed_sep_avx2_opt1"
    if ! "$COMPARE" "$OUT_DIR/mixed_sep_avx2_${shape}_p5.f32" "$OUT_DIR/mixed_sep_avx2_opt1_${shape}_p5.f32" \
        $((M * N)); then
        status=1
    fi
done
echo "elapsed seconds: $SECONDS"
exit "$status"
