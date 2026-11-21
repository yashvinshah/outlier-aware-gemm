#!/bin/bash
# scripts/cpu/run_golden_opt1.sh: Round 5 golden check for mixed_sep_avx2_opt1 (SPEC 13 step 6).
# Golden shape, t = 2, p in {0, 2, 10}, threads in {1, T}, the base variant's parameters. Appends the
# lines to results/cpu/golden/checksums_cpu.txt (the file from run_golden.sh must exist and is not
# truncated) and compares each with the mixed_sep_avx2 line of the same p and threads.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

OUT_C="$DATA_DIR/golden_out/cpu"
CHK="$REPO_DIR/results/cpu/golden/checksums_cpu.txt"
CSV="$OUT_C/golden_opt1_runs.csv"

if [ ! -f "$CHK" ]; then
    echo "run_golden_opt1.sh: $CHK not found; run run_golden.sh first" >&2
    exit 1
fi
if grep -q '^mixed_sep_avx2_opt1 ' "$CHK"; then
    echo "run_golden_opt1.sh: $CHK already has mixed_sep_avx2_opt1 lines; remove them first" >&2
    exit 1
fi
start_job golden_opt1
mkdir -p "$OUT_C"
rm -f "$CSV"

status=0
for p in 0 2 10; do
    for threads in 1 "$T"; do
        cfile="$OUT_C/mixed_sep_avx2_opt1_threads${threads}_p${p}.f32"
        out=$("$BENCH" --variant mixed_sep_avx2_opt1 --shape golden --M 256 --K 256 --N 256 --true-pct 2 \
            --iso-pct "$p" --threads "$threads" --warmups 3 --reps 10 --data-dir "$DATA_DIR" \
            --out-csv "$CSV" --write-c "$cfile")
        hex=$(echo "$out" | sed -n 's/^C_CHECKSUM //p')
        if [ -z "$hex" ]; then
            echo "run_golden_opt1.sh: no C_CHECKSUM for threads=$threads p=$p" >&2
            exit 1
        fi
        echo "mixed_sep_avx2_opt1 threads=$threads p=$p $hex" | tee -a "$CHK"
        base=$(grep "^mixed_sep_avx2 threads=$threads p=$p " "$CHK" | awk '{print $4}')
        if [ "$base" = "$hex" ]; then
            echo "CHECK opt1 vs mixed_sep_avx2 threads=$threads p=$p: equal"
        else
            echo "CHECK opt1 vs mixed_sep_avx2 threads=$threads p=$p: DIFFERENT (base $base)"
            status=1
        fi
    done
done
echo "elapsed seconds: $SECONDS"
exit "$status"
