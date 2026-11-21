#!/bin/bash
# scripts/cpu/run_profile_opt1.sh: Round 5 profile of mixed_sep_avx2_opt1 at sq4096, t = 2, p = 5, T threads,
# in the same style as run_profile.sh (perf stat, then perf record, user-space events only). Text output goes
# to results/cpu/profile/; the large perf .data file and the run CSV go to the git-ignored data/perf/.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PROFILE_DIR="$REPO_DIR/results/cpu/profile"
PERF_DIR="$DATA_DIR/perf"
mkdir -p "$PROFILE_DIR" "$PERF_DIR"
rm -f "$PERF_DIR/perf_opt1_runs.csv"
start_job profile_opt1

echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
perf --version

tag=mixed_sep_avx2_opt1_p5
args=(--variant mixed_sep_avx2_opt1 --shape sq4096 --M 4096 --K 4096 --N 4096 --true-pct 2 --iso-pct 5
    --threads "$T" --warmups 3 --reps 10 --data-dir "$DATA_DIR" --out-csv "$PERF_DIR/perf_opt1_runs.csv")
echo "=== $tag: perf stat"
perf stat -e task-clock,cycles:u,instructions:u,cache-references:u,cache-misses:u \
    -o "$PROFILE_DIR/stat_$tag.txt" -- "$BENCH" "${args[@]}"
echo "=== $tag: perf record"
perf record -F 999 -e cycles:u -o "$PERF_DIR/$tag.data" -- "$BENCH" "${args[@]}"
perf report -i "$PERF_DIR/$tag.data" --stdio --sort symbol > "$PROFILE_DIR/report_$tag.txt" 2>&1
cp "$PERF_DIR/perf_opt1_runs.csv" "$PROFILE_DIR/perf_opt1_runs.csv"
echo "elapsed seconds: $SECONDS"
