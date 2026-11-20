#!/bin/bash
# scripts/cpu/run_profile.sh: Round 5 profiling at sq4096, t = 2, T threads, on the chosen node, for
# mixed_sep_avx2 at p = 5, mixed_fused_avx2 at p = 5, and mixed_sep_avx2 at p = 0 and p = 0.5. Uses perf
# (user-space events only). Text output goes to results/cpu/profile/ (COVERAGE section 5); the large perf
# .data files and the run CSV go to the git-ignored data/perf/. Nothing here changes the benchmark: it runs
# the same bench_cpu command twice per configuration, once under "perf stat" and once under "perf record".
# It also runs mixed_sep_avx2 at p = 0 and p = 5 five times each as separate processes, without perf, to see
# how much the same configuration varies from one process to the next (data/perf/repeat_runs.csv).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PROFILE_DIR="$REPO_DIR/results/cpu/profile"
PERF_DIR="$DATA_DIR/perf"
mkdir -p "$PROFILE_DIR" "$PERF_DIR"
rm -f "$PERF_DIR/perf_runs.csv" "$PERF_DIR/repeat_runs.csv"
start_job profile

echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
which perf
perf --version
perf list > "$PROFILE_DIR/perf_list.txt" 2>&1

# profile_config <variant> <p>: the output files are named <variant>_p<p>
profile_config() {
    local variant="$1" p="$2"
    local tag="${variant}_p${p}"
    local args=(--variant "$variant" --shape sq4096 --M 4096 --K 4096 --N 4096 --true-pct 2 --iso-pct "$p"
        --threads "$T" --warmups 3 --reps 10 --data-dir "$DATA_DIR" --out-csv "$PERF_DIR/perf_runs.csv")
    echo "=== $tag: perf stat"
    perf stat -e task-clock,cycles:u,instructions:u,cache-references:u,cache-misses:u \
        -o "$PROFILE_DIR/stat_$tag.txt" -- "$BENCH" "${args[@]}"
    echo "=== $tag: perf record"
    perf record -F 999 -e cycles:u -o "$PERF_DIR/$tag.data" -- "$BENCH" "${args[@]}"
    perf report -i "$PERF_DIR/$tag.data" --stdio --sort symbol > "$PROFILE_DIR/report_$tag.txt" 2>&1
}

profile_config mixed_sep_avx2 5
profile_config mixed_fused_avx2 5
profile_config mixed_sep_avx2 0
profile_config mixed_sep_avx2 0.5

# repeat_runs: five rounds of one process at p = 0 and one at p = 5. The two alternate, so a slow drift
# of the node over the job affects both the same way.
repeat_runs() {
    local p i
    for i in 1 2 3 4 5; do
        for p in 0 5; do
            echo "=== repeat $i: mixed_sep_avx2 p=$p"
            "$BENCH" --variant mixed_sep_avx2 --shape sq4096 --M 4096 --K 4096 --N 4096 --true-pct 2 \
                --iso-pct "$p" --threads "$T" --warmups 3 --reps 10 --data-dir "$DATA_DIR" \
                --out-csv "$PERF_DIR/repeat_runs.csv"
        done
    done
}

repeat_runs
echo "elapsed seconds: $SECONDS"
