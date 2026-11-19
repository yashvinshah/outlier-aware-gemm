#!/bin/bash
# scripts/cpu/lib.sh: settings and helpers shared by the Round 3 scripts. Source it; do not run it.
# The callers use "set -euo pipefail"; this file sets no shell options itself.

T=16             # physical cores of the chosen node (docs/env_cpu.md)
PARTITION=turin  # Slurm partition and node of every CPU job (docs/env_cpu.md, answer 4c)
NODE=c5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH="$REPO_DIR/cpu/bench_cpu"
DATA_DIR="$REPO_DIR/data"
: "${CSV_DIR:=$REPO_DIR/results/cpu}"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Sets M K N for a shape name (SPEC 5).
shape_dims() {
    case "$1" in
        golden)    M=256;  K=256;  N=256 ;;
        sq1024)    M=1024; K=1024; N=1024 ;;
        sq2048)    M=2048; K=2048; N=2048 ;;
        sq4096)    M=4096; K=4096; N=4096 ;;
        skinny16)  M=16;   K=4096; N=4096 ;;
        skinny128) M=128;  K=4096; N=4096 ;;
        *) echo "lib.sh: unknown shape: $1" >&2; exit 1 ;;
    esac
}

# pick_shapes <allowed shapes> [requested shapes...]: prints the shapes to run, which are all the
# allowed ones when none are requested. A group can be split into several jobs by shape this way.
pick_shapes() {
    local allowed="$1"
    shift
    if [ "$#" -eq 0 ]; then
        echo "$allowed"
        return
    fi
    local s
    for s in "$@"; do
        case " $allowed " in
            *" $s "*) ;;
            *) echo "lib.sh: shape $s is not part of this group ($allowed)" >&2; exit 1 ;;
        esac
    done
    echo "$@"
}

# env_tag <group name> [requested shapes...]: the env file tag; the shapes are part of it when only
# some of them are run, so jobs of one split group do not overwrite each other's env file.
env_tag() {
    local tag="$1"
    shift
    if [ "$#" -gt 0 ]; then
        tag="${tag}_$(echo "$*" | tr ' ' '_')"
    fi
    echo "$tag"
}

# require_fresh <csv name> <shapes...>: exits if the CSV already has rows for one of the shapes,
# so a rerun can never add duplicate rows (COVERAGE section 4, step 2).
require_fresh() {
    local csv="$CSV_DIR/$1"
    shift
    local s
    for s in "$@"; do
        if [ -f "$csv" ] && grep -q ",${s}," "$csv"; then
            echo "lib.sh: $csv already has rows for $s; move or remove them first" >&2
            exit 1
        fi
    done
}

# start_job <env tag>: checks the binary, creates the output folder and writes the env file first (SPEC 14).
start_job() {
    if [ ! -x "$BENCH" ]; then
        echo "lib.sh: $BENCH not found; build it with make -C cpu" >&2
        exit 1
    fi
    mkdir -p "$CSV_DIR"
    bash "$SCRIPT_DIR/write_env.sh" "$1"
}

# Thread list L (COVERAGE section 2): powers of two up to T, plus T if T is not a power of two.
thread_list() {
    local n=1 list=""
    while [ "$n" -le "$T" ]; do
        list="$list $n"
        n=$((n * 2))
    done
    if [ $((n / 2)) -ne "$T" ]; then
        list="$list $T"
    fi
    echo $list
}

# run_cfg <csv name> <variant> <shape> <true pct> <iso pct or NA> <threads>
# Warmups and repetitions follow SPEC 8: 1 and 3 for the slow variants, 3 and 10 for the others.
# Prints one RUN line per configuration with the wall time of the whole process (data loading included).
run_cfg() {
    local csv="$1" variant="$2" shape="$3" t="$4" p="$5" threads="$6"
    local warmups=3 reps=10
    case "$variant" in
        fp32_serial|mixed_sep_scalar) warmups=1; reps=3 ;;
    esac
    shape_dims "$shape"
    local iso=()
    if [ "$p" != "NA" ]; then
        iso=(--iso-pct "$p")
    fi
    local start end
    start=$(date +%s.%N)
    "$BENCH" --variant "$variant" --shape "$shape" --M "$M" --K "$K" --N "$N" --true-pct "$t" \
        ${iso[@]+"${iso[@]}"} --threads "$threads" --warmups "$warmups" --reps "$reps" \
        --data-dir "$DATA_DIR" --out-csv "$CSV_DIR/$csv"
    end=$(date +%s.%N)
    echo "RUN $variant $shape t=$t p=$p threads=$threads elapsed_s=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.2f", e - s }')"
}
