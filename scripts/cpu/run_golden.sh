#!/bin/bash
# scripts/cpu/run_golden.sh: CPU side of the golden cross-check (SPEC 13 steps 1 and 2).
# Golden shape, t = 2. Run it inside a job on the chosen node. Needs cpu/bench_cpu and the golden data.
# Writes results/cpu/golden/checksums_cpu.txt (overwritten on every run) and the C files under
# data/golden_out/cpu/ (git-ignored, SPEC 17.5).
set -euo pipefail

T=16  # physical cores of the chosen node c5 (docs/env_cpu.md); SPEC 2 defines T

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH="$REPO_DIR/cpu/bench_cpu"
DATA_DIR="$REPO_DIR/data"
OUT_C="$DATA_DIR/golden_out/cpu"
CHK_DIR="$REPO_DIR/results/cpu/golden"
CHK="$CHK_DIR/checksums_cpu.txt"
CSV="$OUT_C/golden_runs.csv"

if [ ! -x "$BENCH" ]; then
    echo "run_golden.sh: $BENCH not found; build it with make -C cpu" >&2
    exit 1
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
bash "$SCRIPT_DIR/write_env.sh" golden

mkdir -p "$OUT_C" "$CHK_DIR"
rm -f "$CSV"
: > "$CHK"

SHAPE_ARGS=(--shape golden --M 256 --K 256 --N 256 --true-pct 2 --data-dir "$DATA_DIR" --out-csv "$CSV")

# run_one <variant> <threads> <p or NA> <warmups> <reps>
# Appends "<variant> threads=<n> p=<p> <hex>" to the checksum file (SPEC 13 step 2, 17.5).
run_one() {
    local variant="$1" threads="$2" p="$3" warmups="$4" reps="$5"
    local iso_args=() name_p="pNA"
    if [ "$p" != "NA" ]; then
        iso_args=(--iso-pct "$p")
        name_p="p$p"
    fi
    local cfile="$OUT_C/${variant}_threads${threads}_${name_p}.f32"
    local out hex
    # The ${arr[@]+...} form keeps an empty array legal under "set -u" on older bash versions.
    out=$("$BENCH" --variant "$variant" --threads "$threads" ${iso_args[@]+"${iso_args[@]}"} \
        --warmups "$warmups" --reps "$reps" "${SHAPE_ARGS[@]}" --write-c "$cfile")
    hex=$(echo "$out" | sed -n 's/^C_CHECKSUM //p')
    if [ -z "$hex" ]; then
        echo "run_golden.sh: no C_CHECKSUM from $variant threads=$threads p=$p" >&2
        exit 1
    fi
    echo "$variant threads=$threads p=$p $hex" | tee -a "$CHK"
}

run_one fp32_serial 1 NA 1 3
run_one fp32_omp "$T" NA 3 10
for threads in 1 "$T"; do
    run_one int8_pertensor "$threads" NA 3 10
done
for p in 0 2 10; do
    for threads in 1 "$T"; do
        run_one mixed_sep_scalar "$threads" "$p" 1 3
        run_one mixed_sep_avx2 "$threads" "$p" 3 10
        run_one mixed_fused_avx2 "$threads" "$p" 3 10
    done
done

# Checks that need only this lane (SPEC 13 step 4). The cross-lane comparison is done by the humans.
status=0
serial_hex=$(grep '^fp32_serial ' "$CHK" | awk '{print $4}')
omp_hex=$(grep '^fp32_omp ' "$CHK" | awk '{print $4}')
if [ "$serial_hex" = "$omp_hex" ]; then
    echo "CHECK fp32_omp vs fp32_serial: equal"
else
    echo "CHECK fp32_omp vs fp32_serial: DIFFERENT"
    status=1
fi
for p in 0 2 10; do
    runs=$(grep "^mixed_.* p=$p " "$CHK" | wc -l)
    distinct=$(grep "^mixed_.* p=$p " "$CHK" | awk '{print $4}' | sort -u | wc -l)
    if [ "$distinct" -eq 1 ]; then
        echo "CHECK mixed_* p=$p: $runs runs, 1 distinct checksum"
    else
        echo "CHECK mixed_* p=$p: $runs runs, $distinct distinct checksums (DIFFERENT)"
        status=1
    fi
done
echo "elapsed seconds: $SECONDS"
exit "$status"
