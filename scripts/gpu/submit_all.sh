#!/bin/bash
# scripts/gpu/submit_all.sh <DATA_DIR>
# Run on the login node. Submits groups A, B and C as three independent Slurm jobs (each writes its own CSV)
# and prints the job ids. Slurm logs go to results/gpu/logs/. Needs gpu/bench_gpu built.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: submit_all.sh <DATA_DIR>" >&2
    exit 1
fi
DATA_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_DIR/results/gpu/logs"
mkdir -p "$LOG_DIR"

# sbatch copies the job script, so REPO_DIR is passed through the environment instead of derived in the job.
for group in A_baselines B_sweep C_tiles; do
    jobid=$(sbatch --parsable --export=ALL,REPO_DIR="$REPO_DIR" -o "$LOG_DIR/${group}_%j.out" \
        "$SCRIPT_DIR/run_${group}.sh" "$DATA_DIR")
    echo "$group job $jobid"
done
