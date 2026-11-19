#!/bin/bash
# scripts/cpu/write_env.sh <tag>: writes results/cpu/env_<tag>.txt (SPEC 14).
# No "set -e": an item that is unavailable must be recorded as unavailable, not abort the job.
set -u

if [ "$#" -ne 1 ]; then
    echo "usage: write_env.sh <tag>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$REPO_DIR/results/cpu"
OUT="$OUT_DIR/env_$1.txt"
mkdir -p "$OUT_DIR" || { echo "write_env.sh: cannot create $OUT_DIR" >&2; exit 1; }

section() {
    echo
    echo "## $1"
}

{
    echo "date: $(date)"
    echo "hostname: $(hostname)"
    echo "nproc: $(nproc)"
    echo "cpu_model: $(lscpu | grep 'Model name')"
    # CHECK: these are standard Slurm variable names but none has appeared in pasted ARC output.
    # An unset one is recorded as "not set".
    echo "slurm_job_id: ${SLURM_JOB_ID:-not set}"
    echo "slurm_node_name: ${SLURMD_NODENAME:-not set}"
    echo "slurm_job_nodelist: ${SLURM_JOB_NODELIST:-not set}"
    echo "slurm_job_num_nodes: ${SLURM_JOB_NUM_NODES:-not set}"
    echo "slurm_cpus_on_node: ${SLURM_CPUS_ON_NODE:-not set}"
    echo "OMP_PROC_BIND: ${OMP_PROC_BIND:-not set}"
    echo "OMP_PLACES: ${OMP_PLACES:-not set}"
    section "lscpu"
    lscpu
    section "g++ --version"
    g++ --version
    section "loaded modules"
    module list 2>&1 || echo "module command not available in this shell"
    section "git commit (git rev-parse HEAD)"
    git -C "$REPO_DIR" rev-parse HEAD 2>&1
    # The commands make would run, which shows the exact flags the Makefiles use (CHECK: -n -B).
    section "build commands, common/ (make -n -B)"
    make -C "$REPO_DIR/common" -n -B 2>&1
    section "build commands, cpu/ (make -n -B)"
    make -C "$REPO_DIR/cpu" -n -B 2>&1
} > "$OUT" || { echo "write_env.sh: cannot write $OUT" >&2; exit 1; }

echo "wrote $OUT"
