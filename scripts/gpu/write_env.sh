#!/bin/bash
# scripts/gpu/write_env.sh <tag>
# Writes results/gpu/env_<tag>.txt with the items of SPEC 14 (GPU lane). Every job script calls this first.
# A query that fails is recorded as failed in the file; it does not stop the script.
set -u

if [ "$#" -ne 1 ]; then
    echo "usage: write_env.sh <tag>" >&2
    exit 1
fi
TAG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$REPO_DIR/results/gpu"
ENV_FILE="$REPO_DIR/results/gpu/env_${TAG}.txt"

section() {
    local title="$1"
    shift
    echo "== $title"
    "$@" 2>&1 || echo "(command failed: $*)"
    echo
}

{
    section "date" date
    section "hostname" hostname
    # CHECK: these Slurm variable names did not appear in the pasted discovery output; "unset" means the
    # scheduler did not provide the value.
    echo "== scheduler"
    echo "SLURMD_NODENAME=${SLURMD_NODENAME:-unset}"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
    echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-unset}"
    echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}"
    echo
    section "nproc" nproc
    section "nvidia-smi" nvidia-smi
    section "g++ --version" g++ --version
    section "nvcc --version" nvcc --version
    # CHECK: "module" is a shell function and may not exist inside a script; LOADEDMODULES is the fallback.
    section "module list" module list
    echo "== LOADEDMODULES"
    echo "${LOADEDMODULES:-unset}"
    echo
    section "git commit" git -C "$REPO_DIR" rev-parse HEAD
    section "build flags (gpu/Makefile)" grep -E '^NVCC(FLAGS)?[[:space:]]*=' "$REPO_DIR/gpu/Makefile"
} > "$ENV_FILE"

echo "wrote $ENV_FILE"
