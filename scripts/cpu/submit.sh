#!/bin/bash
# scripts/cpu/submit.sh <script name> <walltime minutes> [script arguments...]
# Submits one of the scripts in this folder as a Slurm batch job on the chosen node (docs/env_cpu.md).
# Example: bash scripts/cpu/submit.sh run_smoke.sh 15 golden sq1024
# The job log goes to $HOME/<script name>_<timestamp>.out.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [ "$#" -lt 2 ]; then
    echo "usage: submit.sh <script name> <walltime minutes> [script arguments...]" >&2
    exit 1
fi
NAME="$1"
MINUTES="$2"
shift 2

if [ ! -f "$SCRIPT_DIR/$NAME" ]; then
    echo "submit.sh: $SCRIPT_DIR/$NAME not found" >&2
    exit 1
fi
case "$MINUTES" in
    ''|*[!0-9]*) echo "submit.sh: walltime must be a whole number of minutes" >&2; exit 1 ;;
esac

LOG="$HOME/${NAME%.sh}_$(date +%Y%m%d_%H%M%S)_$$.out"
sbatch --partition="$PARTITION" --nodelist="$NODE" --exclusive --nodes=1 --time="$MINUTES" \
    --output="$LOG" --wrap="bash $SCRIPT_DIR/$NAME $*"
echo "log: $LOG"
