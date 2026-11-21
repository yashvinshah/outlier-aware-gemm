#!/bin/bash
# scripts/cpu/run_all.sh: runs the CPU groups A, B, C, D in that order inside one job. A group that
# fails does not stop the next one; each group's exit status is printed, and the script exits 1 if
# any group failed. Each group script writes its own env file and CSV (see lib.sh).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

failed=0
for group in run_A_baselines.sh run_B_sweep.sh run_C_ladder.sh run_D_scaling.sh; do
    echo "=== $group started: $(date)"
    bash "$SCRIPT_DIR/$group"
    status=$?
    echo "=== $group exit status $status"
    if [ "$status" -ne 0 ]; then
        failed=1
    fi
done
echo "elapsed seconds: $SECONDS"
exit "$failed"
