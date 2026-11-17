#!/bin/bash
# scripts/cpu/gen_all_data.sh <DATA_DIR> [shape ...]
# Generates the data files of SPEC 4.3 for the given shapes (default: all six of SPEC 5) and prints
# the CHECKSUM lines. Prints the wall time of each shape. Needs common/gen_data to be built.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: gen_all_data.sh <DATA_DIR> [shape ...]" >&2
    exit 1
fi
DATA_DIR="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DATA="$SCRIPT_DIR/../../common/gen_data"
if [ ! -x "$GEN_DATA" ]; then
    echo "gen_all_data.sh: $GEN_DATA not found; build it with make -C common" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    SHAPES="golden sq1024 sq2048 sq4096 skinny16 skinny128"
else
    SHAPES="$*"
fi

# Sets M K N T_LIST. SPEC 5 gives the shapes; SPEC 17.6 says golden uses t = 2 only.
shape_dims() {
    case "$1" in
        golden)    M=256;  K=256;  N=256;  T_LIST="2" ;;
        sq1024)    M=1024; K=1024; N=1024; T_LIST="0.5 2 5" ;;
        sq2048)    M=2048; K=2048; N=2048; T_LIST="0.5 2 5" ;;
        sq4096)    M=4096; K=4096; N=4096; T_LIST="0.5 2 5" ;;
        skinny16)  M=16;   K=4096; N=4096; T_LIST="0.5 2 5" ;;
        skinny128) M=128;  K=4096; N=4096; T_LIST="0.5 2 5" ;;
        *) echo "gen_all_data.sh: unknown shape: $1" >&2; exit 1 ;;
    esac
}

for SHAPE in $SHAPES; do
    shape_dims "$SHAPE"
done

mkdir -p "$DATA_DIR"

for SHAPE in $SHAPES; do
    shape_dims "$SHAPE"
    echo "SHAPE $SHAPE"
    time {
        for T in $T_LIST; do
            "$GEN_DATA" --M "$M" --K "$K" --N "$N" --true-pct "$T" --data-dir "$DATA_DIR"
        done
    }
done
