#!/bin/bash
# Run the graph workloads on the CPU (both modes, limb search, 4 timed A-B-B-A rounds).
# Usage: SEAL_DIR=<seal> ./run.sh [workload ...]   (default: all)
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$HERE/../src
source "$HERE/../../../scripts/protocol.sh"
RESULTS=${RESULTS:-$HULK_ROOT/results_fairg}; BIN=$HERE/bin; mkdir -p "$BIN"
build(){ build_cpu "$SRC/$1.cpp" "$BIN/$1"; }
want(){ [ -z "$SEL" ] || [[ " $SEL " == *" $1 "* ]]; }
SEL="$*"
build Private_Set_Intersection_e2e
want psi && measure psi-bit "$BIN/Private_Set_Intersection_e2e" 9 8 16 --encoding bit
build ANN_e2e
want ann && measure ann-ps9 "$BIN/ANN_e2e"                      9 8 16
