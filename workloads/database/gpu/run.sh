#!/bin/bash
# Run the database workloads on the GPU (both modes, limb search, 4 timed A-B-B-A rounds).
# Usage: HEONGPU_DIR=<heongpu> ./run.sh [workload ...]   (default: all)
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$HERE/../src
source "$HERE/../../../scripts/protocol.sh"
RESULTS=${RESULTS:-$HULK_ROOT/results_gpu_fair}; BIN=$HERE/bin; mkdir -p "$BIN"
build(){ build_gpu "$SRC/$1.cpp" "$BIN/$1"; }
want(){ [ -z "$SEL" ] || [[ " $SEL " == *" $1 "* ]]; }
SEL="$*"
export EXTRA_ENV="TPC_DISABLE_CSE=1"; [ "${RESULTS##*/}" = results_fairg ] && RESULTS=$HULK_ROOT/results_fair6s
for q in TPC-H_Q6:12 TPC-H_Q12:12 TPC-H_Q19:8 TPC-DS_Q3:10 TPC-DS_Q7:8 TPC-DS_Q43:6; do
  n=${q%%:*}; lo=${q##*:}
  want $n || continue
  build ${n}_online; measure $n "$BIN/${n}_online" "" $lo 16
done
unset EXTRA_ENV; [ "${RESULTS##*/}" = results_fair6s ] && RESULTS=$HULK_ROOT/results_fairg
if want TPC-H_Q3_topk; then build TPC-H_Q3_topk_e2e; measure TPC-H_Q3_topk "$BIN/TPC-H_Q3_topk_e2e" 17 11 16; fi
