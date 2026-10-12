#!/bin/bash
# Per-operation unit costs for the stage splits.
# Usage: SEAL_DIR=<seal> ./run.sh cpu      -> $HULK_ROOT/results_bench/bench_ops_cpu.log
#        HEONGPU_DIR=<heongpu> ./run.sh gpu -> $HULK_ROOT/results_gpu_fair/bench_ops.log
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../scripts/protocol.sh"
mkdir -p "$HERE/bin"
case ${1:-cpu} in
  cpu) mkdir -p "$HULK_ROOT/results_bench"; build_cpu "$HERE/bench_ops.cpp" "$HERE/bin/bench_ops"
       OMP_NUM_THREADS=1 taskset -c 0 "$HERE/bin/bench_ops" --lmin 4 --lmax 16 > "$HULK_ROOT/results_bench/bench_ops_cpu.log" ;;
  gpu) mkdir -p "$HULK_ROOT/results_gpu_fair" "$HERE/bin_gpu"; build_gpu "$HERE/bench_ops.cpp" "$HERE/bin_gpu/bench_ops"
       CUDA_MODULE_LOADING=EAGER "$HERE/bin_gpu/bench_ops" --lmin 4 --lmax 16 > "$HULK_ROOT/results_gpu_fair/bench_ops.log" ;;
esac
