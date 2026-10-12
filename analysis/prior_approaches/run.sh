#!/bin/bash
# Table 1 (Sec. 4.2) and the Fermat crossover (Sec. 6.1.3): one bulk AND-K per record.
# B/FV approaches (SEAL): smallest limb count that passes twice, then 4 timed rounds.
# TFHE-rs approaches: 256-record sample, extrapolated to 32,768 records.
# Usage: SEAL_DIR=<seal> ./run.sh
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../scripts/protocol.sh"
R=${RESULTS:-$HULK_ROOT/results_andk}; mkdir -p "$R/logs" "$HERE/bin"
build_cpu "$HERE/andk_e2e.cpp" "$HERE/bin/andk_e2e"
cfg(){ # label args...
  local lab=$1; shift; local L
  for ((L=4; L<=16; L++)); do
    OMP_NUM_THREADS=1 "$HERE/bin/andk_e2e" "$@" --limbs $L > "$R/logs/${lab}_search_a.log" 2>&1 && grep -q "ok=match" "$R/logs/${lab}_search_a.log" &&
    OMP_NUM_THREADS=1 "$HERE/bin/andk_e2e" "$@" --limbs $L > "$R/logs/${lab}_search_b.log" 2>&1 && grep -q "ok=match" "$R/logs/${lab}_search_b.log" && break
  done
  log "SEARCH $lab min_limbs=$L"
  for r in 1 2 3 4; do OMP_NUM_THREADS=1 "$HERE/bin/andk_e2e" "$@" --limbs $L > "$R/logs/${lab}_r$r.log" 2>&1
    log "TIMED $lab r$r $(grep -oE 'eval_ms=[0-9.]+|hmul=[0-9]+|ok=[a-z]+' "$R/logs/${lab}_r$r.log" | tr '\n' ' ')"; done
}
for K in 2 8 16 64 128; do
  cfg A1_K$K  --approach A1  --K $K --N 32768 --p 0       # arithmetic product tree (baseline)
  cfg A3_K$K  --approach A3  --K $K --N 32768 --p 65537   # sum + Fermat zero test
  cfg A7s_K$K --approach A7s --K $K --N 32768 --p 0       # HULK, server-side decoding
  cfg A7c_K$K --approach A7c --K $K --N 32768 --p 0       # HULK, client-side decoding
done
cfg TT31 --approach TT --K 31 --lo 29 --N 32768 --p 65537  # THRESHOLD-31 (>= 29): counting tree
cfg TF31 --approach TF --K 31 --lo 29 --N 32768 --p 65537  #   Fermat
cfg TH31 --approach TH --K 31 --lo 29 --N 32768 --p 65537  #   HULK
cfg TT73 --approach TT --K 73 --lo 0 --hi 9 --N 32768 --p 65537
cfg TF73 --approach TF --K 73 --lo 0 --hi 9 --N 32768 --p 65537
cfg TH73 --approach TH --K 73 --lo 0 --hi 9 --N 32768 --p 65537
# TFHE-rs 1.7.0: multi-input PBS (A5) and gate-by-gate (A6)
( cd "$HERE/tfhe" && RUSTFLAGS="-C target-cpu=native" cargo build --release && ./target/release/andk_tfhe 256 ) | tee "$R/tfhe_results.txt"
