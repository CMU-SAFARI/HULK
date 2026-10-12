#!/bin/bash
# Measurement protocol shared by every run script (CPU and GPU).
#   1. Build the driver (CPU: Microsoft SEAL 4.1; GPU: the same source linked against gpu_shim).
#   2. Limb search: for each mode (arithmetic, hulk), find the smallest limb count whose run
#      passes the plaintext oracle check (ok=match) twice.
#   3. Four timed rounds per mode at that limb count, interleaved A-B-B-A, single thread,
#      pinned to one core (CPU). Each run prints "E2E ... eval_ms=... ok=match" and
#      "STAGE mode= stage= hmul= sq= ..." lines that the parsers in scripts/parse read.
#
# Environment:
#   HULK_ROOT   artifact root (default: parent of this directory)
#   SEAL_DIR    Microsoft SEAL 4.1 source tree, built in $SEAL_DIR/build (CPU)
#   HEONGPU_DIR HEonGPU source tree, built in $HEONGPU_DIR/build (GPU)
set -u
HULK_ROOT=${HULK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HULK_ROOT
log(){ echo "[$(date +%T)] $*"; }

build_cpu(){ # src.cpp out
  : "${SEAL_DIR:?set SEAL_DIR to a Microsoft SEAL 4.1 tree built in \$SEAL_DIR/build}"
  g++ -std=c++20 -O3 -march=native -pthread -I"$HULK_ROOT/common" -I"$(dirname "$1")" \
    -I"$SEAL_DIR/native/src" -I"$SEAL_DIR/build/native/src" -I"$SEAL_DIR/build/thirdparty/msgsl-src/include" \
    "$1" -o "$2" -L"$SEAL_DIR/build/lib" -Wl,-rpath,"$SEAL_DIR/build/lib" -lseal-4.1 -lpthread -ldl
}
build_gpu(){ # src.cpp out
  OUT=$(dirname "$2") SRCS="$1" bash "$HULK_ROOT/gpu_shim/build.sh"
}

# measure NAME BIN PSMIN LO HI [driver args...]
#   PSMIN: HULK_INDICATOR_PS_MIN for the hulk mode (empty = driver default)
#   LO..HI: limb range searched for both modes
measure(){
  local name=$1 bin=$2 psmin=$3 lo=$4 hi=$5; shift 5
  local out=$RESULTS/$name; mkdir -p "$out/logs"
  local core; core=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //;s/[,-].*//'); core=${core:-0}
  run(){ # mode L tag
    local m=$1 L=$2 tag=$3 f ps=""
    if [ "$m" = arithmetic ]; then f=--arithmetic-limbs; else f=--hulk-limbs; [ -n "$psmin" ] && ps="HULK_INDICATOR_PS_MIN=$psmin"; fi
    env -u E2E_SLOT_AGGREGATE -u TPC_SLOT_AGGREGATE -u HULK_DEFER_SCALE -u HULK_POLY_EVAL -u HE_PRIME_BITS -u HULK_INDICATOR_PS_MIN \
        $ps ${EXTRA_ENV:-} OMP_NUM_THREADS=1 taskset -c "$core" "$bin" "$@" --threads 1 --mode "$m" $f "$L" > "$out/logs/$tag.log" 2>&1
    grep -q "ok=match" "$out/logs/$tag.log"
  }
  declare -A MIN
  for m in arithmetic hulk; do
    MIN[$m]=FAIL
    for ((L=lo; L<=hi; L++)); do
      if run $m $L search_${m}_L${L}_a "$@" && run $m $L search_${m}_L${L}_b "$@"; then MIN[$m]=$L; break; fi
    done
    log "$name SEARCH $m min_limbs=${MIN[$m]}"
  done
  [ "${MIN[arithmetic]}" = FAIL ] || [ "${MIN[hulk]}" = FAIL ] && { log "$name SEARCH FAILED"; return 1; }
  for r in 1 2 3 4; do
    if [ $((r % 2)) = 1 ]; then order="arithmetic hulk"; else order="hulk arithmetic"; fi
    for m in $order; do
      run $m ${MIN[$m]} timed_${m}_r$r "$@" || log "$name TIMED $m r$r MISMATCH"
      log "$name TIMED $m L=${MIN[$m]} r$r $(grep -m1 -oE 'eval_ms=[0-9.]+' "$out/logs/timed_${m}_r$r.log")"
    done
  done
}
