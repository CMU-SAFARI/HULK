#!/bin/bash
# Run the healthcare workloads on the GPU (both modes, limb search, 4 timed A-B-B-A rounds).
# Usage: HEONGPU_DIR=<heongpu> ./run.sh [workload ...]   (default: all)
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$HERE/../src
source "$HERE/../../../scripts/protocol.sh"
RESULTS=${RESULTS:-$HULK_ROOT/results_gpu_fair}; BIN=$HERE/bin; mkdir -p "$BIN"
build(){ build_gpu "$SRC/$1.cpp" "$BIN/$1"; }
want(){ [ -z "$SEL" ] || [[ " $SEL " == *" $1 "* ]]; }
SEL="$*"
build Genomics_e2e
want kmer-exact && measure kmer-exact "$BIN/Genomics_e2e" 9  8 16 --workload kmer-exact
want kmer-fuzzy && measure kmer-fuzzy "$BIN/Genomics_e2e" 17 7 16 --workload kmer-fuzzy
want variant    && measure variant    "$BIN/Genomics_e2e" 17 6 16 --workload variant
want kinship      && measure kinship      "$BIN/Genomics_e2e" 17 5 16 --workload kinship
want cohort       && measure cohort       "$BIN/Genomics_e2e" 9  4 16 --workload cohort
want str-exact    && measure str-exact    "$BIN/Genomics_e2e" 9  7 16 --workload str-exact
want str-familial && measure str-familial "$BIN/Genomics_e2e" 9  7 16 --workload str-familial
want websearch    && measure websearch-ps17 "$BIN/Genomics_e2e" 17 4 16 --workload websearch
build GWAS_e2e
want gwas       && measure gwas-ps9   "$BIN/GWAS_e2e"     9  6 16 --patients 128
