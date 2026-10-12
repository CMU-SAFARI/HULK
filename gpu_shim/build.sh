#!/bin/bash
# Build the SEAL-on-HEonGPU shim and one driver for the GPU.
# The drivers are the SAME sources as the CPU runs; only <seal/seal.h> resolves to the shim.
# Usage: OUT=<bin dir> SRCS="<driver.cpp> ..." HEONGPU_DIR=<HEonGPU tree built in $HEONGPU_DIR/build> ./build.sh
set -euo pipefail
export PATH=${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
: "${HEONGPU_DIR:?set HEONGPU_DIR to an HEonGPU tree built in \$HEONGPU_DIR/build}"
SHIM=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$SHIM"); SRC=$HEONGPU_DIR; BLD=${HEONGPU_BUILD:-$SRC/build}
OUT=${OUT:-$SHIM/bin}; mkdir -p "$OUT" "$SHIM/obj"
NVCC=nvcc; ARCH=${CUDA_ARCH:--arch=sm_80}
INC="-I$SHIM/include -I$SRC/src/include -I$SRC/thirdparty/GPU-NTT/src/include -I$SRC/thirdparty/GPU-FFT/src/include \
 -I$SRC/thirdparty/RNGonGPU/src/include -I$BLD/_deps/cccl-src/libcudacxx/include -I$BLD/_deps/cccl-src/cub -I$BLD/_deps/cccl-src/thrust \
 -I$BLD/_deps/spdlog-src/include -I$BLD/_deps/rmm-src/cpp/include -I$BLD/_deps/rmm-build/include -I$BLD/_deps/nvtx3-src/c/include \
 -I$BLD/_deps/rapids_logger-src/include"
DEF="-DCCCL_DISABLE_PDL -DCUB_DISABLE_NAMESPACE_MAGIC -DCUB_IGNORE_NAMESPACE_MAGIC_ERROR -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_DISABLE_ABI_NAMESPACE -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR"
LIBD="-L$BLD/src -L$BLD/thirdparty/GPU-NTT/src -L$BLD/thirdparty/GPU-FFT/src -L$BLD/thirdparty/RNGonGPU/src -L$BLD/_deps/rmm-build -L$BLD/_deps/rapids_logger-build"
LIBS="-lheongpu -lntt-1.0 -lfft-1.0 -lrngongpu-1.0 -lrmm -lrapids_logger -lgmp -lssl -lcrypto -lz -lcurand -lcudart -ldl -lpthread"
RPATH="-Xlinker -rpath -Xlinker $BLD/_deps/rmm-build -Xlinker -rpath -Xlinker $BLD/_deps/rapids_logger-build"
[ -f "$SHIM/obj/seal_heongpu.o" ] || { echo "[build] shim"; $NVCC -x cu -std=c++17 -O3 $ARCH $DEF $INC -c "$SHIM/src/seal_heongpu.cu" -o "$SHIM/obj/seal_heongpu.o"; }
for s in ${SRCS:?}; do
  n=$(basename "$s" .cpp); echo "[build] $n"
  g++ -std=c++20 -O3 -march=native -pthread -I"$SHIM/include" -I"$ROOT/common" -I"$(dirname "$s")" -c "$s" -o "$SHIM/obj/$n.o"
  $NVCC $ARCH "$SHIM/obj/$n.o" "$SHIM/obj/seal_heongpu.o" -o "$OUT/$n" $LIBD $RPATH $LIBS
done
