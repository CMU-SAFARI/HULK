# gpu_shim

A subset of the Microsoft SEAL API implemented on HEonGPU, so that the GPU runs use the
**same driver sources** as the CPU runs. Only `#include "seal/seal.h"` resolves to
`include/seal/seal.h` here.

- Same primes as SEAL; the last prime is the special prime.
- `cudaDeviceSynchronize()` after every operation, so each timed operation includes its kernel time.
- Plaintext constants use scalar multiplication; squaring is `multiply(x, x)`; Galois keys are generated lazily.

Build one or more drivers:

```bash
HEONGPU_DIR=/path/to/HEonGPU OUT=bin SRCS="../workloads/database/src/TPC-DS_Q43_online.cpp" ./build.sh
```

`CUDA_ARCH` (default `-arch=sm_80`) and `HEONGPU_BUILD` (default `$HEONGPU_DIR/build`) can be overridden.
The run scripts in `workloads/*/gpu/` call this script.
