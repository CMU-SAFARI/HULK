# workloads

The end-to-end workloads of the evaluation (Sec. 7.2, Table 3) and additional genomic and
web-search workloads, grouped by domain.
Each domain folder has:

- `src/`: driver sources. One source per workload family; the same source builds for the CPU
  (Microsoft SEAL) and for the GPU (HEonGPU through `gpu_shim`).
- `cpu/run.sh`: builds the drivers with SEAL and measures both modes on the CPU.
- `gpu/run.sh`: builds the drivers with `gpu_shim` and measures both modes on the GPU.

All drivers use 32,768 records (one ciphertext block); latency scales linearly with blocks.
Common driver flags: `--mode {arithmetic|hulk} --arithmetic-limbs L --hulk-limbs L --threads 1`.
