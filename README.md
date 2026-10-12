# HULK: Fast Bulk Bitwise Operations on Homomorphically Encrypted Data

This repository contains the code and scripts of the paper "HULK: An Efficient Algorithm for
Enabling Fast Bulk Bitwise Operations on Homomorphically Encrypted Data" (see [Citation](#citation)).

## Abstract

Many cloud applications, such as database, graph, and genome analytics, spend a large fraction
of their execution time on bulk bitwise operations (e.g., AND, OR, XOR), which apply one wide
Boolean function independently to every record of a large dataset. Homomorphic encryption (HE)
keeps this data private during computation, but existing HE approaches evaluate bulk bitwise
operations with high-latency homomorphic multiplications. HULK is an HE algorithm that
evaluates bulk bitwise operations on bit-sliced B/FV ciphertexts in two steps: (1) it adds the
encrypted operand bits with low-cost homomorphic additions, which compute their Hamming weight
in every ciphertext slot, and (2) it decodes the AND, OR, or XOR result from the Hamming weight
with a single indicator polynomial, which keeps the result in the standard encoding for
subsequent arithmetic. Across 13 end-to-end applications in healthcare, graph, and database
analytics, HULK improves performance over the arithmetic approach of prior works by up to
7.26x on a CPU and 6.65x on a GPU.

Every driver implements **both** approaches (`--mode arithmetic` and `--mode hulk`) on the
same data, encoding, and primes, and checks every result against a plaintext oracle
(`ok=match`).

## Layout

| Folder | Contents | Paper |
|---|---|---|
| `common/` | Shared HE harness: product trees, HULK indicators, comparison, aggregation, op counters | all results |
| `gpu_shim/` | SEAL API on top of HEonGPU; builds the unmodified drivers for the GPU | GPU results |
| `workloads/healthcare/` | Exact k-mer, fuzzy k-mer, variant lookup, GWAS; kinship, cohort, STR, web search | Sec. 8, Table 3 |
| `workloads/graph/` | Bit-sliced PSI, ANN search | Sec. 8, Table 3 |
| `workloads/database/` | TPC-H Q6/Q12/Q19, TPC-DS Q3/Q7/Q43, TPC-H Q3 top-10 | Sec. 8, Table 3, Sec. 9.2 |
| `analysis/prior_approaches/` | AND-K across HE approaches (B/FV, Fermat, TFHE-rs PBS and gates, HULK) | Sec. 4.2, Table 1; Sec. 6.1.3 |
| `analysis/indicator_crossover/` | Root-paired vs. Paterson-Stockmeyer indicator crossover | Sec. 6.1 |
| `analysis/unit_costs/` | Per-operation unit costs (stage split of every breakdown) | Fig. 2, Table 3 |
| `analysis/bitmap_indices/` | TPC-H/TPC-DS precomputability coverage and Table 4 projection | Sec. 9.2, Table 4 |
| `scripts/` | Measurement protocol and log parsers | all results |

Each folder has its own README.

## Dependencies

- Microsoft SEAL 4.1 (CPU), built with `cmake -S . -B build && cmake --build build`.
- HEonGPU (GPU), built in `<HEonGPU>/build`; CUDA 12, an NVIDIA GPU (we used an A100 80 GB).
- TFHE-rs 1.7.0 and a Rust toolchain (Table 1 TFHE rows only).
- g++ with C++20, Python 3.

## Quick start

```bash
export HULK_ROOT=$PWD SEAL_DIR=/path/to/SEAL HEONGPU_DIR=/path/to/HEonGPU
workloads/database/cpu/run.sh TPC-DS_Q43     # one workload, CPU, both modes
workloads/healthcare/gpu/run.sh kmer-exact   # one workload, GPU, both modes
analysis/unit_costs/run.sh cpu               # unit costs for the stage split
python3 scripts/parse/make_results_csv.py    # output/HULK_results_*.csv
```

Each run searches the smallest limb count that passes the oracle check twice for each mode,
then times four interleaved A-B-B-A rounds per mode with one thread on one core
(see `scripts/protocol.sh`). We report homomorphic evaluation time only.

## Citation

Please cite the following paper if you find this repository useful.

Paper: arXiv link coming soon.

```bibtex
@article{kabra2026hulk,
  title   = {{HULK: An Efficient Algorithm for Enabling Fast Bulk Bitwise Operations on Homomorphically Encrypted Data}},
  author  = {Kabra, Mayank and Widdowson, Phillip and Gupta, Harshita and Nadig, Rakesh and
             Tzermpou, Eirini and Bostanc{\i}, Nisa and Kumar, Rakesh and Galanopoulos, Spiros and
             Olgun, Ataberk and Ya{\u{g}}l{\i}k{\c{c}}{\i}, Abdullah Giray and Sadrosadati, Mohammad and
             Mutlu, Onur},
  journal = {arXiv preprint},
  year    = {2026}
}
```
