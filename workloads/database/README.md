# Database analytics

| Query | Source | Structure (Table 3) |
|---|---|---|
| TPC-H Q6 | `src/TPC-H_Q6_online.cpp` | 3 filters; 1 x AND-3; 1 SUM |
| TPC-H Q12 | `src/TPC-H_Q12_online.cpp` | 4 x AND-5; 4 COUNTs |
| TPC-H Q19 | `src/TPC-H_Q19_online.cpp` | 3 x AND-6; 3 SUMs |
| TPC-DS Q3 | `src/TPC-DS_Q3_online.cpp` | 16 x AND-4; 16 SUMs |
| TPC-DS Q7 | `src/TPC-DS_Q7_online.cpp` | 64 x AND-6; COUNT + 4 SUMs per item |
| TPC-DS Q43 | `src/TPC-DS_Q43_online.cpp` | 21 x AND-4; 21 SUMs |
| TPC-H Q3 top-10 | `src/TPC-H_Q3_topk_e2e.cpp` | 1 filter; 63 comparisons per order; THRESHOLD-73 (<= 9); top-10 mask |

Columns are dictionary-coded over their public TPC domains. Both approaches generate every
predicate online with the same comparison polynomial (`common/HE_comparison.h`), so the limb
count of each query depends on its column domains; only the composition of the predicates
(AND/OR) differs between the modes. Runs set `TPC_DISABLE_CSE=1` for both modes.

```bash
SEAL_DIR=/path/to/SEAL cpu/run.sh                       # all seven, CPU
HEONGPU_DIR=/path/to/HEonGPU gpu/run.sh TPC-H_Q19       # one, GPU
```

Logs go to `$HULK_ROOT/results_fair6s/<query>/logs` (CPU TPC), `results_fairg/TPC-H_Q3_topk` (CPU top-10),
and `results_gpu_fair/<query>/logs` (GPU). The bitmap-index study of Sec. 9.2 is in
`analysis/bitmap_indices/`.
