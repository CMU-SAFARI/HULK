# Graph analytics

| Workload | Source and flags | Structure (Table 3) | HULK indicator switch |
|---|---|---|---|
| Bit-sliced PSI | `src/Private_Set_Intersection_e2e.cpp --encoding bit` | 16 x AND-40 (one per server item), then OR-16; 1 cardinality | `HULK_INDICATOR_PS_MIN=9` |
| ANN search | `src/ANN_e2e.cpp` | 8 x AND-12, then OR-8 (candidate); 64 XORs (Hamming distance); [h <= 10]; top-10 | `HULK_INDICATOR_PS_MIN=9` |

```bash
SEAL_DIR=/path/to/SEAL cpu/run.sh            # both, CPU
HEONGPU_DIR=/path/to/HEonGPU gpu/run.sh psi   # one, GPU
```

