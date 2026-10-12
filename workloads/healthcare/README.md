# Healthcare analytics

| Workload | Source and flags | Structure (Table 3) | HULK indicator switch |
|---|---|---|---|
| Exact k-mer matching | `src/Genomics_e2e.cpp --workload kmer-exact` | 16 x AND-62, then OR-16; 1 count | `HULK_INDICATOR_PS_MIN=9` |
| Fuzzy k-mer matching | `src/Genomics_e2e.cpp --workload kmer-fuzzy` | 16 x THRESHOLD-31 (>= 29), then OR-16; 1 count | `HULK_INDICATOR_PS_MIN=17` |
| Variant lookup | `src/Genomics_e2e.cpp --workload variant` | 32 x AND-37, then plaintext-weighted label sum | `HULK_INDICATOR_PS_MIN=17` |
| GWAS | `src/GWAS_e2e.cpp --patients 128` | per patient and SNP: OR-3, AND-8 (case), AND-9 (control); 2x2 table | `HULK_INDICATOR_PS_MIN=9` |

Additional workloads in `src/Genomics_e2e.cpp`:

| Workload | Flags | Structure | HULK indicator switch |
|---|---|---|---|
| Kinship search | `--workload kinship` | 32 SNPs one-hot; [>= 28 of 32 genotypes match] per query genome | `HULK_INDICATOR_PS_MIN=17` |
| Cohort count (Beacon) | `--workload cohort` | 64 SNPs one-hot; AND over 16 (SNP, genotype) conditions; count per query | `HULK_INDICATOR_PS_MIN=9` |
| Forensic STR identity | `--workload str-exact` | 20 loci x 12 bits; AND-240 as 20 per-locus AND-12s | `HULK_INDICATOR_PS_MIN=9` |
| Forensic STR familial search | `--workload str-familial` | per-locus AND-12, then [>= 15 of 20 loci match] | `HULK_INDICATOR_PS_MIN=9` |
| Web-search indexing (BitFunnel) | `--workload websearch` | 128 signature rows; 32 queries of 6 terms x 2 rows = AND-12 | `HULK_INDICATOR_PS_MIN=17` |

```bash
SEAL_DIR=/path/to/SEAL cpu/run.sh                 # all workloads, CPU
HEONGPU_DIR=/path/to/HEonGPU gpu/run.sh kmer-exact # one, GPU
```

Logs go to `$HULK_ROOT/results_fairg/<workload>/logs` (CPU) and `$HULK_ROOT/results_gpu_fair/<workload>/logs` (GPU).
