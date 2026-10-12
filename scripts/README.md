# scripts

- `protocol.sh`: the measurement protocol used by every run script (`build_cpu`, `build_gpu`,
  `measure`). For each mode, it finds the smallest limb count that passes the plaintext oracle
  check twice, then runs four timed rounds per mode in A-B-B-A order with one thread on one core.
- `parse/`: turn run logs into the result CSVs.
  - `make_results_csv.py`: `output/HULK_results_summary.csv` (latency, speedup, limbs per workload)
    and `output/HULK_results_breakdown.csv` (per-stage latency: measured latency split by
    operation counts x unit costs).
  - `fairg_parse.py`, `fair6_parse.py`, `gpu_parse.py`: per-run tables for the CPU healthcare/graph,
    CPU TPC, and GPU runs.
  - `newapps_parse.py`: GWAS and ANN breakdowns.
  - `units.py`: reads the per-operation unit costs from the `bench_ops` logs (`analysis/unit_costs/run.sh`).
  - `opcounts.py`: HMul/HAdd/plaintext-multiplication counts and noise budgets (Table 3).

All scripts read `HULK_ROOT` (default: the artifact root). Raw logs are written to
`$HULK_ROOT/results_fairg`, `results_fair6s`, `results_gpu_fair`, `results_andk`, and `results_bench`;
parsed CSVs go to `$HULK_ROOT/output`.
