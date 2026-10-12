# Encrypted bitmap indices (Sec. 9.2, Table 4)

- `classify/`: precomputability of every predicate of the 22 TPC-H and 99 TPC-DS query templates.
  - `RULES.md`: classification rules.
  - `*.jsonl`: one record per predicate, with its legal-constant set from the official query
    templates and distributions (TPC-H qgen, TPC-DS dsqgen).
  - `coverage.py`: per-query coverage at T = 16/64/256 legal values, written to
    `output/HULK_bitmap_coverage.csv`.
- `project_table4.py`: Tier-1 projected speedups of Table 4 from `output/HULK_results_breakdown.csv`
  (run the database workloads and `scripts/parse/make_results_csv.py` first).
  It removes the comparison stage of the precomputable predicates from the measured latency at the
  same limb count, in proportion to their share of the comparison HMuls.

```bash
python3 classify/coverage.py
python3 project_table4.py
```
