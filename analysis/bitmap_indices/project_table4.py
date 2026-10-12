#!/usr/bin/env python3
"""Sec. 9.2, Table 4: projected speedup of encrypted bitmap indices (Tier-1, CPU).

Projection: remove the comparison (filter) stage of the precomputable predicates from the
measured latency, at the same limb count, in proportion to their share of the comparison
HMuls (HMul + squarings) of the filter stage.
  A+B     = eval_A / (eval_A - f * filter_A)
  H+B     = eval_A / (eval_H - f * filter_H)
  time cut = f * filter_A / eval_A
f = precomputable share of the comparison HMuls (1 for fully precomputable queries).
Input: output/HULK_results_breakdown.csv (from scripts/parse/make_results_csv.py) (CPU, suite=tpc)."""
import csv, os
ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
# precomputable comparison HMuls / all comparison HMuls of the filter stage (T = 64 legal values)
#   TPC-DS Q3: d_moy and d_year predicates (13 of 84); i_manufact_id and the brand group-by stay online
#   TPC-DS Q7: year, marital status, education, promotion, gender (13 of 470); the 64 i_item_id groups stay online
SHARE = {'TPC-H_Q6': 1.0, 'TPC-H_Q12': 1.0, 'TPC-H_Q19': 1.0,
         'TPC-DS_Q3': 13 / 84, 'TPC-DS_Q7': 13 / 470, 'TPC-DS_Q43': 1.0}
rows = {}
for r in csv.DictReader(open(os.path.join(ROOT, 'output', 'HULK_results_breakdown.csv'))):
    if r['platform'] == 'CPU' and r['suite'] == 'tpc' and r['stage'] == 'filter' and r['workload'] in SHARE:
        rows[(r['workload'], r['mode'])] = (float(r['eval_ms']), float(r['stage_ms']))
print(f"{'query':12s} {'A+B':>8s} {'H+B':>8s} {'H+B/A+B':>8s} {'cut':>7s}")
acc = []
for q, f in SHARE.items():
    ea, fa = rows[(q, 'arithmetic')]; eh, fh = rows[(q, 'hulk')]
    ab = ea / (ea - f * fa); hb = ea / (eh - f * fh); cut = f * fa / ea
    acc.append((ab, hb, hb / ab, cut))
    print(f"{q:12s} {ab:7.2f}x {hb:7.2f}x {hb/ab:7.2f}x {100*cut:6.1f}%")
m = [sum(x[i] for x in acc) / len(acc) for i in range(4)]
print(f"{'mean':12s} {m[0]:7.2f}x {m[1]:7.2f}x {m[2]:7.2f}x {100*m[3]:6.1f}%")
