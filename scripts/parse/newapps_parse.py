#!/usr/bin/env python3
"""Stage breakdown for the new end-to-end applications (GWAS, ANN), arithmetic approach.

Same model as make_results_csv.py: median HE-eval latency over the 4 timed rounds, split by
stage as eval_ms x stage_cost / total_cost, where stage_cost = op counts x the CPU system single-thread
unit costs (bench_ops log, scripts/parse/units.py) at the run's limb count. Writes HULK_newapps_breakdown.csv.
"""
import csv, glob, os, re, statistics
HULK_ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
os.makedirs(os.path.join(HULK_ROOT, 'output'), exist_ok=True)

H = HULK_ROOT
import units as _units
UNITS = _units.CPU_LOG
APPS = [('gwas', 'GWAS', 'healthcare'), ('ann', 'ANN', 'search')]
BITWISE = {'and_test', 'xor'}


def unit_vec(g):
    rot = (14 * 0.5 * (g['rot_rows_1'] + g['rot_rows_4096']) + g['rot_cols']) / 15
    return {'hmul': g['hmul_relin'], 'sq': g['sq_relin'], 'pmul': g['mul_plain_const'], 'add': g['hadd'],
            'padd': g['add_plain'], 'rot': rot}


units = {L: unit_vec(g) for L, g in _units.load(UNITS).items()}
rows = []
for d, name, suite in APPS:
    ev, L, st, ok = [], None, {}, True
    for f in sorted(glob.glob(os.path.join(H, 'results_newapps', d, 'logs', 'timed_arithmetic_r*.log'))):
        t = open(f).read()
        e = re.search(r'E2E mode=\S+ limbs=(\d+) .*?eval_ms=([\d.]+).*?ok=(\w+)', t)
        if not e: ok = False; continue
        L = int(e.group(1)); ev.append(float(e.group(2))); ok &= e.group(3) == 'match'
        st = {s.group(1): {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', s.group(2))}
              for s in re.finditer(r'STAGE mode=\S+ stage=(\w+) (.*)', t)}
    if not ev:
        print(f'{name}: no timed rounds'); continue
    em = statistics.median(ev); u = units[L]
    cost = {s: sum(c.get(op, 0) * u[op] for op in u) for s, c in st.items()}
    tot = sum(cost.values())
    print(f'{name}: L={L} eval {em/1000:.2f} s (median of {len(ev)}, all ok={ok}), model/measured {tot/em:.3f}')
    for s, c in st.items():
        pct = 100 * cost[s] / tot
        print(f'   {s:10s} {"bitwise" if s in BITWISE else "other  "} {em*cost[s]/tot/1000:9.2f} s  {pct:6.2f}%  '
              f'HMul={c.get("hmul",0)+c.get("sq",0)} rot={c.get("rot",0)}')
        rows.append({'platform': 'CPU', 'suite': suite, 'workload': name, 'mode': 'arithmetic', 'limbs': L,
                     'eval_ms': round(em, 3), 'stage': s, 'bulk_bitwise': s in BITWISE,
                     'stage_ms': round(em * cost[s] / tot, 3), 'stage_pct': round(pct, 2),
                     'hmul': c.get('hmul', 0), 'squarings': c.get('sq', 0), 'rotations': c.get('rot', 0),
                     'rounds': len(ev), 'all_correct': ok, 'model_over_measured': round(tot / em, 3)})
    print(f'   bulk bitwise total: {sum(100*cost[s]/tot for s in st if s in BITWISE):.1f}%')
if rows:
    out = os.path.join(H, 'output', 'HULK_newapps_breakdown.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('wrote', out)
