#!/usr/bin/env python3
"""GPU fair-run summary (setup A on HEonGPU via the SEAL shim).
Per workload: median HE-eval latency per mode (4 ABBA rounds, same GPU), speedup, stage
split = eval_ms x stage_cost / total_cost (op counts x GPU unit costs from bench_ops at the
mode's limb count), HMul per stage. Writes a CSV next to the results when --csv is given."""
import csv, glob, os, re, statistics, sys
HULK_ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
root = os.path.expanduser(HULK_ROOT + '/results_gpu_fair')
units = {}
for line in open(os.path.join(root, 'bench_ops.log')):
    m = re.match(r'UNIT L=(\d+) op=(\S+) ms=([\d.]+)', line)
    if m: units.setdefault(int(m.group(1)), {})[m.group(2)] = float(m.group(3))
def uv(L):
    g = units[L]
    rot = (14 * 0.5 * (g.get('rot_rows_1', 0) + g.get('rot_rows_4096', 0)) + g.get('rot_cols', 0)) / 15
    return {'hmul': g['hmul_relin'], 'sq': g['sq_relin'], 'pmul': g['mul_plain_const'], 'add': g['hadd'],
            'padd': g['add_plain'], 'rot': rot}
ORDER = ['websearch-ps17', 'kinship', 'kmer-fuzzy', 'kmer-exact', 'variant', 'gwas-ps9', 'psi-bit', 'ann-ps9', 'str-familial', 'cohort', 'str-exact',
         'TPC-DS_Q43', 'TPC-DS_Q3', 'TPC-DS_Q7', 'TPC-H_Q19', 'TPC-H_Q12', 'TPC-H_Q6', 'TPC-H_Q3_topk']
rows = []
for q in ORDER:
    qd = os.path.join(root, q)
    res = {}
    for m in ('arithmetic', 'hulk'):
        ev, L, st, ok = [], None, {}, True
        for f in sorted(glob.glob(f'{qd}/logs/timed_{m}_r*.log')):
            t = open(f).read()
            e = re.search(r'E2E mode=\S+ limbs=(\d+) .*?eval_ms=([\d.]+).*?ok=(\w+)', t)
            if not e: ok = False; continue
            L = int(e.group(1)); ev.append(float(e.group(2))); ok &= e.group(3) == 'match'
            st = {s.group(1): {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', s.group(2))}
                  for s in re.finditer(r'STAGE mode=\S+ stage=(\w+) (.*)', t)}
        if ev: res[m] = (L, statistics.median(ev), st, ok, len(ev))
    if len(res) < 2: print(f'{q}: incomplete'); continue
    sp = res['arithmetic'][1] / res['hulk'][1]
    row = {'workload': q, 'speedup': round(sp, 2)}
    for m in ('arithmetic', 'hulk'):
        L, e, st, ok, n = res[m]; u = uv(L)
        cost = {s: sum(c.get(k, 0) * u[k] for k in u) for s, c in st.items()}
        tot = sum(cost.values()) or 1
        cells = ' | '.join(f'{s}: {100*cost[s]/tot:.1f}% [{st[s].get("hmul",0)+st[s].get("sq",0)} HMul]' for s in st)
        print(f'{q} | {m} | L={L} | {e:.2f} ms | {sp:.2f}x | n={n} ok={ok} | {cells} | model/meas={tot/e:.2f}')
        a = m[:5]
        row.update({f'{a}_limbs': L, f'{a}_eval_ms': round(e, 3), f'{a}_ok': ok, f'{a}_model_ratio': round(tot / e, 2)})
        for s in st:
            row[f'{a}_{s}_pct'] = round(100 * cost[s] / tot, 1)
            row[f'{a}_{s}_hmul'] = st[s].get('hmul', 0) + st[s].get('sq', 0)
    rows.append(row)
if '--csv' in sys.argv and rows:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    out = os.path.join(root, 'HULK_GPU_results_setupA.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('wrote', out)
