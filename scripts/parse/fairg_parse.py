#!/usr/bin/env python3
"""Fair genomics/PSI summary (setup A): median HE-eval latency per mode (4 ABBA rounds,
same node), speedup, and stage split = eval_ms x stage_cost / total_cost (op counts x
the CPU system single-thread unit costs at the mode's limb count)."""
import csv, glob, os, re, statistics, sys
HULK_ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else HULK_ROOT + '/results_fairg')
import units as _units
units = _units.load(_units.CPU_LOG)
def uv(L):
    g = units[L]
    return {'hmul': g['hmul_relin'], 'sq': g['sq_relin'], 'pmul': g['mul_plain_const'], 'add': g['hadd'],
            'padd': g['add_plain'], 'rot': (14 * 0.5 * (g['rot_rows_1'] + g['rot_rows_4096']) + g['rot_cols']) / 15}
for qd in sorted(glob.glob(root + '/*/')):
    q = os.path.basename(qd.rstrip('/')); res = {}
    for m in ('arithmetic', 'hulk'):
        ev, L, st, ok = [], None, {}, True
        for f in sorted(glob.glob(f'{qd}logs/timed_{m}_r*.log')):
            t = open(f).read()
            e = re.search(r'E2E mode=\S+ limbs=(\d+) .*?eval_ms=([\d.]+).*?ok=(\w+)', t)
            if not e: ok = False; continue
            L = int(e.group(1)); ev.append(float(e.group(2))); ok &= e.group(3) == 'match'
            st = {}
            for s in re.finditer(r'STAGE mode=\S+ stage=(\w+) (.*)', t):
                st[s.group(1)] = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', s.group(2))}
        if ev: res[m] = (L, statistics.median(ev), st, ok, len(ev))
    if len(res) < 2: print(f'{q}: incomplete'); continue
    sp = res['arithmetic'][1] / res['hulk'][1]
    for m in ('arithmetic', 'hulk'):
        L, e, st, ok, n = res[m]; u = uv(L)
        cost = {s: sum(c.get(k, 0) * u[k] for k in u) for s, c in st.items()}
        tot = sum(cost.values()) or 1
        cells = ' | '.join(f'{s}: {e*cost[s]/tot/1000:.2f} s ({100*cost[s]/tot:.1f}%) [{st[s].get("hmul",0)+st[s].get("sq",0)} HMul, {st[s].get("rot",0)} rot]' for s in st)
        print(f'{q} | {m} | L={L} | {e/1000:.2f} s | {sp:.2f}x | n={n} ok={ok} | {cells} | model/meas={tot/e:.2f}')
