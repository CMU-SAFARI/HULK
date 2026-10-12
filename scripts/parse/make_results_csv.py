#!/usr/bin/env python3
"""Unified CPU + GPU result CSVs for the final (setup A) HULK evaluation.

Reads the raw timed logs (4 ABBA rounds per mode, same node/GPU) and writes:
  HULK_results_summary.csv    one row per (platform, workload): limbs, median HE-eval
                              latency per mode, speedup, correctness
  HULK_results_breakdown.csv  one row per (platform, workload, mode, stage): stage
                              latency (ms) and share, HMul / squarings / rotations
Stage latency = eval_ms x stage_cost / total_cost, stage_cost = op counts x the
platform's single-thread unit costs at the mode's limb count (CPU: bench_ops log
from bench_ops on the CPU system; GPU: bench_ops on the A100).
"""
import sys
import csv, glob, os, re, statistics
HULK_ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
os.makedirs(os.path.join(HULK_ROOT, 'output'), exist_ok=True)

H = HULK_ROOT
import units as _units
CPU_UNITS = _units.CPU_LOG
GPU_UNITS = _units.GPU_LOG

# (suite, workload label, CPU result dir, GPU result dir)
WORKLOADS = [
    ('genomics', 'kinship',       'results_fairg/kinship',        'results_gpu_fair/kinship'),
    ('genomics', 'kmer-fuzzy',    'results_fairg/kmer-fuzzy',     'results_gpu_fair/kmer-fuzzy'),
    ('genomics', 'kmer-exact',    'results_fairg/kmer-exact',     'results_gpu_fair/kmer-exact'),
    ('genomics', 'variant',       'results_fairg/variant',        'results_gpu_fair/variant'),
    ('genomics', 'str-familial',  'results_fairg/str-familial',   'results_gpu_fair/str-familial'),
    ('genomics', 'str-exact',     'results_fairg/str-exact',      'results_gpu_fair/str-exact'),
    ('genomics', 'cohort',        'results_fairg/cohort',         'results_gpu_fair/cohort'),
    ('genomics', 'GWAS',          'results_fairg/gwas-ps9',       'results_gpu_fair/gwas-ps9'),
    ('psi',      'ANN',           'results_fairg/ann-ps9',        'results_gpu_fair/ann-ps9'),
    ('psi',      'psi-bit-sliced','results_fairg/psi-bit',        'results_gpu_fair/psi-bit'),
    ('websearch','websearch-bitfunnel','results_fairg/websearch-ps17','results_gpu_fair/websearch-ps17'),
    ('tpc',      'TPC-DS_Q43',    'results_fair6s/TPC-DS_Q43',    'results_gpu_fair/TPC-DS_Q43'),
    ('tpc',      'TPC-DS_Q3',     'results_fair6s/TPC-DS_Q3',     'results_gpu_fair/TPC-DS_Q3'),
    ('tpc',      'TPC-DS_Q7',     'results_fair6s/TPC-DS_Q7',     'results_gpu_fair/TPC-DS_Q7'),
    ('tpc',      'TPC-H_Q19',     'results_fair6s/TPC-H_Q19',     'results_gpu_fair/TPC-H_Q19'),
    ('tpc',      'TPC-H_Q12',     'results_fair6s/TPC-H_Q12',     'results_gpu_fair/TPC-H_Q12'),
    ('tpc',      'TPC-H_Q6',      'results_fair6s/TPC-H_Q6',      'results_gpu_fair/TPC-H_Q6'),
    ('tpc',      'TPC-H_Q3_topk', 'results_fairg/TPC-H_Q3_topk',  'results_gpu_fair/TPC-H_Q3_topk'),
]
# unified stage names: TPC filter/and/weight/aggregate; genomics test->and_test, reduce->or_reduce;
# PSI indicator->and_test, or->or_reduce
STAGE_MAP = {'filter': 'filter', 'and': 'and_test', 'test': 'and_test', 'indicator': 'and_test',
             'reduce': 'or_reduce', 'or': 'or_reduce', 'weight': 'weight', 'aggregate': 'aggregate',
             'sort_compare': 'sort_compare', 'sort_rank': 'sort_rank'}
STAGE_ORDER = ['filter', 'and_test', 'or_reduce', 'weight', 'aggregate', 'sort_compare', 'sort_rank']


def unit_vec(g):
    rot = (14 * 0.5 * (g.get('rot_rows_1', 0) + g.get('rot_rows_4096', 0)) + g.get('rot_cols', 0)) / 15
    return {'hmul': g['hmul_relin'], 'sq': g['sq_relin'], 'pmul': g['mul_plain_const'], 'add': g['hadd'],
            'padd': g['add_plain'], 'rot': rot}


cpu_units = {L: unit_vec(g) for L, g in _units.load(CPU_UNITS).items()} if os.path.exists(CPU_UNITS) else {}
gpu_units = {L: unit_vec(g) for L, g in _units.load(GPU_UNITS).items()} if os.path.exists(GPU_UNITS) else {}


def load(d):
    res = {}
    for m in ('arithmetic', 'hulk'):
        ev, L, st, ok = [], None, {}, True
        for f in sorted(glob.glob(os.path.join(d, 'logs', f'timed_{m}_r*.log'))):
            t = open(f).read()
            e = re.search(r'E2E mode=\S+ limbs=(\d+) .*?eval_ms=([\d.]+).*?ok=(\w+)', t)
            if not e: ok = False; continue
            L = int(e.group(1)); ev.append(float(e.group(2))); ok &= e.group(3) == 'match'
            st = {s.group(1): {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', s.group(2))}
                  for s in re.finditer(r'STAGE mode=\S+ stage=(\w+) (.*)', t)}
        if ev: res[m] = dict(limbs=L, eval_ms=statistics.median(ev), rounds=len(ev), ok=ok, stages=st)
    return res


summary, breakdown = [], []
for suite, name, cdir, gdir in WORKLOADS:
    for platform, d, units in (('CPU', cdir, cpu_units), ('GPU', gdir, gpu_units)):
        res = load(os.path.join(H, d))
        if len(res) < 2 or not units:
            print(f'skip {platform} {name}: incomplete runs or missing unit costs'); continue
        a, h = res['arithmetic'], res['hulk']
        sp = a['eval_ms'] / h['eval_ms']
        summary.append({'platform': platform, 'suite': suite, 'workload': name, 'rows': 32768,
                        'arith_limbs': a['limbs'], 'hulk_limbs': h['limbs'],
                        'arith_eval_ms': round(a['eval_ms'], 3), 'hulk_eval_ms': round(h['eval_ms'], 3),
                        'speedup': round(sp, 3), 'rounds_per_mode': min(a['rounds'], h['rounds']),
                        'all_correct': a['ok'] and h['ok']})
        for mode, r in (('arithmetic', a), ('hulk', h)):
            u = units[r['limbs']]
            agg = {s: {'cost': 0.0, 'hmul': 0, 'sq': 0, 'rot': 0} for s in STAGE_ORDER}
            for s, c in r['stages'].items():
                k = STAGE_MAP.get(s, s)
                agg[k]['cost'] += sum(c.get(op, 0) * u[op] for op in u)
                for op in ('hmul', 'sq', 'rot'): agg[k][op] += c.get(op, 0)
            tot = sum(v['cost'] for v in agg.values()) or 1.0
            for s in STAGE_ORDER:
                v = agg[s]
                breakdown.append({'platform': platform, 'suite': suite, 'workload': name, 'mode': mode,
                                  'limbs': r['limbs'], 'eval_ms': round(r['eval_ms'], 3), 'stage': s,
                                  'stage_ms': round(r['eval_ms'] * v['cost'] / tot, 3),
                                  'stage_pct': round(100 * v['cost'] / tot, 2),
                                  'hmul': v['hmul'], 'squarings': v['sq'], 'hmul_total': v['hmul'] + v['sq'],
                                  'rotations': v['rot'], 'model_over_measured': round(tot / r['eval_ms'], 3)})

if not summary: sys.exit('no complete runs found under ' + H)
for fn, rows in (('HULK_results_summary.csv', summary), ('HULK_results_breakdown.csv', breakdown)):
    with open(os.path.join(H, 'output', fn), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('wrote', os.path.join(H, 'output', fn), len(rows), 'rows')
