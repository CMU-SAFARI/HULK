#!/usr/bin/env python3
"""Static bitmap-precomputability coverage over TPC-H (22) and TPC-DS (99) templates.
Inputs: classify/*.jsonl (per-predicate records). Filter predicates = all WHERE/HAVING/ON predicates except
FK equi-joins (joins are materialized at ingest in the bitmap model). A predicate is precomputable at threshold T
iff category == constant and legal_set_size <= T. A query is fully precomputable iff all its filter predicates
are precomputable AND every GROUP BY key has domain <= T."""
import json, glob, csv, collections, os
HULK_ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
os.makedirs(os.path.join(HULK_ROOT, 'output'), exist_ok=True)
D = os.path.expanduser(HULK_ROOT + '/analysis/bitmap_indices/classify')
Q = []
for f in sorted(glob.glob(f'{D}/*.jsonl')):
    for line in open(f):
        if line.strip(): Q.append(json.loads(line))
def norm(q, p):                                   # cross-batch normalizations (distinct values, not raw add lines)
    s, col = p.get('legal_set_size'), str(p.get('column', '')).lower()
    if s is None: return s
    if s == 3108 and 'gmt_offset' in col: return 6
    if s == 3108 and 'state' in col: return 51
    if q['benchmark'] == 'TPC-DS' and q['query'] == 84 and 'income' in col: return min(s, 21)
    if q['benchmark'] == 'TPC-DS' and q['query'] in (53, 63) and 'month_seq' in col and s == 60: return 49
    return s
TS = (16, 64, 256)
rows = []
for q in sorted(Q, key=lambda x: (x['benchmark'], x['query'])):
    filt = [p for p in q['predicates'] if p.get('category') != 'join']
    gb = q.get('group_by', [])
    r = dict(benchmark=q['benchmark'], query=q['query'], n_filter_predicates=len(filt),
             n_join_predicates=sum(p.get('category') == 'join' for p in q['predicates']), n_group_by=len(gb))
    for T in TS:
        pre = [p for p in filt if p.get('category') == 'constant' and norm(q, p) is not None and norm(q, p) <= T]
        bad = collections.Counter()
        for p in filt:
            if p in pre: continue
            c = p.get('category')
            bad[c if c != 'constant' else f'legal set >{T}'] += 1
        gbad = [g for g in gb if (g.get('domain_size_estimate') or 10**9) > T]
        if gbad: bad[f'group-by domain >{T}'] += len(gbad)
        r[f'n_precomputable_T{T}'] = len(pre)
        r[f'coverage_pct_T{T}'] = round(100 * len(pre) / len(filt), 1) if filt else 100.0
        r[f'fully_precomputable_T{T}'] = len(pre) == len(filt) and not gbad
        r[f'non_precomputable_reasons_T{T}'] = '; '.join(f'{k} x{v}' for k, v in sorted(bad.items()))
    rows.append(r)
# required column names = T=64 defaults
for r in rows:
    r['n_precomputable'] = r['n_precomputable_T64']; r['coverage_pct'] = r['coverage_pct_T64']
    r['fully_precomputable'] = r['fully_precomputable_T64']; r['non_precomputable_reasons'] = r['non_precomputable_reasons_T64']
keys = ['benchmark', 'query', 'n_filter_predicates', 'n_precomputable', 'coverage_pct', 'fully_precomputable',
        'non_precomputable_reasons', 'n_join_predicates', 'n_group_by'] + [k for k in rows[0] if k.endswith(('T16', 'T64', 'T256'))]
out = os.path.expanduser(HULK_ROOT + '/output/HULK_bitmap_coverage.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
print('wrote', out, len(rows), 'rows')
for b in ('TPC-H', 'TPC-DS'):
    rs = [r for r in rows if r['benchmark'] == b]
    for T in TS:
        full = [r['query'] for r in rs if r[f'fully_precomputable_T{T}']]
        some = sum(r[f'n_precomputable_T{T}'] > 0 for r in rs)
        filt_only = sum(r[f'n_precomputable_T{T}'] == r['n_filter_predicates'] for r in rs)
        mean = sum(r[f'coverage_pct_T{T}'] for r in rs) / len(rs)
        print(f'{b} T={T}: n={len(rs)} fully={len(full)} ({100*len(full)/len(rs):.0f}%)  all-filters-only={filt_only} '
              f'>=1 pre={some} ({100*some/len(rs):.0f}%)  mean coverage={mean:.1f}%  full list={full}')

# Paper convention (Sec. 9.2): a comparison between two columns of the same row has no constant,
# so it is one fixed bitmap and counts as precomputable.
import re
print('\nPaper convention (column-vs-column predicates as fixed bitmaps):')
for T in ('T16', 'T64', 'T256'):
    for b in ('TPC-H', 'TPC-DS'):
        rs = [x for x in csv.DictReader(open(out)) if x['benchmark'] == b]
        covs, full, some = [], [], 0
        for x in rs:
            reas = x['non_precomputable_reasons_' + T]
            m = re.search(r'col_vs_col x(\d+)', reas); c = int(m.group(1)) if m else 0
            n = int(x['n_filter_predicates']); p = int(x['n_precomputable_' + T]) + c
            covs.append(100 * p / n if n else 0); some += p > 0
            rest = re.sub(r'col_vs_col x\d+;?\s*', '', reas).strip('; ')
            if x['fully_precomputable_' + T] == 'True' or rest == '': full.append(int(x['query']))
        print(f"{b} {T}: fully={len(full)} ({round(100*len(full)/len(rs))}%) >=1 pre={some} mean coverage={sum(covs)/len(covs):.1f}% full list={sorted(full)}")
