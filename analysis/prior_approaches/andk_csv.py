#!/usr/bin/env python3
"""Final AND-K / THRESHOLD comparison CSV (the CPU system, AMD EPYC 7742) -> output/HULK_andK_comparison.csv

SEAL rows: 4 timed rounds per configuration from logs_timed (andk_timed) and logs_timed2 (andk_timed2;
its rounds replace andk_timed's where both exist). Median eval_ms; load1 range; flags.
Also: A3 25-bit infeasible rows at N=32768 (andk2 feasibility), 64-way throughput (andk64),
TFHE A5/A6 (sampled + extrapolated), A4 (analytical)."""
import csv, glob, os, re, statistics, collections
HULK_ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
os.makedirs(os.path.join(HULK_ROOT, 'output'), exist_ok=True)
R = os.path.expanduser(HULK_ROOT + '/results_andk')
P25 = {32768: 33292289, 65536: None}
LINE = re.compile(r'ANDK approach=(\S+) K=(\d+)(?: lo=(\d+) hi=(\d+))? N=(\d+) t=(\d+) limbs=(\d+) total_bits=(\d+) '
                  r'eval_ms=([\d.]+) hmul=(\d+) relin=(\d+) depth=(\d+) ct_bytes=(\d+) noise=(-?\d+) ok=(\w+)')
DESC = {'A1': 'AND-K: arithmetic product tree (relin every HMul)',
        'A2': 'AND-K: arithmetic product tree, lazy relin (SEAL: final relin skipped)',
        'A3': 'AND-K: sum + Fermat zero test 1-(S-K)^(p-1)',
        'A7s': 'AND-K: HULK sum + indicator, server-side decoding',
        'A7c': 'AND-K: HULK sum only, client-side decoding (client tests S==K)',
        'TT': 'counting tree (arithmetic)', 'TF': 'Fermat (sum + one zero test per accepted count)',
        'TH': 'HULK indicator (sum + bounded comparison polynomial)'}

def load_dir(d):
    out = collections.defaultdict(list)
    for f in glob.glob(f'{R}/{d}/*_r[1-4].log'):
        lab = re.sub(r'_r[1-4]\.log$', '', os.path.basename(f)); t = open(f).read()
        m = LINE.search(t); l = re.search(r'load1=([\d.]+)', t)
        out[lab].append((m, float(l.group(1)) if l else None))
    return out

t1 = load_dir('logs_timed')
# N=65536: the in-job timed rounds of andk2 (logs2/t_<label>_K<k>_r<r>.log; load not logged)
t2 = collections.defaultdict(list)
for f in glob.glob(f'{R}/logs2/t_*_K*_r[1-4].log'):
    m = re.match(r't_(\w+?)_K(\d+)_r[1-4]\.log$', os.path.basename(f))
    lab = {'A1N65536': 'A1_p25', 'A3p25': 'A3_p25', 'A3p786433': 'A3_p786433'}[m.group(1)] + f'_N65536_K{m.group(2)}'
    t2[lab].append((LINE.search(open(f).read()), None))
rows = []
for lab in sorted(set(t1) | set(t2)):
    runs, src = (t2[lab], 'andk2 in-job timed rounds (load not logged)') if lab in t2 else (t1[lab], 'andk_timed')
    good = [(m, l) for m, l in runs if m]
    if not good:
        rows.append(dict(config=lab, ok='NO OUTPUT')); continue
    m0 = good[0][0]; a = m0.group(1)
    ev = [float(m.group(9)) for m, _ in good]; ld = [l for _, l in good if l is not None]
    K = int(m0.group(2)); N = int(m0.group(5)); t = int(m0.group(6))
    lo, hi = m0.group(3), m0.group(4)
    op = (f'THRESHOLD-{K} (>= {lo})' if lo and lo != '0' else f'THRESHOLD-{K} (<= {hi})') if lo is not None else f'AND-{K}'
    desc = f'{op}: {DESC[a]}' if lo is not None else DESC[a]
    med = statistics.median(ev)
    rows.append(dict(config=lab, approach=a, operation=op, description=desc, K=K, N=N, p=t,
                     limbs=int(m0.group(7)), total_bits=int(m0.group(8)), depth=int(m0.group(12)),
                     hmul_or_pbs=int(m0.group(10)), relin=int(m0.group(11)), ciphertext_bytes=int(m0.group(13)),
                     records_per_block=N, threads=1, block_ms=round(med, 3), per_record_ms=round(med / N, 7),
                     rounds=len(ev), spread=round(max(ev) / min(ev), 3),
                     load1_min=min(ld) if ld else '', load1_max=max(ld) if ld else '',
                     high_load=bool(ld) and max(ld) > 256, noisy=max(ev) / min(ev) > 1.15,
                     min_noise_bits=min(int(m.group(14)) for m, _ in good),
                     ok=all(m.group(15) == 'match' for m, _ in good), source=f'measured, {src}'))
# A3 at the 25-bit prime, N=32768: infeasibility evidence
for f in sorted(glob.glob(f'{R}/logs2/f32_L*.log')):
    m = LINE.search(open(f).read())
    if m:
        rows.append(dict(config=f'A3_p25_N32768_K2_L{m.group(7)}', approach='A3', operation='AND-2',
                         description=DESC['A3'] + ' at the 25-bit batching prime', K=2, N=32768, p=int(m.group(6)),
                         limbs=int(m.group(7)), total_bits=int(m.group(8)), depth=int(m.group(12)),
                         hmul_or_pbs=int(m.group(10)), min_noise_bits=int(m.group(14)), threads=1,
                         ok='infeasible (noise budget exhausted; 16 limbs = 880 bits is the 128-bit maximum at N=32768)',
                         source='measured (feasibility search)'))
# 64-way throughput
for line in open(f'{R}/andk64.out') if os.path.exists(f'{R}/andk64.out') else []:
    m = re.search(r'AND64 approach=(\S+) K=(\d+) limbs=(\d+) procs=64 ok=(\d+)/64 median_eval_ms=([\d.]+)', line)
    if m:
        a, K, L, okn, me = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))
        rows.append(dict(config=f'{a}_N32768_K{K}_64way', approach=a, operation=f'AND-{K}', description=DESC[a],
                         K=K, N=32768, p=65537 if a == 'A3' else P25[32768], limbs=L, threads=64,
                         records_per_block=32768, block_ms=round(me, 3), per_record_ms=round(me / (64 * 32768), 8),
                         ok=okn == 64, source='measured: 64 concurrent blocks (1 core each); per_record = median block time / (64 x 32768)'))
# TFHE
tf = f'{R}/tfhe_results.txt'
for line in open(tf) if os.path.exists(tf) else []:
    m = re.search(r'TFHE approach=(\S+) params=(\S+)(?: g=(\d+))? K=(\d+) pbs_per_record=(\d+) sample=(\d+) threads=(\d+) .*?'
                  r'per_pbs_ms_1t=([\d.]+) block_ms_1t_extrap=([\d.]+) block_ms_par_extrap=([\d.]+) per_record_ms_1t=([\d.]+) '
                  r'per_record_ms_par=([\d.]+) ok=(\w+)', line)
    if not m: continue
    a, prm, g, K, pbs, samp, thr = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))
    desc = (f'AND-K: TFHE-rs multi-input PBS, groups of g={g} (capped by max_noise_level)' if a == 'A5'
            else 'AND-K: TFHE-rs boolean gate-by-gate (K-1 HomAND)')
    for th, bms, prms in ((1, float(m.group(9)), float(m.group(11))), (thr, float(m.group(10)), float(m.group(12)))):
        rows.append(dict(config=f'{a}_{prm}_K{K}_t{th}', approach=a, operation=f'AND-{K}', description=desc, K=K,
                         params=f'tfhe-rs 1.7.0 {prm}', hmul_or_pbs=pbs, per_pbs_ms_1t=float(m.group(8)), threads=th,
                         records_per_block=32768, block_ms=round(bms, 1), per_record_ms=round(prms, 6),
                         ok=m.group(13) == 'match', source=f'extrapolated from a {samp}-record sample'))
# A4: analytical cost of one equality test with a small plaintext prime p (HElib-style
# extension-field slots); HElib was not run. Columns: K, p, m, phi(m), ord_m(p), slots per
# ciphertext, equality-test depth, equality-test HMuls per ciphertext.
A4 = [(2, 3, 34511, 34510, 17, 2030, 1, 1), (4, 5, 36991, 36400, 10, 3640, 2, 2),
      (8, 11, 41329, 40176, 12, 3348, 4, 4), (16, 17, 58023, 33048, 6, 5508, 4, 4),
      (32, 37, 39463, 35640, 6, 5940, 6, 6), (64, 67, 74085, 35840, 4, 8960, 7, 7),
      (128, 131, 42905, 34320, 4, 8580, 8, 8), (256, 257, 68481, 39096, 6, 6516, 8, 8)]
for K, p, m, phi, ordp, slots, depth, hm in A4:
    rows.append(dict(config=f"A4_K{K}", approach='A4', operation=f"AND-{K}",
                     description='AND-K: sum + one equality test with small p (HElib-style extension-field slots)',
                     K=K, p=p, params=f"m={m} phi(m)={phi} ord_m(p)={ordp}",
                     records_per_block=slots, depth=depth, hmul_or_pbs=hm, source='analytical (not run)'))
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
rows.sort(key=lambda r: (str(r.get('operation', '')), str(r.get('approach', '')), str(r.get('N', '')), str(r.get('p', '')), str(r.get('threads', ''))))
out = os.path.expanduser(HULK_ROOT + '/output/HULK_andK_comparison.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print('wrote', out, len(rows), 'rows')
