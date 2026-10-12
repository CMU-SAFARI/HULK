import glob, os, re
HULK_ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
H=HULK_ROOT
W=[('kmer-exact','results_fairg/kmer-exact'),('kmer-fuzzy','results_fairg/kmer-fuzzy'),('variant','results_fairg/variant'),
   ('GWAS','results_fairg/gwas-ps9'),('PSI','results_fairg/psi-bit'),('ANN','results_fairg/ann-ps9'),
   ('TPC-H_Q6','results_fair6s/TPC-H_Q6'),('TPC-H_Q12','results_fair6s/TPC-H_Q12'),('TPC-H_Q19','results_fair6s/TPC-H_Q19'),
   ('TPC-DS_Q3','results_fair6s/TPC-DS_Q3'),('TPC-DS_Q7','results_fair6s/TPC-DS_Q7'),('TPC-DS_Q43','results_fair6s/TPC-DS_Q43'),
   ('TPC-H_Q3_topk','results_fairg/TPC-H_Q3_topk')]
def noise(t, m):
    v=[int(x) for x in re.findall(r'noise budget left: (-?\d+) bits', t)]
    v+=[int(a) for a in re.findall(r'(?:HULK|arithmetic) noise budget \(min/max bits remaining\): (-?\d+)', t)]
    return min(v) if v else None
print('workload,mode,limbs,rounds,min_noise_bits_timed,hmul_total,rotations,hadd,ct_pt_mul,pt_add,stage_sum_check')
for name,d in W:
    for m in ('arithmetic','hulk'):
        fs=sorted(glob.glob(f'{H}/{d}/logs/timed_{m}_r*.log'))
        if not fs: print(f'{name},{m},MISSING'); continue
        nb=[]; L=hm=rot=None; add=pm=pa=0
        for i,f in enumerate(fs):
            t=open(f).read(); n=noise(t,m); nb.append(n)
            e=re.search(r'E2E mode=\S+ limbs=(\d+) .*?hmul=(\d+) rot=(\d+)',t)
            if e: L,hm,rot=e.group(1),int(e.group(2)),int(e.group(3))
            if i==0:
                s_h=0
                for s in re.finditer(r'STAGE mode=\S+ stage=\w+ (.*)',t):
                    c={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',s.group(1))}
                    add+=c.get('add',0); pm+=c.get('pmul',0); pa+=c.get('padd',0); s_h+=c.get('hmul',0)+c.get('sq',0)
        nbv=[x for x in nb if x is not None]
        print(f'{name},{m},{L},{len(fs)},{min(nbv) if nbv else "not logged"},{hm},{rot},{add},{pm},{pa},{s_h}')
