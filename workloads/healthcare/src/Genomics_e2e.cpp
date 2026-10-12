/*
 * Genomics workloads with PSI-like structure: bit-sliced encrypted client data,
 * plaintext reference/query data, a wide Boolean test per (client, reference)
 * pair, then a reduction.  Each workload is evaluated by the arithmetic approach
 * and by HULK over the same checks, and every output slot is verified against a
 * plaintext oracle.
 *
 * Workloads (--workload):
 *   kmer-exact    k-mer screening, k=31, 2 bits/base (62 bit-planes); AND-62 vs
 *                 each of 16 reference k-mers; OR over the references;
 *                 returns the NUMBER of matching reads (homomorphic sum_slots).
 *   kmer-fuzzy    k-mer screening with <=2 mismatched bases; one-hot bases
 *                 (124 planes); [>=29 of 31 bases match]; OR over 16 references;
 *                 returns the NUMBER of matching reads (homomorphic sum_slots).
 *   variant       pathogenic-variant panel; 37-bit (chr,pos,ref,alt) keys;
 *                 AND-37 vs 32 panel variants; returns the matched variant's label.
 *   kinship       relative search over an encrypted cohort; 32 SNPs, one-hot
 *                 genotypes (96 planes); [>=28 of 32 genotypes match] per query.
 *   cohort        Beacon/count query; 64 SNPs one-hot (192 planes); AND over 16
 *                 (SNP, genotype) conditions per query; returns the COUNT of
 *                 matching individuals per query (homomorphic sum_slots).
 *   str-exact     forensic STR identity; 20 loci x (min,max) 6-bit alleles
 *                 (240 planes); AND-240 per query profile, as all 20 per-locus
 *                 AND-12s (a single degree-240 HULK indicator exceeds the noise
 *                 budget at N=32768; per-locus grouping keeps it shallow).
 *   str-familial  forensic familial search; per-locus AND-12, then [>=15 of 20
 *                 loci match] per query profile.
 *
 * A check is either an AND over one group of terms, or a threshold
 * [#groups fully true >= need].  A term is a bit-plane or its complement.
 *   arithmetic : AND = balanced HMul product tree (an AND of all groups, as in
 *                str-exact, is one flat tree over every term); threshold = balanced
 *                tree that merges mismatch-count distributions truncated at
 *                D = n - need (exact, depth ceil(log2 n), HMul-lean leaf and root
 *                merges); OR = 1 - prod(1 - x).
 *   HULK       : AND = sum of terms, then one indicator [S = m];
 *                threshold = sum of group values, then one indicator [S >= need];
 *                OR = sum, then one nonzero indicator.
 *   COUNT (both modes, identical): sum_slots = log2(N/2) row rotations + 1 column
 *                rotation, each + HAdd, inside the timed region.
 *
 * Timing (per mode): enc_ms = client encryption of the bit-planes (same data, same
 * ciphertext count in both modes); eval_ms = homomorphic operations only, from the
 * first HE op to the returned ciphertexts (incl. sum_slots); dec_ms = decryption +
 * decode of the returned ciphertexts. Verification, noise-budget queries and
 * printing are outside all three. One line per mode:
 *   E2E mode=<m> limbs=<L> enc_ms=.. eval_ms=.. dec_ms=.. upload_bytes=.. hmul=.. rot=.. ok=..
 * --verify-records (count workloads only): additionally decrypt the per-record
 * indicators before sum_slots and check them slot by slot (debug; the ciphertext
 * copies it needs then land inside eval_ms, so do not use it for timing).
 */
#include "HE_workload_common.h"
#include <random>
#include <unordered_set>
using namespace hulk;

struct GCfg : RunConfig { string workload = "kmer-exact"; bool oracle_only = false; bool verify_records = false; };

static GCfg parse_g(int argc, char **argv){
    GCfg c; c.threads = 1;
    for (int i=1;i<argc;++i){ string a=argv[i];
        auto nx=[&]{ return stoull(argv[++i]); };
        if(a=="--workload") c.workload=argv[++i];
        else if(a=="--oracle-only") c.oracle_only=true;
        else if(a=="--verify-records") c.verify_records=true;
        else if(a=="--seed") c.seed=nx();
        else if(a=="--threads") c.threads=max<size_t>(1,nx());
        else if(a=="--mode") c.mode=parse_mode_value(argv[++i]);
        else if(a=="--limbs"){ size_t L=max<size_t>(1,nx()); c.arithmetic_limbs=L; c.hulk_limbs=L; }
        else if(a=="--arithmetic-limbs") c.arithmetic_limbs=max<size_t>(1,nx());
        else if(a=="--hulk-limbs") c.hulk_limbs=max<size_t>(1,nx());
        else throw invalid_argument("unknown arg: "+a);
    }
    return c;
}

struct Term  { int plane; int pol; };                 // pol=1: X[plane]; pol=0: 1-X[plane]
struct Check { vector<vector<Term>> groups; size_t need; };
enum class Agg { OR, LABEL, PER };

struct Workload {
    string desc; Agg agg = Agg::PER;
    vector<vector<uint64_t>> planes;                  // plane x slot, 0/1
    vector<Check> checks;                             // one per reference item / query
    vector<uint64_t> labels;                          // Agg::LABEL
    vector<vector<uint64_t>> direct;                  // semantic oracle: output x slot
    // count=true: the application returns the NUMBER of positive records per output
    // (kmer-exact / kmer-fuzzy: matching reads; cohort: count per query), computed
    // homomorphically with sum_slots inside the timed region in both modes.
    bool count = false;
};

// ---------------------------------------------------------------- data generation
static void bit_planes(Workload &w, const vector<uint64_t> &v, size_t width){
    const size_t base = w.planes.size(); w.planes.resize(base+width, vector<uint64_t>(v.size()));
    for(size_t t=0;t<width;++t) for(size_t i=0;i<v.size();++i) w.planes[base+t][i]=(v[i]>>t)&1ull;
}
static void onehot_planes(Workload &w, const vector<vector<int>> &sym, size_t alph){
    const size_t N=sym.size(), P=sym[0].size(); w.planes.assign(P*alph, vector<uint64_t>(N,0));
    for(size_t i=0;i<N;++i) for(size_t p=0;p<P;++p) w.planes[p*alph+sym[i][p]][i]=1;
}
static vector<Term> key_terms(uint64_t y, size_t width, size_t base=0){
    vector<Term> g; for(size_t t=0;t<width;++t) g.push_back({int(base+t), int((y>>t)&1ull)}); return g;
}
static vector<uint64_t> distinct_keys(mt19937_64 &rng, size_t K, size_t width){
    uniform_int_distribution<uint64_t> d(0, (width>=64)?~0ull:((1ull<<width)-1));
    unordered_set<uint64_t> s; while(s.size()<K) s.insert(d(rng)); return {s.begin(),s.end()};
}
static void mutate(mt19937_64 &rng, vector<int> &s, size_t count, int alph){
    vector<size_t> idx(s.size()); iota(idx.begin(),idx.end(),0); shuffle(idx.begin(),idx.end(),rng);
    for(size_t c=0;c<count;++c){ int old=s[idx[c]]; int nv; do nv=int(rng()%alph); while(nv==old); s[idx[c]]=nv; }
}

static Workload make_workload(const string &name, size_t N, uint64_t seed){
    Workload w; mt19937_64 rng(seed);
    if(name=="kmer-exact" || name=="variant"){
        const bool kmer = name=="kmer-exact";
        const size_t width = kmer? 62 : 37, K = kmer? 16 : 32;
        w.desc = kmer? "k-mer screening, exact (k=31, 2 bits/base, AND-62, OR over 16 references)"
                     : "pathogenic-variant panel (37-bit keys, AND-37, labelled over 32 panel variants)";
        w.agg = kmer? Agg::OR : Agg::LABEL; w.count = kmer;
        auto ref = distinct_keys(rng, K, width);
        uniform_int_distribution<uint64_t> d(0,(1ull<<width)-1), lab(1,65535);
        vector<uint64_t> cl(N); for(size_t i=0;i<N;++i) cl[i] = (i%5==0)? ref[i%K] : d(rng);
        bit_planes(w, cl, width);
        for(uint64_t y: ref){ w.checks.push_back({{key_terms(y,width)},1}); w.labels.push_back(lab(rng)); }
        w.direct.assign(1, vector<uint64_t>(N,0));
        for(size_t i=0;i<N;++i) for(size_t j=0;j<K;++j) if(cl[i]==ref[j]) w.direct[0][i] = kmer? 1 : w.labels[j];
        if(kmer) w.labels.clear();
    } else if(name=="kmer-fuzzy"){
        const size_t k=31, K=16, D=2; w.desc="k-mer screening, fuzzy (k=31, <=2 mismatched bases, one-hot, OR over 16 references)";
        w.agg=Agg::OR; w.count=true;
        vector<vector<int>> ref(K, vector<int>(k)); for(auto &r:ref) for(auto &b:r) b=int(rng()%4);
        vector<vector<int>> cl(N, vector<int>(k));
        for(size_t i=0;i<N;++i){ size_t r=i%8;
            if(r<5){ cl[i]=ref[(i/8)%K]; mutate(rng, cl[i], r, 4); } else for(auto &b:cl[i]) b=int(rng()%4); }
        onehot_planes(w, cl, 4);
        for(auto &y: ref){ Check c; c.need=k-D; for(size_t p=0;p<k;++p) c.groups.push_back({{int(p*4+y[p]),1}}); w.checks.push_back(c); }
        w.direct.assign(1, vector<uint64_t>(N,0));
        for(size_t i=0;i<N;++i) for(auto &y: ref){ size_t mm=0; for(size_t p=0;p<k;++p) mm+= cl[i][p]!=y[p]; if(mm<=D) w.direct[0][i]=1; }
    } else if(name=="kinship" || name=="cohort"){
        const bool kin = name=="kinship";
        const size_t M = kin? 32 : 64, Q = kin? 2 : 4;
        w.desc = kin? "kinship search (32 SNPs one-hot, [>=28 of 32 genotypes match], 2 query genomes)"
                    : "cohort count / Beacon (64 SNPs one-hot, AND over 16 (SNP,genotype) conditions, 4 queries)";
        w.agg=Agg::PER;
        vector<vector<int>> db(N, vector<int>(M)); for(auto &r:db) for(auto &g:r) g=int(rng()%3);
        if(kin){
            vector<vector<int>> qg(Q, vector<int>(M)); for(auto &r:qg) for(auto &g:r) g=int(rng()%3);
            for(size_t i=0;i<N;++i){ size_t r=i%10; if(r<8){ db[i]=qg[(i/10)%Q]; mutate(rng, db[i], r, 3); } }
            onehot_planes(w, db, 3);
            for(auto &h: qg){ Check c; c.need=M-4; for(size_t p=0;p<M;++p) c.groups.push_back({{int(p*3+h[p]),1}}); w.checks.push_back(c); }
            w.direct.assign(Q, vector<uint64_t>(N,0));
            for(size_t q=0;q<Q;++q) for(size_t i=0;i<N;++i){ size_t m=0; for(size_t p=0;p<M;++p) m+= db[i][p]==qg[q][p]; w.direct[q][i]= m>=M-4; }
        } else {
            vector<vector<pair<int,int>>> cond(Q);
            for(auto &cq: cond){ vector<size_t> idx(M); iota(idx.begin(),idx.end(),0); shuffle(idx.begin(),idx.end(),rng);
                for(size_t c=0;c<16;++c) cq.push_back({int(idx[c]), int(rng()%3)}); }
            for(size_t i=0;i<N;++i){ size_t r=i%9, q=(i/9)%Q;
                if(r<=1){ for(auto &[s,g]: cond[q]) db[i][s]=g; if(r==1){ auto &[s,g]=cond[q][i%16]; db[i][s]=(g+1)%3; } } }
            onehot_planes(w, db, 3);
            w.count=true;
            for(auto &cq: cond){ Check c; c.need=1; vector<Term> g; for(auto &[s,gt]: cq) g.push_back({s*3+gt,1}); c.groups.push_back(g); w.checks.push_back(c); }
            w.direct.assign(Q, vector<uint64_t>(N,0));
            for(size_t q=0;q<Q;++q) for(size_t i=0;i<N;++i){ bool ok=true; for(auto &[s,g]: cond[q]) ok = ok && db[i][s]==g; w.direct[q][i]=ok; }
        }
    } else if(name=="str-exact" || name=="str-familial"){
        const bool fam = name=="str-familial"; const size_t L=20, Q=2, need=15;
        w.desc = fam? "forensic STR familial search (20 loci x 12 bits, per-locus AND-12, [>=15 of 20 loci], 2 queries)"
                    : "forensic STR identity (20 loci x 12 bits, AND-240, 2 query profiles)";
        w.agg=Agg::PER;
        auto locus = [&](){ int a=5+int(rng()%40), b=5+int(rng()%40); return make_pair(min(a,b),max(a,b)); };
        auto code  = [](pair<int,int> p){ return uint64_t(p.first) | (uint64_t(p.second)<<6); };
        vector<vector<pair<int,int>>> qp(Q, vector<pair<int,int>>(L)); for(auto &r:qp) for(auto &x:r) x=locus();
        vector<vector<pair<int,int>>> db(N, vector<pair<int,int>>(L));
        for(size_t i=0;i<N;++i){ size_t r=i%12;
            if(r<=7){ db[i]=qp[(i/12)%Q]; vector<size_t> idx(L); iota(idx.begin(),idx.end(),0); shuffle(idx.begin(),idx.end(),rng);
                      for(size_t c=0;c<r;++c){ auto nv=locus(); while(nv==db[i][idx[c]]) nv=locus(); db[i][idx[c]]=nv; } }
            else for(auto &x:db[i]) x=locus(); }
        for(size_t l=0;l<L;++l){ vector<uint64_t> v(N); for(size_t i=0;i<N;++i) v[i]=code(db[i][l]); bit_planes(w, v, 12); }
        for(auto &q: qp){ Check c;
            if(fam){ c.need=need; for(size_t l=0;l<L;++l) c.groups.push_back(key_terms(code(q[l]),12,l*12)); }
            else   { c.need=L; for(size_t l=0;l<L;++l) c.groups.push_back(key_terms(code(q[l]),12,l*12)); }   // identity: all 20 loci match (same AND-240)
            w.checks.push_back(c); }
        w.direct.assign(Q, vector<uint64_t>(N,0));
        for(size_t q=0;q<Q;++q) for(size_t i=0;i<N;++i){ size_t m=0; for(size_t l=0;l<L;++l) m+= db[i][l]==qp[q][l]; w.direct[q][i]= fam? (m>=need) : (m==L); }
    } else if(name=="websearch"){
        // BitFunnel-style web-search indexing (Goodwin et al., SIGIR 2017): a bit-sliced
        // signature index. At ingest the data owner hashes every term of every document
        // into R=2 of the 128 signature rows (Bloom-filter style) and encrypts the rows
        // (row r = one ciphertext, document i = slot i). A query of 6 terms probes its
        // 12 rows; a document is a candidate iff all 12 bits are set (AND-12). Query row
        // indices are public, the corpus is private. 32 queries; per-query candidate mask.
        const size_t ROWS=128, R=2, TERMS=6, Q=32;
        w.desc="web-search indexing, BitFunnel signatures (128 rows, 32 queries, 6 terms x 2 rows = AND-12)";
        w.agg=Agg::PER;
        w.planes.assign(ROWS, vector<uint64_t>(N,0));
        bernoulli_distribution bit(0.35);                    // row density from the other terms of each document
        for(auto &row: w.planes) for(auto &b: row) b = bit(rng);
        vector<vector<int>> qrows(Q);
        for(auto &qr: qrows){ vector<int> idx(ROWS); iota(idx.begin(),idx.end(),0); shuffle(idx.begin(),idx.end(),rng);
            qr.assign(idx.begin(), idx.begin()+TERMS*R); }  // 12 distinct rows per query
        for(size_t i=0;i<N;++i) if(i%16==0) for(int r: qrows[(i/16)%Q]) w.planes[r][i]=1;   // documents containing every query term
        for(auto &qr: qrows){ Check c; c.need=1; vector<Term> g; for(int r: qr) g.push_back({r,1}); c.groups.push_back(g); w.checks.push_back(c); }
        w.direct.assign(Q, vector<uint64_t>(N,0));
        for(size_t q=0;q<Q;++q) for(size_t i=0;i<N;++i){ bool ok=true; for(int r: qrows[q]) ok = ok && w.planes[r][i]; w.direct[q][i]=ok; }
    } else throw invalid_argument("unknown workload: "+name);
    return w;
}

// generic plaintext oracle from the check structure (cross-checked against w.direct)
static vector<vector<uint64_t>> generic_oracle(const Workload &w, size_t N){
    vector<vector<uint64_t>> per(w.checks.size(), vector<uint64_t>(N,0));
    for(size_t j=0;j<w.checks.size();++j) for(size_t i=0;i<N;++i){
        const Check &c=w.checks[j]; size_t ok=0;
        for(auto &g: c.groups){ bool all=true; for(auto &t: g) all = all && (w.planes[t.plane][i]==uint64_t(t.pol)); ok+=all; }
        per[j][i] = (c.groups.size()==1)? ok : (ok>=c.need);
    }
    if(w.agg==Agg::PER) return per;
    vector<vector<uint64_t>> out(1, vector<uint64_t>(N,0));
    for(size_t i=0;i<N;++i) for(size_t j=0;j<per.size();++j){
        if(w.agg==Agg::OR) out[0][i] |= per[j][i];
        else out[0][i] += per[j][i]*w.labels[j];
    }
    return out;
}

// ---------------------------------------------------------------- arithmetic approach
// 1 - x: the shared harness one_minus_bit (negate + cached plaintext add; no
// encryption, no plaintext multiply), the same complement the PSI driver uses.
static Ciphertext not_bit(const HEEnv &env, const Ciphertext &x, OpCount &ops){ return one_minus_bit(env,x,ops); }
// OR = 1 - prod(1 - x), with the same cheap complement
static pair<Ciphertext,OpCount> arith_or(const vector<Ciphertext> &bits, const HEEnv &env){
    OpCount ops; vector<Ciphertext> inv; inv.reserve(bits.size());
    for(auto &b: bits) inv.push_back(not_bit(env,b,ops));
    auto p=arithmetic_all_true(inv,env); ops+=p.second;
    return {not_bit(env,p.first,ops), ops};
}
static Ciphertext arith_term(const vector<Ciphertext> &X, const vector<Ciphertext> &NX, Term t){ return t.pol? X[t.plane] : NX[t.plane]; }

static Ciphertext arith_group(const HEEnv &env, const vector<Ciphertext> &X, const vector<Ciphertext> &NX, const vector<Term> &g, OpCount &ops){
    if(g.size()==1) return arith_term(X,NX,g[0]);
    vector<Ciphertext> v; v.reserve(g.size()); for(auto &t: g) v.push_back(arith_term(X,NX,t));
    auto r=arithmetic_all_true(v,env); ops+=r.second; return r.first;
}

// [#true >= need] by merging mismatch-count distributions truncated at D = n - need
// (exact; balanced tree, depth ceil(log2 n)). Entry k of a node over m inputs is
// the encrypted indicator [#mismatches among its m inputs == k], k <= D.
// HMul-lean merges (audit G8), all exact on 0/1 inputs:
//   leaf x leaf  : p=v*w; C0=p, C1=v+w-2p, C2=1-v-w+p              (1 HMul)
//   leaf w x A   : C_k = A_{k-1} + w*(A_k - A_{k-1}),
//                  C_|A| = A_{|A|-1} - sum_k w*(A_k - A_{k-1})     (|A| HMul)
//   root (last)  : P(<=D) = sum_a L_a * prefix(R)_{D-a}; a prefix that covers a
//                  complete (untruncated) distribution is exactly 1 -> no HMul.
struct ThrNode { vector<Ciphertext> c; size_t m; };   // m = #inputs; a leaf (m==1) stores only {v}
static Ciphertext arith_threshold(const HEEnv &env, const vector<Ciphertext> &vals, size_t need, OpCount &ops){
    const size_t n=vals.size(), D=n-need;
    if(n==1) return vals[0];
    auto entries =[&](const ThrNode &x){ return min(x.m, D)+1; };          // logical length (truncated at D)
    auto complete=[&](const ThrNode &x){ return x.m<=D; };                  // untruncated: entries sum to 1
    auto sub=[&](const Ciphertext &a, const Ciphertext &b){ Ciphertext r; env.eval->sub(a,b,r); ops.ct_add++; return r; };
    auto add=[&](const Ciphertext &a, const Ciphertext &b){ Ciphertext r=a; add_inplace(env,r,b,ops); return r; };
    auto mul=[&](const Ciphertext &a, const Ciphertext &b){ return multiply_ciphertexts(env,a,b,ops); };
    auto merge_leaves=[&](const Ciphertext &v, const Ciphertext &w) -> ThrNode {
        ThrNode out; out.m=2; const size_t len=entries(out);
        Ciphertext p=mul(v,w); out.c.push_back(p);                                   // C0 = vw
        if(len>=2){ Ciphertext t=add(v,w); out.c.push_back(sub(sub(t,p),p));         // C1 = v+w-2p
            if(len>=3){ Ciphertext c2=sub(p,t); add_plain_inplace(env,c2,1,ops); out.c.push_back(move(c2)); } }  // C2 = 1-v-w+p
        return out;
    };
    auto merge_leaf=[&](const Ciphertext &w, const ThrNode &A) -> ThrNode {         // leaf w x distribution A
        ThrNode out; out.m=A.m+1; const size_t len=entries(out), na=A.c.size();
        Ciphertext qsum;
        for(size_t k=0;k<na && k<len;++k){
            Ciphertext q = mul(w, k? sub(A.c[k],A.c[k-1]) : A.c[0]);
            out.c.push_back(k? add(A.c[k-1],q) : q);
            if(k==0) qsum=q; else add_inplace(env,qsum,q,ops);
        }
        if(len>na) out.c.push_back(sub(A.c[na-1],qsum));                              // (1-w)*A_last
        return out;
    };
    auto merge=[&](const ThrNode &L, const ThrNode &R) -> ThrNode {
        if(L.m==1 && R.m==1) return merge_leaves(L.c[0],R.c[0]);
        if(L.m==1) return merge_leaf(L.c[0],R);
        if(R.m==1) return merge_leaf(R.c[0],L);
        ThrNode out; out.m=L.m+R.m; const size_t len=entries(out);
        vector<Ciphertext> C(len); vector<bool> init(len,false);
        for(size_t a=0;a<L.c.size();++a) for(size_t b=0;b<R.c.size() && a+b<len;++b){
            Ciphertext p=mul(L.c[a],R.c[b]);
            if(!init[a+b]){ C[a+b]=move(p); init[a+b]=true; } else add_inplace(env,C[a+b],p,ops);
        }
        out.c=move(C); return out;
    };
    // root: P(mismatches <= D) = sum_{a<=D} L_a * prefR_{min(D-a,|R|-1)}; the full prefix of a
    // complete R is exactly 1, so that term is L_a itself (no HMul).
    auto root_cost=[&](const ThrNode &L, const ThrNode &R){
        size_t cost=0; const size_t nl=entries(L), nr=entries(R);
        for(size_t a=0;a<nl && a<=D;++a){ const size_t k=min(D-a,nr-1); if(!(k==nr-1 && complete(R))) ++cost; }
        return cost;
    };
    auto root=[&](ThrNode L, const ThrNode &R) -> Ciphertext {
        if(L.m==1 && D>=1) L.c.push_back(not_bit(env,L.c[0],ops));                   // leaf L needs 1-v
        const size_t nr=entries(R), top = complete(R)? nr-1 : nr;                    // prefixes actually used
        vector<Ciphertext> pref; pref.reserve(top);
        for(size_t k=0;k<top;++k) pref.push_back(k? add(pref[k-1],R.c[k]) : R.c[0]);
        Ciphertext res;
        for(size_t a=0;a<L.c.size() && a<=D;++a){
            const size_t k=min(D-a,nr-1);
            Ciphertext t = (k==nr-1 && complete(R)) ? L.c[a] : mul(L.c[a],pref[k]);
            if(a==0) res=move(t); else add_inplace(env,res,t,ops);
        }
        return res;
    };
    vector<ThrNode> layer; layer.reserve(n);
    for(auto &v: vals) layer.push_back(ThrNode{{v},1});
    while(layer.size()>2){
        vector<ThrNode> next; next.reserve((layer.size()+1)/2);
        for(size_t i=0;i<layer.size();i+=2){
            if(i+1==layer.size()){ next.push_back(move(layer[i])); continue; }
            next.push_back(merge(layer[i],layer[i+1]));
        }
        layer.swap(next);
    }
    return root_cost(layer[1],layer[0]) < root_cost(layer[0],layer[1]) ? root(layer[1],layer[0]) : root(layer[0],layer[1]);
}

// ---------------------------------------------------------------- HULK
// S = sum over terms of (X or 1-X): additions/subtractions and one plaintext constant, no HMul
static Ciphertext affine_sum(const HEEnv &env, const vector<Ciphertext> &X, const vector<Term> &terms, OpCount &ops){
    Ciphertext S; bool init=false; uint64_t konst=0; const uint64_t p=env.plain_modulus;
    for(auto &t: terms){
        if(t.pol){ if(!init){ S=X[t.plane]; init=true; } else add_inplace(env,S,X[t.plane],ops); }
        else { ++konst;
               if(!init){ S=multiply_plain_const(env,X[t.plane],mod_sub(0,1,p),ops); init=true; }
               else { env.eval->sub_inplace(S,X[t.plane]); ops.ct_add++; } }
    }
    if(konst) add_plain_inplace(env,S,konst,ops);
    return S;
}

// Deferred 1/K! (see HulkDeferScaleScope). `final_scale` receives the constant the client
// multiplies the decrypted value by (1 if nothing was deferred); `defer_final` = false
// for checks whose value is consumed further (OR / label reductions).
//  * single AND group, final: indicator evaluated in a deferral scope.
//  * threshold over equal-width AND groups: each group indicator is deferred by the same
//    constant c (value = I/c), so S' = S/c and [S >= need] = f(c*S') is evaluated with
//    coefficients a_i c^i -- exact, and it removes the ~log2(t) noise bits of the inner
//    1/K! multiplies (coefficients are full-size plaintext scalars either way).
static Ciphertext hulk_check(const HEEnv &env, const vector<Ciphertext> &X, const Check &c, OpCount &ops,
                             bool defer_final = false, uint64_t *final_scale = nullptr){
    const uint64_t p=env.plain_modulus;
    if(final_scale) *final_scale=1;
    if(c.groups.size()==1){
        const size_t m=c.groups[0].size(); Ciphertext S=affine_sum(env,X,c.groups[0],ops);
        if(m==1) return S;
        HulkDeferScaleScope defer(defer_final);
        auto r=hulk_indicator_from_sum(S,m,m,env); ops+=r.second;
        if(final_scale) *final_scale=defer.skipped();
        return r.first;
    }
    const size_t n=c.groups.size(); Ciphertext S; bool init=false;
    const size_t w0=c.groups[0].size(); bool uniform = w0>=2;
    for(auto &g: c.groups) uniform = uniform && g.size()==w0;
    uint64_t gscale=1; bool gscale_set=false;
    for(auto &g: c.groups){
        Ciphertext v;
        if(g.size()==1) v=affine_sum(env,X,g,ops);
        else { Ciphertext s=affine_sum(env,X,g,ops);
               HulkDeferScaleScope defer(uniform);
               auto r=hulk_indicator_from_sum(s,g.size(),g.size(),env); ops+=r.second; v=move(r.first);
               const uint64_t sk=defer.skipped();
               if(!gscale_set){ gscale=sk; gscale_set=true; }
               else if(gscale!=sk) throw logic_error("hulk_check: group scales differ"); }
        if(!init){ S=move(v); init=true; } else add_inplace(env,S,v,ops);
    }
    // [S >= need] via BSGS (depth ~log2 n + 1); with deferred groups S holds S_true/gscale.
    if(uniform && gscale!=1){
        vector<uint64_t> a=bounded_ge_plain_coefficients(env,n,c.need); uint64_t cp=1;
        for(auto &ai: a){ ai=mod_mul(ai%p,cp,p); cp=mod_mul(cp,gscale,p); }
        auto r=eval_poly_bsgs(env,S,a); ops+=r.second; return r.first;
    }
    auto r=he_ge_plain_bounded(env,S,n,c.need); ops+=r.second; return r.first;
}

// ---------------------------------------------------------------- driver
// What the application returns, per output: a count (w.count) or a per-record
// slot vector. returned_oracle(w, per_record_oracle)[o] is a 1-element {count}
// vector for count outputs, else the N-slot vector.
static vector<vector<uint64_t>> returned_oracle(const Workload &w, const vector<vector<uint64_t>> &per){
    if(!w.count) return per;
    vector<vector<uint64_t>> out;
    for(auto &v: per){ uint64_t c=0; for(auto x: v) c+=x; out.push_back({c}); }
    return out;
}

int main(int argc, char **argv){
    const GCfg cfg=parse_g(argc,argv);
    if(cfg.oracle_only){                                   // plaintext check of the workload definition only
        const size_t N=kPolyDegree; Workload w=make_workload(cfg.workload,N,cfg.seed); auto o=generic_oracle(w,N);
        size_t hits=0; for(auto &v: o) for(auto x: v) hits+= x!=0;
        const auto ret=returned_oracle(w,o);
        cout<<cfg.workload<<": planes="<<w.planes.size()<<" checks="<<w.checks.size()<<" positives="<<hits
            <<" returns="<<(w.count? "count":"per-record");
        if(w.count){ cout<<" counts="; for(size_t q=0;q<ret.size();++q) cout<<(q?",":"")<<ret[q][0]; }
        cout<<" oracle-consistent="<<(o==w.direct?"yes":"NO")<<"\n"; return o==w.direct?0:1; }
    const BenchmarkEnvs envs=make_benchmark_envs(cfg);
    const size_t N=envs.slot_count;
    Workload w=make_workload(cfg.workload, N, cfg.seed);
    const auto oracle=generic_oracle(w,N);
    if(oracle!=w.direct){ cerr<<"internal error: generic oracle disagrees with semantic oracle\n"; return 2; }
    const auto expect=returned_oracle(w,oracle);
    size_t hits=0; for(auto &o: oracle) for(auto v: o) hits += v!=0;
    cout<<"Genomics workload: "<<cfg.workload<<" -- "<<w.desc<<"\n"
        <<"slots: "<<N<<", planes: "<<w.planes.size()<<", checks: "<<w.checks.size()<<", positive outputs: "<<hits
        <<", returns: "<<(w.count? "count per output (homomorphic sum_slots)" : "per-record slots")<<"\n";
    if(w.count){ cout<<"oracle counts:"; for(auto &e: expect) cout<<" "<<e[0]; cout<<"\n"; }
    cout<<fixed<<setprecision(3);
    bool all_ok=true; double t_ar=0,t_hl=0;
    for(int pass=0;pass<2;++pass){
        const EvalMode m = pass? EvalMode::hulk : EvalMode::arithmetic;
        if(m==EvalMode::arithmetic && !wants_arithmetic(cfg.mode)) continue;
        if(m==EvalMode::hulk && !wants_hulk(cfg.mode)) continue;
        const HEEnv &env=envs.for_mode(m); const uint64_t p=env.plain_modulus;
        if(w.count) for(auto &e: expect) if(e[0]>=p)
            throw runtime_error(cfg.workload+": oracle count "+to_string(e[0])+" >= t (aggregate >= t)");

        // ---- client: encrypt the bit-planes (identical data and work in both modes; timed as enc_ms)
        Stopwatch enc_sw;
        vector<Ciphertext> X; X.reserve(w.planes.size()); for(auto &pl: w.planes) X.push_back(encrypt_slots(env,pl));
        const double enc_ms=enc_sw.ms();
        const size_t upload_bytes=ciphertext_upload_bytes(X);

        // ---- server: homomorphic evaluation only (eval_ms)
        OpCount ops; const auto t0=steady_clock::now();
        vector<Ciphertext> NX;                                  // complements, arithmetic only (computed once)
        vector<uint64_t> out_scale;                             // client-side constants (deferred 1/K!), HULK only
        vector<Ciphertext> vals; vals.reserve(w.checks.size());
        if(m==EvalMode::arithmetic){
            vector<bool> need_not(X.size(),false);
            for(auto &c: w.checks) for(auto &g: c.groups) for(auto &t: g) if(!t.pol) need_not[t.plane]=true;
            NX.resize(X.size()); for(size_t k=0;k<X.size();++k) if(need_not[k]) NX[k]=not_bit(env,X[k],ops);
            for(auto &c: w.checks){
                if(c.groups.size()==1) vals.push_back(arith_group(env,X,NX,c.groups[0],ops));
                else if(c.need==c.groups.size()){   // AND of all groups: one flat balanced tree (depth ceil(log2 #terms)), audit G7
                    vector<Ciphertext> v; for(auto &g: c.groups) for(auto &t: g) v.push_back(arith_term(X,NX,t));
                    auto r=arithmetic_all_true(v,env); ops+=r.second; vals.push_back(move(r.first)); }
                else { vector<Ciphertext> gv; for(auto &g: c.groups) gv.push_back(arith_group(env,X,NX,g,ops)); vals.push_back(arith_threshold(env,gv,c.need,ops)); }
            }
        } else for(auto &c: w.checks){ uint64_t sc=1; vals.push_back(hulk_check(env,X,c,ops,w.agg==Agg::PER,&sc)); out_scale.push_back(sc); }
        const OpCount ops_after_test=ops;                       // stage snapshot (instrumentation only)

        vector<Ciphertext> outs;
        if(w.agg==Agg::PER) outs=move(vals);
        else if(w.agg==Agg::OR){
            out_scale.assign(1,1);
            HulkDeferScaleScope defer(m==EvalMode::hulk);     // final OR: 1/K! deferred to the client
            auto r = (m==EvalMode::hulk)? hulk_any_true(vals,env) : arith_or(vals,env); ops+=r.second; outs.push_back(move(r.first));
            out_scale[0]=defer.skipped(); }
        else { out_scale.assign(1,1); Ciphertext acc; for(size_t j=0;j<vals.size();++j){ Ciphertext t=multiply_plain_const(env,vals[j],w.labels[j],ops); if(j==0) acc=move(t); else add_inplace(env,acc,t,ops); } outs.push_back(move(acc)); }
        // count outputs: rotate-and-sum (identical in both modes) so every slot holds the count
        vector<Ciphertext> records;                             // --verify-records only (copies are inside the timer)
        OpCount agg_ops; double agg_ms=0;
        const OpCount ops_after_reduce=ops;                     // stage snapshot (instrumentation only)
        // Default: client-side slot reduction (the server returns the per-record ciphertext,
        // no rotations; the client sums the slots). E2E_SLOT_AGGREGATE=1 restores sum_slots.
        const bool slot_agg = env_flag_enabled("E2E_SLOT_AGGREGATE");
        if(w.count){
            if(cfg.verify_records) records=outs;
            Stopwatch agg_sw;
            if(slot_agg) for(auto &o: outs){ auto r=sum_slots(move(o),env); o=move(r.first); agg_ops+=r.second; }
            agg_ms=agg_sw.ms(); ops+=agg_ops;
        }
        const double eval_ms=duration<double,milli>(steady_clock::now()-t0).count();

        // ---- client: decrypt + decode the returned values (dec_ms)
        Stopwatch dec_sw;
        vector<vector<uint64_t>> got; got.reserve(outs.size());
        for(size_t oi=0; oi<outs.size(); ++oi){
            auto d=decrypt_slots(env,outs[oi]);
            const uint64_t sc = oi<out_scale.size()? out_scale[oi]%p : 1;
            if(w.count){
                uint64_t v=0;
                if(slot_agg) v=d[0]%p; else for(auto x: d) v=(v+x%p)%p;
                got.push_back({mod_mul(v,sc,p)});
            } else { if(sc!=1) for(auto &x: d) x=mod_mul(x%p,sc,p); got.push_back(move(d)); }
        }
        const double dec_ms=dec_sw.ms();

        // ---- verification (untimed): every returned value against the plaintext oracle
        bool ok = got.size()==expect.size();
        for(size_t o=0;ok && o<got.size();++o) ok = got[o]==expect[o];
        if(w.count && slot_agg){   // sum_slots leaves the count in EVERY slot; a Galois-key failure breaks this
            for(auto &o: outs){ auto d=decrypt_slots(env,o); for(auto v: d) ok = ok && v==d[0]; }
        }
        bool rec_ok=true;
        if(cfg.verify_records && w.count){
            for(size_t o=0;o<records.size();++o){ auto d=decrypt_slots(env,records[o]); for(size_t i=0;i<N;++i) rec_ok = rec_ok && (d[i]%p)==oracle[o][i]; }
            ok = ok && rec_ok;
        }
        all_ok = all_ok && ok; (m==EvalMode::hulk? t_hl : t_ar)=eval_ms;

        const char *mn = m==EvalMode::hulk?"hulk":"arithmetic";
        cout<<"\n["<<mn<<"] limbs="<<env.coeff_modulus_primes<<"\n"
            <<"  encrypt (client) : "<<enc_ms<<" ms   upload "<<X.size()<<" ciphertexts, "<<upload_bytes<<" B\n"
            <<"  HE eval (server) : "<<eval_ms<<" ms   HMul="<<ops.total_ct_mul()<<"   rotations="<<ops.rotations;
        if(w.count) cout<<"   (sum_slots: "<<agg_ms<<" ms, "<<agg_ops.rotations<<" rotations)";
        cout<<"\n  decrypt (client) : "<<dec_ms<<" ms   ["<<matches_string(ok)<<"]\n";
        if(w.count){ cout<<"  returned counts  :"; for(auto &g: got) cout<<" "<<g[0]; cout<<"\n"; }
        if(cfg.verify_records && w.count) cout<<"  per-record check : "<<(rec_ok?"match":"MISMATCH")<<"\n";
        int nb=INT_MAX; for(auto &o: outs) nb=min(nb, env.dec->invariant_noise_budget(o));
        cout<<"  noise budget left: "<<nb<<" bits (min over outputs)\n";
        print_ops(mn, ops);
        // per-stage op counts (sum exactly to `ops`): test = per-pair AND / threshold per
        // reference or query (incl. arithmetic complements); reduce = OR over references or
        // label weighting (none for per-query outputs); aggregate = sum_slots of count outputs
        print_stage_line(mn, "test", ops_after_test);
        print_stage_line(mn, "reduce", op_diff(ops_after_reduce, ops_after_test));
        print_stage_line(mn, "aggregate", agg_ops);
        E2EReport rep; rep.mode=m; rep.limbs=env.coeff_modulus_primes; rep.enc_ms=enc_ms; rep.eval_ms=eval_ms; rep.dec_ms=dec_ms;
        rep.upload_bytes=upload_bytes; rep.hmul=ops.total_ct_mul(); rep.rot=ops.rotations; rep.ok=ok;
        print_e2e_line(rep);
    }
    if(t_ar>0&&t_hl>0) cout<<"\nspeedup (arith/HULK, HE eval) = "<<(t_ar/t_hl)<<"x\n";
    return all_ok?0:1;
}
