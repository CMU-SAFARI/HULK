/*
 * TPC-H Q3 top-10 under encryption (setup A): the one evaluated TPC query with a
 * server-side ORDER BY ... LIMIT.
 *
 *   select o_orderkey, revenue ... where c_mktsegment = 'BUILDING'
 *   order by revenue desc limit 10
 *
 * Encrypted columns (dictionary-coded, one order per slot): c_mktsegment (D=5) and
 * the order's revenue score, quantized at ingest to 0..M (default M=15). The other Q3
 * predicates (o_orderdate, l_shipdate over D~2500) are not evaluated here: together
 * with the sort they exceed the depth budget of N=32768 without bootstrapping.
 *
 * Sorting = rank-based top-k ("sorting by counting", Cetin, Doroz, Sunar, Savas,
 * LATINCRYPT 2015), evaluated per shard of n candidate orders (default n=64):
 *   q_i   = [segment_i == BUILDING]                      (comparison, BSGS, D=5)
 *   c_ij  = [s_p > s_i] * q_p,  p = (i+j) mod n,  j=1..n-1
 *           (partner via rotation by 1, repeated; GT interpolated on d in [-M, M])
 *   S_i   = sum_j c_ij + k*(1 - q_i)                     (affine)
 *   top_i = [S_i <= k-1]                                 (rank threshold)
 * top_i = 1 iff order i qualifies and fewer than k qualifying orders of its shard have a
 * strictly larger score (ties keep all tied orders, so the flags are a superset of the
 * top-k; the client, which owns the data, breaks ties and merges the shard winners).
 * Layout: the data owner stores every shard twice (2n slots: candidates, then a copy),
 * so a partner is reached by rotation without masks; 16384 orders per ciphertext.
 *
 * Both modes compute q, every c_ij and S_i identically; they differ ONLY in the rank
 * threshold, a threshold over (n-1)+k bits:
 *   arithmetic : exact counting tree over {1 - c_ij} and k copies of q_i, [#true >= n]
 *                (the same baseline as the genomics thresholds)
 *   HULK       : S_i (free), then one indicator [S <= k-1] (BSGS)
 * Output: the encrypted top_i flags (decrypted by the client). HE-eval only is reported.
 * Env: Q3_SHARD (n, default 64), Q3_SCORE_MAX (M, default 15), Q3_TOPK (k, default 10).
 */
#include "HE_workload_common.h"
#include <random>
using namespace hulk;

static size_t env_size(const char *k, size_t d){ const char *v=getenv(k); return v? strtoul(v,nullptr,10) : d; }
static Ciphertext not_bit(const HEEnv &env, const Ciphertext &x, OpCount &ops){ return one_minus_bit(env,x,ops); }

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

// Lagrange interpolation over Z_t: the polynomial (degree <= |xs|-1) with p(xs[i]) = ys[i].
static vector<uint64_t> interpolate(const vector<uint64_t> &xs, const vector<uint64_t> &ys, uint64_t t){
    const size_t n=xs.size(); vector<uint64_t> coef(n,0);
    for(size_t i=0;i<n;++i){
        if(ys[i]%t==0) continue;
        vector<uint64_t> num{1}; uint64_t den=1;                      // prod_{j!=i} (x - xs[j])
        for(size_t j=0;j<n;++j){ if(j==i) continue;
            vector<uint64_t> nx(num.size()+1,0);
            for(size_t a=0;a<num.size();++a){ nx[a+1]=(nx[a+1]+num[a])%t; nx[a]=(nx[a]+mod_mul(num[a], mod_sub(0,xs[j],t), t))%t; }
            num.swap(nx); den=mod_mul(den, mod_sub(xs[i],xs[j],t), t); }
        const uint64_t sc=mod_mul(ys[i]%t, mod_inv(den,t), t);
        for(size_t a=0;a<num.size();++a) coef[a]=(coef[a]+mod_mul(num[a],sc,t))%t;
    }
    return coef;
}

int main(int argc, char **argv){
    RunConfig cfg = parse_basic_args(argc, argv);
    const size_t n=env_size("Q3_SHARD",64), M=env_size("Q3_SCORE_MAX",15), k=env_size("Q3_TOPK",10);
    const size_t SEG=5, BUILDING=1;
    const BenchmarkEnvs envs = make_benchmark_envs(cfg);
    const size_t N=envs.slot_count, block=2*n, shards=N/block;
    if(N%block) throw invalid_argument("2*Q3_SHARD must divide the slot count");
    // ---- data (data owner): segment code and quantized revenue per order, shard-duplicated layout
    mt19937_64 rng(cfg.seed);
    vector<uint64_t> seg(N), score(N);
    for(size_t b=0;b<shards;++b) for(size_t i=0;i<n;++i){
        const uint64_t g=rng()%SEG, s=rng()%(M+1);
        seg[b*block+i]=seg[b*block+n+i]=g; score[b*block+i]=score[b*block+n+i]=s; }
    // ---- plaintext oracle (first-half slots of every shard)
    vector<uint64_t> expect(N,0); size_t flagged=0;
    for(size_t b=0;b<shards;++b) for(size_t i=0;i<n;++i){
        const size_t base=b*block; const bool qi = seg[base+i]==BUILDING; size_t S = qi? 0 : k;
        for(size_t j=1;j<n;++j){ const size_t p=(i+j)%n; S += (seg[base+p]==BUILDING && score[base+p]>score[base+i]); }
        expect[base+i] = S<=k-1; flagged += expect[base+i]; }
    cout<<"TPC-H Q3 top-"<<k<<" (rank-based, setup A): orders="<<shards*n<<" shards="<<shards<<" n="<<n
        <<" score domain 0.."<<M<<" segment D="<<SEG<<" flagged="<<flagged<<"\n";
    bool all_ok=true; double t_ar=0,t_hl=0;
    for(int pass=0;pass<2;++pass){
        const EvalMode m = pass? EvalMode::hulk : EvalMode::arithmetic;
        if(m==EvalMode::arithmetic && !wants_arithmetic(cfg.mode)) continue;
        if(m==EvalMode::hulk && !wants_hulk(cfg.mode)) continue;
        const HEEnv &env=envs.for_mode(m); const uint64_t t=env.plain_modulus;
        // plaintext setup outside the timer: GT interpolant on d in [-M, M]
        vector<uint64_t> xs, ys; for(long v=-long(M); v<=long(M); ++v){ xs.push_back(v<0? t-uint64_t(-v) : uint64_t(v)); ys.push_back(v>0); }
        const vector<uint64_t> gt = interpolate(xs, ys, t);
        const vector<uint64_t> eqc = bounded_eq_coefficients(env, SEG-1, BUILDING);
        Stopwatch enc_sw;
        Ciphertext Xs = encrypt_slots(env, score), Xg = encrypt_slots(env, seg);
        const double enc_ms=enc_sw.ms();
        { Ciphertext warm; env.eval->rotate_rows(Xs, 1, env.gk, warm); }      // untimed: key material warm-up (GPU lazy keys)
        // ---- server: HE evaluation only
        OpCount ops; const auto t0=steady_clock::now();
        auto q = eval_poly_bsgs(env, Xg, eqc); ops += q.second;              // filter
        const OpCount ops_filter = ops;
        vector<Ciphertext> c; c.reserve(n-1);
        Ciphertext rs = Xs, rq = q.first;
        for(size_t j=1;j<n;++j){                                             // sort: compare with every partner
            Ciphertext ns, nq;
            env.eval->rotate_rows(rs, 1, env.gk, ns); env.eval->rotate_rows(rq, 1, env.gk, nq); ops.rotations += 2;
            rs = move(ns); rq = move(nq);
            Ciphertext d; env.eval->sub(rs, Xs, d); ops.ct_add++;            // d = s_p - s_i in [-M, M]
            auto g = eval_poly_bsgs(env, d, gt); ops += g.second;
            c.push_back(multiply_ciphertexts(env, g.first, rq, ops));        // * q_p
        }
        const OpCount ops_compare = op_diff(ops, ops_filter), ops_after_compare = ops;
        Ciphertext top;
        if(m==EvalMode::arithmetic){                                         // counting tree, [#true >= n]
            vector<Ciphertext> v; v.reserve(n-1+k);
            for(auto &cj: c) v.push_back(not_bit(env,cj,ops));
            for(size_t r=0;r<k;++r) v.push_back(q.first);
            top = arith_threshold(env, v, n, ops);
        } else {                                                             // S = sum c + k(1-q), [S <= k-1]
            Ciphertext S = c[0]; for(size_t j=1;j<c.size();++j) add_inplace(env,S,c[j],ops);
            Ciphertext kq = multiply_plain_const(env, q.first, mod_sub(0,k,t), ops); add_plain_inplace(env,kq,k,ops);
            add_inplace(env,S,kq,ops);
            auto r = he_le_plain_bounded(env, S, n-1+k, k-1); ops += r.second; top = move(r.first);
        }
        const OpCount ops_rank = op_diff(ops, ops_after_compare);
        const double eval_ms=duration<double,milli>(steady_clock::now()-t0).count();
        Stopwatch dec_sw; auto dec = decrypt_slots(env, top); const double dec_ms=dec_sw.ms();
        bool ok=true; for(size_t b=0;b<shards;++b) for(size_t i=0;i<n;++i) ok = ok && dec[b*block+i]%t == expect[b*block+i];
        all_ok = all_ok && ok; (m==EvalMode::hulk? t_hl : t_ar) = eval_ms;
        const char *mn = m==EvalMode::hulk? "hulk":"arithmetic";
        cout<<"\n["<<mn<<"] limbs="<<env.coeff_modulus_primes<<"  HE eval: "<<eval_ms<<" ms  HMul="<<ops.total_ct_mul()
            <<"  rotations="<<ops.rotations<<"  ["<<(ok?"match":"MISMATCH")<<"]\n";
        cout<<"  noise budget left: "<<env.dec->invariant_noise_budget(top)<<" bits\n";
        print_stage_line(mn, "filter", ops_filter);
        print_stage_line(mn, "sort_compare", ops_compare);
        print_stage_line(mn, "sort_rank", ops_rank);
        E2EReport rep; rep.mode=m; rep.limbs=env.coeff_modulus_primes; rep.enc_ms=enc_ms; rep.eval_ms=eval_ms; rep.dec_ms=dec_ms;
        rep.upload_bytes=ciphertext_upload_bytes(vector<Ciphertext>{Xs,Xg}); rep.hmul=ops.total_ct_mul(); rep.rot=ops.rotations; rep.ok=ok;
        print_e2e_line(rep);
    }
    if(t_ar>0&&t_hl>0) cout<<"\nspeedup (arith/HULK, HE eval) = "<<(t_ar/t_hl)<<"x\n";
    return all_ok?0:1;
}
