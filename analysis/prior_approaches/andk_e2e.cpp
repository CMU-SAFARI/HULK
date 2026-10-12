/*
 * AND-K across B/FV approaches (one bulk AND-K per record, 32,768 records per block).
 *
 * Every record i holds K encrypted bits a_{i,1..K}; bit j of all records is one ciphertext
 * (bit-sliced, one record per slot). Output: the per-record AND, checked slot by slot against
 * a plaintext oracle. HE-eval latency only (the client's encryption/decryption are not timed).
 *
 *   A1   arithmetic product tree: balanced HMul tree, relinearize after every HMul (baseline)
 *   A2   A1 with lazy relinearization: SEAL 4.1 only has relin keys for size-3 ciphertexts, so
 *        every HMul input must be relinearized; the only relin that can be dropped is the one
 *        after the final product (decryption accepts a size-3 ciphertext)
 *   A3   sum + Fermat zero test: S = sum_j a_j (HAdd), AND = 1 - (S - K)^(p-1), p = 65537
 *        (p - 1 = 2^16: 16 squarings); batching needs p = 1 mod 2N, so 65537 works up to N = 32768.
 *        At N = 65536 use --p 786433 (= 3*2^18 + 1, the smallest batching prime there: 18 squarings
 *        + 1 HMul). --p selects the modulus; z^(p-1) is evaluated by square-and-multiply.
 *   A7s  HULK, server-side decoding: S = sum (HAdd), then one indicator [S = K] (harness
 *        hulk_indicator_from_sum: root-paired up to the PS threshold, BSGS above)
 *   TT/TF/TH  THRESHOLD [lo <= #true <= hi] over K bits (--lo/--hi): TT arithmetic counting tree
 *        (the genomics baseline), TF sum + one Fermat zero test per accepted value, TH HULK sum + one
 *        bounded comparison polynomial (BSGS)
 *   A7c  HULK, client-side decoding: the server only computes S = sum (HAdd); the client decrypts
 *        S and tests S == K (reference point; the server returns S, which reveals the count)
 *
 * Output line (one per run):
 *   ANDK approach=<a> K=<K> N=<N> t=<t> limbs=<L> total_bits=<B> eval_ms=<x> hmul=<n> relin=<r>
 *        depth=<d> ct_bytes=<b> noise=<bits> ok=<match|MISMATCH>
 */
#include "HE_workload_common.h"
#include <random>
using namespace hulk;

struct AndKCfg { string approach = "A1"; size_t K = 16; size_t limbs = 8; size_t N = 32768; uint64_t seed = 42; uint64_t p = 65537; size_t lo = 0, hi = 0; };

static AndKCfg parse(int argc, char **argv){
    AndKCfg c;
    for(int i=1;i<argc;++i){ string a=argv[i]; auto nx=[&]{ return string(argv[++i]); };
        if(a=="--approach") c.approach=nx(); else if(a=="--K") c.K=stoul(nx());
        else if(a=="--limbs") c.limbs=stoul(nx()); else if(a=="--N") c.N=stoul(nx());
        else if(a=="--seed") c.seed=stoull(nx());
        else if(a=="--p") c.p=stoull(nx());
        else if(a=="--lo") c.lo=stoul(nx()); else if(a=="--hi") c.hi=stoul(nx());
        else if(a=="--threads"||a=="--mode") ++i;                 // accepted for runner compatibility, ignored
        else throw invalid_argument("unknown arg: "+a); }
    return c;
}

// BFV env with an explicit N and plaintext modulus (same structure as make_env).
static HEEnv make_env_np(size_t N, uint64_t t, size_t limbs, EvalMode mode){
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(N);
    // 57-bit primes while they fit SEAL's 128-bit bound for this N, else 55-bit (as the harness does at N=32768)
    const size_t maxbits = N==32768? 881 : N==65536? 1761 : N==16384? 438 : 218;
    const int bits = limbs*57 <= maxbits ? 57 : 55; (void)mode;
    parms.set_coeff_modulus(CoeffModulus::Create(N, vector<int>(limbs, bits)));
    parms.set_plain_modulus(Modulus(t));
    // SEAL's built-in 128-bit table stops at N=32768. For N=65536 we use the same 128-bit bound
    // from the HomomorphicEncryption.org / lattice-estimator extension (log2 q <= 1761) and
    // check it here, because SEAL itself cannot (sec_level_type::none).
    if(N > 32768 && size_t(limbs)*size_t(bits) > maxbits) throw runtime_error("coefficient modulus exceeds the 128-bit bound for this N");
    auto ctx = make_shared<SEALContext>(parms, true, N > 32768 ? sec_level_type::none : sec_level_type::tc128);
    if(!ctx->parameters_set()) throw runtime_error(string("params rejected: ")+ctx->parameter_error_message());
    if(!ctx->first_context_data()->qualifiers().using_batching) throw runtime_error("plain modulus does not batch at this N");
    KeyGenerator kg(*ctx); HEEnv env; env.ctx=ctx; env.sk=kg.secret_key();
    kg.create_public_key(env.pk); kg.create_relin_keys(env.rk);
    env.enc=make_unique<Encryptor>(*ctx,env.pk); env.eval=make_unique<Evaluator>(*ctx);
    env.dec=make_unique<Decryptor>(*ctx,env.sk); env.batch=make_unique<BatchEncoder>(*ctx);
    env.mode=mode; env.cache_enabled=true; env.cache=make_shared<HEEnv::CacheState>();
    env.plain_modulus=t; env.slot_count=env.batch->slot_count(); env.max_threads=1;
    env.coeff_modulus_primes=parms.coeff_modulus().size();
    for(const Modulus &q: parms.coeff_modulus()) env.coeff_modulus_total_bits += q.bit_count();
    warm_env_cache(env);
    return env;
}

// A2: balanced tree, relinearize every product except the final one
static pair<Ciphertext,OpCount> lazy_product_tree(const vector<Ciphertext> &bits, const HEEnv &env){
    OpCount ops; vector<Ciphertext> layer=bits; int depth=0;
    while(layer.size()>1){
        const bool last = layer.size()==2;
        vector<Ciphertext> next;
        for(size_t i=0;i+1<layer.size();i+=2){
            Ciphertext out; env.eval->multiply(layer[i], layer[i+1], out); ops.ct_ct_mul++;
            if(!last){ env.eval->relinearize_inplace(out, env.rk); ops.relin++; }
            next.push_back(move(out));
        }
        if(layer.size()%2) next.push_back(layer.back());
        layer.swap(next); depth++;
    }
    ops.depth=depth; return {move(layer[0]), ops};
}

static Ciphertext not_bit(const HEEnv &env, const Ciphertext &x, OpCount &ops){ return one_minus_bit(env,x,ops); }
// ---- THRESHOLD approaches (TT counting tree, TF Fermat, TH HULK) -- same baseline code as the genomics driver
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


// Fermat zero test 1 - z^(p-1), depth-optimal (shared by A3 and TF)
static Ciphertext fermat_is_zero(const HEEnv &env, const Ciphertext &z, uint64_t t, OpCount &ops, int &depth){
    const uint64_t e = t-1; int top=63; while(!((e>>top)&1)) --top;
    vector<pair<int,Ciphertext>> fac; Ciphertext pw = z;
    for(int b=0;b<=top;++b){ if(b>0) pw = square_ciphertext(env, pw, ops); if((e>>b)&1) fac.push_back({b, pw}); }
    while(fac.size()>1){
        sort(fac.begin(), fac.end(), [](const auto &x, const auto &y){ return x.first < y.first; });
        auto a = move(fac[0]), bq = move(fac[1]); fac.erase(fac.begin(), fac.begin()+2);
        fac.push_back({max(a.first,bq.first)+1, multiply_ciphertexts(env, a.second, bq.second, ops)});
    }
    depth = fac[0].first; return one_minus_bit(env, fac[0].second, ops);
}

// [lo <= #true <= hi] over K bits; one of the two ends is open (lo == 0 or hi == K)
static int run_threshold(const AndKCfg &c, HEEnv &env, uint64_t t){
    const size_t S = env.slot_count, K = c.K, lo = c.lo, hi = c.hi ? c.hi : c.K;
    if(!(lo==0 || hi==K) || lo>hi || hi>K) throw invalid_argument("threshold needs --lo 0 or --hi K");
    mt19937_64 rng(c.seed);
    vector<vector<uint64_t>> bits(K, vector<uint64_t>(S,0)); vector<uint64_t> expect(S,0);
    for(size_t i=0;i<S;++i){                                    // counts concentrated around the threshold edge
        const long edge = lo>0? long(lo) : long(hi); long cnt = edge + long(rng()%9) - 4;
        cnt = max(0L, min(long(K), cnt)); vector<size_t> idx(K); iota(idx.begin(),idx.end(),0); shuffle(idx.begin(),idx.end(),rng);
        for(long r=0;r<cnt;++r) bits[idx[r]][i]=1;
        expect[i] = size_t(cnt)>=lo && size_t(cnt)<=hi; }
    vector<Ciphertext> X; for(auto &b: bits) X.push_back(encrypt_slots(env,b));
    OpCount ops; Ciphertext out; int depth=0; const auto t0=steady_clock::now();
    if(c.approach=="TT"){
        if(lo>0) out = arith_threshold(env, X, lo, ops);
        else { vector<Ciphertext> nx; for(auto &x: X) nx.push_back(not_bit(env,x,ops)); out = arith_threshold(env, nx, K-hi, ops); }
        int d=0; while((size_t(1)<<d)<K) ++d; depth=d;
    } else if(c.approach=="TF"){
        auto s=sum_bits(X,env); ops+=s.second; bool init=false;
        for(size_t v=lo; v<=hi; ++v){
            Ciphertext z=s.first; add_plain_inplace(env, z, mod_sub(0, v, t), ops);
            int d=0; Ciphertext ind = fermat_is_zero(env, z, t, ops, d); depth=max(depth,d);
            if(!init){ out=move(ind); init=true; } else add_inplace(env,out,ind,ops);
        }
    } else if(c.approach=="TH"){
        auto s=sum_bits(X,env); ops+=s.second;
        auto r = lo>0? he_ge_plain_bounded(env, s.first, K, lo) : he_le_plain_bounded(env, s.first, K, hi);
        ops+=r.second; out=move(r.first); depth=ops.depth;
    } else throw invalid_argument("threshold approach must be TT, TF or TH");
    const double eval_ms=duration<double,milli>(steady_clock::now()-t0).count();
    auto d=decrypt_slots(env,out); bool ok=true; for(size_t i=0;i<S;++i) ok = ok && d[i]%t==expect[i];
    cout<<"ANDK approach="<<c.approach<<" K="<<K<<" lo="<<lo<<" hi="<<hi<<" N="<<c.N<<" t="<<t<<" limbs="<<env.coeff_modulus_primes
        <<" total_bits="<<env.coeff_modulus_total_bits<<fixed<<setprecision(3)<<" eval_ms="<<eval_ms
        <<" hmul="<<ops.total_ct_mul()<<" relin="<<ops.relin<<" depth="<<depth
        <<" ct_bytes="<<ciphertext_upload_bytes(X[0])<<" noise="<<env.dec->invariant_noise_budget(out)
        <<" ok="<<(ok?"match":"MISMATCH")<<"\n";
    return ok?0:1;
}

int main(int argc, char **argv){
    const AndKCfg c = parse(argc, argv);
    const bool fermat = c.approach=="A3";
    // --p 0 (or any non-Fermat approach): the applications' 25-bit batching prime for this N
    const uint64_t t = c.p ? c.p : PlainModulus::Batching(c.N, kPlainModulusBits).value();
    if(c.K >= t) throw invalid_argument("K must be below the plaintext modulus");
    const EvalMode mode = (c.approach=="A7s"||c.approach=="A7c"||c.approach=="TH")? EvalMode::hulk : EvalMode::arithmetic;
    HEEnv env = make_env_np(c.N, t, c.limbs, mode);
    if(c.approach[0]=='T') return run_threshold(c, env, t);
    const size_t S = env.slot_count;

    // data: 1/8 of the records all-ones, the rest with one to three zero bits (near misses)
    mt19937_64 rng(c.seed);
    vector<vector<uint64_t>> bits(c.K, vector<uint64_t>(S,1));
    vector<uint64_t> expect(S,1);
    for(size_t i=0;i<S;++i){ if(i%8==0) continue;
        const size_t z = 1 + rng()%3; for(size_t r=0;r<z;++r) bits[rng()%c.K][i]=0;
        bool all=true; for(size_t j=0;j<c.K;++j) all = all && bits[j][i]; expect[i]=all; }
    vector<Ciphertext> X; X.reserve(c.K); for(auto &b: bits) X.push_back(encrypt_slots(env,b));

    OpCount ops; Ciphertext out; const auto t0=steady_clock::now();
    if(c.approach=="A1"){ auto r=arithmetic_all_true(X,env); ops=r.second; out=move(r.first); }
    else if(c.approach=="A2"){ auto r=lazy_product_tree(X,env); ops=r.second; out=move(r.first); }
    else if(c.approach=="A3"){
        auto s=sum_bits(X,env); ops+=s.second; Ciphertext z=move(s.first);
        add_plain_inplace(env, z, mod_sub(0, c.K, t), ops);                 // z = S - K
        // z^(p-1), depth-optimal: z^(2^i) by repeated squaring (depth i), then the factors of the
        // set bits of p-1 are multiplied pairing the two shallowest first (optimal depth for
        // this factor set; p = 65537: 16 squarings, depth 16; p = 2^18*127 + 1: 24 sq + 6 HMul, depth 25)
        const uint64_t e = t-1; int top=63; while(!((e>>top)&1)) --top;
        vector<pair<int,Ciphertext>> fac;                                // (depth, ciphertext)
        Ciphertext pw = z;
        for(int b=0;b<=top;++b){
            if(b>0) pw = square_ciphertext(env, pw, ops);
            if((e>>b)&1) fac.push_back({b, pw});
        }
        while(fac.size()>1){
            sort(fac.begin(), fac.end(), [](const auto &x, const auto &y){ return x.first < y.first; });
            auto a = move(fac[0]), bq = move(fac[1]); fac.erase(fac.begin(), fac.begin()+2);
            fac.push_back({max(a.first,bq.first)+1, multiply_ciphertexts(env, a.second, bq.second, ops)});
        }
        Ciphertext r = move(fac[0].second); const int dep = fac[0].first;
        out = one_minus_bit(env, r, ops); ops.depth=dep;
    }
    else if(c.approach=="A7s"){
        auto s=sum_bits(X,env); ops+=s.second;
        HulkDeferScaleScope defer(true);                                     // final indicator: 1/K! to the client
        auto r=hulk_indicator_from_sum(s.first, c.K, c.K, env); ops+=r.second; out=move(r.first);
        const uint64_t sc = defer.skipped();
        const double eval_ms=duration<double,milli>(steady_clock::now()-t0).count();
        auto d=decrypt_slots(env,out); bool ok=true;
        for(size_t i=0;i<S;++i) ok = ok && mod_mul(d[i]%t, sc, t)==expect[i];
        cout<<"ANDK approach="<<c.approach<<" K="<<c.K<<" N="<<c.N<<" t="<<t<<" limbs="<<env.coeff_modulus_primes
            <<" total_bits="<<env.coeff_modulus_total_bits<<fixed<<setprecision(3)<<" eval_ms="<<eval_ms
            <<" hmul="<<ops.total_ct_mul()<<" relin="<<ops.relin<<" depth="<<ops.depth
            <<" ct_bytes="<<ciphertext_upload_bytes(X[0])<<" noise="<<env.dec->invariant_noise_budget(out)
            <<" ok="<<(ok?"match":"MISMATCH")<<"\n";
        return ok?0:1;
    }
    else if(c.approach=="A7c"){ auto s=sum_bits(X,env); ops=s.second; out=move(s.first); }
    else throw invalid_argument("approach must be A1, A2, A3, A7s or A7c");
    const double eval_ms=duration<double,milli>(steady_clock::now()-t0).count();

    auto d=decrypt_slots(env,out); bool ok=true;
    for(size_t i=0;i<S;++i){
        const uint64_t v = d[i]%t;
        const uint64_t got = (c.approach=="A7c")? uint64_t(v==c.K) : v;     // A7c: client tests S == K
        ok = ok && got==expect[i];
    }
    if(c.approach=="A1"||c.approach=="A2"){ int dd=0; while((size_t(1)<<dd)<c.K) ++dd; ops.depth=dd; }
    cout<<"ANDK approach="<<c.approach<<" K="<<c.K<<" N="<<c.N<<" t="<<t<<" limbs="<<env.coeff_modulus_primes
        <<" total_bits="<<env.coeff_modulus_total_bits<<fixed<<setprecision(3)<<" eval_ms="<<eval_ms
        <<" hmul="<<ops.total_ct_mul()<<" relin="<<ops.relin<<" depth="<<ops.depth
        <<" ct_bytes="<<ciphertext_upload_bytes(X[0])<<" noise="<<env.dec->invariant_noise_budget(out)
        <<" ok="<<(ok?"match":"MISMATCH")<<"\n";
    return ok?0:1;
}
