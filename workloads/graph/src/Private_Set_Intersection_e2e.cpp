/*
 * End-to-end PSI: indicator generation + reduction, both under encryption.
 *
 * Model (server set is the server's own plaintext data, as in labeled/circuit PSI):
 *   client  -> bit-decomposes its items, encrypts ell bit-columns (one per bit
 *              position), each packing `rows` client items across the slots.
 *   server  -> holds K plaintext items.
 *
 * STAGE 1 (indicator generation) -- the stage the previous driver omitted.
 *   For plaintext server item c_j, the count of matching bit positions is
 *      S_j = sum_t [ x_t == c_j,t ]
 *          = sum_{t: c=1} x_t  -  sum_{t: c=0} x_t  +  |{t : c_j,t = 0}|
 *   which is a *pure affine map* on the encrypted bits: HAdd/HSub plus one
 *   plaintext constant. Zero homomorphic multiplications.
 *   Then eq_j = [ S_j == ell ].
 *     arithmetic: product over the ell matched-bit ciphertexts (ell-1 HMul)
 *     HULK      : indicator polynomial on S_j (sum-then-indicate)
 *
 * STAGE 2 (reduction)
 *   match = OR_j eq_j   (arithmetic: 1 - prod(1 - eq_j); HULK: nonzero indicator
 *                        of sum_j eq_j).
 * STAGE 3 (aggregation, identical in both modes)
 *   cardinality = sum_slots(match): log2(N/2) row rotations + 1 column rotation,
 *   each + HAdd. The server returns ONE ciphertext whose every slot holds |X n Y|.
 *
 * Timing per mode: enc_ms = client encryption of its columns (same data, same
 * ciphertext count in both modes); eval_ms = stages 1-3, homomorphic ops only;
 * dec_ms = decryption + decode of the returned ciphertext. Verification, noise
 * budget and printing are outside all three. One parseable line per mode:
 *   E2E mode=<m> limbs=<L> enc_ms=.. eval_ms=.. dec_ms=.. upload_bytes=.. hmul=.. rot=.. ok=..
 * (upload_bytes = exact serialized size of the uploaded ciphertexts, no compression).
 * Oracle: the returned cardinality must equal the plaintext intersection size and
 * every slot of the returned ciphertext must hold it. --verify-records additionally
 * checks the stage-2 match vector slot by slot (debug: the ciphertext copy it needs
 * lands inside eval_ms, so do not use it for timing).
 * Padding slots (rows < slot_count) encode item 0; a server set containing 0 would
 * make them count, so that configuration is rejected.
 *
 * --encoding digit : field-encoded baseline (Iliashenko-Zucca, PETS'21 style).
 *   Each item is split into D = ceil(ell/w) digits of w bits; ciphertext d holds
 *   digit d of every client item. Equality with server item y is
 *      eq = prod_d ( 1 - (x_d - y_d)^(p-1) )          (Fermat, per digit)
 *   followed by the same OR stage (arithmetic, or HULK with --mode hulk).
 *   With full batching the plaintext modulus must satisfy p = 1 mod 2N; the
 *   cheapest Fermat prime at N=32768 is p=65537 (p-1 = 2^16: 16 squarings).
 */
#include "HE_workload_common.h"
#include <random>
#include <unordered_set>
using namespace hulk;

struct E2ECfg : RunConfig {
    size_t server_size = 16; size_t item_bits = 40;
    string encoding = "bit";          // bit | digit
    size_t digit_bits = 16;           // digit encoding only
    uint64_t field_plain = 65537;     // digit encoding only
    size_t field_poly = 32768;        // digit encoding only
    int field_prime_bits = 55;        // digit encoding only
    bool verify_records = false;      // also check the stage-2 match vector (debug)
};

static E2ECfg parse_e2e(int argc, char **argv){
    E2ECfg c;
    for (int i=1;i<argc;++i){ string a=argv[i];
        auto nx=[&]{ return stoull(argv[++i]); };
        if(a=="--rows") c.rows=(size_t)nx();
        else if(a=="--seed") c.seed=nx();
        else if(a=="--threads") c.threads=max<size_t>(1,nx());
        else if(a=="--server-size") c.server_size=(size_t)nx();
        else if(a=="--item-bits") c.item_bits=(size_t)nx();
        else if(a=="--mode") c.mode=parse_mode_value(argv[++i]);
        else if(a=="--limbs"){ size_t L=max<size_t>(1,nx()); c.arithmetic_limbs=L; c.hulk_limbs=L; }
        else if(a=="--arithmetic-limbs") c.arithmetic_limbs=max<size_t>(1,nx());
        else if(a=="--hulk-limbs") c.hulk_limbs=max<size_t>(1,nx());
        else if(a=="--encoding") c.encoding=argv[++i];
        else if(a=="--digit-bits") c.digit_bits=(size_t)nx();
        else if(a=="--plain-modulus") c.field_plain=nx();
        else if(a=="--poly-degree") c.field_poly=(size_t)nx();
        else if(a=="--prime-bits") c.field_prime_bits=(int)nx();
        else if(a=="--verify-records") c.verify_records=true;
        else throw invalid_argument("unknown arg: "+a);
    }
    if(c.item_bits<1||c.item_bits>62) throw invalid_argument("item-bits must be in [1,62]");
    return c;
}


static void make_sets(const E2ECfg &cfg, vector<uint64_t> &server, vector<uint64_t> &client, uint64_t &reference, vector<uint64_t> &expect){
  const size_t L = cfg.item_bits, K = cfg.server_size, N = cfg.rows;
  mt19937_64 rng(cfg.seed);
  const uint64_t mask = (L>=64)?~0ull:((1ull<<L)-1);
  uniform_int_distribution<uint64_t> dist(0, mask);
  { unordered_set<uint64_t> st; while(st.size()<K) st.insert(dist(rng)); server.assign(st.begin(),st.end()); }
  client.assign(N,0); reference=0; expect.assign(N,0);
  for(size_t i=0;i<N;++i){
    uint64_t v = (i%5==0) ? server[i%K] : dist(rng);
    client[i]=v;
    bool hit=false; for(uint64_t x:server) if(x==v){hit=true;break;}
    reference += hit?1:0; expect[i] = hit?1:0;
  }
}

static HEEnv make_field_env(size_t N, uint64_t p, size_t limbs, int bits, bool hulk_mode){
  EncryptionParameters parms(scheme_type::bfv);
  parms.set_poly_modulus_degree(N);
  parms.set_coeff_modulus(CoeffModulus::Create(N, vector<int>(limbs, bits)));
  parms.set_plain_modulus(Modulus(p));
  auto ctx = make_shared<SEALContext>(parms, true, sec_level_type::tc128);
  if(!ctx->parameters_set()) throw runtime_error(string("field params rejected: ")+ctx->parameter_error_message());
  if(!ctx->first_context_data()->qualifiers().using_batching) throw runtime_error("plain modulus does not support batching (need p = 1 mod 2N)");
  KeyGenerator kg(*ctx); HEEnv env; env.ctx=ctx; env.sk=kg.secret_key();
  kg.create_public_key(env.pk); kg.create_relin_keys(env.rk);
  { // stage-3 sum_slots needs only the positive power-of-two row steps and the column
    // swap (step 0): half of SEAL's default key set (Galois keys dominate memory at 16 limbs)
    vector<int> steps; for(size_t st=1; st<N/2; st<<=1) steps.push_back(int(st)); steps.push_back(0);
    kg.create_galois_keys(steps, env.gk); }
  env.enc=make_unique<Encryptor>(*ctx,env.pk); env.eval=make_unique<Evaluator>(*ctx);
  env.dec=make_unique<Decryptor>(*ctx,env.sk); env.batch=make_unique<BatchEncoder>(*ctx);
  env.mode = hulk_mode ? EvalMode::hulk : EvalMode::arithmetic;
  // constant/plan caching for BOTH modes (audit H1/G5): same harness optimisations either way
  env.cache_enabled = true; env.cache=make_shared<HEEnv::CacheState>();
  env.plain_modulus=p; env.slot_count=env.batch->slot_count(); env.max_threads=1;
  env.coeff_modulus_primes=parms.coeff_modulus().size();
  for(const Modulus &q: parms.coeff_modulus()) env.coeff_modulus_total_bits += q.bit_count();
  warm_env_cache(env);   // identical pre-timer warm-up for both modes
  return env;
}

// slot-by-slot plaintext oracle (audit G10): slot i < rows must equal expect[i] exactly
// (0/1), every padding slot must be 0; a sum or nonzero count alone could hide errors.
static bool slots_match(const vector<uint64_t> &dec, const vector<uint64_t> &expect, uint64_t p){
  for(size_t i=0;i<dec.size();++i){ const uint64_t want = i<expect.size()? expect[i] : 0; if(dec[i]%p != want) return false; }
  return true;
}

// z^e by left-to-right square-and-multiply; returns multiplicative depth via *depth
static Ciphertext pow_ct(const HEEnv &env, const Ciphertext &z, uint64_t e, OpCount &ops, int *depth){
  int top = 63 - __builtin_clzll(e); Ciphertext r = z; int d = 0;
  for(int b=top-1;b>=0;--b){ r = square_ciphertext(env,r,ops); ++d;
    if((e>>b)&1ull){ r = multiply_ciphertexts(env,r,z,ops); ++d; } }
  *depth = d; return r;
}

// Stage 3 + client decryption + oracle check + report, identical for every path
// and mode. `match` is the stage-2 per-record match vector; t0 is the start of
// the HE evaluation. Returns ok; *eval_ms receives the HE-evaluation time.
static bool finish_psi(const HEEnv &env, EvalMode mode, const string &label, Ciphertext match, OpCount &ops,
                       steady_clock::time_point t0, double stage1_ms, double stage2_ms, double enc_ms,
                       const vector<Ciphertext> &uploaded, const vector<uint64_t> &expect, uint64_t reference,
                       bool verify_records, double *eval_ms_out, const OpCount &ops_stage1, const OpCount &ops_stage2,
                       const string &extra = "", uint64_t scale = 1){
  const uint64_t p = env.plain_modulus;
  if(reference >= p) throw runtime_error(label+": intersection size "+to_string(reference)+" >= t (aggregate >= t)");
  Ciphertext record; if(verify_records) record = match;           // debug copy (inside the timer)
  // Stage 3. Default: client-side slot reduction -- the server returns the match
  // ciphertext itself (no rotations) and the client sums its slots after decryption.
  // E2E_SLOT_AGGREGATE=1 restores the homomorphic rotate-and-sum (sum_slots).
  const bool slot_agg = env_flag_enabled("E2E_SLOT_AGGREGATE");
  Stopwatch agg_sw;
  pair<Ciphertext, OpCount> card{move(match), OpCount{}};
  if(slot_agg){ card = sum_slots(move(card.first), env); ops += card.second; }
  const double agg_ms = agg_sw.ms();
  const double eval_ms = duration<double,milli>(steady_clock::now()-t0).count();

  Stopwatch dec_sw;
  const vector<uint64_t> dec = decrypt_slots(env, card.first);
  uint64_t got = 0;
  if(slot_agg) got = dec.empty()? 0 : dec[0] % p;
  else for(uint64_t v: dec) got = (got + v % p) % p;
  got = mod_mul(got, scale % p, p);                 // deferred 1/K! of the final OR indicator (client side)
  const double dec_ms = dec_sw.ms();

  bool consistent = true; if(slot_agg) for(uint64_t v: dec) consistent = consistent && v==dec[0];
  bool good = consistent && got==reference;
  bool rec_ok = true;
  if(verify_records){ rec_ok = slots_match(decrypt_slots(env, record), expect, p); good = good && rec_ok; }
  const size_t upload_bytes = ciphertext_upload_bytes(uploaded);
  cout<<"\n["<<label<<"] limbs="<<env.coeff_modulus_primes<<"\n"
      <<"  encrypt (client)     : "<<enc_ms<<" ms   upload "<<uploaded.size()<<" ciphertexts, "<<upload_bytes<<" B ("
      <<double(upload_bytes)/double(max<size_t>(1,expect.size()))<<" B/item)\n"
      <<"  stage1 indicator-gen : "<<stage1_ms<<" ms"<<extra<<"\n"
      <<"  stage2 OR reduction  : "<<stage2_ms<<" ms\n"
      <<"  stage3 cardinality   : "<<agg_ms<<" ms   ("<<card.second.rotations<<" rotations)\n"
      <<"  HE eval (server)     : "<<eval_ms<<" ms   HMul="<<ops.total_ct_mul()<<"\n"
      <<"  decrypt (client)     : "<<dec_ms<<" ms   cardinality="<<got<<" (oracle "<<reference<<")"
      <<(consistent? "" : " [slots disagree]")<<" ["<<matches_string(good)<<"]\n";
  if(verify_records) cout<<"  per-record check     : "<<(rec_ok?"match":"MISMATCH")<<"\n";
  print_ops(label, ops);
  {   // per-stage op counts (instrumentation only; they sum exactly to `ops`)
    const string mn = mode==EvalMode::hulk ? "hulk" : "arithmetic";
    print_stage_line(mn, "indicator", ops_stage1);
    print_stage_line(mn, "or", ops_stage2);
    print_stage_line(mn, "aggregate", card.second);
  }
  cout<<"  noise budget left: "<<env.dec->invariant_noise_budget(card.first)<<" bits\n";
  E2EReport rep; rep.mode=mode; rep.limbs=env.coeff_modulus_primes; rep.enc_ms=enc_ms; rep.eval_ms=eval_ms; rep.dec_ms=dec_ms;
  rep.upload_bytes=upload_bytes; rep.hmul=ops.total_ct_mul(); rep.rot=ops.rotations; rep.ok=good;
  print_e2e_line(rep);
  *eval_ms_out = eval_ms;
  return good;
}

// Padding slots hold item 0; reject a server set that contains it.
static void check_padding(const vector<uint64_t> &server, size_t rows, size_t slots){
  if(rows < slots) for(uint64_t y: server) if(y==0)
    throw invalid_argument("server set contains item 0 and rows < slot_count: padding slots would be counted");
}

static int run_field(const E2ECfg &cfg){
  const size_t L = cfg.item_bits, K = cfg.server_size, N = cfg.rows, w = cfg.digit_bits;
  const size_t D = (L + w - 1) / w; const uint64_t p = cfg.field_plain;
  if(w >= 63 || (1ull<<w) > p) throw invalid_argument("digit-bits must satisfy 2^w <= plain modulus");
  vector<uint64_t> server, client, expect; uint64_t reference=0; make_sets(cfg, server, client, reference, expect);
  cout<<"End-to-end PSI, field-encoded baseline (digit-wise Fermat)\n"
      <<"client rows: "<<N<<", server size: "<<K<<", item bits: "<<L<<", digit bits: "<<w
      <<", digits: "<<D<<", plain modulus: "<<p<<", N: "<<cfg.field_poly<<"\n"
      <<"reference intersection size: "<<reference<<"\n"<<fixed<<setprecision(3);
  bool ok=true;
  for(int pass=0;pass<2;++pass){
    const bool hulk_or = pass==1;
    if(!hulk_or && !wants_arithmetic(cfg.mode)) continue;
    if( hulk_or && !wants_hulk(cfg.mode)) continue;
    const size_t limbs = (hulk_or? cfg.hulk_limbs : cfg.arithmetic_limbs).value_or(16);
    HEEnv env = make_field_env(cfg.field_poly, p, limbs, cfg.field_prime_bits, hulk_or);
    if(N > env.slot_count) throw invalid_argument("rows exceed slot count");
    check_padding(server, N, env.slot_count);
    const uint64_t dmask = (1ull<<w)-1;
    Stopwatch enc_sw;                                    // client encryption (enc_ms)
    vector<Ciphertext> X; X.reserve(D);
    for(size_t d=0; d<D; ++d){ vector<uint64_t> col(N);
      for(size_t i=0;i<N;++i) col[i]=(client[i]>>(d*w))&dmask;
      X.push_back(encrypt_slots(env,col)); }
    const double enc_ms = enc_sw.ms();

    OpCount ops; int fermat_depth=0; const auto t0=steady_clock::now();
    vector<Ciphertext> eq; eq.reserve(K);
    for(size_t j=0;j<K;++j){
      vector<Ciphertext> e; e.reserve(D);
      for(size_t d=0; d<D; ++d){
        Ciphertext z = X[d];
        add_plain_inplace(env, z, mod_sub(0, (server[j]>>(d*w))&dmask, p), ops);   // z = x_d - y_d
        Ciphertext zp = pow_ct(env, z, p-1, ops, &fermat_depth);                    // z^(p-1)
        e.push_back(one_minus_bit(env, zp, ops));   // [z == 0] = 1 - z^(p-1): negate + cached add_plain, no encryption (G4)
      }
      auto prod = arithmetic_all_true(e, env); ops += prod.second; eq.push_back(move(prod.first));
    }
    const double stage1_ms = duration<double,milli>(steady_clock::now()-t0).count();
    const OpCount ops_stage1 = ops;                          // stage snapshot (instrumentation only)
    const auto s2=steady_clock::now();
    uint64_t or_scale = 1;
    pair<Ciphertext, OpCount> match;
    { HulkDeferScaleScope defer(hulk_or);                 // final OR: 1/K! deferred to the client
      match = hulk_or ? hulk_any_true(eq,env) : arithmetic_any_true(eq,env); or_scale = defer.skipped(); }
    ops += match.second;
    const double stage2_ms = duration<double,milli>(steady_clock::now()-s2).count();
    double eval_ms = 0;
    const bool good = finish_psi(env, hulk_or? EvalMode::hulk : EvalMode::arithmetic, hulk_or?"field-hulkOR":"field-arith",
                                 move(match.first), ops, t0, stage1_ms, stage2_ms, enc_ms, X, expect, reference,
                                 cfg.verify_records, &eval_ms, ops_stage1, match.second,
                                 "   (Fermat depth per digit: "+to_string(fermat_depth)+")", or_scale);
    ok = ok && good;
  }
  return ok?0:1;
}

int main(int argc,char**argv){
  const E2ECfg cfg = parse_e2e(argc,argv);
  if(cfg.encoding=="digit") return run_field(cfg);
  if(cfg.encoding!="bit") throw invalid_argument("encoding must be bit or digit");
  const BenchmarkEnvs envs = make_benchmark_envs(cfg);
  const size_t L = cfg.item_bits, K = cfg.server_size, N = cfg.rows;

  mt19937_64 rng(cfg.seed);
  const uint64_t mask = (L>=64)?~0ull:((1ull<<L)-1);
  uniform_int_distribution<uint64_t> dist(0, mask);

  // ---- server set: K distinct plaintext items ----
  vector<uint64_t> server; { unordered_set<uint64_t> s;
    while(s.size()<K) s.insert(dist(rng)); server.assign(s.begin(),s.end()); }

  // ---- client items: every 5th one is planted as a match ----
  vector<uint64_t> client(N), expect(N,0); uint64_t reference=0;
  for(size_t i=0;i<N;++i){
    uint64_t v = (i%5==0) ? server[i%K] : dist(rng);
    client[i]=v;
    bool hit=false; for(uint64_t s:server) if(s==v){hit=true;break;}
    reference += hit?1:0; expect[i] = hit?1:0;
  }

  cout<<"End-to-end PSI (indicator generation + reduction)\n"
      <<"client rows: "<<N<<", server size: "<<K<<", item bits: "<<L
      <<", slot_count: "<<envs.slot_count<<", mode: "<<mode_name(cfg.mode)<<"\n"
      <<"reference intersection size: "<<reference<<"\n"<<fixed<<setprecision(3);

  bool ok=true; double t_ar=0,t_hl=0;
  for(int pass=0;pass<2;++pass){
    EvalMode m = pass? EvalMode::hulk : EvalMode::arithmetic;
    if(m==EvalMode::arithmetic && !wants_arithmetic(cfg.mode)) continue;
    if(m==EvalMode::hulk && !wants_hulk(cfg.mode)) continue;
    const HEEnv &env = envs.for_mode(m);
    const uint64_t p = env.plain_modulus;
    if(N > env.slot_count) throw invalid_argument("rows exceed slot count");
    check_padding(server, N, env.slot_count);

    // client uploads: one ciphertext per bit position, rows packed across slots (enc_ms)
    Stopwatch enc_sw;
    vector<Ciphertext> X; X.reserve(L);
    for(size_t t=0;t<L;++t){ vector<uint64_t> col(N);
      for(size_t i=0;i<N;++i) col[i]=(client[i]>>t)&1ull;
      X.push_back(encrypt_slots(env,col)); }
    const double enc_ms = enc_sw.ms();

    OpCount ops; const auto t0=steady_clock::now();
    double stage1_ms=0;

    vector<Ciphertext> eq; eq.reserve(K);
    const auto s1=steady_clock::now();
    // arithmetic complements 1 - x_t, computed once per bit-plane that some server
    // item has as 0 (inside the timed region; negate + cached add_plain, audit G2)
    vector<Ciphertext> NX;
    if(m==EvalMode::arithmetic){
      vector<bool> need_not(L,false);
      for(size_t j=0;j<K;++j) for(size_t t=0;t<L;++t) if(!((server[j]>>t)&1ull)) need_not[t]=true;
      NX.resize(L); for(size_t t=0;t<L;++t) if(need_not[t]) NX[t]=one_minus_bit(env,X[t],ops);
    }
    for(size_t j=0;j<K;++j){
      if(m==EvalMode::hulk){
        // S_j = sum_t [x_t == c_jt]  -- affine, no multiplications
        Ciphertext S; bool init=false; uint64_t konst=0;
        for(size_t t=0;t<L;++t){
          if((server[j]>>t)&1ull){ if(!init){S=X[t];init=true;} else add_inplace(env,S,X[t],ops); }
          else { ++konst;
                 if(!init){ S=multiply_plain_const(env,X[t],mod_sub(0,1,p),ops); init=true; }
                 else { env.eval->sub_inplace(S,X[t]); ops.ct_add++; } }
        }
        add_plain_inplace(env,S,konst,ops);
        // [S == L] via hulk_indicator_from_sum (audit G11). For L >= 17 (default
        // HULK_INDICATOR_PS_MIN) it takes the shallower of the root-paired product
        // tree and depth-optimal eval_poly_bsgs, ties -> fewer HMul. At L=40 both
        // have depth 6, so BSGS is used: depth 6, 13 HMul (root-paired: 6, 20);
        // the arithmetic 40-bit AND tree is depth 6, 39 HMul.
        // Ablation is through the harness switches, not this driver:
        //   HULK_POLY_EVAL=bsgs       always BSGS
        //   HULK_POLY_EVAL=streaming  pre-audit Horner-over-giant-steps PS (depth 9 at L=40)
        //   HULK_INDICATOR_PS_MIN=41  root-paired tree for L=40
        auto ind = hulk_indicator_from_sum(S,L,L,env);
        ops += ind.second; eq.push_back(move(ind.first));
      } else {
        // arithmetic: explicit matched-bit ciphertexts, then an AND-tree
        vector<Ciphertext> terms; terms.reserve(L);
        for(size_t t=0;t<L;++t)
          terms.push_back(((server[j]>>t)&1ull) ? X[t] : NX[t]);
        auto prod = arithmetic_all_true(terms,env);
        ops += prod.second; eq.push_back(move(prod.first));
      }
    }
    stage1_ms = duration<double,milli>(steady_clock::now()-s1).count();
    const OpCount ops_stage1 = ops;                          // stage snapshot (instrumentation only)

    const auto s2=steady_clock::now();
    uint64_t or_scale = 1;
    pair<Ciphertext, OpCount> match;
    { HulkDeferScaleScope defer(m==EvalMode::hulk);       // final OR: 1/K! deferred to the client
      match = (m==EvalMode::hulk)? hulk_any_true(eq,env) : arithmetic_any_true(eq,env); or_scale = defer.skipped(); }
    ops += match.second;
    const double stage2_ms = duration<double,milli>(steady_clock::now()-s2).count();

    double eval_ms = 0;
    const bool good = finish_psi(env, m, m==EvalMode::hulk?"hulk":"arithmetic", move(match.first), ops, t0,
                                 stage1_ms, stage2_ms, enc_ms, X, expect, reference, cfg.verify_records, &eval_ms,
                                 ops_stage1, match.second, "", or_scale);
    ok = ok && good;
    (m==EvalMode::hulk? t_hl : t_ar) = eval_ms;
  }
  if(t_ar>0&&t_hl>0) cout<<"\nspeedup (arith/HULK, HE eval) = "<<(t_ar/t_hl)<<"x\n";
  return ok?0:1;
}
