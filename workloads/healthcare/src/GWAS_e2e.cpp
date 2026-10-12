/*
 * GWAS case/control allelic association test, end to end under encryption.
 *
 * Setting: outsourced GWAS (as in Blatt et al., PNAS 2020): every participant uploads its own
 * encrypted record and the server runs the whole study without interaction. The phenotype is
 * not given; it is derived on the server from encrypted EHR flags with a rule-based phenotype
 * algorithm (eMERGE/PheKB style), as in EHR-linked biobanks.
 *
 * Layout (patient-major, one SNP per slot, 32,768 SNPs per ciphertext block):
 *   per patient i:  G1_i, G2_i   genotype one-hot planes (heterozygous, homozygous-alternate)
 *                   C_i          per-SNP call plane (1 = genotype called and passed call QC)
 *                   EHR / QC flags, each one ciphertext with the flag in every slot
 *
 * Pipeline (arithmetic approach; Boolean AND = HMul, OR = 1 - prod(1 - x), NOT = 1 - x):
 *   and_test   bulk bitwise   phenotype algorithm and inclusion masks
 *                               anymed = insulin OR metformin
 *                               case   = T2D_dx AND (insulin OR metformin OR HbA1c_high) AND NOT T1D_dx
 *                               ctrl   = NOT T2D_dx AND NOT T1D_dx AND NOT anymed
 *                               qc     = callrate_ok AND ancestry_ok AND unrelated AND sex_ok
 *                               caseS  = case AND qc AND C_i,  ctrlS = ctrl AND qc AND C_i
 *   weight     arithmetic     alternate-allele dosage x_i = G1_i + 2 G2_i (additions only);
 *                               case dosage caseS * x_i, control dosage ctrlS * x_i (2 HMul)
 *   aggregate  arithmetic     sums over patients (HAdd): A_case, A_ctrl, N_case, N_ctrl per SNP
 *   stats      arithmetic     allelic 2x2 table a = A_case, b = 2 N_case - A_case,
 *                               c = A_ctrl, d = 2 N_ctrl - A_ctrl; returns
 *                               D = ad - bc, M1 = (a+b)(c+d), M2 = (a+c)(b+d), T = a+b+c+d
 *                             The client finishes chi2 = T D^2 / (M1 M2): D^2 and the products
 *                             exceed the 25-bit plaintext modulus, every returned value stays below it.
 *
 * Timing as in the other e2e drivers: enc_ms = client encryption, eval_ms = homomorphic
 * evaluation only, dec_ms = client decryption. Every returned slot is checked against a
 * plaintext oracle.
 *
 * Arithmetic vs HULK: both evaluate the same Boolean phenotype; only and_test differs.
 *   arithmetic   AND = HMul tree, OR = 1 - prod(1 - x), with QC shared by case and control
 *   HULK         each definition flattened into one symmetric function:
 *                  evid = [ins + met + hba1c >= 1]
 *                  case = [T2D + (1-T1D) + 4 QC flags + call + evid = 8]
 *                  ctrl = [(1-T2D) + (1-T1D) + (1-ins) + (1-met) + 4 QC flags + call = 9]
 *                (NOT anymed = NOT ins AND NOT met). weight / aggregate / stats are identical.
 */
#include "HE_workload_common.h"
#include <array>
#include <random>
using namespace hulk;

struct GwasCfg : RunConfig { size_t patients = 128; };

static GwasCfg parse_cfg(int argc, char **argv){
    GwasCfg c; c.threads = 1;
    for(int i=1;i<argc;++i){ string a=argv[i];
        auto nx=[&]{ return stoull(argv[++i]); };
        if(a=="--patients") c.patients=max<size_t>(2,nx());
        else if(a=="--seed") c.seed=nx();
        else if(a=="--threads") c.threads=max<size_t>(1,nx());
        else if(a=="--mode") c.mode=parse_mode_value(argv[++i]);
        else if(a=="--limbs"||a=="--arithmetic-limbs") c.arithmetic_limbs=max<size_t>(1,nx());
        else if(a=="--hulk-limbs") c.hulk_limbs=max<size_t>(1,nx());
        else throw invalid_argument("unknown arg: "+a);
    }
    return c;
}

enum Flag { T2D, T1D, INSULIN, METFORMIN, HBA1C, CALLRATE, ANCESTRY, UNRELATED, SEX, NFLAGS };

struct Patient { array<uint64_t, NFLAGS> flag{}; vector<uint64_t> g1, g2, call; };

static vector<Patient> make_cohort(size_t P, size_t S, uint64_t seed){
    mt19937_64 rng(seed); uniform_real_distribution<double> U(0,1);
    vector<double> maf(S); for(auto &m: maf) m = 0.05 + 0.45*U(rng);
    vector<bool> assoc(S); for(size_t s=0;s<S;++s) assoc[s] = U(rng) < 0.01;   // 1% truly associated SNPs
    vector<Patient> pts(P);
    for(auto &p: pts){
        const bool t2d = U(rng) < 0.45, t1d = U(rng) < 0.05;
        p.flag[T2D]=t2d; p.flag[T1D]=t1d;
        p.flag[INSULIN]   = U(rng) < (t2d? 0.35 : 0.04);
        p.flag[METFORMIN] = U(rng) < (t2d? 0.70 : 0.05);
        p.flag[HBA1C]     = U(rng) < (t2d? 0.60 : 0.08);
        p.flag[CALLRATE]  = U(rng) < 0.97; p.flag[ANCESTRY] = U(rng) < 0.92;
        p.flag[UNRELATED] = U(rng) < 0.95; p.flag[SEX]      = U(rng) < 0.99;
        p.g1.assign(S,0); p.g2.assign(S,0); p.call.assign(S,0);
        for(size_t s=0;s<S;++s){
            const double q = min(0.95, maf[s] * ((assoc[s] && t2d)? 1.5 : 1.0));
            const int alt = (U(rng)<q) + (U(rng)<q);
            p.g1[s] = alt==1; p.g2[s] = alt==2; p.call[s] = U(rng) < 0.98;
        }
    }
    return pts;
}

struct Outputs { vector<int64_t> D; vector<uint64_t> M1, M2, T; };

static Outputs oracle(const vector<Patient> &pts, size_t S){
    vector<uint64_t> Ac(S,0), Ao(S,0), Nc(S,0), No(S,0);
    for(auto &p: pts){
        const auto &f = p.flag;
        const bool cs = f[T2D] && (f[INSULIN]||f[METFORMIN]||f[HBA1C]) && !f[T1D];
        const bool ct = !f[T2D] && !f[T1D] && !(f[INSULIN]||f[METFORMIN]);
        const bool qc = f[CALLRATE] && f[ANCESTRY] && f[UNRELATED] && f[SEX];
        for(size_t s=0;s<S;++s){
            if(!qc || !p.call[s]) continue;
            const uint64_t x = p.g1[s] + 2*p.g2[s];
            if(cs){ Ac[s]+=x; Nc[s]++; }
            if(ct){ Ao[s]+=x; No[s]++; }
        }
    }
    Outputs o; o.D.resize(S); o.M1.resize(S); o.M2.resize(S); o.T.resize(S);
    for(size_t s=0;s<S;++s){
        const int64_t a=Ac[s], b=2*Nc[s]-Ac[s], c=Ao[s], d=2*No[s]-Ao[s];
        o.D[s]=a*d-b*c; o.M1[s]=uint64_t((a+b)*(c+d)); o.M2[s]=uint64_t((a+c)*(b+d)); o.T[s]=uint64_t(a+b+c+d);
    }
    return o;
}

int main(int argc, char **argv){
    const GwasCfg cfg = parse_cfg(argc, argv);
    const BenchmarkEnvs envs = make_benchmark_envs(cfg);
    const size_t S = envs.slot_count, P = cfg.patients;
    const auto pts = make_cohort(P, S, cfg.seed);
    const Outputs expect = oracle(pts, S);
    size_t ncase=0, nctrl=0;
    for(auto &q: pts){ const auto &f=q.flag;
        const bool qc=f[CALLRATE]&&f[ANCESTRY]&&f[UNRELATED]&&f[SEX];
        ncase += qc && f[T2D] && (f[INSULIN]||f[METFORMIN]||f[HBA1C]) && !f[T1D];
        nctrl += qc && !f[T2D] && !f[T1D] && !(f[INSULIN]||f[METFORMIN]); }
    cout << "GWAS e2e: " << P << " patients x " << S << " SNPs (one ciphertext block); "
         << ncase << " cases, " << nctrl << " controls after phenotyping and QC\n" << fixed << setprecision(3);
    bool all_ok = true;

    for(int pass=0; pass<2; ++pass){
        const EvalMode m = pass ? EvalMode::hulk : EvalMode::arithmetic;
        if(m==EvalMode::arithmetic && !wants_arithmetic(cfg.mode)) continue;
        if(m==EvalMode::hulk && !wants_hulk(cfg.mode)) continue;
        const HEEnv &env = envs.for_mode(m); const uint64_t p = env.plain_modulus;
        const char *mn = m==EvalMode::hulk ? "hulk" : "arithmetic";
        if(uint64_t(4*P)*uint64_t(4*P) >= p/2) throw runtime_error("too many patients for the plaintext modulus");

        // ---- client side: every participant encrypts its own record (enc_ms)
        Stopwatch enc_sw;
        struct EncPatient { array<Ciphertext, NFLAGS> flag; Ciphertext g1, g2, call; };
        vector<EncPatient> X(P);
        for(size_t i=0;i<P;++i){
            for(int f=0;f<NFLAGS;++f) X[i].flag[f] = encrypt_slots(env, vector<uint64_t>(S, pts[i].flag[f]));
            X[i].g1 = encrypt_slots(env, pts[i].g1); X[i].g2 = encrypt_slots(env, pts[i].g2);
            X[i].call = encrypt_slots(env, pts[i].call);
        }
        const double enc_ms = enc_sw.ms();
        size_t upload_bytes = 0;
        for(auto &e: X){ upload_bytes += ciphertext_upload_bytes(e.g1)+ciphertext_upload_bytes(e.g2)+ciphertext_upload_bytes(e.call);
                         for(auto &c: e.flag) upload_bytes += ciphertext_upload_bytes(c); }

        // ---- server: homomorphic evaluation (eval_ms); per-stage op counts are instrumentation only
        OpCount ops_bit, ops_weight, ops_agg, ops_stats;
        double tm_bit=0, tm_w=0, tm_agg=0;                     // diagnostic wall clock per stage
        auto lap=[](steady_clock::time_point &t){ auto n=steady_clock::now(); double d=duration<double,milli>(n-t).count(); t=n; return d; };
        const auto t0 = steady_clock::now();
        Ciphertext Ac, Ao, Nc, No; bool init=false;
        for(size_t i=0;i<P;++i){
            const auto &F = X[i].flag;
            auto tl = steady_clock::now();
            // and_test: phenotype algorithm, QC and per-SNP call mask (bulk bitwise)
            Ciphertext cs, ct;
            Ciphertext not_t1d = one_minus_bit(env, F[T1D], ops_bit);
            Ciphertext not_t2d = one_minus_bit(env, F[T2D], ops_bit);
            if(m==EvalMode::arithmetic){
                auto anymed = arithmetic_any_true({F[INSULIN], F[METFORMIN]}, env);                     ops_bit += anymed.second;
                auto evid   = arithmetic_any_true({F[INSULIN], F[METFORMIN], F[HBA1C]}, env);           ops_bit += evid.second;
                Ciphertext not_med = one_minus_bit(env, anymed.first, ops_bit);
                auto qc = arithmetic_all_true({F[CALLRATE], F[ANCESTRY], F[UNRELATED], F[SEX]}, env);   ops_bit += qc.second;
                auto a  = arithmetic_all_true({F[T2D], evid.first, not_t1d, qc.first, X[i].call}, env); ops_bit += a.second;
                auto b  = arithmetic_all_true({not_t2d, not_t1d, not_med, qc.first, X[i].call}, env);   ops_bit += b.second;
                cs = move(a.first); ct = move(b.first);
            } else {
                Ciphertext not_ins = one_minus_bit(env, F[INSULIN], ops_bit);
                Ciphertext not_met = one_minus_bit(env, F[METFORMIN], ops_bit);
                auto evid = hulk_any_true({F[INSULIN], F[METFORMIN], F[HBA1C]}, env);                   ops_bit += evid.second;
                auto a = hulk_all_true({F[T2D], not_t1d, F[CALLRATE], F[ANCESTRY], F[UNRELATED], F[SEX],
                                        X[i].call, evid.first}, env);                                   ops_bit += a.second;
                auto b = hulk_all_true({not_t2d, not_t1d, not_ins, not_met, F[CALLRATE], F[ANCESTRY],
                                        F[UNRELATED], F[SEX], X[i].call}, env);                         ops_bit += b.second;
                cs = move(a.first); ct = move(b.first);
            }
            tm_bit += lap(tl);
            // weight: alternate-allele dosage, then case / control dosage
            Ciphertext x = multiply_plain_const(env, X[i].g2, 2, ops_weight); add_inplace(env, x, X[i].g1, ops_weight);
            Ciphertext ac = multiply_ciphertexts(env, cs, x, ops_weight);
            Ciphertext ao = multiply_ciphertexts(env, ct, x, ops_weight);
            tm_w += lap(tl);
            // aggregate: sums over patients
            if(!init){ Ac=move(ac); Ao=move(ao); Nc=cs; No=ct; init=true; }
            else { add_inplace(env, Ac, ac, ops_agg); add_inplace(env, Ao, ao, ops_agg);
                   add_inplace(env, Nc, cs, ops_agg); add_inplace(env, No, ct, ops_agg); }
            tm_agg += lap(tl);
        }
        // stats: allelic 2x2 table and the chi-square components, SIMD over all SNPs
        Ciphertext b = multiply_plain_const(env, Nc, 2, ops_stats); b = subtract_ciphertexts(env, b, Ac, ops_stats);
        Ciphertext d = multiply_plain_const(env, No, 2, ops_stats); d = subtract_ciphertexts(env, d, Ao, ops_stats);
        Ciphertext ad = multiply_ciphertexts(env, Ac, d, ops_stats);
        Ciphertext bc = multiply_ciphertexts(env, b, Ao, ops_stats);                    // b * c, c = A_ctrl
        Ciphertext D  = subtract_ciphertexts(env, ad, bc, ops_stats);
        Ciphertext ab = Ac; add_inplace(env, ab, b, ops_stats);                         // a + b = 2 N_case
        Ciphertext cd = Ao; add_inplace(env, cd, d, ops_stats);                         // c + d = 2 N_ctrl
        Ciphertext ac2 = Ac; add_inplace(env, ac2, Ao, ops_stats);                      // a + c
        Ciphertext bd = b; add_inplace(env, bd, d, ops_stats);                          // b + d
        Ciphertext M1 = multiply_ciphertexts(env, ab, cd, ops_stats);
        Ciphertext M2 = multiply_ciphertexts(env, ac2, bd, ops_stats);
        Ciphertext T  = ab; add_inplace(env, T, cd, ops_stats);
        const double eval_ms = duration<double, milli>(steady_clock::now() - t0).count();

        // ---- client: decrypt and verify against the plaintext oracle
        Stopwatch dec_sw;
        auto dD = decrypt_slots(env, D), dM1 = decrypt_slots(env, M1), dM2 = decrypt_slots(env, M2), dT = decrypt_slots(env, T);
        const double dec_ms = dec_sw.ms();
        bool ok = true;
        for(size_t s=0; s<S && ok; ++s){
            const int64_t Dv = dD[s] % p > p/2 ? int64_t(dD[s] % p) - int64_t(p) : int64_t(dD[s] % p);
            ok = Dv == expect.D[s] && dM1[s] % p == expect.M1[s] && dM2[s] % p == expect.M2[s] && dT[s] % p == expect.T[s];
        }
        all_ok = all_ok && ok;

        OpCount ops; ops += ops_bit; ops += ops_weight; ops += ops_agg; ops += ops_stats;
        int nb = INT_MAX; for(auto *c: {&D,&M1,&M2,&T}) nb = min(nb, env.dec->invariant_noise_budget(*c));
        cout << "\n[" << mn << "] limbs=" << env.coeff_modulus_primes << "\n"
             << "  encrypt (client) : " << enc_ms << " ms   upload " << upload_bytes << " B\n"
             << "  HE eval (server) : " << eval_ms << " ms   HMul=" << ops.total_ct_mul() << "   rotations=" << ops.rotations << "\n"
             << "  decrypt (client) : " << dec_ms << " ms   [" << (ok? "match":"MISMATCH") << "]\n"
             << "  noise budget left: " << nb << " bits (min over outputs)\n";
        print_ops(mn, ops);
        print_stage_line(mn, "and_test", ops_bit);
        print_stage_line(mn, "weight", ops_weight);
        print_stage_line(mn, "aggregate", ops_agg);
        print_stage_line(mn, "stats", ops_stats);
        cout << "TIMER mode=" << mn << " and_test_ms=" << tm_bit << " weight_ms=" << tm_w << " aggregate_ms=" << tm_agg
             << " stats_ms=" << (eval_ms - tm_bit - tm_w - tm_agg) << "\n";
        E2EReport rep; rep.mode=m; rep.limbs=env.coeff_modulus_primes; rep.enc_ms=enc_ms; rep.eval_ms=eval_ms;
        rep.dec_ms=dec_ms; rep.upload_bytes=upload_bytes; rep.hmul=ops.total_ct_mul(); rep.rot=ops.rotations; rep.ok=ok;
        print_e2e_line(rep);
    }
    return all_ok? 0 : 1;
}
