/*
 * bench_ops.cpp -- single-thread unit costs of the homomorphic primitives the
 * workload drivers use, at the benchmark ring (BFV, N=32768, 25-bit batching
 * plaintext modulus), for every limb count L in [--lmin, --lmax] (default 4..16).
 *
 * Every op is called exactly as the harness calls it (HE_workload_common.h):
 *   hmul_relin       multiply_ciphertexts   (multiply + relinearize)
 *   sq_relin         square_ciphertext      (square + relinearize)
 *   mul_plain        Evaluator::multiply_plain with a NON-constant (random-slot) plaintext
 *   mul_plain_const  multiply_plain_const   (cached constant plaintext = constant polynomial)
 *   add_plain        add_plain_inplace      (cached constant plaintext)
 *   hadd             add_inplace
 *   relin            relinearize_inplace of a size-3 product (reference only)
 *   negate           Evaluator::negate      (reference only; one_minus_bit, uncounted by OpCount)
 *   rot_rows_1       rotate_rows(step 1)
 *   rot_rows_4096    rotate_rows(step 4096)
 *   rot_cols         rotate_columns
 *   sum_slots        harness sum_slots: 14 row rotations + 1 column rotation + 15 HAdd
 *                    (reference, to check the per-op model of the aggregation stage)
 * Each op: 2 untimed warm-up calls, then --reps timed calls (default 15, >= 10);
 * the median is reported. Output, one line per (L, op):
 *   UNIT L=<L> op=<name> ms=<median>
 * plus a header "BENCH ..." and per-L "UNITINFO L=<L> prime_bits=<b> total_bits=<B> ..." lines.
 * Galois keys are generated for every power-of-two row step 1..N/4 (incl. 1 and 4096)
 * and for the column rotation (step 0), i.e. exactly what sum_slots needs.
 *
 * Prime size: --prime-bits 55 (default, 55-bit primes for every L), or
 * --prime-bits harness (the drivers' rule: 57-bit primes for L <= 15, 55-bit at 16).
 * SEAL stores every prime in one 64-bit word, so 55 vs 57 bits does not change the
 * per-limb work; the option is there to confirm that on the target node.
 */
#include "HE_workload_common.h"
#include <random>
#include <unistd.h>
using namespace hulk;

struct BenchCfg { size_t lmin = 4, lmax = 16, reps = 15; string prime_bits = "55"; uint64_t seed = 42; };

static BenchCfg parse(int argc, char **argv)
{
    BenchCfg c;
    for (int i = 1; i < argc; ++i) {
        const string a = argv[i];
        auto nx = [&]() -> string { if (i + 1 >= argc) throw invalid_argument("missing value for " + a); return argv[++i]; };
        if (a == "--lmin") c.lmin = stoul(nx());
        else if (a == "--lmax") c.lmax = stoul(nx());
        else if (a == "--reps") c.reps = stoul(nx());
        else if (a == "--prime-bits") c.prime_bits = nx();
        else if (a == "--seed") c.seed = stoull(nx());
        else if (a == "--threads") { if (stoul(nx()) != 1) throw invalid_argument("bench_ops is single-thread only"); }
        else throw invalid_argument("unknown arg: " + a);
    }
    if (c.reps < 10) throw invalid_argument("--reps must be >= 10");
    if (c.lmin < 1 || c.lmin > c.lmax) throw invalid_argument("bad --lmin/--lmax");
    return c;
}

static int prime_bits_for(const BenchCfg &c, size_t L)
{
    if (c.prime_bits == "harness") return coeff_prime_bits(EvalMode::arithmetic, L);
    const int b = stoi(c.prime_bits);
    if (b < 20 || b > 60) throw invalid_argument("--prime-bits must be 20..60 or 'harness'");
    return b;
}

static HEEnv make_bench_env(size_t L, int bits)
{
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(kPolyDegree);
    parms.set_coeff_modulus(CoeffModulus::Create(kPolyDegree, vector<int>(L, bits)));
    parms.set_plain_modulus(PlainModulus::Batching(kPolyDegree, kPlainModulusBits));
    auto ctx = make_shared<SEALContext>(parms, true, sec_level_type::tc128);
    if (!ctx->parameters_set()) throw runtime_error(string("params rejected: ") + ctx->parameter_error_message());
    KeyGenerator kg(*ctx);
    HEEnv env;
    env.ctx = ctx;
    env.sk = kg.secret_key();
    kg.create_public_key(env.pk);
    kg.create_relin_keys(env.rk);
    vector<int> steps;
    for (size_t st = 1; st < kPolyDegree / 2; st <<= 1) steps.push_back(int(st));   // 1 .. 8192 (incl. 4096)
    steps.push_back(0);                                                             // column rotation
    kg.create_galois_keys(steps, env.gk);
    env.enc = make_unique<Encryptor>(*ctx, env.pk);
    env.eval = make_unique<Evaluator>(*ctx);
    env.dec = make_unique<Decryptor>(*ctx, env.sk);
    env.batch = make_unique<BatchEncoder>(*ctx);
    env.mode = EvalMode::arithmetic;
    env.cache_enabled = true;                 // as in the drivers: cached constant plaintexts
    env.cache = make_shared<HEEnv::CacheState>();
    env.plain_modulus = parms.plain_modulus().value();
    env.slot_count = env.batch->slot_count();
    env.max_threads = 1;
    env.coeff_modulus_primes = parms.coeff_modulus().size();
    for (const Modulus &q : parms.coeff_modulus()) env.coeff_modulus_total_bits += q.bit_count();
    warm_env_cache(env);
    return env;
}

template <class F>
static double median_ms(size_t reps, F &&f)
{
    f(); f();                                   // warm-up (allocations, caches)
    vector<double> t; t.reserve(reps);
    for (size_t i = 0; i < reps; ++i) {
        const auto s = steady_clock::now();
        f();
        t.push_back(duration<double, milli>(steady_clock::now() - s).count());
    }
    sort(t.begin(), t.end());
    return (reps % 2) ? t[reps / 2] : 0.5 * (t[reps / 2 - 1] + t[reps / 2]);
}

int main(int argc, char **argv)
{
    const BenchCfg cfg = parse(argc, argv);
    char host[256] = "?"; gethostname(host, sizeof host - 1);
    cout << "BENCH N=" << kPolyDegree << " t_bits=" << kPlainModulusBits << " prime_bits=" << cfg.prime_bits
         << " lmin=" << cfg.lmin << " lmax=" << cfg.lmax << " reps=" << cfg.reps << " threads=1 host=" << host << "\n";
    mt19937_64 rng(cfg.seed);
    for (size_t L = cfg.lmin; L <= cfg.lmax; ++L) {
        const int bits = prime_bits_for(cfg, L);
        Stopwatch kg_sw;
        HEEnv env = make_bench_env(L, bits);
        const double keygen_ms = kg_sw.ms();
        const uint64_t t = env.plain_modulus;
        uniform_int_distribution<uint64_t> d(0, t - 1);
        auto rand_slots = [&]() { vector<uint64_t> v(env.slot_count); for (auto &x : v) x = d(rng); return v; };
        const Ciphertext a = encrypt_slots(env, rand_slots()), b = encrypt_slots(env, rand_slots());
        Plaintext pt_rand; env.batch->encode(rand_slots(), pt_rand);       // non-constant plaintext
        const uint64_t konst = 12345 % t;
        (void)constant_plaintext(env, konst);                               // cache, as prewarm does
        Ciphertext prod3; env.eval->multiply(a, b, prod3);                  // size-3 input for relin
        OpCount ops;
        cout << "UNITINFO L=" << L << " prime_bits=" << bits << " total_bits=" << env.coeff_modulus_total_bits
             << " keygen_ms=" << fixed << setprecision(1) << keygen_ms << "\n";
        cout.unsetf(ios::floatfield);
        const auto emit = [&](const char *name, double ms) {
            cout << "UNIT L=" << L << " op=" << name << " ms=" << fixed << setprecision(4) << ms << "\n" << flush;
            cout.unsetf(ios::floatfield);
        };
        { Ciphertext r; emit("hmul_relin",      median_ms(cfg.reps, [&] { r = multiply_ciphertexts(env, a, b, ops); })); }
        { Ciphertext r; emit("sq_relin",        median_ms(cfg.reps, [&] { r = square_ciphertext(env, a, ops); })); }
        { Ciphertext r; emit("mul_plain",       median_ms(cfg.reps, [&] { env.eval->multiply_plain(a, pt_rand, r); })); }
        { Ciphertext r; emit("mul_plain_const", median_ms(cfg.reps, [&] { r = multiply_plain_const(env, a, konst, ops); })); }
        { Ciphertext r = a; emit("add_plain",   median_ms(cfg.reps, [&] { add_plain_inplace(env, r, konst, ops); })); }
        { Ciphertext r = a; emit("hadd",        median_ms(cfg.reps, [&] { add_inplace(env, r, b, ops); })); }
        { Ciphertext r; emit("relin",           median_ms(cfg.reps, [&] { r = prod3; env.eval->relinearize_inplace(r, env.rk); })); }
        { Ciphertext r; emit("negate",          median_ms(cfg.reps, [&] { env.eval->negate(a, r); })); }
        { Ciphertext r; emit("rot_rows_1",      median_ms(cfg.reps, [&] { env.eval->rotate_rows(a, 1, env.gk, r); })); }
        { Ciphertext r; emit("rot_rows_4096",   median_ms(cfg.reps, [&] { env.eval->rotate_rows(a, 4096, env.gk, r); })); }
        { Ciphertext r; emit("rot_cols",        median_ms(cfg.reps, [&] { env.eval->rotate_columns(a, env.gk, r); })); }
        { Ciphertext r; emit("sum_slots",       median_ms(cfg.reps, [&] { r = sum_slots(a, env).first; })); }
    }
    return 0;
}
