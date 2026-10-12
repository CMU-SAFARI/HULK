#pragma once

#include <seal/seal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hulk {

using namespace seal;
using namespace std;
using namespace std::chrono;

static constexpr size_t kPolyDegree = 32768;
static constexpr int kPlainModulusBits = 25;
static constexpr size_t kDefaultPackedRows = 32768;
static constexpr uint64_t kDefaultSeed = 42;
static constexpr size_t kDefaultThreads = 8;

struct RunConfig {
    size_t rows = kDefaultPackedRows;
    uint64_t seed = kDefaultSeed;
    size_t threads = kDefaultThreads;
    size_t query_batch = 1;
    size_t top_k = 10;
    optional<size_t> arithmetic_limbs;
    optional<size_t> hulk_limbs;
    enum class Mode {
        both,
        arithmetic,
        hulk
    } mode = Mode::both;
};

enum class EvalMode {
    arithmetic,
    hulk
};

inline RunConfig::Mode parse_mode_value(const string &value)
{
    if (value == "both") {
        return RunConfig::Mode::both;
    }
    if (value == "arithmetic") {
        return RunConfig::Mode::arithmetic;
    }
    if (value == "hulk") {
        return RunConfig::Mode::hulk;
    }
    throw invalid_argument("unknown mode: " + value);
}

inline bool wants_arithmetic(RunConfig::Mode mode)
{
    return mode == RunConfig::Mode::both || mode == RunConfig::Mode::arithmetic;
}

inline bool wants_hulk(RunConfig::Mode mode)
{
    return mode == RunConfig::Mode::both || mode == RunConfig::Mode::hulk;
}

inline string mode_name(RunConfig::Mode mode)
{
    switch (mode) {
        case RunConfig::Mode::both: return "both";
        case RunConfig::Mode::arithmetic: return "arithmetic";
        case RunConfig::Mode::hulk: return "hulk";
    }
    throw invalid_argument("unreachable mode");
}

inline RunConfig parse_basic_args(int argc, char **argv)
{
    RunConfig cfg;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--rows" && i + 1 < argc) {
            cfg.rows = static_cast<size_t>(stoull(argv[++i]));
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            cfg.seed = stoull(argv[++i]);
            continue;
        }
        if (arg == "--threads" && i + 1 < argc) {
            cfg.threads = max<size_t>(1, stoull(argv[++i]));
            continue;
        }
        if (arg == "--query-batch" && i + 1 < argc) {
            cfg.query_batch = max<size_t>(1, stoull(argv[++i]));
            continue;
        }
        if (arg == "--top-k" && i + 1 < argc) {
            cfg.top_k = stoull(argv[++i]);
            continue;
        }
        if (arg == "--mode" && i + 1 < argc) {
            cfg.mode = parse_mode_value(argv[++i]);
            continue;
        }
        if (arg == "--limbs" && i + 1 < argc) {
            const size_t limbs = max<size_t>(1, stoull(argv[++i]));
            cfg.arithmetic_limbs = limbs;
            cfg.hulk_limbs = limbs;
            continue;
        }
        if (arg == "--arithmetic-limbs" && i + 1 < argc) {
            cfg.arithmetic_limbs = max<size_t>(1, stoull(argv[++i]));
            continue;
        }
        if (arg == "--hulk-limbs" && i + 1 < argc) {
            cfg.hulk_limbs = max<size_t>(1, stoull(argv[++i]));
            continue;
        }
        if (arg == "--help") {
            cout << "Options:\n"
                 << "  --rows <n>   Number of synthetic rows (default "
                 << kDefaultPackedRows << ")\n"
                 << "  --seed <n>   RNG seed (default " << kDefaultSeed << ")\n"
                 << "  --threads <n> Threads for block/output parallelism (default " << kDefaultThreads << ")\n"
                 << "  --query-batch <n> Number of full independent query copies to execute in parallel (default 1)\n"
                 << "  --top-k <n> Number of ranked outputs to report after aggregation (default 10)\n"
                 << "  --mode <m>   arithmetic | hulk | both (default both)\n"
                 << "  --limbs <n>  Override coeff-modulus prime count for both modes\n"
                 << "  --arithmetic-limbs <n>  Override arithmetic coeff-modulus prime count\n"
                 << "  --hulk-limbs <n>        Override HULK coeff-modulus prime count\n";
            exit(0);
        }
        throw invalid_argument("unknown argument: " + arg);
    }
    return cfg;
}

struct OpCount {
    int ct_ct_mul = 0;
    int ct_sq = 0;
    int plain_mul = 0;
    int ct_add = 0;
    int plain_add = 0;
    int relin = 0;
    int rotations = 0;
    int depth = 0;
    // Breakdown subset counters (each is a subset of the totals above):
    int weight_muls = 0;   // ct*ct muls from weight-column application only
    int agg_adds = 0;      // ct-adds that fold masked partials into per-output totals
                           // (zero guard + HAdd over blocks / workers); instrumentation only

    int total_ct_mul() const
    {
        return ct_ct_mul + ct_sq;
    }

    OpCount &operator+=(const OpCount &other)
    {
        ct_ct_mul += other.ct_ct_mul;
        ct_sq += other.ct_sq;
        plain_mul += other.plain_mul;
        ct_add += other.ct_add;
        plain_add += other.plain_add;
        relin += other.relin;
        rotations += other.rotations;
        depth = max(depth, other.depth);
        weight_muls += other.weight_muls;
        agg_adds += other.agg_adds;
        return *this;
    }
};

// ---- per-stage latency decomposition (instrumentation only; no HE work) ----
// a - b field by field for the counters of the stage lines (depth is not a count
// and is left 0). Used to snapshot/diff an OpCount at existing stage boundaries.
inline OpCount op_diff(const OpCount &a, const OpCount &b)
{
    OpCount d;
    d.ct_ct_mul = a.ct_ct_mul - b.ct_ct_mul;
    d.ct_sq = a.ct_sq - b.ct_sq;
    d.plain_mul = a.plain_mul - b.plain_mul;
    d.ct_add = a.ct_add - b.ct_add;
    d.plain_add = a.plain_add - b.plain_add;
    d.relin = a.relin - b.relin;
    d.rotations = a.rotations - b.rotations;
    d.weight_muls = a.weight_muls - b.weight_muls;
    d.agg_adds = a.agg_adds - b.agg_adds;
    return d;
}

// "STAGE mode=<arithmetic|hulk> stage=<name> hmul=.. sq=.. pmul=.. add=.. padd=.. rot=.. relin=.."
// The stage lines of one mode sum exactly to that mode's reported OpCount totals.
inline void print_stage_line(const string &mode, const string &stage, const OpCount &o, ostream &os = cout)
{
    os << "STAGE mode=" << mode << " stage=" << stage
       << " hmul=" << o.ct_ct_mul << " sq=" << o.ct_sq << " pmul=" << o.plain_mul
       << " add=" << o.ct_add << " padd=" << o.plain_add << " rot=" << o.rotations
       << " relin=" << o.relin << "\n";
}

struct PolyBlockTerm {
    uint16_t power = 0;
    uint64_t coeff = 0;
};

struct PolyEvalPlan {
    size_t block_size = 1;
    vector<uint64_t> block_constants;
    vector<vector<PolyBlockTerm>> block_terms;
};

struct FactorPlan {
    vector<uint64_t> constants;
    uint64_t scale = 1;
};

inline void print_ops(const string &label, const OpCount &ops)
{
    cout << "  " << label << ":\n"
         << "    ct-ct mul (incl sq): " << ops.total_ct_mul()
         << " (" << ops.ct_ct_mul << " mul + " << ops.ct_sq << " sq)\n"
         << "    plain-mul:           " << ops.plain_mul << "\n"
         << "    ct-add:              " << ops.ct_add << "\n"
         << "    plain-add:           " << ops.plain_add << "\n"
         << "    relin:               " << ops.relin << "\n"
         << "    rotations:           " << ops.rotations << "\n"
         << "    depth:               " << ops.depth << "\n";
}

// Print a 3-way breakdown: shared-filter DAG muls | per-output predicate muls | weight muls.
// shared_ops reflects shared-filter node evaluation amortized over all outputs.
// For arithmetic: pred_muls = HMul-tree muls; for HULK: pred_muls = indicator polynomial muls.
inline void print_ops_breakdown(
    const string &label,
    const OpCount &total_ops,
    const OpCount &shared_filter_ops)
{
    const int shared_muls   = shared_filter_ops.total_ct_mul();
    const int weight_muls   = total_ops.weight_muls;
    const int pred_muls     = total_ops.total_ct_mul() - shared_muls - weight_muls;
    cout << "  " << label << " breakdown (CSE | pred | weight):\n"
         << "    shared-filter DAG muls (CSE, amortized): " << shared_muls << "\n"
         << "    per-output predicate muls (HMul-tree or indicator): " << pred_muls << "\n"
         << "    weight-column muls: " << weight_muls << "\n"
         << "    total ct-ct mul: " << total_ops.total_ct_mul() << "\n";
}

struct HEEnv {
    struct CacheState {
        mutable mutex plain_mutex;
        mutable mutex encrypted_mutex;
        mutable mutex coeff_mutex;
        mutable mutex plan_mutex;
        mutable mutex scalar_mutex;
        unordered_map<uint64_t, Plaintext> plain_cache;
        unordered_map<uint64_t, Ciphertext> encrypted_cache;
        unordered_map<string, vector<uint64_t>> coeff_cache;
        unordered_map<string, shared_ptr<const PolyEvalPlan>> poly_plan_cache;
        unordered_map<string, shared_ptr<const FactorPlan>> factor_plan_cache;
        unordered_map<string, uint64_t> scalar_cache;
    };

    shared_ptr<SEALContext> ctx;
    PublicKey pk;
    SecretKey sk;
    RelinKeys rk;
    GaloisKeys gk;
    unique_ptr<Encryptor> enc;
    unique_ptr<Evaluator> eval;
    unique_ptr<Decryptor> dec;
    unique_ptr<BatchEncoder> batch;
    shared_ptr<CacheState> cache;
    EvalMode mode = EvalMode::arithmetic;
    bool cache_enabled = false;
    uint64_t plain_modulus = 0;
    size_t slot_count = 0;
    size_t max_threads = 1;
    size_t coeff_modulus_primes = 0;
    size_t coeff_modulus_total_bits = 0;
};

// Coefficient-modulus prime size: ONE rule for both modes (audit F1/G6).
// The prime size depends only on the limb count, never on the evaluation mode,
// so "minimum limbs" is an apples-to-apples count: 57-bit primes while
// limbs*57 fits the 881-bit 128-bit-security cap for N=32768 (<= 15 limbs),
// 55-bit primes at 16 limbs. Report limbs together with coeff_modulus_total_bits.
//   HE_PRIME_BITS=<b>      force b-bit primes for both modes
//   HE_PRIME_BITS=legacy   restore the pre-audit per-mode sizes (arith 55, HULK 57)
static constexpr int kArithPrimeBits = 55;   // legacy arithmetic size (HE_PRIME_BITS=legacy)
static constexpr int kHulkPrimeBits  = 57;   // legacy HULK size (HE_PRIME_BITS=legacy)
static constexpr int kPrimeBits      = 57;   // shared default for <= 15 limbs
static constexpr int kPrimeBitsWide  = 55;   // shared default for 16 limbs

inline int coeff_prime_bits(EvalMode mode, size_t limb_count)
{
    const char *raw = getenv("HE_PRIME_BITS");
    if (raw && *raw) {
        const string value(raw);
        if (value == "legacy") {
            return (mode == EvalMode::hulk) ? kHulkPrimeBits : kArithPrimeBits;
        }
        const long bits = strtol(raw, nullptr, 10);
        if (bits < 20 || bits > 60) {
            throw invalid_argument("HE_PRIME_BITS must be 'legacy' or an integer in [20, 60]");
        }
        return static_cast<int>(bits);
    }
    return limb_count * kPrimeBits <= 881 ? kPrimeBits : kPrimeBitsWide;
}

inline vector<int> coeff_modulus_bits(EvalMode mode, optional<size_t> limb_override = nullopt)
{
    const size_t limb_count = limb_override.value_or(mode == EvalMode::hulk ? 7 : 8);
    if (limb_count == 0) {
        throw invalid_argument("coeff modulus must have at least one limb");
    }
    return vector<int>(limb_count, coeff_prime_bits(mode, limb_count));
}

inline optional<size_t> limb_override_for(EvalMode mode, const RunConfig &cfg)
{
    return mode == EvalMode::hulk ? cfg.hulk_limbs : cfg.arithmetic_limbs;
}

inline Ciphertext constant_ciphertext(const HEEnv &env, uint64_t value);

// Identical cache warm-up for both modes, before any timer: the encrypted
// constants 0 (refresh_zero_guard) and 1 (constant terms) are created once here,
// so no timed region in either mode pays a public-key encryption for them.
inline void warm_env_cache(const HEEnv &env)
{
    if (!env.cache_enabled) {
        return;
    }
    (void)constant_ciphertext(env, 0);
    (void)constant_ciphertext(env, 1);
}

inline HEEnv make_env(EvalMode mode, size_t max_threads = 1, optional<size_t> limb_override = nullopt)
{
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(kPolyDegree);
    parms.set_coeff_modulus(CoeffModulus::Create(kPolyDegree, coeff_modulus_bits(mode, limb_override)));
    parms.set_plain_modulus(PlainModulus::Batching(kPolyDegree, kPlainModulusBits));

    auto ctx = make_shared<SEALContext>(parms, true, sec_level_type::tc128);
    if (!ctx->parameters_set()) {
        throw runtime_error(ctx->parameter_error_message());
    }

    KeyGenerator keygen(*ctx);
    HEEnv env;
    env.ctx = ctx;
    env.sk = keygen.secret_key();
    keygen.create_public_key(env.pk);
    keygen.create_relin_keys(env.rk);
    keygen.create_galois_keys(env.gk);
    env.enc = make_unique<Encryptor>(*ctx, env.pk);
    env.eval = make_unique<Evaluator>(*ctx);
    env.dec = make_unique<Decryptor>(*ctx, env.sk);
    env.batch = make_unique<BatchEncoder>(*ctx);
    env.mode = mode;
    // Constant/coefficient/plan caching is on for BOTH modes (audit H1/G5/T4):
    // previously only HULK cached, so the arithmetic baseline re-encoded constants
    // and re-encrypted constant ciphertexts inside timed regions.
    env.cache_enabled = true;
    env.cache = make_shared<HEEnv::CacheState>();
    env.plain_modulus = ctx->key_context_data()->parms().plain_modulus().value();
    env.slot_count = env.batch->slot_count();
    env.max_threads = max<size_t>(1, max_threads);
    env.coeff_modulus_primes = parms.coeff_modulus().size();
    for (const Modulus &prime : parms.coeff_modulus()) {
        env.coeff_modulus_total_bits += prime.bit_count();
    }
    warm_env_cache(env);
    return env;
}

inline HEEnv make_env(size_t max_threads = 1)
{
    return make_env(EvalMode::arithmetic, max_threads);
}

inline HEEnv make_worker_env(const HEEnv &source, size_t max_threads_override = 0)
{
    HEEnv env;
    env.ctx = source.ctx;
    env.pk = source.pk;
    env.sk = source.sk;
    env.rk = source.rk;
    env.gk = source.gk;
    env.enc = make_unique<Encryptor>(*env.ctx, env.pk);
    env.eval = make_unique<Evaluator>(*env.ctx);
    env.dec = make_unique<Decryptor>(*env.ctx, env.sk);
    env.batch = make_unique<BatchEncoder>(*env.ctx);
    env.mode = source.mode;
    env.cache_enabled = source.cache_enabled;
    env.cache = source.cache;
    env.plain_modulus = source.plain_modulus;
    env.slot_count = source.slot_count;
    env.max_threads = max_threads_override == 0 ? source.max_threads : max<size_t>(1, max_threads_override);
    env.coeff_modulus_primes = source.coeff_modulus_primes;
    env.coeff_modulus_total_bits = source.coeff_modulus_total_bits;
    return env;
}

// Build `count` single-threaded worker envs. make_worker_env deep-copies pk/sk/rk
// and the FULL GaloisKeys (~L^2 bytes; >1 GB at N=32768), which is NOT a
// homomorphic operation, so callers must build the pool BEFORE starting any
// HE-evaluation timer. Returns an empty pool when count <= 1: a single worker
// uses the caller's env directly (no copy at all under --threads 1).
inline vector<HEEnv> make_worker_pool(const HEEnv &source, size_t count)
{
    vector<HEEnv> pool;
    if (count <= 1) {
        return pool;
    }
    pool.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        pool.push_back(make_worker_env(source, 1));
    }
    return pool;
}

struct BenchmarkEnvs {
    unique_ptr<HEEnv> arithmetic;
    unique_ptr<HEEnv> hulk;
    size_t slot_count = 0;

    const HEEnv &for_mode(EvalMode mode) const
    {
        if (mode == EvalMode::arithmetic && arithmetic) {
            return *arithmetic;
        }
        if (mode == EvalMode::hulk && hulk) {
            return *hulk;
        }
        throw invalid_argument("requested HE environment was not constructed");
    }
};

inline BenchmarkEnvs make_benchmark_envs(RunConfig::Mode mode, size_t max_threads = 1)
{
    BenchmarkEnvs envs;
    if (wants_arithmetic(mode)) {
        envs.arithmetic = make_unique<HEEnv>(make_env(EvalMode::arithmetic, max_threads));
        envs.slot_count = envs.arithmetic->slot_count;
    }
    if (wants_hulk(mode)) {
        envs.hulk = make_unique<HEEnv>(make_env(EvalMode::hulk, max_threads));
        envs.slot_count = envs.hulk->slot_count;
    }
    return envs;
}

inline BenchmarkEnvs make_benchmark_envs(const RunConfig &cfg)
{
    BenchmarkEnvs envs;
    if (wants_arithmetic(cfg.mode)) {
        envs.arithmetic = make_unique<HEEnv>(
            make_env(EvalMode::arithmetic, cfg.threads, limb_override_for(EvalMode::arithmetic, cfg)));
        envs.slot_count = envs.arithmetic->slot_count;
    }
    if (wants_hulk(cfg.mode)) {
        envs.hulk = make_unique<HEEnv>(
            make_env(EvalMode::hulk, cfg.threads, limb_override_for(EvalMode::hulk, cfg)));
        envs.slot_count = envs.hulk->slot_count;
    }
    return envs;
}

inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod)
{
    return (a + b) % mod;
}

inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod)
{
    return (a + mod - (b % mod)) % mod;
}

inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod)
{
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % mod);
}

inline uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod)
{
    uint64_t result = 1 % mod;
    base %= mod;
    while (exp > 0) {
        if (exp & 1U) {
            result = mod_mul(result, base, mod);
        }
        base = mod_mul(base, base, mod);
        exp >>= 1U;
    }
    return result;
}

inline uint64_t mod_inv(uint64_t value, uint64_t mod)
{
    return mod_pow(value % mod, mod - 2, mod);
}

inline Plaintext constant_plaintext(const HEEnv &env, uint64_t value)
{
    const uint64_t key = value % env.plain_modulus;
    if (!env.cache_enabled) {
        vector<uint64_t> slots(env.slot_count, key);
        Plaintext pt;
        env.batch->encode(slots, pt);
        return pt;
    }
    {
        lock_guard<mutex> guard(env.cache->plain_mutex);
        auto it = env.cache->plain_cache.find(key);
        if (it != env.cache->plain_cache.end()) {
            return it->second;
        }
    }

    vector<uint64_t> slots(env.slot_count, key);
    Plaintext pt;
    env.batch->encode(slots, pt);

    lock_guard<mutex> guard(env.cache->plain_mutex);
    auto [it, inserted] = env.cache->plain_cache.emplace(key, pt);
    return it->second;
}

inline Ciphertext constant_ciphertext(const HEEnv &env, uint64_t value)
{
    const uint64_t key = value % env.plain_modulus;
    if (!env.cache_enabled) {
        Plaintext pt = constant_plaintext(env, key);
        Ciphertext ct;
        env.enc->encrypt(pt, ct);
        return ct;
    }
    {
        lock_guard<mutex> guard(env.cache->encrypted_mutex);
        auto it = env.cache->encrypted_cache.find(key);
        if (it != env.cache->encrypted_cache.end()) {
            return it->second;
        }
    }

    Plaintext pt = constant_plaintext(env, key);
    Ciphertext ct;
    env.enc->encrypt(pt, ct);

    lock_guard<mutex> guard(env.cache->encrypted_mutex);
    auto [it, inserted] = env.cache->encrypted_cache.emplace(key, ct);
    return it->second;
}

inline Ciphertext encrypt_slots(const HEEnv &env, const vector<uint64_t> &values)
{
    vector<uint64_t> slots(env.slot_count, 0);
    for (size_t i = 0; i < values.size() && i < env.slot_count; ++i) {
        slots[i] = values[i] % env.plain_modulus;
    }
    Plaintext pt;
    env.batch->encode(slots, pt);
    Ciphertext ct;
    env.enc->encrypt(pt, ct);
    return ct;
}

inline vector<uint64_t> decrypt_slots(const HEEnv &env, const Ciphertext &ct)
{
    Plaintext pt;
    env.dec->decrypt(ct, pt);
    vector<uint64_t> values;
    env.batch->decode(pt, values);
    for (uint64_t &value : values) {
        value %= env.plain_modulus;
    }
    return values;
}

inline uint64_t decrypt_scalar(const HEEnv &env, const Ciphertext &ct)
{
    return decrypt_slots(env, ct)[0] % env.plain_modulus;
}

inline uint64_t sum_plain_slots(const vector<uint64_t> &values)
{
    return accumulate(values.begin(), values.end(), uint64_t{0});
}

struct EncryptedTable {
    vector<vector<Ciphertext>> columns;
    size_t rows = 0;

    size_t block_count() const
    {
        return columns.empty() ? 0 : columns.front().size();
    }
};

inline Ciphertext multiply_ciphertexts(const HEEnv &env, const Ciphertext &lhs, const Ciphertext &rhs, OpCount &ops);

inline size_t append_encrypted_column(EncryptedTable &table, vector<Ciphertext> blocks)
{
    if (!table.columns.empty() && blocks.size() != table.block_count()) {
        throw invalid_argument("derived column block count does not match encrypted table");
    }
    table.columns.push_back(move(blocks));
    return table.columns.size() - 1;
}

inline vector<Plaintext> make_live_row_mask_column(const HEEnv &env, size_t rows)
{
    vector<Plaintext> blocks;
    for (size_t offset = 0; offset < rows || blocks.empty(); offset += env.slot_count) {
        vector<uint64_t> slice(env.slot_count, 0);
        const size_t live = rows > offset ? min(env.slot_count, rows - offset) : 0;
        fill(slice.begin(), slice.begin() + static_cast<ptrdiff_t>(live), uint64_t{1});
        Plaintext pt;
        env.batch->encode(slice, pt);
        blocks.push_back(move(pt));
        if (rows == 0) {
            break;
        }
    }
    return blocks;
}

inline void mask_ciphertext_blocks_inplace(
    const HEEnv &env,
    vector<Ciphertext> &blocks,
    const vector<Plaintext> &live_mask,
    OpCount &ops)
{
    if (blocks.size() != live_mask.size()) {
        throw invalid_argument("live mask block count does not match derived column");
    }
    for (size_t block = 0; block < blocks.size(); ++block) {
        Ciphertext masked;
        env.eval->multiply_plain(blocks[block], live_mask[block], masked);
        ops.plain_mul++;
        blocks[block] = move(masked);
    }
}

inline vector<Ciphertext> encrypt_column(const HEEnv &env, const vector<uint64_t> &column)
{
    vector<Ciphertext> blocks;
    for (size_t offset = 0; offset < column.size(); offset += env.slot_count) {
        const size_t len = min(env.slot_count, column.size() - offset);
        vector<uint64_t> slice(column.begin() + static_cast<ptrdiff_t>(offset),
                               column.begin() + static_cast<ptrdiff_t>(offset + len));
        blocks.push_back(encrypt_slots(env, slice));
    }
    if (blocks.empty()) {
        blocks.push_back(constant_ciphertext(env, 0));
    }
    return blocks;
}

inline vector<vector<Ciphertext>> encrypt_columns(const HEEnv &env, const vector<vector<uint64_t>> &columns)
{
    vector<vector<Ciphertext>> encrypted;
    encrypted.reserve(columns.size());
    for (const auto &column : columns) {
        encrypted.push_back(encrypt_column(env, column));
    }
    return encrypted;
}

inline Ciphertext multiply_plain_const(const HEEnv &env, const Ciphertext &ct, uint64_t scalar, OpCount &ops)
{
    if ((scalar % env.plain_modulus) == 0) {
        return constant_ciphertext(env, 0);
    }
    Plaintext pt = constant_plaintext(env, scalar % env.plain_modulus);
    Ciphertext out;
    env.eval->multiply_plain(ct, pt, out);
    ops.plain_mul++;
    return out;
}

inline void add_inplace(const HEEnv &env, Ciphertext &lhs, const Ciphertext &rhs, OpCount &ops)
{
    env.eval->add_inplace(lhs, rhs);
    ops.ct_add++;
}

inline void add_plain_inplace(const HEEnv &env, Ciphertext &ct, uint64_t scalar, OpCount &ops)
{
    if ((scalar % env.plain_modulus) == 0) {
        return;
    }
    Plaintext pt = constant_plaintext(env, scalar);
    env.eval->add_plain_inplace(ct, pt);
    ops.plain_add++;
}

inline void refresh_zero_guard(const HEEnv &env, Ciphertext &ct, OpCount &ops)
{
    Ciphertext zero = constant_ciphertext(env, 0);
    add_inplace(env, ct, zero, ops);
    ops.agg_adds++;   // instrumentation: this HAdd belongs to the aggregation stage
}

inline Ciphertext multiply_ciphertexts(const HEEnv &env, const Ciphertext &lhs, const Ciphertext &rhs, OpCount &ops)
{
    Ciphertext out;
    env.eval->multiply(lhs, rhs, out);
    env.eval->relinearize_inplace(out, env.rk);
    ops.ct_ct_mul++;
    ops.relin++;
    return out;
}

inline Ciphertext square_ciphertext(const HEEnv &env, const Ciphertext &ct, OpCount &ops)
{
    Ciphertext out;
    env.eval->square(ct, out);
    env.eval->relinearize_inplace(out, env.rk);
    ops.ct_sq++;
    ops.relin++;
    return out;
}

inline pair<Ciphertext, OpCount> balanced_sum(vector<Ciphertext> values, const HEEnv &env)
{
    if (values.empty()) {
        throw invalid_argument("balanced_sum needs at least one ciphertext");
    }
    OpCount ops;
    size_t active = values.size();
    while (active > 1) {
        const size_t half = active / 2;
        for (size_t i = 0; i < half; ++i) {
            env.eval->add_inplace(values[i], values[i + half]);
            ops.ct_add++;
        }
        if (active % 2 == 1) {
            values[half] = move(values[active - 1]);
        }
        active = half + (active % 2);
    }
    return {move(values[0]), ops};
}

inline pair<Ciphertext, OpCount> sum_slots(Ciphertext ct, const HEEnv &env)
{
    OpCount ops;
    const size_t row_size = env.slot_count / 2;
    for (size_t step = 1; step < row_size; step <<= 1U) {
        Ciphertext rotated;
        env.eval->rotate_rows(ct, static_cast<int>(step), env.gk, rotated);
        ops.rotations++;
        add_inplace(env, ct, rotated, ops);
    }
    Ciphertext swapped;
    env.eval->rotate_columns(ct, env.gk, swapped);
    ops.rotations++;
    add_inplace(env, ct, swapped, ops);
    return {move(ct), ops};
}

inline Ciphertext broadcast_slot_ciphertext(
    const HEEnv &env,
    const Ciphertext &block,
    size_t slot_index,
    OpCount &ops)
{
    if (slot_index >= env.slot_count) {
        throw invalid_argument("slot index outside packed block");
    }

    Ciphertext rotated = block;
    if (slot_index != 0) {
        env.eval->rotate_rows(block, -static_cast<int>(slot_index), env.gk, rotated);
        ops.rotations++;
    }

    vector<uint64_t> mask(env.slot_count, 0);
    mask[0] = 1;
    Plaintext pt;
    env.batch->encode(mask, pt);

    Ciphertext isolated;
    env.eval->multiply_plain(rotated, pt, isolated);
    ops.plain_mul++;

    auto summed = sum_slots(move(isolated), env);
    ops += summed.second;
    return move(summed.first);
}

inline pair<vector<Ciphertext>, OpCount> broadcast_rows_from_blocks(
    const HEEnv &env,
    const vector<Ciphertext> &blocks,
    size_t rows)
{
    vector<Ciphertext> outputs;
    outputs.reserve(rows);
    OpCount ops;

    for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
        const size_t offset = block_idx * env.slot_count;
        if (offset >= rows) {
            break;
        }
        const size_t live = min(env.slot_count, rows - offset);
        for (size_t slot = 0; slot < live; ++slot) {
            outputs.push_back(broadcast_slot_ciphertext(env, blocks[block_idx], slot, ops));
        }
    }

    return {move(outputs), ops};
}

inline pair<Ciphertext, OpCount> parallel_product_tree(vector<Ciphertext> terms, const HEEnv &env)
{
    if (terms.empty()) {
        return {constant_ciphertext(env, 1), OpCount{}};
    }
    if (terms.size() == 1) {
        return {move(terms[0]), OpCount{}};
    }

    OpCount ops;
    int depth = 0;
    // Worker evaluation contexts are built ONCE for the whole tree. Previously a
    // fresh HEEnv (Encryptor/Evaluator/Decryptor/BatchEncoder, each with its own
    // NTT precomputation) was constructed per worker *per layer*; for wide trees
    // called in a loop that overhead dominated, making the threaded path up to
    // 2x slower than the sequential one.
    // Deliberately SEQUENTIAL. Splitting a single ciphertext's product tree
    // across threads is not a deployment-realistic execution model and it
    // distorts the measurement: per-worker HEEnv construction (Encryptor /
    // Evaluator / Decryptor / BatchEncoder, each with NTT precomputation)
    // dominated the actual homomorphic work, making the threaded path up to 2x
    // slower than the sequential one. Thread-level parallelism belongs at the
    // ciphertext/block granularity (see run_packed_aggregates_preencrypted),
    // where each thread owns whole ciphertexts and throughput scales cleanly.
    vector<HEEnv> worker_pool;   // intentionally left empty
    while (terms.size() > 1) {
        const size_t pairs = terms.size() / 2;
        vector<Ciphertext> next((terms.size() + 1) / 2);
        OpCount layer_ops;
        const size_t worker_count = min(worker_pool.size(), max<size_t>(1, pairs));

        if (worker_count <= 1 || pairs < 2) {
            for (size_t i = 0; i < pairs; ++i) {
                next[i] = multiply_ciphertexts(env, terms[2 * i], terms[2 * i + 1], layer_ops);
            }
        } else {
            struct ChunkResult {
                size_t begin_pair = 0;
                vector<Ciphertext> products;
                OpCount ops;
            };

            const size_t chunk = (pairs + worker_count - 1) / worker_count;
            vector<future<ChunkResult>> futures;
            futures.reserve(worker_count);
            for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
                const size_t begin_pair = worker_idx * chunk;
                const size_t end_pair = min(pairs, begin_pair + chunk);
                if (begin_pair >= end_pair) {
                    continue;
                }
                futures.push_back(async(launch::async, [&, worker_idx, begin_pair, end_pair]() {
                    const HEEnv &worker = worker_pool[worker_idx];
                    ChunkResult result;
                    result.begin_pair = begin_pair;
                    result.products.reserve(end_pair - begin_pair);
                    for (size_t pair_idx = begin_pair; pair_idx < end_pair; ++pair_idx) {
                        result.products.push_back(multiply_ciphertexts(
                            worker,
                            terms[2 * pair_idx],
                            terms[2 * pair_idx + 1],
                            result.ops));
                    }
                    return result;
                }));
            }

            for (auto &future_chunk : futures) {
                ChunkResult chunk_result = future_chunk.get();
                layer_ops += chunk_result.ops;
                for (size_t i = 0; i < chunk_result.products.size(); ++i) {
                    next[chunk_result.begin_pair + i] = move(chunk_result.products[i]);
                }
            }
        }

        if (terms.size() % 2 == 1) {
            next[pairs] = move(terms.back());
        }
        terms.swap(next);
        ops += layer_ops;
        depth++;
    }
    ops.depth = depth;
    return {move(terms[0]), ops};
}

inline Ciphertext linear_combo_2(
    const HEEnv &env,
    const Ciphertext &a,
    uint64_t ca,
    const Ciphertext &b,
    uint64_t cb,
    OpCount &ops)
{
    Ciphertext lhs = multiply_plain_const(env, a, ca, ops);
    Ciphertext rhs = multiply_plain_const(env, b, cb, ops);
    add_inplace(env, lhs, rhs, ops);
    return lhs;
}

inline vector<uint64_t> lagrange_coefficients(const vector<uint64_t> &values, uint64_t mod)
{
    const size_t n = values.size();
    vector<uint64_t> coeffs(n, 0);

    for (size_t i = 0; i < n; ++i) {
        if ((values[i] % mod) == 0) {
            continue;
        }

        vector<uint64_t> basis(1, 1);
        uint64_t denom = 1;
        for (size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            vector<uint64_t> next(basis.size() + 1, 0);
            for (size_t k = 0; k < basis.size(); ++k) {
                next[k] = mod_add(next[k], mod_sub(0, mod_mul(static_cast<uint64_t>(j), basis[k], mod), mod), mod);
                next[k + 1] = mod_add(next[k + 1], basis[k], mod);
            }
            basis.swap(next);
            denom = mod_mul(denom, mod_sub(static_cast<uint64_t>(i), static_cast<uint64_t>(j), mod), mod);
        }

        const uint64_t scale = mod_mul(values[i] % mod, mod_inv(denom, mod), mod);
        for (size_t k = 0; k < basis.size(); ++k) {
            coeffs[k] = mod_add(coeffs[k], mod_mul(scale, basis[k], mod), mod);
        }
    }

    while (coeffs.size() > 1 && coeffs.back() == 0) {
        coeffs.pop_back();
    }
    return coeffs;
}

inline vector<uint64_t> trim_coefficients(vector<uint64_t> coeffs)
{
    while (coeffs.size() > 1 && coeffs.back() == 0) {
        coeffs.pop_back();
    }
    return coeffs;
}

inline vector<uint64_t> scale_coefficients(vector<uint64_t> coeffs, uint64_t scale, uint64_t mod)
{
    for (uint64_t &coeff : coeffs) {
        coeff = mod_mul(coeff, scale, mod);
    }
    return trim_coefficients(move(coeffs));
}

template <typename Builder>
inline vector<uint64_t> cached_coefficients(const HEEnv &env, const string &key, Builder builder)
{
    if (!env.cache_enabled) {
        return builder();
    }
    {
        lock_guard<mutex> guard(env.cache->coeff_mutex);
        auto it = env.cache->coeff_cache.find(key);
        if (it != env.cache->coeff_cache.end()) {
            return it->second;
        }
    }

    vector<uint64_t> coeffs = builder();
    lock_guard<mutex> guard(env.cache->coeff_mutex);
    auto [it, inserted] = env.cache->coeff_cache.emplace(key, move(coeffs));
    return it->second;
}

template <typename Builder>
inline uint64_t cached_scalar(const HEEnv &env, const string &key, Builder builder)
{
    if (!env.cache_enabled) {
        return builder();
    }
    {
        lock_guard<mutex> guard(env.cache->scalar_mutex);
        auto it = env.cache->scalar_cache.find(key);
        if (it != env.cache->scalar_cache.end()) {
            return it->second;
        }
    }

    const uint64_t value = builder();
    lock_guard<mutex> guard(env.cache->scalar_mutex);
    auto [it, inserted] = env.cache->scalar_cache.emplace(key, value);
    return it->second;
}

template <typename Builder>
inline shared_ptr<const FactorPlan> cached_factor_plan(const HEEnv &env, const string &key, Builder builder)
{
    if (!env.cache_enabled) {
        return builder();
    }
    {
        lock_guard<mutex> guard(env.cache->plan_mutex);
        auto it = env.cache->factor_plan_cache.find(key);
        if (it != env.cache->factor_plan_cache.end()) {
            return it->second;
        }
    }

    shared_ptr<const FactorPlan> plan = builder();
    lock_guard<mutex> guard(env.cache->plan_mutex);
    auto [it, inserted] = env.cache->factor_plan_cache.emplace(key, move(plan));
    return it->second;
}

inline shared_ptr<const PolyEvalPlan> cached_poly_eval_plan(
    const HEEnv &env,
    const string &key,
    const vector<uint64_t> &coeffs)
{
    if (!env.cache_enabled) {
        auto plan = make_shared<PolyEvalPlan>();
        if (coeffs.empty()) {
            plan->block_constants = {0};
        } else if (coeffs.size() == 1) {
            plan->block_constants = {coeffs[0]};
        } else {
            const size_t degree = coeffs.size() - 1;
            plan->block_size = static_cast<size_t>(ceil(sqrt(static_cast<double>(degree + 1))));
            const size_t block_count = (coeffs.size() + plan->block_size - 1) / plan->block_size;
            plan->block_constants.assign(block_count, 0);
            plan->block_terms.resize(block_count);
            for (size_t block = 0; block < block_count; ++block) {
                const size_t base = block * plan->block_size;
                if (base < coeffs.size()) {
                    plan->block_constants[block] = coeffs[base];
                }
                for (size_t i = 1; i < plan->block_size && base + i < coeffs.size(); ++i) {
                    if (coeffs[base + i] == 0) {
                        continue;
                    }
                    plan->block_terms[block].push_back(PolyBlockTerm{
                        static_cast<uint16_t>(i),
                        coeffs[base + i]
                    });
                }
            }
        }
        return plan;
    }
    {
        lock_guard<mutex> guard(env.cache->plan_mutex);
        auto it = env.cache->poly_plan_cache.find(key);
        if (it != env.cache->poly_plan_cache.end()) {
            return it->second;
        }
    }

    auto plan = make_shared<PolyEvalPlan>();
    if (coeffs.empty()) {
        plan->block_constants = {0};
    } else if (coeffs.size() == 1) {
        plan->block_constants = {coeffs[0]};
    } else {
        const size_t degree = coeffs.size() - 1;
        plan->block_size = static_cast<size_t>(ceil(sqrt(static_cast<double>(degree + 1))));
        const size_t block_count = (coeffs.size() + plan->block_size - 1) / plan->block_size;
        plan->block_constants.assign(block_count, 0);
        plan->block_terms.resize(block_count);
        for (size_t block = 0; block < block_count; ++block) {
            const size_t base = block * plan->block_size;
            if (base < coeffs.size()) {
                plan->block_constants[block] = coeffs[base];
            }
            for (size_t i = 1; i < plan->block_size && base + i < coeffs.size(); ++i) {
                if (coeffs[base + i] == 0) {
                    continue;
                }
                plan->block_terms[block].push_back(PolyBlockTerm{
                    static_cast<uint16_t>(i),
                    coeffs[base + i]
                });
            }
        }
    }

    lock_guard<mutex> guard(env.cache->plan_mutex);
    auto [it, inserted] = env.cache->poly_plan_cache.emplace(key, move(plan));
    return it->second;
}

inline vector<uint64_t> multiply_linear_factors(const vector<uint64_t> &constants, uint64_t mod)
{
    vector<uint64_t> poly(1, 1);
    for (uint64_t constant : constants) {
        vector<uint64_t> next(poly.size() + 1, 0);
        for (size_t i = 0; i < poly.size(); ++i) {
            next[i] = mod_add(next[i], mod_mul(poly[i], constant, mod), mod);
            next[i + 1] = mod_add(next[i + 1], poly[i], mod);
        }
        poly.swap(next);
    }
    return trim_coefficients(move(poly));
}

inline uint64_t factorial_mod(size_t n, uint64_t mod)
{
    uint64_t value = 1;
    for (size_t i = 2; i <= n; ++i) {
        value = mod_mul(value, static_cast<uint64_t>(i), mod);
    }
    return value;
}

inline uint64_t factorial_inverse_cached(const HEEnv &env, size_t n)
{
    return cached_scalar(env, "factorial_inv:" + to_string(n), [&]() {
        return mod_inv(factorial_mod(n, env.plain_modulus), env.plain_modulus);
    });
}

// Deferred indicator normalization (HULK). The root-paired AND indicators end with a
// multiply by 1/K! mod t, a full-size (~log2 t bit) constant that multiplies the noise
// by ~t/2 and costs ~24 bits of budget. For an indicator whose result is FINAL (it only
// feeds the weight HMul, HAdd over blocks, the slot-sum and decryption), the composition
// evaluates it inside a HulkDeferScaleScope: the multiply is skipped, the ciphertext
// holds K!*I (a small integer), and the skipped constant is applied by the client to
// the decrypted aggregate (value * skipped mod t, exact). Indicators that feed a later
// sum (CSE parents, OR-of-AND columns) are never evaluated inside a scope.
// HULK_DEFER_SCALE=0 restores the in-circuit multiply.
struct HulkDeferScaleState {
    bool active = false;
    uint64_t skipped = 1;
};
inline HulkDeferScaleState &hulk_defer_scale_state()
{
    static thread_local HulkDeferScaleState state;
    return state;
}
inline bool hulk_defer_scale_enabled()
{
    const char *raw = getenv("HULK_DEFER_SCALE");
    return !(raw && string(raw) == "0");
}
class HulkDeferScaleScope {
public:
    explicit HulkDeferScaleScope(bool enable) : saved_(hulk_defer_scale_state())
    {
        auto &st = hulk_defer_scale_state();
        st.active = enable && hulk_defer_scale_enabled();
        st.skipped = 1;
    }
    ~HulkDeferScaleScope() { hulk_defer_scale_state() = saved_; }
    uint64_t skipped() const { return hulk_defer_scale_state().skipped; }
    HulkDeferScaleScope(const HulkDeferScaleScope &) = delete;
    HulkDeferScaleScope &operator=(const HulkDeferScaleScope &) = delete;
private:
    HulkDeferScaleState saved_;
};
inline Ciphertext hulk_scale_or_defer(const HEEnv &env, const Ciphertext &ct, uint64_t scale, OpCount &ops)
{
    auto &st = hulk_defer_scale_state();
    if (st.active) {
        st.skipped = mod_mul(st.skipped, scale, env.plain_modulus);
        return ct;
    }
    return multiply_plain_const(env, ct, scale, ops);
}

// 1 + scale*ct (negate: 1 - scale*ct), the nonzero-indicator epilogue. Deferred:
// (+/-)ct + scale^{-1}, i.e. the result divided by scale; the client multiplies by scale.
// The constant is only added (no noise growth); the sign flip is a negation.
inline Ciphertext hulk_one_plus_scaled_or_defer(const HEEnv &env, const Ciphertext &ct, uint64_t scale,
                                                bool negate, OpCount &ops)
{
    const uint64_t mod = env.plain_modulus;
    auto &st = hulk_defer_scale_state();
    if (st.active) {
        Ciphertext out = ct;
        if (negate) {
            env.eval->negate_inplace(out);
            ops.plain_add++;   // negation: counted with the cheap plaintext-side ops
        }
        add_plain_inplace(env, out, mod_inv(scale, mod), ops);
        st.skipped = mod_mul(st.skipped, scale, mod);
        return out;
    }
    Ciphertext out = multiply_plain_const(env, ct, negate ? mod_mul(scale, mod_sub(0, 1, mod), mod) : scale, ops);
    add_plain_inplace(env, out, 1, ops);
    return out;
}

inline vector<uint64_t> indicator_coefficients(const HEEnv &env, size_t max_sum, size_t target)
{
    const string key = "indicator:" + to_string(max_sum) + ":" + to_string(target);
    return cached_coefficients(env, key, [&]() {
        vector<uint64_t> values(max_sum + 1, 0);
        if (target <= max_sum) {
            values[target] = 1;
        }
        return lagrange_coefficients(values, env.plain_modulus);
    });
}

inline vector<uint64_t> nonzero_coefficients(const HEEnv &env, size_t max_sum)
{
    const string key = "nonzero:" + to_string(max_sum);
    return cached_coefficients(env, key, [&]() {
        vector<uint64_t> values(max_sum + 1, 1);
        if (!values.empty()) {
            values[0] = 0;
        }
        return lagrange_coefficients(values, env.plain_modulus);
    });
}

inline shared_ptr<const PolyEvalPlan> indicator_poly_plan(const HEEnv &env, size_t max_sum, size_t target)
{
    const string key = "indicator-plan:" + to_string(max_sum) + ":" + to_string(target);
    return cached_poly_eval_plan(env, key, indicator_coefficients(env, max_sum, target));
}

inline shared_ptr<const PolyEvalPlan> nonzero_poly_plan(const HEEnv &env, size_t max_sum)
{
    const string key = "nonzero-plan:" + to_string(max_sum);
    return cached_poly_eval_plan(env, key, nonzero_coefficients(env, max_sum));
}

inline pair<Ciphertext, OpCount> eval_poly_ps_streaming(
    const HEEnv &env,
    const Ciphertext &x,
    const PolyEvalPlan &plan);

inline pair<Ciphertext, OpCount> eval_poly_bsgs(const HEEnv &env, const Ciphertext &x, const vector<uint64_t> &coeffs_in);

// Polynomial evaluator for HULK indicators and bounded thresholds (audit H4/H5/G1).
//   HULK_POLY_EVAL unset/auto  depth-optimal eval_poly_bsgs; for AND/OR indicators at
//                              fan-in >= indicator_ps_min_fanin() the shallower of
//                              BSGS and the root-paired tree (ties: fewer HMul)
//   HULK_POLY_EVAL=bsgs        always BSGS on those paths (ablation)
//   HULK_POLY_EVAL=streaming   pre-audit behaviour: Horner-over-giant-steps PS
//                              (eval_poly_ps_streaming), reproduces earlier numbers
enum class HulkPolyEval { automatic, bsgs, streaming };
inline HulkPolyEval hulk_poly_eval()
{
    static const HulkPolyEval v = [] {
        const char *e = getenv("HULK_POLY_EVAL");
        if (!e || !*e) return HulkPolyEval::automatic;
        const string s(e);
        if (s == "streaming" || s == "ps") return HulkPolyEval::streaming;
        if (s == "bsgs") return HulkPolyEval::bsgs;
        if (s == "auto") return HulkPolyEval::automatic;
        throw invalid_argument("HULK_POLY_EVAL must be auto, bsgs or streaming");
    }();
    return v;
}

inline vector<uint64_t> bounded_ge_plain_coefficients(const HEEnv &env, size_t max_value, size_t threshold)
{
    const string key = "bounded-ge:" + to_string(max_value) + ":" + to_string(threshold);
    return cached_coefficients(env, key, [&]() {
        vector<uint64_t> values(max_value + 1, 0);
        for (size_t x = threshold; x <= max_value; ++x) {
            values[x] = 1;
        }
        return lagrange_coefficients(values, env.plain_modulus);
    });
}

inline vector<uint64_t> bounded_eq_coefficients(const HEEnv &env, size_t max_value, size_t target)
{
    const string key = "bounded-eq:" + to_string(max_value) + ":" + to_string(target);
    return cached_coefficients(env, key, [&]() {
        vector<uint64_t> values(max_value + 1, 0);
        if (target <= max_value) {
            values[target] = 1;
        }
        return lagrange_coefficients(values, env.plain_modulus);
    });
}

inline shared_ptr<const PolyEvalPlan> bounded_ge_plain_plan(const HEEnv &env, size_t max_value, size_t threshold)
{
    const string key = "bounded-ge-plan:" + to_string(max_value) + ":" + to_string(threshold);
    return cached_poly_eval_plan(env, key, bounded_ge_plain_coefficients(env, max_value, threshold));
}

inline shared_ptr<const PolyEvalPlan> bounded_eq_plan(const HEEnv &env, size_t max_value, size_t target)
{
    const string key = "bounded-eq-plan:" + to_string(max_value) + ":" + to_string(target);
    return cached_poly_eval_plan(env, key, bounded_eq_coefficients(env, max_value, target));
}

inline pair<Ciphertext, OpCount> he_ge_plain_bounded(
    const HEEnv &env,
    const Ciphertext &x,
    size_t max_value,
    size_t threshold)
{
    if (threshold > max_value + 1) {
        throw invalid_argument("threshold outside bounded compare range");
    }
    if (threshold == 0) {
        return {constant_ciphertext(env, 1), OpCount{}};
    }
    if (threshold > max_value) {
        return {constant_ciphertext(env, 0), OpCount{}};
    }
    if (hulk_poly_eval() == HulkPolyEval::streaming) {
        return eval_poly_ps_streaming(env, x, *bounded_ge_plain_plan(env, max_value, threshold));
    }
    return eval_poly_bsgs(env, x, bounded_ge_plain_coefficients(env, max_value, threshold));
}

inline pair<Ciphertext, OpCount> he_eq_plain_bounded(
    const HEEnv &env,
    const Ciphertext &x,
    size_t max_value,
    size_t target)
{
    if (target > max_value) {
        return {constant_ciphertext(env, 0), OpCount{}};
    }
    if (hulk_poly_eval() == HulkPolyEval::streaming) {
        return eval_poly_ps_streaming(env, x, *bounded_eq_plan(env, max_value, target));
    }
    return eval_poly_bsgs(env, x, bounded_eq_coefficients(env, max_value, target));
}

inline pair<Ciphertext, OpCount> he_ge_ciphertexts_small_domain(
    const HEEnv &env,
    const Ciphertext &lhs,
    const Ciphertext &rhs,
    size_t max_value)
{
    if (max_value > 64) {
        throw invalid_argument("small-domain ciphertext compare supports max_value <= 64");
    }

    OpCount ops;
    vector<Ciphertext> lhs_eq(max_value + 1);
    vector<Ciphertext> rhs_eq(max_value + 1);

    for (size_t i = 0; i <= max_value; ++i) {
        auto lhs_ind = he_eq_plain_bounded(env, lhs, max_value, i);
        auto rhs_ind = he_eq_plain_bounded(env, rhs, max_value, i);
        lhs_eq[i] = move(lhs_ind.first);
        rhs_eq[i] = move(rhs_ind.first);
        ops += lhs_ind.second;
        ops += rhs_ind.second;
    }

    bool have = false;
    Ciphertext out;
    for (size_t i = 0; i <= max_value; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            Ciphertext term = multiply_ciphertexts(env, lhs_eq[i], rhs_eq[j], ops);
            if (!have) {
                out = move(term);
                have = true;
            } else {
                add_inplace(env, out, term, ops);
            }
        }
    }
    if (!have) {
        out = constant_ciphertext(env, 0);
    }
    return {move(out), ops};
}

// 1 - x as negate + plaintext add (audit H2): no encryption, no plaintext multiply.
// Same cost for every caller in both modes.
inline Ciphertext one_minus_bit(const HEEnv &env, const Ciphertext &bit, OpCount &ops)
{
    Ciphertext out;
    env.eval->negate(bit, out);
    add_plain_inplace(env, out, 1, ops);
    return out;
}

inline pair<Ciphertext, OpCount> he_lt_plain_bounded(
    const HEEnv &env,
    const Ciphertext &x,
    size_t max_value,
    size_t threshold)
{
    if (threshold == 0) {
        return {constant_ciphertext(env, 0), OpCount{}};
    }
    if (threshold > max_value) {
        return {constant_ciphertext(env, 1), OpCount{}};
    }
    auto ge = he_ge_plain_bounded(env, x, max_value, threshold);
    Ciphertext out = one_minus_bit(env, ge.first, ge.second);
    return {move(out), ge.second};
}

inline pair<Ciphertext, OpCount> he_le_plain_bounded(
    const HEEnv &env,
    const Ciphertext &x,
    size_t max_value,
    size_t threshold)
{
    if (threshold >= max_value) {
        return {constant_ciphertext(env, 1), OpCount{}};
    }
    return he_lt_plain_bounded(env, x, max_value, threshold + 1);
}

inline pair<Ciphertext, OpCount> he_gt_plain_bounded(
    const HEEnv &env,
    const Ciphertext &x,
    size_t max_value,
    size_t threshold)
{
    if (threshold >= max_value) {
        return {constant_ciphertext(env, 0), OpCount{}};
    }
    return he_ge_plain_bounded(env, x, max_value, threshold + 1);
}

inline pair<Ciphertext, OpCount> he_lt_ciphertexts_small_domain(
    const HEEnv &env,
    const Ciphertext &lhs,
    const Ciphertext &rhs,
    size_t max_value)
{
    auto ge = he_ge_ciphertexts_small_domain(env, lhs, rhs, max_value);
    Ciphertext out = one_minus_bit(env, ge.first, ge.second);
    return {move(out), ge.second};
}

inline Ciphertext subtract_ciphertexts(const HEEnv &env, const Ciphertext &lhs, const Ciphertext &rhs, OpCount &ops)
{
    Ciphertext out;
    env.eval->sub(lhs, rhs, out);
    ops.ct_add++;
    return out;
}

inline Ciphertext he_select_by_bit(
    const HEEnv &env,
    const Ciphertext &bit,
    const Ciphertext &when_one,
    const Ciphertext &when_zero,
    OpCount &ops)
{
    Ciphertext diff = subtract_ciphertexts(env, when_one, when_zero, ops);
    Ciphertext scaled = multiply_ciphertexts(env, bit, diff, ops);
    Ciphertext out = when_zero;
    add_inplace(env, out, scaled, ops);
    return out;
}

inline vector<Ciphertext> derive_eq_plain_column(
    const HEEnv &env,
    const vector<Ciphertext> &source,
    size_t max_value,
    size_t target,
    OpCount &ops)
{
    vector<Ciphertext> out;
    out.reserve(source.size());
    for (const Ciphertext &block : source) {
        auto derived = he_eq_plain_bounded(env, block, max_value, target);
        ops += derived.second;
        out.push_back(move(derived.first));
    }
    return out;
}

inline vector<Ciphertext> derive_ge_plain_column(
    const HEEnv &env,
    const vector<Ciphertext> &source,
    size_t max_value,
    size_t threshold,
    OpCount &ops)
{
    vector<Ciphertext> out;
    out.reserve(source.size());
    for (const Ciphertext &block : source) {
        auto derived = he_ge_plain_bounded(env, block, max_value, threshold);
        ops += derived.second;
        out.push_back(move(derived.first));
    }
    return out;
}

inline vector<Ciphertext> derive_gt_plain_column(
    const HEEnv &env,
    const vector<Ciphertext> &source,
    size_t max_value,
    size_t threshold,
    OpCount &ops)
{
    vector<Ciphertext> out;
    out.reserve(source.size());
    for (const Ciphertext &block : source) {
        auto derived = he_gt_plain_bounded(env, block, max_value, threshold);
        ops += derived.second;
        out.push_back(move(derived.first));
    }
    return out;
}

inline vector<Ciphertext> derive_lt_plain_column(
    const HEEnv &env,
    const vector<Ciphertext> &source,
    size_t max_value,
    size_t threshold,
    OpCount &ops)
{
    vector<Ciphertext> out;
    out.reserve(source.size());
    for (const Ciphertext &block : source) {
        auto derived = he_lt_plain_bounded(env, block, max_value, threshold);
        ops += derived.second;
        out.push_back(move(derived.first));
    }
    return out;
}

inline vector<Ciphertext> derive_le_plain_column(
    const HEEnv &env,
    const vector<Ciphertext> &source,
    size_t max_value,
    size_t threshold,
    OpCount &ops)
{
    vector<Ciphertext> out;
    out.reserve(source.size());
    for (const Ciphertext &block : source) {
        auto derived = he_le_plain_bounded(env, block, max_value, threshold);
        ops += derived.second;
        out.push_back(move(derived.first));
    }
    return out;
}

inline vector<Ciphertext> derive_lt_ciphertext_column_small_domain(
    const HEEnv &env,
    const vector<Ciphertext> &lhs,
    const vector<Ciphertext> &rhs,
    size_t max_value,
    OpCount &ops)
{
    if (lhs.size() != rhs.size()) {
        throw invalid_argument("ciphertext columns must have the same block count");
    }
    vector<Ciphertext> out;
    out.reserve(lhs.size());
    for (size_t block = 0; block < lhs.size(); ++block) {
        auto derived = he_lt_ciphertexts_small_domain(env, lhs[block], rhs[block], max_value);
        ops += derived.second;
        out.push_back(move(derived.first));
    }
    return out;
}

inline pair<vector<Ciphertext>, OpCount> he_sort_desc_small_domain(
    const HEEnv &env,
    vector<Ciphertext> scores,
    size_t max_value)
{
    OpCount ops;
    if (scores.size() <= 1) {
        return {move(scores), ops};
    }

    // Small-N encrypted compare-exchange network (bubble-sort style).
    for (size_t pass = 0; pass + 1 < scores.size(); ++pass) {
        for (size_t i = 0; i + 1 < scores.size() - pass; ++i) {
            auto ge = he_ge_ciphertexts_small_domain(env, scores[i], scores[i + 1], max_value);
            Ciphertext bit_ge = move(ge.first);
            ops += ge.second;
            Ciphertext bit_lt = one_minus_bit(env, bit_ge, ops);

            Ciphertext hi = he_select_by_bit(env, bit_ge, scores[i], scores[i + 1], ops);
            Ciphertext lo = he_select_by_bit(env, bit_lt, scores[i], scores[i + 1], ops);
            scores[i] = move(hi);
            scores[i + 1] = move(lo);
        }
    }
    return {move(scores), ops};
}

inline pair<vector<Ciphertext>, OpCount> he_topk_desc_small_domain_streaming(
    const HEEnv &env,
    const vector<Ciphertext> &scores,
    size_t k,
    size_t max_value)
{
    if (k == 0 || scores.empty()) {
        return {{}, OpCount{}};
    }

    const size_t keep = min(k, scores.size());
    OpCount ops;
    vector<Ciphertext> topk(keep, constant_ciphertext(env, 0));
    for (const Ciphertext &score : scores) {
        Ciphertext carry = score;
        for (size_t pos = 0; pos < keep; ++pos) {
            auto ge = he_ge_ciphertexts_small_domain(env, carry, topk[pos], max_value);
            ops += ge.second;

            const Ciphertext hi = he_select_by_bit(env, ge.first, carry, topk[pos], ops);
            const Ciphertext lo = he_select_by_bit(env, ge.first, topk[pos], carry, ops);
            topk[pos] = hi;
            carry = lo;
        }
    }
    return {move(topk), ops};
}

inline pair<vector<Ciphertext>, OpCount> he_topk_desc_small_domain(
    const HEEnv &env,
    const vector<Ciphertext> &scores,
    size_t k,
    size_t max_value)
{
    if (k == 0 || scores.empty()) {
        return {{}, OpCount{}};
    }
    if (scores.size() > 32) {
        return he_topk_desc_small_domain_streaming(env, scores, k, max_value);
    }

    auto sorted = he_sort_desc_small_domain(env, scores, max_value);
    vector<Ciphertext> topk;
    topk.reserve(min(k, sorted.first.size()));
    for (size_t i = 0; i < min(k, sorted.first.size()); ++i) {
        topk.push_back(move(sorted.first[i]));
    }
    return {move(topk), sorted.second};
}

inline pair<Ciphertext, OpCount> eval_poly_ps_streaming(
    const HEEnv &env,
    const Ciphertext &x,
    const PolyEvalPlan &plan)
{
    OpCount ops;
    if (plan.block_constants.empty()) {
        return {constant_ciphertext(env, 0), ops};
    }
    if (plan.block_constants.size() == 1 && plan.block_terms.empty()) {
        return {constant_ciphertext(env, plan.block_constants[0]), ops};
    }

    const size_t block_size = plan.block_size;
    vector<Ciphertext> small_powers(block_size);
    // Track the HMul depth of each baby-step power for accurate reporting.
    vector<int> baby_depths(block_size, 0);
    small_powers[0] = constant_ciphertext(env, 1);  // x^0 = 1
    if (block_size > 1) {
        small_powers[1] = x;  // x^1 = x (depth 0)
    }
    for (size_t i = 2; i < block_size; ++i) {
        if (i == 2 || (i & (i - 1)) == 0) {
            // Power of two: x^{2j} = sq(x^j) — always one squaring deeper.
            small_powers[i] = square_ciphertext(env, small_powers[i >> 1], ops);
            baby_depths[i] = baby_depths[i >> 1] + 1;
        } else {
            // Non-power-of-two: split at the highest set bit for minimum depth.
            // x^i = x^{hi} * x^{i-hi} where hi = floor(log2(i)) rounded to a power of 2.
            // This achieves depth max(d[hi], d[i-hi])+1, which is always ≤ depth of
            // the naive sequential chain x^{i-1}*x = d[i-1]+1.
            size_t hi = 2;
            while (hi * 2 <= i) {
                hi *= 2;
            }
            small_powers[i] = multiply_ciphertexts(env, small_powers[hi], small_powers[i - hi], ops);
            baby_depths[i] = max(baby_depths[hi], baby_depths[i - hi]) + 1;
        }
    }

    // Giant step: x^{block_size}.
    // When block_size is a power of two, sq(x^{block_size/2}) saves one depth level
    // compared to x^{block_size-1}*x.
    Ciphertext giant;
    int giant_depth = 0;
    if (block_size == 1) {
        giant = x;
        giant_depth = 0;
    } else if ((block_size & (block_size - 1)) == 0) {
        // block_size is a power of two: x^{block_size} = sq(x^{block_size/2}).
        giant = square_ciphertext(env, small_powers[block_size / 2], ops);
        giant_depth = baby_depths[block_size / 2] + 1;
    } else {
        // General case: x^{block_size} = x^{block_size-1} * x.
        giant = multiply_ciphertexts(env, small_powers[block_size - 1], x, ops);
        giant_depth = baby_depths[block_size - 1] + 1;
    }

    const size_t block_count = plan.block_constants.size();
    auto build_block = [&](size_t block) {
        bool have_term = false;
        Ciphertext acc;

        if (plan.block_constants[block] != 0) {
            acc = constant_ciphertext(env, plan.block_constants[block]);
            have_term = true;
        }

        for (const PolyBlockTerm &term_spec : plan.block_terms[block]) {
            Ciphertext term = multiply_plain_const(env, small_powers[term_spec.power], term_spec.coeff, ops);
            if (!have_term) {
                acc = move(term);
                have_term = true;
            } else {
                add_inplace(env, acc, term, ops);
            }
        }

        if (!have_term) {
            acc = constant_ciphertext(env, 0);
        }
        return acc;
    };

    // Horner evaluation: result = giant * (... giant * (giant * g_{N-1} + g_{N-2}) + ...) + g_0
    // After k Horner steps the result depth = giant_depth + k (since giant_depth > 0 after
    // the first step and each subsequent multiply adds exactly 1 level).
    Ciphertext result = build_block(block_count - 1);
    int result_depth = 0;
    for (size_t idx = block_count; idx-- > 1;) {
        result = multiply_ciphertexts(env, result, giant, ops);
        result_depth++;
        Ciphertext next_block = build_block(idx - 1);
        add_inplace(env, result, next_block, ops);
    }
    // Actual depth = giant_depth + Horner steps (result_depth).
    // The old formula used (block_size-1) as baby depth, which overestimated by ~2x
    // since squarings achieve ceil(log2(block_size)) depth, not block_size-1.
    ops.depth = max(ops.depth, giant_depth + result_depth);
    return {move(result), ops};
}

// Depth-optimal baby-step/giant-step evaluation of sum_i c_i x^i.
// Baby powers x^1..x^k (k = 2^a >= sqrt(d+1)) at depth ceil(log2 j); giant powers
// G_i = x^{k 2^i} by squaring; p is split recursively as p = r + G_{L-1} * q.
// Depth = a + m ~ ceil(log2 d) + 1, versus giant_depth + (#blocks - 1) for the
// Horner-over-giant-steps form of eval_poly_ps_streaming (depth ~ sqrt(d)).
// Nonscalar multiplications ~ k + m + d/k, the same order as Paterson-Stockmeyer.
// A sub-polynomial that is a bare constant stays a plaintext scalar (audit G9):
// q scalar -> multiply_plain on G (no HMul, no extra level); r scalar -> add_plain.
struct BsgsShape {
    size_t d = 0, k = 1, kb = 0, m = 0;
};

inline BsgsShape bsgs_shape(size_t d)
{
    BsgsShape s; s.d = d;
    size_t a = 0; while ((size_t(1) << (2 * a)) < d + 1) ++a;
    s.k = size_t(1) << a;
    s.kb = min(s.k, d);                                            // baby powers actually needed
    while ((s.k << s.m) <= d) ++s.m;                               // k 2^m > d
    return s;
}

inline vector<uint64_t> bsgs_normalise(const vector<uint64_t> &coeffs_in, uint64_t p)
{
    vector<uint64_t> c(coeffs_in.size()); for (size_t i=0;i<c.size();++i) c[i]=coeffs_in[i]%p;
    while (c.size()>1 && c.back()==0) c.pop_back();
    return c;
}

// Exact (depth, nonscalar-multiplication) cost of eval_poly_bsgs on these
// coefficients, without evaluating anything. Mirrors eval_poly_bsgs step by step;
// test_bsgs checks that the two agree.
struct PolyCost { int depth = 0; int muls = 0; };

inline PolyCost bsgs_cost(const vector<uint64_t> &coeffs_in, uint64_t p)
{
    PolyCost cost;
    const vector<uint64_t> c = bsgs_normalise(coeffs_in, p);
    if (c.size() <= 1) return cost;
    const BsgsShape sh = bsgs_shape(c.size()-1);
    vector<int> bd(sh.k+1,0);
    for (size_t j=2;j<=sh.kb;++j) { size_t hi=1; while (hi*2<j) hi*=2; bd[j]=max(bd[hi],bd[j-hi])+1; cost.muls++; }
    vector<int> gd(sh.m,0);
    if (sh.m>0) { gd[0]=bd[sh.k]; for (size_t i=1;i<sh.m;++i){ gd[i]=gd[i-1]+1; cost.muls++; } }
    // kind: 0 none, 1 scalar, 2 ciphertext (with depth)
    function<pair<int,int>(size_t,size_t,size_t)> rec = [&](size_t lo, size_t len, size_t level) -> pair<int,int> {
        if (level==0) {
            bool ct=false, sc=false; int dep=0;
            for (size_t j=0;j<len && lo+j<c.size();++j) { if (!c[lo+j]) continue; if (j==0) sc=true; else { ct=true; dep=max(dep,bd[j]); } }
            return ct ? pair<int,int>{2,dep} : pair<int,int>{sc?1:0,0};
        }
        const size_t half = sh.k<<(level-1);
        if (len<=half) return rec(lo,len,level-1);
        auto r=rec(lo,half,level-1); auto q=rec(lo+half,len-half,level-1);
        if (q.first==0) return r;
        int dep = (q.first==1) ? gd[level-1] : max(q.second,gd[level-1])+1;
        if (q.first==2) cost.muls++;
        if (r.first==2) dep=max(dep,r.second);
        return {2,dep};
    };
    auto res=rec(0,c.size(),sh.m);
    cost.depth = res.first==2 ? res.second : 0;
    return cost;
}

// Baby powers x^1..x^kb and giant powers G_i = x^{k 2^i} of one ciphertext, for the
// BSGS shape of degree d. Several polynomials in the SAME ciphertext x (e.g. the
// predicates of one column: item = 0, item = 1, ...) can share one BsgsPowers: each
// then costs only its giant-step combination (see eval_poly_bsgs_with).
struct BsgsPowers {
    BsgsShape sh;
    vector<Ciphertext> B, G;
    vector<int> bd, gd;
};

inline BsgsPowers bsgs_powers(const HEEnv &env, const Ciphertext &x, size_t d, OpCount &ops)
{
    BsgsPowers P;
    P.sh = bsgs_shape(d);
    const size_t k = P.sh.k, kb = P.sh.kb, m = P.sh.m;
    // x^j for j <= kb; when m > 0 (d >= k) kb == k and x^k is also the first giant power.
    P.B.resize(k+1); P.bd.assign(k+1,0); P.B[1]=x;
    for (size_t j=2;j<=kb;++j) {
        size_t hi=1; while (hi*2<j) hi*=2;                        // largest power of two < j
        if (j==2*hi) P.B[j]=square_ciphertext(env,P.B[hi],ops); else P.B[j]=multiply_ciphertexts(env,P.B[hi],P.B[j-hi],ops);
        P.bd[j]=max(P.bd[hi],P.bd[j-hi])+1;
    }
    P.G.resize(m); P.gd.assign(m,0);
    if (m>0) { P.G[0]=P.B[k]; P.gd[0]=P.bd[k]; for (size_t i=1;i<m;++i){ P.G[i]=square_ciphertext(env,P.G[i-1],ops); P.gd[i]=P.gd[i-1]+1; } }
    return P;
}

// Evaluate sum_i c_i x^i from precomputed powers. The powers must cover the polynomial:
// normalised degree <= P.sh.d (a smaller degree descends the same recursion).
inline pair<Ciphertext, OpCount> eval_poly_bsgs_with(const HEEnv &env, const BsgsPowers &P, const vector<uint64_t> &coeffs_in)
{
    OpCount ops;
    const vector<uint64_t> c = bsgs_normalise(coeffs_in, env.plain_modulus);
    if (c.empty()) return {constant_ciphertext(env,0), ops};
    if (c.size()==1) return {constant_ciphertext(env,c[0]), ops};
    if (c.size()-1 > P.sh.d) throw invalid_argument("eval_poly_bsgs_with: polynomial degree exceeds the shared powers");
    const size_t k = P.sh.k, m = P.sh.m;
    const auto &B = P.B; const auto &bd = P.bd; const auto &G = P.G; const auto &gd = P.gd;
    struct Node { int kind = 0; uint64_t scalar = 0; Ciphertext ct; int dep = 0; };   // 0 none, 1 scalar, 2 ct
    function<Node(size_t,size_t,size_t)> rec = [&](size_t lo, size_t len, size_t level) -> Node {
        Node out;
        if (level==0) {
            bool have=false; uint64_t c0=0;
            for (size_t j=0;j<len && lo+j<c.size();++j) {
                const uint64_t cj=c[lo+j]; if (!cj) continue;
                if (j==0) { c0=cj; continue; }
                Ciphertext t=multiply_plain_const(env,B[j],cj,ops); out.dep=max(out.dep,bd[j]);
                if (!have) { out.ct=move(t); have=true; } else add_inplace(env,out.ct,t,ops);
            }
            if (have) { out.kind=2; if (c0) add_plain_inplace(env,out.ct,c0,ops); }
            else if (c0) { out.kind=1; out.scalar=c0; }
            return out;
        }
        const size_t half = k<<(level-1);
        if (len<=half) return rec(lo,len,level-1);
        Node r=rec(lo,half,level-1); Node q=rec(lo+half,len-half,level-1);
        if (q.kind==0) return r;
        if (q.kind==1) { out.ct=multiply_plain_const(env,G[level-1],q.scalar,ops); out.dep=gd[level-1]; }
        else { out.ct=multiply_ciphertexts(env,q.ct,G[level-1],ops); out.dep=max(q.dep,gd[level-1])+1; }
        out.kind=2;
        if (r.kind==1) add_plain_inplace(env,out.ct,r.scalar,ops);
        else if (r.kind==2) { add_inplace(env,out.ct,r.ct,ops); out.dep=max(out.dep,r.dep); }
        return out;
    };
    Node res=rec(0,c.size(),m);
    if (res.kind==0) return {constant_ciphertext(env,0), ops};
    if (res.kind==1) return {constant_ciphertext(env,res.scalar), ops};
    ops.depth=res.dep;
    return {move(res.ct), ops};
}

inline pair<Ciphertext, OpCount> eval_poly_bsgs(const HEEnv &env, const Ciphertext &x, const vector<uint64_t> &coeffs_in)
{
    const vector<uint64_t> c = bsgs_normalise(coeffs_in, env.plain_modulus);
    if (c.size() <= 1) return eval_poly_bsgs_with(env, BsgsPowers{}, c);
    OpCount ops;
    const BsgsPowers P = bsgs_powers(env, x, c.size()-1, ops);
    auto r = eval_poly_bsgs_with(env, P, c);
    r.second += ops;
    return r;
}

inline pair<Ciphertext, OpCount> eval_poly_ps_streaming(
    const HEEnv &env,
    const Ciphertext &x,
    const vector<uint64_t> &coeffs)
{
    PolyEvalPlan plan;
    if (coeffs.empty()) {
        plan.block_constants = {0};
    } else if (coeffs.size() == 1) {
        plan.block_constants = {coeffs[0]};
    } else {
        const size_t degree = coeffs.size() - 1;
        plan.block_size = static_cast<size_t>(ceil(sqrt(static_cast<double>(degree + 1))));
        const size_t block_count = (coeffs.size() + plan.block_size - 1) / plan.block_size;
        plan.block_constants.assign(block_count, 0);
        plan.block_terms.resize(block_count);
        for (size_t block = 0; block < block_count; ++block) {
            const size_t base = block * plan.block_size;
            if (base < coeffs.size()) {
                plan.block_constants[block] = coeffs[base];
            }
            for (size_t i = 1; i < plan.block_size && base + i < coeffs.size(); ++i) {
                if (coeffs[base + i] == 0) {
                    continue;
                }
                plan.block_terms[block].push_back(PolyBlockTerm{
                    static_cast<uint16_t>(i),
                    coeffs[base + i]
                });
            }
        }
    }
    return eval_poly_ps_streaming(env, x, plan);
}

inline pair<Ciphertext, OpCount> arithmetic_all_true(const vector<Ciphertext> &bits, const HEEnv &env)
{
    if (bits.empty()) {
        throw invalid_argument("arithmetic_all_true needs at least one bit");
    }
    if (bits.size() == 1) {
        return {bits[0], OpCount{}};
    }

    // Balanced binary HMul tree: AND(a,b) = a × b (one full ciphertext multiplication per pair).
    // This is the correct arithmetic baseline — uses HMul not squarings.
    // Squarings exploit the bit-idempotent identity (x²=x for {0,1} inputs), which is the same
    // algebraic structure HULK's indicator polynomial uses. Applying squarings here would make
    // arithmetic a HULK-style evaluation, obscuring the comparison.
    vector<Ciphertext> layer = bits;
    OpCount ops;
    int depth = 0;
    while (layer.size() > 1) {
        vector<Ciphertext> next;
        next.reserve((layer.size() + 1) / 2);
        for (size_t i = 0; i + 1 < layer.size(); i += 2) {
            next.push_back(multiply_ciphertexts(env, layer[i], layer[i + 1], ops));
        }
        if (layer.size() % 2 == 1) {
            next.push_back(layer.back());
        }
        layer.swap(next);
        depth++;
    }
    ops.depth = depth;
    return {move(layer[0]), ops};
}

inline pair<Ciphertext, OpCount> arithmetic_any_true(const vector<Ciphertext> &bits, const HEEnv &env)
{
    vector<Ciphertext> inverted;
    inverted.reserve(bits.size());
    OpCount ops;
    for (const Ciphertext &bit : bits) {
        inverted.push_back(one_minus_bit(env, bit, ops));   // negate + add_plain(1)
    }
    auto prod = arithmetic_all_true(inverted, env);
    ops += prod.second;
    Ciphertext result = one_minus_bit(env, prod.first, ops);
    ops.depth = prod.second.depth;
    return {move(result), ops};
}

inline pair<Ciphertext, OpCount> sum_bits(const vector<Ciphertext> &bits, const HEEnv &env)
{
    return balanced_sum(bits, env);
}

inline vector<uint64_t> paired_all_true_even_constants(const HEEnv &env, size_t max_sum)
{
    const string key = "paired_all_true_even_constants:" + to_string(max_sum);
    return cached_coefficients(env, key, [&]() {
        vector<uint64_t> constants;
        constants.reserve(max_sum / 2);
        for (size_t i = 0; i < max_sum / 2; ++i) {
            constants.push_back(mod_mul(
                static_cast<uint64_t>(i),
                static_cast<uint64_t>(max_sum - 1 - i),
                env.plain_modulus));
        }
        return constants;
    });
}

inline vector<uint64_t> paired_all_true_odd_constants(const HEEnv &env, size_t max_sum)
{
    const string key = "paired_all_true_odd_constants:" + to_string(max_sum);
    return cached_coefficients(env, key, [&]() {
        const size_t radius = max_sum / 2;
        vector<uint64_t> constants;
        constants.reserve(radius);
        for (size_t i = 1; i <= radius; ++i) {
            constants.push_back(mod_sub(0, mod_mul(
                static_cast<uint64_t>(i),
                static_cast<uint64_t>(i),
                env.plain_modulus), env.plain_modulus));
        }
        return constants;
    });
}

inline vector<uint64_t> paired_nonzero_even_constants(const HEEnv &env, size_t max_sum)
{
    const string key = "paired_nonzero_even_constants:" + to_string(max_sum);
    return cached_coefficients(env, key, [&]() {
        vector<uint64_t> constants;
        constants.reserve(max_sum / 2);
        for (size_t i = 1; i <= max_sum / 2; ++i) {
            constants.push_back(mod_mul(
                static_cast<uint64_t>(i),
                static_cast<uint64_t>(max_sum + 1 - i),
                env.plain_modulus));
        }
        return constants;
    });
}

inline vector<uint64_t> paired_nonzero_odd_constants(const HEEnv &env, size_t max_sum)
{
    const string key = "paired_nonzero_odd_constants:" + to_string(max_sum);
    return cached_coefficients(env, key, [&]() {
        const size_t radius = max_sum / 2;
        vector<uint64_t> constants;
        constants.reserve(radius);
        for (size_t i = 1; i <= radius; ++i) {
            constants.push_back(mod_sub(0, mod_mul(
                static_cast<uint64_t>(i),
                static_cast<uint64_t>(i),
                env.plain_modulus), env.plain_modulus));
        }
        return constants;
    });
}

inline shared_ptr<const FactorPlan> paired_all_true_even_plan(const HEEnv &env, size_t max_sum)
{
    const string key = "paired-all-true-even-plan:" + to_string(max_sum);
    return cached_factor_plan(env, key, [&]() {
        auto plan = make_shared<FactorPlan>();
        plan->constants = paired_all_true_even_constants(env, max_sum);
        plan->scale = factorial_inverse_cached(env, max_sum);
        return shared_ptr<const FactorPlan>(move(plan));
    });
}

inline shared_ptr<const FactorPlan> paired_all_true_odd_plan(const HEEnv &env, size_t max_sum)
{
    const string key = "paired-all-true-odd-plan:" + to_string(max_sum);
    return cached_factor_plan(env, key, [&]() {
        auto plan = make_shared<FactorPlan>();
        plan->constants = paired_all_true_odd_constants(env, max_sum);
        plan->scale = factorial_inverse_cached(env, max_sum);
        return shared_ptr<const FactorPlan>(move(plan));
    });
}

inline shared_ptr<const FactorPlan> paired_nonzero_even_plan(const HEEnv &env, size_t max_sum)
{
    const string key = "paired-nonzero-even-plan:" + to_string(max_sum);
    return cached_factor_plan(env, key, [&]() {
        auto plan = make_shared<FactorPlan>();
        plan->constants = paired_nonzero_even_constants(env, max_sum);
        plan->scale = factorial_inverse_cached(env, max_sum);
        return shared_ptr<const FactorPlan>(move(plan));
    });
}

inline shared_ptr<const FactorPlan> paired_nonzero_odd_plan(const HEEnv &env, size_t max_sum)
{
    const string key = "paired-nonzero-odd-plan:" + to_string(max_sum);
    return cached_factor_plan(env, key, [&]() {
        auto plan = make_shared<FactorPlan>();
        plan->constants = paired_nonzero_odd_constants(env, max_sum);
        plan->scale = factorial_inverse_cached(env, max_sum);
        return shared_ptr<const FactorPlan>(move(plan));
    });
}

inline pair<Ciphertext, OpCount> eval_poly_compact(
    const HEEnv &env,
    const Ciphertext &x,
    const vector<uint64_t> &coeffs)
{
    return eval_poly_ps_streaming(env, x, coeffs);
}

inline pair<Ciphertext, OpCount> hulk_indicator_large_all_true(
    const Ciphertext &sum,
    size_t max_sum,
    const HEEnv &env)
{
    OpCount ops;
    const uint64_t mod = env.plain_modulus;

    if (max_sum % 2 == 0) {
        Ciphertext x2 = square_ciphertext(env, sum, ops);
        Ciphertext y = linear_combo_2(env, x2, 1, sum, mod_sub(0, max_sum - 1, mod), ops);
        const auto plan = paired_all_true_even_plan(env, max_sum);
        vector<Ciphertext> factors;
        factors.reserve(max_sum / 2);
        for (uint64_t constant : plan->constants) {
            Ciphertext factor = y;
            add_plain_inplace(env, factor, constant, ops);
            factors.push_back(move(factor));
        }
        auto product = parallel_product_tree(move(factors), env);
        ops += product.second;
        Ciphertext result = hulk_scale_or_defer(env, product.first, plan->scale, ops);
        ops.depth = max(ops.depth, product.second.depth + 1);
        return {move(result), ops};
    }

    const uint64_t center = static_cast<uint64_t>(max_sum / 2);
    Ciphertext z = sum;
    add_plain_inplace(env, z, mod_sub(0, center, mod), ops);
    Ciphertext w = square_ciphertext(env, z, ops);
    const auto plan = paired_all_true_odd_plan(env, max_sum);
    vector<Ciphertext> factors;
    factors.reserve(max_sum / 2);
    for (uint64_t constant : plan->constants) {
        Ciphertext factor = w;
        add_plain_inplace(env, factor, constant, ops);
        factors.push_back(move(factor));
    }
    auto product = parallel_product_tree(move(factors), env);
    ops += product.second;
    Ciphertext result = multiply_ciphertexts(env, product.first, z, ops);
    result = hulk_scale_or_defer(env, result, plan->scale, ops);
    ops.depth = max(ops.depth, product.second.depth + 2);
    return {move(result), ops};
}

inline pair<Ciphertext, OpCount> hulk_nonzero_large(
    const Ciphertext &sum,
    size_t max_sum,
    const HEEnv &env)
{
    OpCount ops;
    const uint64_t mod = env.plain_modulus;

    if (max_sum % 2 == 0) {
        Ciphertext x2 = square_ciphertext(env, sum, ops);
        Ciphertext y = linear_combo_2(env, x2, 1, sum, mod_sub(0, max_sum + 1, mod), ops);
        const auto plan = paired_nonzero_even_plan(env, max_sum);
        vector<Ciphertext> factors;
        factors.reserve(max_sum / 2);
        for (uint64_t constant : plan->constants) {
            Ciphertext factor = y;
            add_plain_inplace(env, factor, constant, ops);
            factors.push_back(move(factor));
        }
        auto product = parallel_product_tree(move(factors), env);
        ops += product.second;
        Ciphertext result = hulk_one_plus_scaled_or_defer(env, product.first, plan->scale, true, ops);
        ops.depth = max(ops.depth, product.second.depth + 1);
        return {move(result), ops};
    }

    const uint64_t center = static_cast<uint64_t>((max_sum + 1) / 2);
    Ciphertext z = sum;
    add_plain_inplace(env, z, mod_sub(0, center, mod), ops);
    Ciphertext w = square_ciphertext(env, z, ops);
    const auto plan = paired_nonzero_odd_plan(env, max_sum);
    vector<Ciphertext> factors;
    factors.reserve(max_sum / 2);
    for (uint64_t constant : plan->constants) {
        Ciphertext factor = w;
        add_plain_inplace(env, factor, constant, ops);
        factors.push_back(move(factor));
    }
    auto product = parallel_product_tree(move(factors), env);
    ops += product.second;
    Ciphertext root = multiply_ciphertexts(env, product.first, z, ops);
    root = hulk_one_plus_scaled_or_defer(env, root, plan->scale, false, ops);
    ops.depth = max(ops.depth, product.second.depth + 2);
    return {move(root), ops};
}

// Root-paired tree vs polynomial evaluation for the AND/OR indicators (K = fan-in).
// Root-paired (K >= 9): even K -> K/2 HMul at depth 1+ceil(log2(K/2));
//                       odd K  -> floor(K/2)+1 HMul at depth 2+ceil(log2(floor(K/2))).
// eval_poly_bsgs on the same degree-K polynomial: depth ~ceil(log2 K)+1, see bsgs_cost()
// (depth/HMul, BSGS vs root-paired: K=16 4/9 vs 4/8; K=19 5/10 vs 6/10; K=20 5/10 vs
// 5/10; K=40 6/13 vs 6/20; below 17 BSGS is also shallower at K=11,13,15, which
// HULK_INDICATOR_PS_MIN=9 enables; the default 17 keeps K<=16 unchanged).
// The old Horner-over-giant-steps PS (HULK_POLY_EVAL=streaming) was much deeper
// (K=16: depth 6, K=40: depth 9). From this fan-in on, HULK_POLY_EVAL=auto takes the
// shallower of root-paired and BSGS (ties -> fewer HMul); below it root-paired is kept.
inline size_t indicator_ps_min_fanin()
{
    static const size_t v = [] {
        const char *e = getenv("HULK_INDICATOR_PS_MIN");
        return e ? static_cast<size_t>(strtoul(e, nullptr, 10)) : size_t{17};
    }();
    return v;
}

// (depth, HMul) of hulk_indicator_large_all_true / hulk_nonzero_large for fan-in K >= 2
// (both have the same shape: one square, a product tree over the paired factors and,
// for odd K, one extra HMul by the centred root).
inline PolyCost root_paired_cost(size_t max_sum)
{
    auto clog2 = [](size_t n) { int r = 0; while ((size_t(1) << r) < n) ++r; return r; };
    const size_t pairs = max_sum / 2;
    PolyCost c;
    if (max_sum % 2 == 0) { c.depth = 1 + clog2(pairs); c.muls = static_cast<int>(pairs); }
    else                  { c.depth = 2 + clog2(pairs); c.muls = static_cast<int>(pairs) + 1; }
    return c;
}

inline bool prefer_bsgs_over_root_paired(const HEEnv &env, const vector<uint64_t> &coeffs, size_t max_sum)
{
    const PolyCost b = bsgs_cost(coeffs, env.plain_modulus);
    const PolyCost r = root_paired_cost(max_sum);
    return b.depth < r.depth || (b.depth == r.depth && b.muls < r.muls);
}

inline pair<Ciphertext, OpCount> hulk_indicator_from_sum(
    const Ciphertext &sum,
    size_t max_sum,
    size_t target,
    const HEEnv &env)
{
    OpCount ops;
    const uint64_t mod = env.plain_modulus;

    if (max_sum == 1) {
        return {sum, ops};
    }
    if (max_sum == 2 && target == 2) {
        Ciphertext x2 = square_ciphertext(env, sum, ops);
        Ciphertext result = linear_combo_2(env, x2, 1, sum, mod_sub(0, 1, mod), ops);
        result = hulk_scale_or_defer(env, result, factorial_inverse_cached(env, 2), ops);
        ops.depth = 1;
        return {move(result), ops};
    }
    if (max_sum == 3 && target == 3) {
        Ciphertext x2 = square_ciphertext(env, sum, ops);
        Ciphertext x3 = multiply_ciphertexts(env, x2, sum, ops);
        Ciphertext result = multiply_plain_const(env, x3, 1, ops);
        Ciphertext neg3x2 = multiply_plain_const(env, x2, mod_sub(0, 3, mod), ops);
        Ciphertext plus2x = multiply_plain_const(env, sum, 2, ops);
        add_inplace(env, result, neg3x2, ops);
        add_inplace(env, result, plus2x, ops);
        result = hulk_scale_or_defer(env, result, factorial_inverse_cached(env, 3), ops);
        ops.depth = 2;
        return {move(result), ops};
    }
    if (max_sum == 4 && target == 4) {
        Ciphertext x2 = square_ciphertext(env, sum, ops);
        Ciphertext y = linear_combo_2(env, x2, 1, sum, mod_sub(0, 3, mod), ops);
        Ciphertext y2 = square_ciphertext(env, y, ops);
        Ciphertext two_y = multiply_plain_const(env, y, 2, ops);
        add_inplace(env, y2, two_y, ops);
        y2 = hulk_scale_or_defer(env, y2, factorial_inverse_cached(env, 4), ops);
        ops.depth = 2;
        return {move(y2), ops};
    }
    if (max_sum == 5 && target == 5) {
        // t = S-2, u = t^2, u2 = u^2; expand (u-4)(u-1) = u2-5u+4 without HMul.
        // Schedule: 2 sq + 1 HMul, depth 3 (replaces old 1 sq + 2 HMul).
        Ciphertext t = sum;
        add_plain_inplace(env, t, mod_sub(0, 2, mod), ops);           // t = S-2, depth 0
        Ciphertext u  = square_ciphertext(env, t, ops);               // u = t^2, depth 1
        Ciphertext u2 = square_ciphertext(env, u, ops);               // u^2, depth 2
        // P(u) = u^2 - 5u + 4 = (u-4)(u-1): plaintext linear combo of u2 and u
        Ciphertext P = linear_combo_2(env, u2, 1, u, mod_sub(0, 5, mod), ops);
        add_plain_inplace(env, P, 4, ops);                            // P = u2-5u+4, depth 2
        // I_5 = t * P(u) / 120: t at depth 0, P at depth 2 -> depth 3
        Ciphertext term = multiply_ciphertexts(env, t, P, ops);
        term = hulk_scale_or_defer(env, term, factorial_inverse_cached(env, 5), ops);
        ops.depth = 3;
        return {move(term), ops};
    }
    if (max_sum == 6 && target == 6) {
        Ciphertext x2 = square_ciphertext(env, sum, ops);
        Ciphertext y = linear_combo_2(env, x2, 1, sum, mod_sub(0, 5, mod), ops);
        Ciphertext y2 = square_ciphertext(env, y, ops);
        Ciphertext y3 = multiply_ciphertexts(env, y2, y, ops);
        Ciphertext ten_y2 = multiply_plain_const(env, y2, 10, ops);
        Ciphertext twentyfour_y = multiply_plain_const(env, y, 24, ops);
        add_inplace(env, y3, ten_y2, ops);
        add_inplace(env, y3, twentyfour_y, ops);
        y3 = hulk_scale_or_defer(env, y3, factorial_inverse_cached(env, 6), ops);
        ops.depth = 3;
        return {move(y3), ops};
    }
    if (max_sum == 7 && target == 7) {
        // y = S(S-6) depth 1; xm3 = S-3 depth 0; y+5 = (S-1)(S-5) depth 1; y+8 = (S-2)(S-4) depth 1.
        // Balanced tree: p1 = y*(S-3) depth 2; p2 = (y+5)*(y+8) depth 2; result = p1*p2 depth 3.
        // Replaces old sequential chain that reached depth 4.
        Ciphertext x2 = square_ciphertext(env, sum, ops);             // depth 1
        Ciphertext y  = linear_combo_2(env, x2, 1, sum,
                             mod_sub(0, 6, mod), ops);                 // y = S(S-6), depth 1
        Ciphertext xm3 = sum;
        add_plain_inplace(env, xm3, mod_sub(0, 3, mod), ops);         // S-3, depth 0
        Ciphertext yp5 = y;
        add_plain_inplace(env, yp5, 5, ops);                          // (S-1)(S-5), depth 1
        Ciphertext yp8 = y;
        add_plain_inplace(env, yp8, 8, ops);                          // (S-2)(S-4), depth 1
        Ciphertext p1   = multiply_ciphertexts(env, y,   xm3, ops);   // depth 2
        Ciphertext p2   = multiply_ciphertexts(env, yp5, yp8, ops);   // depth 2
        Ciphertext term = multiply_ciphertexts(env, p1,  p2,  ops);   // depth 3
        term = hulk_scale_or_defer(env, term, factorial_inverse_cached(env, 7), ops);
        ops.depth = 3;
        return {move(term), ops};
    }
    if (max_sum == 8 && target == 8) {
        Ciphertext x2 = square_ciphertext(env, sum, ops);
        Ciphertext y = linear_combo_2(env, x2, 1, sum, mod_sub(0, 7, mod), ops);
        Ciphertext y2 = square_ciphertext(env, y, ops);
        Ciphertext twentyeight_y = multiply_plain_const(env, y, 28, ops);
        Ciphertext inner = y2;
        add_inplace(env, inner, twentyeight_y, ops);
        Ciphertext term = multiply_ciphertexts(env, y2, inner, ops);
        Ciphertext twofiftytwo_y2 = multiply_plain_const(env, y2, 252, ops);
        Ciphertext seventwenty_y = multiply_plain_const(env, y, 720, ops);
        add_inplace(env, term, twofiftytwo_y2, ops);
        add_inplace(env, term, seventwenty_y, ops);
        term = hulk_scale_or_defer(env, term, factorial_inverse_cached(env, 8), ops);
        ops.depth = 3;
        return {move(term), ops};
    }

    const HulkPolyEval eval_kind = hulk_poly_eval();
    if (target == max_sum) {
        if (max_sum >= indicator_ps_min_fanin()) {
            if (eval_kind == HulkPolyEval::streaming) {
                return eval_poly_ps_streaming(env, sum, *indicator_poly_plan(env, max_sum, target));
            }
            vector<uint64_t> coeffs = indicator_coefficients(env, max_sum, target);
            if (eval_kind == HulkPolyEval::bsgs || prefer_bsgs_over_root_paired(env, coeffs, max_sum)) {
                return eval_poly_bsgs(env, sum, coeffs);
            }
        }
        return hulk_indicator_large_all_true(sum, max_sum, env);
    }

    if (eval_kind == HulkPolyEval::streaming) {
        return eval_poly_ps_streaming(env, sum, *indicator_poly_plan(env, max_sum, target));
    }
    return eval_poly_bsgs(env, sum, indicator_coefficients(env, max_sum, target));
}

inline pair<Ciphertext, OpCount> hulk_nonzero_from_sum(const Ciphertext &sum, size_t max_sum, const HEEnv &env)
{
    // Root-paired below the crossover; from it on, the shallower of root-paired and
    // BSGS (or legacy PS with HULK_POLY_EVAL=streaming). See indicator_ps_min_fanin().
    if (max_sum >= indicator_ps_min_fanin()) {
        const HulkPolyEval eval_kind = hulk_poly_eval();
        if (eval_kind == HulkPolyEval::streaming) {
            return eval_poly_ps_streaming(env, sum, *nonzero_poly_plan(env, max_sum));
        }
        vector<uint64_t> coeffs = nonzero_coefficients(env, max_sum);
        if (eval_kind == HulkPolyEval::bsgs || prefer_bsgs_over_root_paired(env, coeffs, max_sum)) {
            return eval_poly_bsgs(env, sum, coeffs);
        }
    }
    return hulk_nonzero_large(sum, max_sum, env);
}

inline pair<Ciphertext, OpCount> hulk_all_true(const vector<Ciphertext> &bits, const HEEnv &env)
{
    if (bits.empty()) {
        throw invalid_argument("hulk_all_true needs at least one bit");
    }
    auto sum = sum_bits(bits, env);
    auto indicator = hulk_indicator_from_sum(sum.first, bits.size(), bits.size(), env);
    indicator.second += sum.second;
    indicator.second.depth = max(indicator.second.depth, sum.second.depth);
    return indicator;
}

inline pair<Ciphertext, OpCount> hulk_any_true(const vector<Ciphertext> &bits, const HEEnv &env)
{
    if (bits.empty()) {
        throw invalid_argument("hulk_any_true needs at least one bit");
    }
    auto sum = sum_bits(bits, env);
    auto nonzero = hulk_nonzero_from_sum(sum.first, bits.size(), env);
    nonzero.second += sum.second;
    nonzero.second.depth = max(nonzero.second.depth, sum.second.depth);
    return nonzero;
}

enum class FilterKind {
    all_true,
    any_true
};

struct AggregateOutput {
    string name;
    vector<size_t> predicate_cols;
    FilterKind filter_kind = FilterKind::all_true;
    int weight_col = -1;
    int shared_filter = -1;
};

struct SharedFilter {
    string name;
    vector<size_t> predicate_cols;
    FilterKind filter_kind = FilterKind::all_true;
    int parent_filter = -1;
};

inline vector<uint64_t> repeat_values(const vector<uint64_t> &values, size_t copies)
{
    vector<uint64_t> out;
    out.reserve(values.size() * copies);
    for (size_t i = 0; i < copies; ++i) {
        out.insert(out.end(), values.begin(), values.end());
    }
    return out;
}

inline vector<SharedFilter> replicate_shared_filters(const vector<SharedFilter> &filters, size_t copies)
{
    if (copies <= 1) {
        return filters;
    }
    vector<SharedFilter> out;
    out.reserve(filters.size() * copies);
    for (size_t copy_idx = 0; copy_idx < copies; ++copy_idx) {
        const int base = static_cast<int>(out.size());
        for (const SharedFilter &filter : filters) {
            SharedFilter dup = filter;
            dup.name += "__qb" + to_string(copy_idx);
            if (dup.parent_filter >= 0) {
                dup.parent_filter += base;
            }
            out.push_back(move(dup));
        }
    }
    return out;
}

inline vector<AggregateOutput> replicate_outputs(
    const vector<AggregateOutput> &outputs,
    size_t copies,
    size_t shared_filter_count_per_copy = 0)
{
    if (copies <= 1) {
        return outputs;
    }
    vector<AggregateOutput> out;
    out.reserve(outputs.size() * copies);
    for (size_t copy_idx = 0; copy_idx < copies; ++copy_idx) {
        const int shared_base = static_cast<int>(copy_idx * shared_filter_count_per_copy);
        for (const AggregateOutput &output : outputs) {
            AggregateOutput dup = output;
            dup.name += "__qb" + to_string(copy_idx);
            if (dup.shared_filter >= 0) {
                dup.shared_filter += shared_base;
            }
            out.push_back(move(dup));
        }
    }
    return out;
}

struct PackedTable {
    vector<string> column_names;
    vector<vector<uint64_t>> columns;

    size_t rows() const
    {
        return columns.empty() ? 0 : columns.front().size();
    }

    size_t col_index(const string &name) const
    {
        auto it = find(column_names.begin(), column_names.end(), name);
        if (it == column_names.end()) {
            throw invalid_argument("missing column: " + name);
        }
        return static_cast<size_t>(distance(column_names.begin(), it));
    }
};

inline EncryptedTable encrypt_table(const HEEnv &env, const PackedTable &table)
{
    return {encrypt_columns(env, table.columns), table.rows()};
}

struct AggregateRunResult {
    vector<uint64_t> values;
    vector<size_t> topk_indices;
    vector<uint64_t> topk_values;
    OpCount ops;
    // ops from shared-filter DAG node evaluation only (subset of ops, for CSE breakdown)
    OpCount shared_filter_ops;
    double elapsed_ms = 0.0;
    double decrypt_ms = 0.0;
    double online_e2e_ms = 0.0;
    size_t block_count = 0;
    size_t input_ciphertexts = 0;
    // Noise budget analysis: min/max remaining bits across all output ciphertexts.
    // positive = decryption OK; if min_noise_budget > bits_per_limb → could use 1 fewer limb.
    int min_noise_budget_bits = -1;
    int max_noise_budget_bits = -1;
    int bits_per_limb = 0;  // prime size used (55 for arith, 57 for HULK)
    // Homomorphic slot aggregation (AggregateOptions::slot_aggregate / HE_SLOT_AGGREGATE=1).
    // When true, values[o] is slot 0 of sum_slots(HAdd_blocks(masked_o)), i.e. the
    // aggregate reduced mod t, computed inside the timed region; aggregate_ops holds
    // the rotate-and-sum ops (also included in ops). aggregate_slots_consistent is
    // false if any decrypted aggregate ciphertext did not hold the same value in
    // every slot (a sum_slots / Galois-key failure).
    bool slot_aggregated = false;
    bool aggregate_slots_consistent = true;
    OpCount aggregate_ops;
    // Wall time of the rotate-and-sum stage alone (0 when slot_aggregated is
    // false). Included in elapsed_ms; report it as its own component.
    double aggregate_ms = 0.0;
    // Cache warm-up (AggregateOptions::warm_first_block / HE_WARM_RUN=1), untimed.
    bool warmed = false;
    double warm_ms = 0.0;
};

// Oracle check for run_packed_aggregates_preencrypted results. With slot
// aggregation the HE value is the aggregate REDUCED MOD t, so an oracle
// aggregate >= t can never match; that is a workload/encoding limit, not an HE
// failure. This throws a clear "aggregate >= t" error in that case (instead of a
// silent MISMATCH) and also fails on inconsistent aggregate slots. Otherwise it
// returns values == oracle.
inline bool verify_aggregates(
    const AggregateRunResult &res, const vector<uint64_t> &oracle, const HEEnv &env,
    const string &label = "aggregate")
{
    if (res.slot_aggregated) {
        for (size_t i = 0; i < oracle.size(); ++i) {
            if (oracle[i] >= env.plain_modulus) {
                throw runtime_error(
                    label + ": plaintext-oracle aggregate[" + to_string(i) + "]=" + to_string(oracle[i]) +
                    " >= t=" + to_string(env.plain_modulus) +
                    " (aggregate >= t): slot aggregation reduces mod t, so this output cannot be "
                    "verified; rescale/split the output or disable HE_SLOT_AGGREGATE");
            }
        }
        if (!res.aggregate_slots_consistent) {
            cerr << label << ": slot-aggregated ciphertext slots disagree (sum_slots / Galois-key failure)\n";
            return false;
        }
    }
    return res.values == oracle;
}

// Print remaining noise budget. If min > bits_per_limb: could drop 1 limb.
// If min > 2×bits_per_limb: could drop 2 limbs. Negative means decryption fails.
inline void print_noise_budget(const string &label, const AggregateRunResult &r)
{
    if (r.min_noise_budget_bits < 0) {
        return;
    }
    const int min_b = r.min_noise_budget_bits;
    const int max_b = r.max_noise_budget_bits;
    const int bpl   = r.bits_per_limb;
    const int limbs_spare = (bpl > 0) ? (min_b / bpl) : 0;
    cout << "  " << label << " noise budget (min/max bits remaining): "
         << min_b << " / " << max_b;
    if (bpl > 0) {
        cout << "  [prime=" << bpl << "b";
        if (limbs_spare > 0) {
            cout << " → could drop " << limbs_spare << " limb(s)!";
        } else {
            cout << " → tight, 0 spare";
        }
        cout << "]";
    }
    cout << "\n";
}

inline vector<size_t> select_topk_indices(const vector<uint64_t> &values, size_t k)
{
    if (k == 0 || values.empty()) {
        return {};
    }

    k = min(k, values.size());
    vector<size_t> indices(values.size());
    iota(indices.begin(), indices.end(), 0);

    // Engorgio-style online ranking integration: select top-k candidates,
    // then deterministically order them by score and stable tie-break.
    const auto better = [&](size_t lhs, size_t rhs) {
        if (values[lhs] != values[rhs]) {
            return values[lhs] > values[rhs];
        }
        return lhs < rhs;
    };

    if (k < indices.size()) {
        nth_element(indices.begin(), indices.begin() + static_cast<ptrdiff_t>(k), indices.end(), better);
        indices.resize(k);
    }
    sort(indices.begin(), indices.end(), better);
    return indices;
}

inline size_t configured_top_k()
{
    const char *raw = getenv("HULK_TOP_K");
    if (!raw) {
        return 10;
    }
    try {
        return stoull(string(raw));
    } catch (...) {
        return 10;
    }
}

inline void print_topk_summary(
    EvalMode mode,
    const vector<AggregateOutput> &outputs,
    const vector<uint64_t> &values,
    size_t top_k)
{
    const vector<size_t> indices = select_topk_indices(values, top_k);
    if (indices.empty()) {
        return;
    }

    cout << (mode == EvalMode::hulk ? "HULK" : "arithmetic")
         << " top-" << indices.size() << " outputs:";
    for (size_t idx : indices) {
        const string &name = idx < outputs.size() ? outputs[idx].name : string("output_") + to_string(idx);
        cout << " [" << name << "=" << values[idx] << "]";
    }
    cout << "\n";
}

inline pair<Ciphertext, OpCount> apply_filter(
    EvalMode mode,
    FilterKind kind,
    const vector<Ciphertext> &predicates,
    const HEEnv &env)
{
    if (kind == FilterKind::all_true) {
        return mode == EvalMode::arithmetic
            ? arithmetic_all_true(predicates, env)
            : hulk_all_true(predicates, env);
    }
    return mode == EvalMode::arithmetic
        ? arithmetic_any_true(predicates, env)
        : hulk_any_true(predicates, env);
}

// apply_filter / indicator-from-sum evaluated inside a HulkDeferScaleScope(defer):
// `scale` receives the constant the client must multiply the decrypted value by
// (1 when nothing was deferred). Only for FINAL filters (see HulkDeferScaleScope).
inline pair<Ciphertext, OpCount> apply_filter_deferred(
    EvalMode mode, FilterKind kind, const vector<Ciphertext> &predicates, const HEEnv &env,
    bool defer, uint64_t &scale)
{
    HulkDeferScaleScope scope(defer && mode == EvalMode::hulk);
    auto r = apply_filter(mode, kind, predicates, env);
    scale = scope.skipped();
    return r;
}
inline pair<Ciphertext, OpCount> hulk_filter_from_sum_deferred(
    FilterKind kind, const Ciphertext &sum, size_t width, const HEEnv &env, bool defer, uint64_t &scale)
{
    HulkDeferScaleScope scope(defer);
    auto r = kind == FilterKind::all_true
        ? hulk_indicator_from_sum(sum, width, width, env)
        : hulk_nonzero_from_sum(sum, width, env);
    scale = scope.skipped();
    return r;
}

// Boolean env flag: "1", "true"/"TRUE", "yes"/"YES" enable it; anything else (or unset) disables it.
inline bool env_flag_enabled(const char *name)
{
    const char *raw = getenv(name);
    if (!raw) {
        return false;
    }
    const string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

inline bool disable_shared_filter_hoisting() { return env_flag_enabled("HULK_DISABLE_SHARED_HOISTING"); }

// HE_SLOT_AGGREGATE=1: default for AggregateOptions::slot_aggregate (default OFF).
inline bool slot_aggregate_enabled() { return env_flag_enabled("HE_SLOT_AGGREGATE"); }
// HE_WARM_RUN=1: default for AggregateOptions::warm_first_block (default OFF).
inline bool warm_run_enabled() { return env_flag_enabled("HE_WARM_RUN"); }

// Options for run_packed_aggregates_preencrypted. Every option acts identically
// in both evaluation modes.
struct AggregateOptions {
    // After an output's masked (weighted) block ciphertexts are HAdded, apply
    // sum_slots (log2(N/2) row rotations + 1 column rotation, each followed by an
    // HAdd) INSIDE the timed region, then decrypt only the aggregate (slot 0).
    // nullopt -> HE_SLOT_AGGREGATE. The aggregate is reduced mod t: callers must
    // keep every true aggregate below the plaintext modulus.
    optional<bool> slot_aggregate;
    // Before the timer starts, evaluate the whole query once on block 0 only and
    // discard the result. This populates every plaintext / coefficient / plan /
    // encrypted-constant cache the timed run will touch, so first-use encodes are
    // outside the timed region in both modes. nullopt -> HE_WARM_RUN.
    optional<bool> warm_first_block;
    // Suppress the top-k summary print (used by the warm-up pass).
    bool quiet = false;
};

// Pre-encode constant plaintexts (and, optionally, encrypted constants) into the
// env cache before t0. Use for plaintext coefficients / constants that a driver
// applies itself (e.g. comparison-polynomial coefficients). Zero coefficients are
// never encoded by the evaluators (multiply_plain_const / add_plain_inplace skip
// them), so they are skipped here too.
inline void warm_plain_constants(const HEEnv &env, const vector<uint64_t> &values, bool also_encrypt = false)
{
    if (!env.cache_enabled) {
        return;
    }
    for (uint64_t value : values) {
        if ((value % env.plain_modulus) == 0 && !also_encrypt) {
            continue;
        }
        (void)constant_plaintext(env, value);
        if (also_encrypt) {
            (void)constant_ciphertext(env, value);
        }
    }
}

inline void warm_coefficients(const HEEnv &env, const vector<uint64_t> &coeffs)
{
    warm_plain_constants(env, coeffs, /*also_encrypt=*/false);
}

inline AggregateRunResult run_packed_aggregates_preencrypted(
    const EncryptedTable &encrypted_table,
    const vector<AggregateOutput> &outputs,
    EvalMode mode,
    const HEEnv &env,
    const vector<SharedFilter> &shared_filters = {},
    size_t query_group_size = 0,
    const AggregateOptions &opts = {})
{
    if (encrypted_table.columns.empty()) {
        throw invalid_argument("run_packed_aggregates_preencrypted needs at least one column");
    }

    if (mode == EvalMode::hulk && !shared_filters.empty() && disable_shared_filter_hoisting()) {
        vector<AggregateOutput> expanded_outputs;
        expanded_outputs.reserve(outputs.size());

        function<void(int, FilterKind, vector<size_t> &)> append_shared_predicates =
            [&](int shared_idx, FilterKind target_kind, vector<size_t> &preds) {
                if (shared_idx < 0 || shared_idx >= static_cast<int>(shared_filters.size())) {
                    throw invalid_argument("aggregate output references missing shared filter");
                }
                const SharedFilter &filter = shared_filters[static_cast<size_t>(shared_idx)];
                if (filter.filter_kind != target_kind) {
                    throw invalid_argument("cannot disable shared hoisting with mixed filter kinds in a shared chain");
                }
                if (filter.parent_filter >= 0) {
                    append_shared_predicates(filter.parent_filter, target_kind, preds);
                }
                preds.insert(preds.end(), filter.predicate_cols.begin(), filter.predicate_cols.end());
            };

        for (const AggregateOutput &output : outputs) {
            AggregateOutput expanded = output;
            if (output.shared_filter >= 0) {
                vector<size_t> merged_preds;
                merged_preds.reserve(output.predicate_cols.size() + 8);
                append_shared_predicates(output.shared_filter, output.filter_kind, merged_preds);
                merged_preds.insert(merged_preds.end(), output.predicate_cols.begin(), output.predicate_cols.end());
                expanded.predicate_cols = move(merged_preds);
                expanded.shared_filter = -1;
            }
            expanded_outputs.push_back(move(expanded));
        }

        return run_packed_aggregates_preencrypted(
            encrypted_table, expanded_outputs, mode, env, {}, query_group_size, opts);
    }

    const bool do_slot_aggregate = opts.slot_aggregate.value_or(slot_aggregate_enabled());
    const bool do_warm = opts.warm_first_block.value_or(warm_run_enabled());

    AggregateRunResult result;
    const auto &encrypted = encrypted_table.columns;
    result.block_count = encrypted_table.block_count();
    result.input_ciphertexts = encrypted.size() * result.block_count;
    result.values.reserve(outputs.size());
    result.slot_aggregated = do_slot_aggregate;

    // Untimed, mode-symmetric cache warm-up: the same query on block 0 only.
    // Per-block work is identical across blocks, so this touches every cached
    // plaintext / coefficient / plan the timed run uses. The output is discarded.
    if (do_warm && result.block_count > 0) {
        const auto warm_start = steady_clock::now();
        EncryptedTable first_block;
        first_block.rows = min(encrypted_table.rows, env.slot_count);
        first_block.columns.reserve(encrypted.size());
        for (const auto &column : encrypted) {
            first_block.columns.push_back({column.front()});
        }
        AggregateOptions warm_opts;
        warm_opts.slot_aggregate = false;   // rotations use no cache
        warm_opts.warm_first_block = false;
        warm_opts.quiet = true;
        (void)run_packed_aggregates_preencrypted(
            first_block, outputs, mode, env, shared_filters, query_group_size, warm_opts);
        result.warmed = true;
        result.warm_ms = duration<double, milli>(steady_clock::now() - warm_start).count();
    }

    // Worker envs are built OUTSIDE the timed region (key copies are not HE ops).
    // Upper bound on either path's worker_count: flat uses min(threads, outputs*blocks),
    // shared uses min(threads, task_groups) with task_groups <= outputs.
    const size_t max_pool_workers = min(max<size_t>(1, env.max_threads),
                                        max<size_t>(1, outputs.size() * result.block_count));
    const vector<HEEnv> worker_pool = make_worker_pool(env, max_pool_workers);
    const auto pool_worker = [&](size_t worker_count, size_t idx) -> const HEEnv & {
        if (worker_count <= 1) {
            return env;
        }
        if (idx >= worker_pool.size()) {
            throw logic_error("worker pool smaller than worker_count");
        }
        return worker_pool[idx];
    };

    // Deferred indicator normalization (see HulkDeferScaleScope): the client-side constant of
    // every output, identical for all blocks (it depends on the filter structure only).
    vector<uint64_t> output_scale(outputs.size(), 0);
    mutex output_scale_mutex;
    const auto record_output_scale = [&](size_t output_idx, uint64_t scale) {
        lock_guard<mutex> lk(output_scale_mutex);
        if (output_scale[output_idx] == 0) {
            output_scale[output_idx] = scale;
        } else if (output_scale[output_idx] != scale) {
            throw logic_error("deferred indicator scale differs across blocks");
        }
    };

    const auto start = steady_clock::now();
    const auto process_output_block = [&](const AggregateOutput &output, size_t block, const HEEnv &worker) {
        vector<Ciphertext> preds;
        preds.reserve(output.predicate_cols.size());
        for (size_t col : output.predicate_cols) {
            preds.push_back(encrypted[col][block]);
        }

        OpCount block_ops;
        uint64_t out_scale = 1;
        auto filtered = apply_filter_deferred(mode, output.filter_kind, preds, worker, true, out_scale);
        record_output_scale(static_cast<size_t>(&output - outputs.data()), out_scale);
        block_ops += filtered.second;

        Ciphertext masked = filtered.first;
        if (output.weight_col >= 0) {
            masked = multiply_ciphertexts(
                worker,
                masked,
                encrypted[static_cast<size_t>(output.weight_col)][block],
                block_ops);
            block_ops.depth = max(block_ops.depth, filtered.second.depth + 1);
            block_ops.weight_muls++;
        }
        refresh_zero_guard(worker, masked, block_ops);
        return pair<Ciphertext, OpCount>{move(masked), block_ops};
    };

    vector<Ciphertext> final_totals(outputs.size());
    vector<bool> final_initialized(outputs.size(), false);
    vector<OpCount> final_ops(outputs.size());
    OpCount shared_ops_total;

    if (query_group_size > outputs.size() || (query_group_size != 0 && outputs.size() % query_group_size != 0)) {
        throw invalid_argument("query_group_size must divide the output count");
    }

    if (shared_filters.empty()) {
        // Parallelize across ALL (output, block) pairs so every thread stays busy
        // regardless of output count. Blocks are embarrassingly parallel — each
        // processes a disjoint row range and contributes a partial ciphertext that
        // is HAdded (depth-0, free noise) into the per-output running total.
        //
        // Bandwidth safety: SEAL NTT is compute-bound (~125 MB/s per thread vs
        // 307 GB/s DDR5 on EPYC 9554). Concurrency is capped at max_threads which
        // is set by the caller; the caller should choose a value ≤ available cores
        // to avoid NUMA cross-socket pressure.

        const size_t n_outputs = outputs.size();
        const size_t n_blocks  = result.block_count;
        const size_t total_tasks = n_outputs * n_blocks;

        // Flatten task list: task t → output o = t / n_blocks, block b = t % n_blocks
        // Interleave by block-major order so threads working on the same output
        // are spread across the block dimension → better cache reuse per thread.
        const size_t worker_count = min(max<size_t>(1, env.max_threads),
                                        max<size_t>(1, total_tasks));

        // Per-output accumulator protected by a fine-grained mutex.
        // HAdd takes ~1ms vs ~40ms per HMul so lock contention is negligible.
        vector<Ciphertext>  out_totals(n_outputs);
        vector<bool>        out_init(n_outputs, false);
        vector<OpCount>     out_ops(n_outputs);
        vector<mutex>       out_mutex(n_outputs);

        vector<future<void>> futures;
        futures.reserve(worker_count);
        for (size_t w = 0; w < worker_count; ++w) {
            futures.push_back(async(launch::async, [&, w]() {
                const HEEnv &worker = pool_worker(worker_count, w);
                // Round-robin over tasks: worker w handles tasks w, w+W, w+2W, ...
                for (size_t t = w; t < total_tasks; t += worker_count) {
                    const size_t output_idx = t / n_blocks;
                    const size_t block      = t % n_blocks;
                    auto [partial, ops] = process_output_block(
                        outputs[output_idx], block, worker);
                    // Accumulate: lock is brief (one HAdd), contention is low
                    lock_guard<mutex> lk(out_mutex[output_idx]);
                    ops += out_ops[output_idx];
                    if (!out_init[output_idx]) {
                        out_totals[output_idx] = move(partial);
                        out_init[output_idx]   = true;
                    } else {
                        add_inplace(worker, out_totals[output_idx], partial, ops);
                        ops.agg_adds++;
                    }
                    out_ops[output_idx] = ops;
                }
            }));
        }
        for (auto &f : futures) f.get();

        for (size_t o = 0; o < n_outputs; ++o) {
            final_totals[o]     = move(out_totals[o]);
            final_initialized[o]= out_init[o];
            final_ops[o]        = out_ops[o];
        }
        result.values.resize(n_outputs, 0);
    } else {
        for (const AggregateOutput &output : outputs) {
            if (output.shared_filter >= static_cast<int>(shared_filters.size())) {
                throw invalid_argument("aggregate output references missing shared filter");
            }
        }
        for (const SharedFilter &filter : shared_filters) {
            if (filter.parent_filter >= static_cast<int>(shared_filters.size())) {
                throw invalid_argument("shared filter references missing parent filter");
            }
        }
        // A shared filter is FINAL when no other filter uses it as a parent and every
        // output that references it uses it alone (empty local predicates): its value
        // then only feeds weight / aggregation / decryption and may stay unnormalized.
        vector<bool> shared_final(shared_filters.size(), true);
        for (const SharedFilter &filter : shared_filters) {
            if (filter.parent_filter >= 0) shared_final[static_cast<size_t>(filter.parent_filter)] = false;
        }
        for (const AggregateOutput &output : outputs) {
            if (output.shared_filter >= 0 && !output.predicate_cols.empty()) {
                shared_final[static_cast<size_t>(output.shared_filter)] = false;
            }
        }

        struct WorkerAggregateState {
            vector<Ciphertext> totals;
            vector<bool> initialized;
            vector<OpCount> ops;
            OpCount shared_ops;
        };

        const auto process_block = [&](size_t block, const HEEnv &worker) {
            vector<Ciphertext> shared_values(shared_filters.size());
            vector<uint64_t> shared_scale(shared_filters.size(), 1);
            vector<bool> shared_initialized(shared_filters.size(), false);
            vector<int> shared_depths(shared_filters.size(), 0);
            vector<Ciphertext> shared_sum_values(shared_filters.size());
            vector<bool> shared_sum_initialized(shared_filters.size(), false);
            vector<bool> shared_sum_supported(shared_filters.size(), false);
            vector<int> shared_sum_depths(shared_filters.size(), 0);
            vector<size_t> shared_sum_widths(shared_filters.size(), 0);
            unordered_map<string, Ciphertext> deferred_mask_cache;
            unordered_map<string, uint64_t> deferred_scale_cache;
            OpCount block_shared_ops;

            const auto make_deferred_mask_key = [](size_t shared_idx,
                                                   FilterKind filter_kind,
                                                   const vector<size_t> &predicate_cols) {
                string key = to_string(shared_idx);
                key.push_back('|');
                key.push_back(filter_kind == FilterKind::all_true ? 'T' : 'N');
                for (size_t col : predicate_cols) {
                    key.push_back('|');
                    key += to_string(col);
                }
                return key;
            };

            function<bool(size_t)> realize_shared_sum = [&](size_t shared_idx) -> bool {
                if (shared_sum_initialized[shared_idx]) {
                    return shared_sum_supported[shared_idx];
                }
                shared_sum_initialized[shared_idx] = true;

                if (mode != EvalMode::hulk) {
                    return false;
                }

                const SharedFilter &filter = shared_filters[shared_idx];
                if (filter.parent_filter >= 0) {
                    const size_t parent_idx = static_cast<size_t>(filter.parent_filter);
                    if (shared_filters[parent_idx].filter_kind != filter.filter_kind) {
                        return false;
                    }
                    if (!realize_shared_sum(parent_idx)) {
                        return false;
                    }
                }

                vector<Ciphertext> local_preds;
                local_preds.reserve(filter.predicate_cols.size());
                for (size_t col : filter.predicate_cols) {
                    local_preds.push_back(encrypted[col][block]);
                }

                Ciphertext total_sum;
                bool initialized = false;
                int total_depth = 0;
                size_t total_width = 0;
                if (filter.parent_filter >= 0) {
                    const size_t parent_idx = static_cast<size_t>(filter.parent_filter);
                    total_sum = shared_sum_values[parent_idx];
                    initialized = true;
                    total_depth = shared_sum_depths[parent_idx];
                    total_width = shared_sum_widths[parent_idx];
                }

                if (!local_preds.empty()) {
                    pair<Ciphertext, OpCount> local_sum =
                        local_preds.size() == 1
                        ? pair<Ciphertext, OpCount>{local_preds.front(), OpCount{}}
                        : sum_bits(local_preds, worker);
                    block_shared_ops += local_sum.second;
                    if (!initialized) {
                        total_sum = move(local_sum.first);
                        initialized = true;
                    } else {
                        add_inplace(worker, total_sum, local_sum.first, block_shared_ops);
                    }
                    total_width += local_preds.size();
                }

                if (!initialized) {
                    throw invalid_argument("shared filter needs at least one predicate");
                }

                shared_sum_values[shared_idx] = move(total_sum);
                shared_sum_depths[shared_idx] = total_depth;
                shared_sum_widths[shared_idx] = total_width;
                shared_sum_supported[shared_idx] = true;
                return true;
            };

            function<const Ciphertext &(size_t)> realize_shared = [&](size_t shared_idx) -> const Ciphertext & {
                if (shared_initialized[shared_idx]) {
                    return shared_values[shared_idx];
                }

                const SharedFilter &filter = shared_filters[shared_idx];

                // HULK 2-bit ROOT node optimization: for root shared filters with exactly 2 predicate
                // columns, use arithmetic AND-2 (HMul) instead of indicator(sum_2,2,2) (squaring).
                // HMul noise: ε²×N (product of two fresh {0,1} ciphertexts).
                // Indicator sq noise: (2ε)²×N/2 = 2ε²×N  (sum then square).
                // HMul is 2× less noisy → log₂(2)=1 extra bit saved → may allow 1 fewer limb.
                // Cost: HMul ≈ 1.5× sq (slight increase, compensated by fewer limbs downstream).
                if (mode == EvalMode::hulk && filter.parent_filter < 0 &&
                    filter.predicate_cols.size() == 2 &&
                    filter.filter_kind == FilterKind::all_true) {
                    vector<Ciphertext> root_preds;
                    root_preds.reserve(2);
                    for (size_t col : filter.predicate_cols) {
                        root_preds.push_back(encrypted[col][block]);
                    }
                    auto root_filtered = arithmetic_all_true(root_preds, worker);  // 1 HMul, lower noise
                    shared_values[shared_idx] = move(root_filtered.first);
                    block_shared_ops += root_filtered.second;
                    shared_depths[shared_idx] = root_filtered.second.depth;
                    shared_initialized[shared_idx] = true;
                    return shared_values[shared_idx];
                }

                // HULK optimization: when a parent indicator is already in {0,1} and we only add
                // a few local columns, use apply_filter({parent_indicator, local_cols}) instead of
                // hulk_indicator_from_sum(accumulated_sum_of_all_bits, total_width).
                if (mode == EvalMode::hulk && filter.parent_filter >= 0 &&
                    filter.filter_kind == FilterKind::all_true) {
                    const size_t parent_idx = static_cast<size_t>(filter.parent_filter);
                    // Build predicates: parent's 0/1 indicator result + local raw bit columns.
                    vector<Ciphertext> preds;
                    preds.reserve(1 + filter.predicate_cols.size());
                    preds.push_back(realize_shared(parent_idx));  // parent ∈ {0,1} (indicator)
                    for (size_t col : filter.predicate_cols) {
                        preds.push_back(encrypted[col][block]);   // fresh bits ∈ {0,1}
                    }
                    // For HULK shared-filter intermediate nodes (parent_indicator + local_col):
                    // Use arithmetic AND-2 (HMul) instead of indicator squaring when preds.size()==2.
                    // HMul noise: noise_a × noise_b × N = ε²N (fresh parent indicator × local bit).
                    // Squaring noise: ((noise_a + noise_b)²×N) ≈ 2ε²N for 2-bit nodes.
                    // HMul is 2× less noisy → can save 1 limb for shallow chains (k=2,3).
                    // Cost: 1 HMul vs 1 sq (slight increase, but limb savings dominate).
                    const int input_depth = shared_depths[parent_idx];
                    if (preds.size() == 1) {
                        shared_values[shared_idx] = preds.front();
                        shared_depths[shared_idx] = input_depth;
                    } else {
                        // Use arithmetic AND for HULK 2-input shared nodes for lower noise budget.
                        // For 3+ inputs, use HULK indicator (cheaper).
                        const EvalMode node_eval_mode =
                            (mode == EvalMode::hulk && preds.size() == 2)
                            ? EvalMode::arithmetic   // HMul: 2× less noise than indicator sq
                            : mode;
                        auto filtered = apply_filter_deferred(node_eval_mode, filter.filter_kind, preds, worker,
                                                              shared_final[shared_idx], shared_scale[shared_idx]);
                        shared_values[shared_idx] = move(filtered.first);
                        filtered.second.depth = input_depth + filtered.second.depth;
                        block_shared_ops += filtered.second;
                        shared_depths[shared_idx] = filtered.second.depth;
                    }
                    shared_initialized[shared_idx] = true;
                    return shared_values[shared_idx];
                }

                if (realize_shared_sum(shared_idx)) {
                    if (shared_sum_widths[shared_idx] == 1) {
                        shared_values[shared_idx] = shared_sum_values[shared_idx];
                        shared_depths[shared_idx] = shared_sum_depths[shared_idx];
                    } else {
                        auto filtered = hulk_filter_from_sum_deferred(filter.filter_kind, shared_sum_values[shared_idx],
                            shared_sum_widths[shared_idx], worker, shared_final[shared_idx], shared_scale[shared_idx]);
                        shared_values[shared_idx] = move(filtered.first);
                        filtered.second.depth = shared_sum_depths[shared_idx] + filtered.second.depth;
                        block_shared_ops += filtered.second;
                        shared_depths[shared_idx] = filtered.second.depth;
                    }
                    shared_initialized[shared_idx] = true;
                    return shared_values[shared_idx];
                }

                vector<Ciphertext> preds;
                preds.reserve(filter.predicate_cols.size() + (filter.parent_filter >= 0 ? 1 : 0));
                if (filter.parent_filter >= 0) {
                    preds.push_back(realize_shared(static_cast<size_t>(filter.parent_filter)));
                }
                for (size_t col : filter.predicate_cols) {
                    preds.push_back(encrypted[col][block]);
                }

                if (preds.empty()) {
                    throw invalid_argument("shared filter needs at least one predicate");
                }

                const int input_depth =
                    filter.parent_filter >= 0 ? shared_depths[static_cast<size_t>(filter.parent_filter)] : 0;
                if (preds.size() == 1) {
                    shared_values[shared_idx] = preds.front();
                    shared_depths[shared_idx] = input_depth;
                } else {
                    const EvalMode nmode = (mode == EvalMode::hulk && preds.size() == 2) ? EvalMode::arithmetic : mode;
                    auto filtered = apply_filter_deferred(nmode, filter.filter_kind, preds, worker,
                                                         shared_final[shared_idx], shared_scale[shared_idx]);
                    shared_values[shared_idx] = move(filtered.first);
                    filtered.second.depth = input_depth + filtered.second.depth;
                    block_shared_ops += filtered.second;
                    shared_depths[shared_idx] = filtered.second.depth;
                }
                shared_initialized[shared_idx] = true;
                return shared_values[shared_idx];
            };

            vector<Ciphertext> block_totals(outputs.size());
            vector<bool> block_initialized(outputs.size(), false);
            vector<OpCount> block_ops(outputs.size());

            for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
                const AggregateOutput &output = outputs[output_idx];
                Ciphertext masked;
                OpCount block_output_ops;
                bool used_deferred_filter = false;
                uint64_t out_scale = 1;
                if (mode == EvalMode::hulk && output.shared_filter >= 0 && !output.predicate_cols.empty()) {
                    const size_t shared_idx = static_cast<size_t>(output.shared_filter);
                    const SharedFilter &shared_filter = shared_filters[shared_idx];
                    if (shared_filter.filter_kind == output.filter_kind && realize_shared_sum(shared_idx)) {
                        // Compute expected CSE chain depth from filter structure without relying on
                        // shared_depths[] (which is only set after realize_shared is called).
                        // Each chain level adds indicator(sum_2,2,2)=1 or indicator(sum_k,k,k) depth.
                        int expected_cse_d = 0;
                        for (size_t cur = shared_idx; ; ) {
                            const SharedFilter &cf = shared_filters[cur];
                            size_t nw = (cf.parent_filter >= 0 ? 1u : 0u) + cf.predicate_cols.size();
                            expected_cse_d += (nw <= 1) ? 0 : (nw == 2) ? 1 : (nw <= 4) ? 2 : 3;
                            if (cf.parent_filter < 0) break;
                            cur = static_cast<size_t>(cf.parent_filter);
                        }
                        const int cse_per_output_depth = expected_cse_d + 1;
                        const size_t deferred_total_width =
                            shared_sum_widths[shared_idx] + output.predicate_cols.size();
                        const int deferred_ind_depth = (deferred_total_width <= 4) ? 2 : 3;
                        // Use deferred only when it saves a depth level AND k ≤ 6
                        // (k=7,8 indicator cost is 4 ops per output — too expensive without cache)
                        const bool use_deferred =
                            (deferred_ind_depth < cse_per_output_depth) &&
                            (deferred_total_width <= 6);
                        if (use_deferred) {
                            const string deferred_key = make_deferred_mask_key(
                                shared_idx, output.filter_kind, output.predicate_cols);
                            auto cached_mask = deferred_mask_cache.find(deferred_key);
                            if (cached_mask != deferred_mask_cache.end()) {
                                masked = cached_mask->second;
                                out_scale = deferred_scale_cache.at(deferred_key);
                            } else {
                                Ciphertext total_sum = shared_sum_values[shared_idx];
                                const int input_depth = shared_sum_depths[shared_idx];
                                size_t total_width2 = shared_sum_widths[shared_idx];
                                vector<Ciphertext> local_preds;
                                local_preds.reserve(output.predicate_cols.size());
                                for (size_t col : output.predicate_cols) {
                                    local_preds.push_back(encrypted[col][block]);
                                }
                                pair<Ciphertext, OpCount> local_sum =
                                    local_preds.size() == 1
                                    ? pair<Ciphertext, OpCount>{local_preds.front(), OpCount{}}
                                    : sum_bits(local_preds, worker);
                                block_output_ops += local_sum.second;
                                add_inplace(worker, total_sum, local_sum.first, block_output_ops);
                                total_width2 += local_preds.size();
                                auto filtered = hulk_filter_from_sum_deferred(output.filter_kind, total_sum, total_width2, worker, true, out_scale);
                                block_output_ops += filtered.second;
                                block_output_ops.depth = max(block_output_ops.depth, input_depth + filtered.second.depth);
                                masked = move(filtered.first);
                                deferred_mask_cache.emplace(deferred_key, masked);
                                deferred_scale_cache[deferred_key] = out_scale;
                            }
                            used_deferred_filter = true;
                        }
                    }
                }

                if (!used_deferred_filter) {
                    vector<Ciphertext> preds;
                    preds.reserve(output.predicate_cols.size() + (output.shared_filter >= 0 ? 1 : 0));
                    if (output.shared_filter >= 0) {
                        preds.push_back(realize_shared(static_cast<size_t>(output.shared_filter)));
                    }
                    for (size_t col : output.predicate_cols) {
                        preds.push_back(encrypted[col][block]);
                    }

                    const int input_depth =
                        output.shared_filter >= 0 ? shared_depths[static_cast<size_t>(output.shared_filter)] : 0;
                    if (preds.empty()) {
                        throw invalid_argument("aggregate output needs at least one predicate");
                    }
                    if (preds.size() == 1) {
                        masked = preds.front();
                        if (output.shared_filter >= 0) out_scale = shared_scale[static_cast<size_t>(output.shared_filter)];
                        block_output_ops.depth = input_depth;
                    } else {
                        auto filtered = apply_filter_deferred(mode, output.filter_kind, preds, worker, true, out_scale);
                        block_output_ops += filtered.second;
                        block_output_ops.depth = max(block_output_ops.depth, input_depth + filtered.second.depth);
                        masked = move(filtered.first);
                    }
                }

                record_output_scale(output_idx, out_scale);

                if (output.weight_col >= 0) {
                    masked = multiply_ciphertexts(
                        worker,
                        masked,
                        encrypted[static_cast<size_t>(output.weight_col)][block],
                        block_output_ops);
                    block_output_ops.depth += 1;
                    block_output_ops.weight_muls++;
                }
                refresh_zero_guard(worker, masked, block_output_ops);
                block_totals[output_idx] = move(masked);
                block_initialized[output_idx] = true;
                block_ops[output_idx] = block_output_ops;
            }

            return tuple<vector<Ciphertext>, vector<bool>, vector<OpCount>, OpCount>{
                move(block_totals), move(block_initialized), move(block_ops), block_shared_ops};
        };

        // Build worker assignment groups. In HULK mode without explicit query grouping,
        // co-locate outputs that can reuse shared/deferred masks.
        vector<vector<size_t>> assignment_groups;
        if (query_group_size != 0) {
            for (size_t begin_output = 0; begin_output < outputs.size(); begin_output += query_group_size) {
                vector<size_t> group;
                const size_t end_output = min(outputs.size(), begin_output + query_group_size);
                group.reserve(end_output - begin_output);
                for (size_t output_idx = begin_output; output_idx < end_output; ++output_idx) {
                    group.push_back(output_idx);
                }
                assignment_groups.push_back(move(group));
            }
        } else if (mode == EvalMode::hulk) {
            unordered_map<string, size_t> group_index_by_key;
            assignment_groups.reserve(outputs.size());
            for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
                const AggregateOutput &output = outputs[output_idx];
                string key;
                if (output.shared_filter >= 0) {
                    key = "S|" + to_string(output.shared_filter);
                    key.push_back('|');
                    key.push_back(output.filter_kind == FilterKind::all_true ? 'T' : 'N');
                    if (output.predicate_cols.empty()) {
                        key += "|BASE";
                    } else {
                        key += "|LOCAL";
                        for (size_t col : output.predicate_cols) {
                            key.push_back('|');
                            key += to_string(col);
                        }
                    }
                } else {
                    key = "U|" + to_string(output_idx);
                }

                auto [it, inserted] = group_index_by_key.emplace(key, assignment_groups.size());
                if (inserted) {
                    assignment_groups.push_back({});
                }
                assignment_groups[it->second].push_back(output_idx);
            }
        } else {
            assignment_groups.reserve(outputs.size());
            for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
                assignment_groups.push_back({output_idx});
            }
        }

        const size_t task_groups = assignment_groups.size();
        const size_t worker_count = min(max<size_t>(1, env.max_threads), max<size_t>(1, task_groups));
        if (worker_count == 1) {
            for (size_t block = 0; block < result.block_count; ++block) {
                auto [block_totals, block_initialized, block_ops, block_shared_ops] = process_block(block, env);
                shared_ops_total += block_shared_ops;
                for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
                    final_ops[output_idx] += block_ops[output_idx];
                    if (!block_initialized[output_idx]) {
                        continue;
                    }
                    if (!final_initialized[output_idx]) {
                        final_totals[output_idx] = move(block_totals[output_idx]);
                        final_initialized[output_idx] = true;
                    } else {
                        add_inplace(env, final_totals[output_idx], block_totals[output_idx], final_ops[output_idx]);
                        final_ops[output_idx].agg_adds++;
                    }
                }
            }
        } else {
            vector<future<WorkerAggregateState>> futures;
            futures.reserve(worker_count);
            for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
                futures.push_back(async(launch::async, [&, worker_idx]() {
                    const HEEnv &worker = pool_worker(worker_count, worker_idx);
                    WorkerAggregateState state;
                    state.totals.resize(outputs.size());
                    state.initialized.assign(outputs.size(), false);
                    state.ops.resize(outputs.size());

                    vector<size_t> assigned_outputs;
                    assigned_outputs.reserve(outputs.size() / worker_count + 1);
                    for (size_t task_group = worker_idx; task_group < task_groups; task_group += worker_count) {
                        for (size_t output_idx : assignment_groups[task_group]) {
                            assigned_outputs.push_back(output_idx);
                        }
                    }

                    for (size_t block = 0; block < result.block_count; ++block) {
                        vector<Ciphertext> shared_values(shared_filters.size());
                        vector<uint64_t> shared_scale(shared_filters.size(), 1);
                        vector<bool> shared_initialized(shared_filters.size(), false);
                        vector<int> shared_depths(shared_filters.size(), 0);
                        vector<Ciphertext> shared_sum_values(shared_filters.size());
                        vector<bool> shared_sum_initialized(shared_filters.size(), false);
                        vector<bool> shared_sum_supported(shared_filters.size(), false);
                        vector<int> shared_sum_depths(shared_filters.size(), 0);
                        vector<size_t> shared_sum_widths(shared_filters.size(), 0);
                        unordered_map<string, Ciphertext> deferred_mask_cache;
            unordered_map<string, uint64_t> deferred_scale_cache;
                        OpCount block_shared_ops;

                        const auto make_deferred_mask_key = [](size_t shared_idx,
                                                               FilterKind filter_kind,
                                                               const vector<size_t> &predicate_cols) {
                            string key = to_string(shared_idx);
                            key.push_back('|');
                            key.push_back(filter_kind == FilterKind::all_true ? 'T' : 'N');
                            for (size_t col : predicate_cols) {
                                key.push_back('|');
                                key += to_string(col);
                            }
                            return key;
                        };

                        function<bool(size_t)> realize_shared_sum = [&](size_t shared_idx) -> bool {
                            if (shared_sum_initialized[shared_idx]) {
                                return shared_sum_supported[shared_idx];
                            }
                            shared_sum_initialized[shared_idx] = true;

                            if (mode != EvalMode::hulk) {
                                return false;
                            }

                            const SharedFilter &filter = shared_filters[shared_idx];
                            if (filter.parent_filter >= 0) {
                                const size_t parent_idx = static_cast<size_t>(filter.parent_filter);
                                if (shared_filters[parent_idx].filter_kind != filter.filter_kind) {
                                    return false;
                                }
                                if (!realize_shared_sum(parent_idx)) {
                                    return false;
                                }
                            }

                            vector<Ciphertext> local_preds;
                            local_preds.reserve(filter.predicate_cols.size());
                            for (size_t col : filter.predicate_cols) {
                                local_preds.push_back(encrypted[col][block]);
                            }

                            Ciphertext total_sum;
                            bool initialized = false;
                            int total_depth = 0;
                            size_t total_width = 0;
                            if (filter.parent_filter >= 0) {
                                const size_t parent_idx = static_cast<size_t>(filter.parent_filter);
                                total_sum = shared_sum_values[parent_idx];
                                initialized = true;
                                total_depth = shared_sum_depths[parent_idx];
                                total_width = shared_sum_widths[parent_idx];
                            }

                            if (!local_preds.empty()) {
                                pair<Ciphertext, OpCount> local_sum =
                                    local_preds.size() == 1
                                    ? pair<Ciphertext, OpCount>{local_preds.front(), OpCount{}}
                                    : sum_bits(local_preds, worker);
                                block_shared_ops += local_sum.second;
                                if (!initialized) {
                                    total_sum = move(local_sum.first);
                                    initialized = true;
                                } else {
                                    add_inplace(worker, total_sum, local_sum.first, block_shared_ops);
                                }
                                total_width += local_preds.size();
                            }

                            if (!initialized) {
                                throw invalid_argument("shared filter needs at least one predicate");
                            }

                            shared_sum_values[shared_idx] = move(total_sum);
                            shared_sum_depths[shared_idx] = total_depth;
                            shared_sum_widths[shared_idx] = total_width;
                            shared_sum_supported[shared_idx] = true;
                            return true;
                        };

                        function<const Ciphertext &(size_t)> realize_shared = [&](size_t shared_idx) -> const Ciphertext & {
                            if (shared_initialized[shared_idx]) {
                                return shared_values[shared_idx];
                            }

                            const SharedFilter &filter = shared_filters[shared_idx];
                            // 2-bit root HMul optimization (multi-worker path)
                            if (mode == EvalMode::hulk && filter.parent_filter < 0 &&
                                filter.predicate_cols.size() == 2 &&
                                filter.filter_kind == FilterKind::all_true) {
                                vector<Ciphertext> root_preds;
                                root_preds.reserve(2);
                                for (size_t col : filter.predicate_cols) {
                                    root_preds.push_back(encrypted[col][block]);
                                }
                                auto root_f = arithmetic_all_true(root_preds, worker);
                                shared_values[shared_idx] = move(root_f.first);
                                block_shared_ops += root_f.second;
                                shared_depths[shared_idx] = root_f.second.depth;
                                shared_initialized[shared_idx] = true;
                                return shared_values[shared_idx];
                            }
                            // HULK optimization: use apply_filter({parent_indicator, local_cols})
                            if (mode == EvalMode::hulk && filter.parent_filter >= 0 &&
                                filter.filter_kind == FilterKind::all_true) {
                                const size_t parent_idx = static_cast<size_t>(filter.parent_filter);
                                vector<Ciphertext> preds;
                                preds.reserve(1 + filter.predicate_cols.size());
                                preds.push_back(realize_shared(parent_idx));
                                for (size_t col : filter.predicate_cols) {
                                    preds.push_back(encrypted[col][block]);
                                }
                                const int input_depth = shared_depths[parent_idx];
                                if (preds.size() == 1) {
                                    shared_values[shared_idx] = preds.front();
                                    shared_depths[shared_idx] = input_depth;
                                } else {
                                    const EvalMode nmode = (mode == EvalMode::hulk && preds.size() == 2) ? EvalMode::arithmetic : mode;
                    auto filtered = apply_filter_deferred(nmode, filter.filter_kind, preds, worker,
                                                         shared_final[shared_idx], shared_scale[shared_idx]);
                                    shared_values[shared_idx] = move(filtered.first);
                                    filtered.second.depth = input_depth + filtered.second.depth;
                                    block_shared_ops += filtered.second;
                                    shared_depths[shared_idx] = filtered.second.depth;
                                }
                                shared_initialized[shared_idx] = true;
                                return shared_values[shared_idx];
                            }

                            if (realize_shared_sum(shared_idx)) {
                                if (shared_sum_widths[shared_idx] == 1) {
                                    shared_values[shared_idx] = shared_sum_values[shared_idx];
                                    shared_depths[shared_idx] = shared_sum_depths[shared_idx];
                                } else {
                                    auto filtered = hulk_filter_from_sum_deferred(filter.filter_kind, shared_sum_values[shared_idx],
                            shared_sum_widths[shared_idx], worker, shared_final[shared_idx], shared_scale[shared_idx]);
                                    shared_values[shared_idx] = move(filtered.first);
                                    filtered.second.depth = shared_sum_depths[shared_idx] + filtered.second.depth;
                                    block_shared_ops += filtered.second;
                                    shared_depths[shared_idx] = filtered.second.depth;
                                }
                                shared_initialized[shared_idx] = true;
                                return shared_values[shared_idx];
                            }

                            vector<Ciphertext> preds;
                            preds.reserve(filter.predicate_cols.size() + (filter.parent_filter >= 0 ? 1 : 0));
                            if (filter.parent_filter >= 0) {
                                preds.push_back(realize_shared(static_cast<size_t>(filter.parent_filter)));
                            }
                            for (size_t col : filter.predicate_cols) {
                                preds.push_back(encrypted[col][block]);
                            }

                            if (preds.empty()) {
                                throw invalid_argument("shared filter needs at least one predicate");
                            }

                            const int input_depth =
                                filter.parent_filter >= 0 ? shared_depths[static_cast<size_t>(filter.parent_filter)] : 0;
                            if (preds.size() == 1) {
                                shared_values[shared_idx] = preds.front();
                                shared_depths[shared_idx] = input_depth;
                            } else {
                                const EvalMode nmode = (mode == EvalMode::hulk && preds.size() == 2) ? EvalMode::arithmetic : mode;
                    auto filtered = apply_filter_deferred(nmode, filter.filter_kind, preds, worker,
                                                         shared_final[shared_idx], shared_scale[shared_idx]);
                                shared_values[shared_idx] = move(filtered.first);
                                filtered.second.depth = input_depth + filtered.second.depth;
                                block_shared_ops += filtered.second;
                                shared_depths[shared_idx] = filtered.second.depth;
                            }
                            shared_initialized[shared_idx] = true;
                            return shared_values[shared_idx];
                        };

                        for (size_t output_idx : assigned_outputs) {
                            const AggregateOutput &output = outputs[output_idx];
                            Ciphertext masked;
                            OpCount block_output_ops;
                            bool used_deferred_filter = false;
                            uint64_t out_scale = 1;
                            if (mode == EvalMode::hulk && output.shared_filter >= 0 && !output.predicate_cols.empty()) {
                                const size_t shared_idx = static_cast<size_t>(output.shared_filter);
                                const SharedFilter &shared_filter = shared_filters[shared_idx];
                                if (shared_filter.filter_kind == output.filter_kind && realize_shared_sum(shared_idx)) {
                                    int mw_cse_d = 0;
                                    for (size_t cur2 = shared_idx; ; ) {
                                        const SharedFilter &cf2 = shared_filters[cur2];
                                        size_t nw2 = (cf2.parent_filter >= 0 ? 1u : 0u) + cf2.predicate_cols.size();
                                        mw_cse_d += (nw2 <= 1) ? 0 : (nw2 == 2) ? 1 : (nw2 <= 4) ? 2 : 3;
                                        if (cf2.parent_filter < 0) break;
                                        cur2 = static_cast<size_t>(cf2.parent_filter);
                                    }
                                    const size_t mw_total_w =
                                        shared_sum_widths[shared_idx] + output.predicate_cols.size();
                                    const int mw_ind_d = (mw_total_w <= 4) ? 2 : 3;
                                    const bool mw_use_deferred =
                                        (mw_ind_d < mw_cse_d + 1) && (mw_total_w <= 6);
                                    if (mw_use_deferred) {
                                        const string deferred_key = make_deferred_mask_key(
                                            shared_idx, output.filter_kind, output.predicate_cols);
                                        auto cached_mask = deferred_mask_cache.find(deferred_key);
                                        if (cached_mask != deferred_mask_cache.end()) {
                                            masked = cached_mask->second;
                                            out_scale = deferred_scale_cache.at(deferred_key);
                                        } else {
                                            Ciphertext total_sum = shared_sum_values[shared_idx];
                                            const int input_depth = shared_sum_depths[shared_idx];
                                            size_t total_width = shared_sum_widths[shared_idx];
                                            vector<Ciphertext> local_preds;
                                            local_preds.reserve(output.predicate_cols.size());
                                            for (size_t col : output.predicate_cols) {
                                                local_preds.push_back(encrypted[col][block]);
                                            }
                                            pair<Ciphertext, OpCount> local_sum =
                                                local_preds.size() == 1
                                                ? pair<Ciphertext, OpCount>{local_preds.front(), OpCount{}}
                                                : sum_bits(local_preds, worker);
                                            block_output_ops += local_sum.second;
                                            add_inplace(worker, total_sum, local_sum.first, block_output_ops);
                                            total_width += local_preds.size();
                                            auto filtered = hulk_filter_from_sum_deferred(output.filter_kind, total_sum, total_width, worker, true, out_scale);
                                            block_output_ops += filtered.second;
                                            block_output_ops.depth = max(block_output_ops.depth, input_depth + filtered.second.depth);
                                            masked = move(filtered.first);
                                            deferred_mask_cache.emplace(deferred_key, masked);
                                deferred_scale_cache[deferred_key] = out_scale;
                                        }
                                        used_deferred_filter = true;
                                    }
                                }
                            }

                            if (!used_deferred_filter) {
                                vector<Ciphertext> preds;
                                preds.reserve(output.predicate_cols.size() + (output.shared_filter >= 0 ? 1 : 0));
                                if (output.shared_filter >= 0) {
                                    preds.push_back(realize_shared(static_cast<size_t>(output.shared_filter)));
                                }
                                for (size_t col : output.predicate_cols) {
                                    preds.push_back(encrypted[col][block]);
                                }

                                const int input_depth =
                                    output.shared_filter >= 0 ? shared_depths[static_cast<size_t>(output.shared_filter)] : 0;
                                if (preds.empty()) {
                                    throw invalid_argument("aggregate output needs at least one predicate");
                                }
                                if (preds.size() == 1) {
                                    masked = preds.front();
                                    if (output.shared_filter >= 0) out_scale = shared_scale[static_cast<size_t>(output.shared_filter)];
                                    block_output_ops.depth = input_depth;
                                } else {
                                    auto filtered = apply_filter_deferred(mode, output.filter_kind, preds, worker, true, out_scale);
                                    block_output_ops += filtered.second;
                                    block_output_ops.depth = max(block_output_ops.depth, input_depth + filtered.second.depth);
                                    masked = move(filtered.first);
                                }
                            }

                            record_output_scale(output_idx, out_scale);

                            if (output.weight_col >= 0) {
                                masked = multiply_ciphertexts(
                                    worker,
                                    masked,
                                    encrypted[static_cast<size_t>(output.weight_col)][block],
                                    block_output_ops);
                                block_output_ops.depth += 1;
                                block_output_ops.weight_muls++;
                            }
                            refresh_zero_guard(worker, masked, block_output_ops);

                            state.ops[output_idx] += block_output_ops;
                            if (!state.initialized[output_idx]) {
                                state.totals[output_idx] = move(masked);
                                state.initialized[output_idx] = true;
                            } else {
                                add_inplace(worker, state.totals[output_idx], masked, state.ops[output_idx]);
                                state.ops[output_idx].agg_adds++;
                            }
                        }

                        state.shared_ops += block_shared_ops;
                    }
                    return state;
                }));
            }

            vector<WorkerAggregateState> worker_states;
            worker_states.reserve(worker_count);
            for (auto &future_state : futures) {
                worker_states.push_back(future_state.get());
            }

            for (auto &worker_state : worker_states) {
                shared_ops_total += worker_state.shared_ops;
            }

            for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
                Ciphertext total = constant_ciphertext(env, 0);
                bool initialized = false;
                OpCount output_ops;

                for (auto &worker_state : worker_states) {
                    output_ops += worker_state.ops[output_idx];
                    if (!worker_state.initialized[output_idx]) {
                        continue;
                    }
                    if (!initialized) {
                        total = move(worker_state.totals[output_idx]);
                        initialized = true;
                    } else {
                        add_inplace(env, total, worker_state.totals[output_idx], output_ops);
                        output_ops.agg_adds++;
                    }
                }
                final_totals[output_idx] = move(total);
                final_initialized[output_idx] = initialized;
                final_ops[output_idx] = output_ops;
            }
        }
    }

    // Homomorphic aggregation (identical in both modes): final_totals[o] already
    // holds HAdd over all blocks of output o; rotate-and-sum folds its slots so
    // every slot holds the aggregate. Timed as part of HE evaluation.
    if (do_slot_aggregate) {
        const auto aggregate_start = steady_clock::now();
        for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
            if (!final_initialized[output_idx]) {
                continue;   // an all-zero output stays zero
            }
            auto folded = sum_slots(move(final_totals[output_idx]), env);
            final_totals[output_idx] = move(folded.first);
            final_ops[output_idx] += folded.second;
            result.aggregate_ops += folded.second;
        }
        result.aggregate_ms = duration<double, milli>(steady_clock::now() - aggregate_start).count();
    }

    const auto query_end = steady_clock::now();
    result.elapsed_ms = duration<double, milli>(query_end - start).count();
    result.values.clear();
    result.shared_filter_ops = shared_ops_total;
    result.ops += shared_ops_total;

    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        if (!final_initialized[output_idx]) {
            final_totals[output_idx] = constant_ciphertext(env, 0);
        }
        result.ops += final_ops[output_idx];
    }

    // Decryption (timed separately): one ciphertext per output. With slot
    // aggregation the value is slot 0 (the homomorphic aggregate, mod t);
    // otherwise the legacy client-side sum of all decrypted slots.
    const auto decrypt_start = steady_clock::now();
    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const vector<uint64_t> slots = decrypt_slots(env, final_totals[output_idx]);
        if (do_slot_aggregate) {
            const uint64_t aggregate = slots.empty() ? 0 : slots[0];
            for (uint64_t v : slots) {
                if (v != aggregate) {
                    result.aggregate_slots_consistent = false;
                    break;
                }
            }
            result.values.push_back(aggregate);
        } else {
            result.values.push_back(sum_plain_slots(slots));
        }
        // Client-side completion of deferred indicator normalization (exact, mod t).
        if (output_scale[output_idx] > 1) {
            result.values.back() = mod_mul(result.values.back() % env.plain_modulus,
                                           output_scale[output_idx], env.plain_modulus);
        }
    }
    const auto end = steady_clock::now();
    result.decrypt_ms = duration<double, milli>(end - decrypt_start).count();
    result.online_e2e_ms = result.elapsed_ms + result.decrypt_ms;

    // Noise budget of every output ciphertext (diagnostic, outside both timers).
    // invariant_noise_budget(ct) returns remaining bits (positive = OK).
    // If min_budget > bits_per_prime: the computation fits in 1 fewer limb.
    result.bits_per_limb = static_cast<int>(env.coeff_modulus_total_bits / env.coeff_modulus_primes);
    int min_budget = INT_MAX, max_budget = INT_MIN;
    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const int budget = env.dec->invariant_noise_budget(final_totals[output_idx]);
        min_budget = min(min_budget, budget);
        max_budget = max(max_budget, budget);
    }
    result.min_noise_budget_bits = (min_budget == INT_MAX) ? -1 : min_budget;
    result.max_noise_budget_bits = (max_budget == INT_MIN) ? -1 : max_budget;

    const size_t top_k = configured_top_k();
    result.topk_indices = select_topk_indices(result.values, top_k);
    result.topk_values.reserve(result.topk_indices.size());
    for (size_t idx : result.topk_indices) {
        result.topk_values.push_back(result.values[idx]);
    }
    if (!opts.quiet) {
        print_topk_summary(mode, outputs, result.values, top_k);
    }

    return result;
}

// ---------------------------------------------------------------------------
// End-to-end timing / reporting helpers for drivers (mode-symmetric).
// ---------------------------------------------------------------------------

struct Stopwatch {
    steady_clock::time_point t0 = steady_clock::now();
    void reset() { t0 = steady_clock::now(); }
    double ms() const { return duration<double, milli>(steady_clock::now() - t0).count(); }
};

// Exact serialized size of a ciphertext as uploaded (no compression: fresh BFV
// ciphertexts are uniformly random and do not compress).
inline size_t ciphertext_upload_bytes(const Ciphertext &ct)
{
    return static_cast<size_t>(ct.save_size(compr_mode_type::none));
}

inline size_t ciphertext_upload_bytes(const vector<Ciphertext> &cts)
{
    size_t total = 0;
    for (const Ciphertext &ct : cts) {
        total += ciphertext_upload_bytes(ct);
    }
    return total;
}

inline size_t ciphertext_upload_bytes(const EncryptedTable &table)
{
    size_t total = 0;
    for (const auto &column : table.columns) {
        total += ciphertext_upload_bytes(column);
    }
    return total;
}

// Client-side encryption of a packed table, timed (enc_ms).
inline EncryptedTable encrypt_table_timed(const HEEnv &env, const PackedTable &table, double &enc_ms)
{
    Stopwatch sw;
    EncryptedTable encrypted = encrypt_table(env, table);
    enc_ms = sw.ms();
    return encrypted;
}

struct E2EReport {
    EvalMode mode = EvalMode::arithmetic;
    size_t limbs = 0;
    double enc_ms = 0.0;    // client encryption
    double eval_ms = 0.0;   // homomorphic operations only
    double dec_ms = 0.0;    // client decryption (+ decode)
    size_t upload_bytes = 0;
    long long hmul = 0;     // ct-ct multiplications incl. squarings
    long long rot = 0;      // Galois rotations
    bool ok = false;        // output matches the plaintext oracle
};

inline E2EReport make_e2e_report(
    EvalMode mode, const HEEnv &env, double enc_ms, size_t upload_bytes,
    const AggregateRunResult &res, bool ok)
{
    E2EReport r;
    r.mode = mode;
    r.limbs = env.coeff_modulus_primes;
    r.enc_ms = enc_ms;
    r.eval_ms = res.elapsed_ms;
    r.dec_ms = res.decrypt_ms;
    r.upload_bytes = upload_bytes;
    r.hmul = res.ops.total_ct_mul();
    r.rot = res.ops.rotations;
    r.ok = ok;
    return r;
}

// One parseable line per mode:
// E2E mode=<arithmetic|hulk> limbs=<L> enc_ms=<x> eval_ms=<y> dec_ms=<z> upload_bytes=<b> hmul=<n> rot=<r> ok=<match|MISMATCH>
inline void print_e2e_line(const E2EReport &r, ostream &os = cout)
{
    const auto flags = os.flags();
    const auto prec = os.precision();
    os << "E2E mode=" << (r.mode == EvalMode::hulk ? "hulk" : "arithmetic")
       << " limbs=" << r.limbs
       << fixed << setprecision(3)
       << " enc_ms=" << r.enc_ms
       << " eval_ms=" << r.eval_ms
       << " dec_ms=" << r.dec_ms
       << " upload_bytes=" << r.upload_bytes
       << " hmul=" << r.hmul
       << " rot=" << r.rot
       << " ok=" << (r.ok ? "match" : "MISMATCH")
       << "\n";
    os.flags(flags);
    os.precision(prec);
}

inline AggregateRunResult run_packed_aggregates(
    const PackedTable &table,
    const vector<AggregateOutput> &outputs,
    EvalMode mode,
    const HEEnv &env,
    const vector<SharedFilter> &shared_filters = {},
    size_t query_group_size = 0)
{
    const EncryptedTable encrypted = encrypt_table(env, table);
    return run_packed_aggregates_preencrypted(encrypted, outputs, mode, env, shared_filters, query_group_size);
}

// Verify that arithmetic with a shared-filter DAG produces the same values as
// arithmetic with a flat output list (no shared filters).  Both are correct ways
// to compute the same query; the DAG form applies CSE for efficiency.  Call this
// with a small encrypted table to keep cost low.
//
// Fair-comparison invariant: CSE reuse (shared_filter_dag + dag_outputs) must
// not change arithmetic results — only the mul-count profile changes.
inline void verify_arithmetic_cse_equivalence(
    const string &workload_name,
    const PackedTable &plain_table,
    const vector<AggregateOutput> &dag_outputs,
    const vector<SharedFilter> &dag_shared_filters,
    const vector<AggregateOutput> &flat_outputs)
{
    // Always use the full 8-limb safe environment so the CSE check is independent
    // of whatever optimized limb count the main benchmark is using.
    const HEEnv safe_env = make_env(EvalMode::arithmetic, /*max_threads=*/1);
    const EncryptedTable small_table = encrypt_table(safe_env, plain_table);
    const auto dag_result  = run_packed_aggregates_preencrypted(
        small_table, dag_outputs, EvalMode::arithmetic, safe_env, dag_shared_filters);
    const auto flat_result = run_packed_aggregates_preencrypted(
        small_table, flat_outputs, EvalMode::arithmetic, safe_env, {});

    if (dag_result.values != flat_result.values) {
        cerr << "[CSE-check FAIL] " << workload_name
             << ": arithmetic DAG and flat results differ!\n";
        for (size_t i = 0; i < min(dag_result.values.size(), flat_result.values.size()); ++i) {
            if (dag_result.values[i] != flat_result.values[i]) {
                cerr << "  output[" << i << "]: DAG=" << dag_result.values[i]
                     << " flat=" << flat_result.values[i] << "\n";
            }
        }
        throw runtime_error("arithmetic CSE equivalence check failed for " + workload_name);
    }
    cout << "[CSE-check] " << workload_name
         << ": arithmetic DAG == flat [pass] "
         << "(dag shared_muls=" << dag_result.shared_filter_ops.total_ct_mul()
         << " flat_total_muls=" << flat_result.ops.total_ct_mul() << ")\n";
}

struct PointwiseRunResult {
    vector<uint64_t> values;
    vector<Ciphertext> block_ciphertexts;
    OpCount ops;
    double elapsed_ms = 0.0;
    double decrypt_ms = 0.0;
    double online_e2e_ms = 0.0;
    size_t block_count = 0;
    size_t input_ciphertexts = 0;
};

inline PointwiseRunResult run_pointwise_mask_preencrypted(
    const EncryptedTable &encrypted_table,
    const vector<size_t> &predicate_cols,
    FilterKind kind,
    int weight_col,
    EvalMode mode,
    const HEEnv &env,
    size_t query_batch = 1)
{
    if (encrypted_table.columns.empty()) {
        throw invalid_argument("run_pointwise_mask_preencrypted needs at least one column");
    }

    const auto &encrypted = encrypted_table.columns;
    const size_t block_count = encrypted_table.block_count();
    const size_t input_ciphertexts = encrypted.size() * block_count;

    // Worker envs are built OUTSIDE the timed region (key copies are not HE ops).
    const size_t query_workers = query_batch <= 1 ? 1 : min(max<size_t>(1, env.max_threads), query_batch);
    const vector<HEEnv> query_pool = make_worker_pool(env, query_workers);
    // Block-level workers only arise when run_single runs on `env` itself
    // (query workers are single-threaded).
    const size_t env_block_workers = query_batch <= 1
        ? min(max<size_t>(1, env.max_threads), max<size_t>(1, block_count)) : 1;
    const vector<HEEnv> block_pool = make_worker_pool(env, env_block_workers);

    auto run_single = [&](const HEEnv &worker) {
        PointwiseRunResult single;
        single.block_count = block_count;
        single.input_ciphertexts = input_ciphertexts;
        single.values.reserve(encrypted_table.rows);

        const auto process_block = [&](size_t block) {
            vector<Ciphertext> preds;
            preds.reserve(predicate_cols.size());
            for (size_t col : predicate_cols) {
                preds.push_back(encrypted[col][block]);
            }

            OpCount block_ops;
            auto filtered = apply_filter(mode, kind, preds, worker);
            block_ops += filtered.second;
            Ciphertext masked = filtered.first;
            if (weight_col >= 0) {
                masked = multiply_ciphertexts(
                    worker,
                    masked,
                    encrypted[static_cast<size_t>(weight_col)][block],
                    block_ops);
                block_ops.depth = max(block_ops.depth, filtered.second.depth + 1);
            }
            refresh_zero_guard(worker, masked, block_ops);
            return tuple<size_t, Ciphertext, OpCount>{block, move(masked), block_ops};
        };

        const size_t worker_count = min(max<size_t>(1, worker.max_threads), max<size_t>(1, block_count));
        vector<Ciphertext> block_outputs(block_count);
        if (worker_count == 1) {
            for (size_t block = 0; block < block_count; ++block) {
                auto [block_idx, masked, block_ops] = process_block(block);
                single.ops += block_ops;
                block_outputs[block_idx] = move(masked);
            }
        } else {
            vector<future<vector<tuple<size_t, Ciphertext, OpCount>>>> futures;
            futures.reserve(worker_count);
            for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
                futures.push_back(async(launch::async, [&, worker_idx]() {
                    if (&worker != &env || worker_idx >= block_pool.size()) {
                        throw logic_error("pointwise block worker pool not prebuilt");
                    }
                    const HEEnv &block_worker = block_pool[worker_idx];
                    vector<tuple<size_t, Ciphertext, OpCount>> blocks;
                    for (size_t block = worker_idx; block < block_count; block += worker_count) {
                        vector<Ciphertext> preds;
                        preds.reserve(predicate_cols.size());
                        for (size_t col : predicate_cols) {
                            preds.push_back(encrypted[col][block]);
                        }

                        OpCount block_ops;
                        auto filtered = apply_filter(mode, kind, preds, block_worker);
                        block_ops += filtered.second;
                        Ciphertext masked = filtered.first;
                        if (weight_col >= 0) {
                            masked = multiply_ciphertexts(
                                block_worker,
                                masked,
                                encrypted[static_cast<size_t>(weight_col)][block],
                                block_ops);
                            block_ops.depth = max(block_ops.depth, filtered.second.depth + 1);
                        }
                        refresh_zero_guard(block_worker, masked, block_ops);
                        blocks.push_back(tuple<size_t, Ciphertext, OpCount>{block, move(masked), block_ops});
                    }
                    return blocks;
                }));
            }

            for (auto &future_blocks : futures) {
                auto blocks = future_blocks.get();
                for (auto &block_result : blocks) {
                    const size_t block = get<0>(block_result);
                    block_outputs[block] = move(get<1>(block_result));
                    single.ops += get<2>(block_result);
                }
            }
        }

        for (size_t block = 0; block < block_count; ++block) {
            vector<uint64_t> slots = decrypt_slots(worker, block_outputs[block]);
            const size_t offset = block * worker.slot_count;
            const size_t live = min(worker.slot_count, encrypted_table.rows - offset);
            slots.resize(live);
            single.values.insert(single.values.end(), slots.begin(), slots.end());
        }
        single.block_ciphertexts = move(block_outputs);
        return single;
    };

    PointwiseRunResult result;
    const auto start = steady_clock::now();
    if (query_batch <= 1) {
        result = run_single(env);
    } else {
        result.block_count = block_count;
        result.input_ciphertexts = input_ciphertexts;
        vector<future<pair<size_t, PointwiseRunResult>>> futures;
        futures.reserve(query_workers);
        for (size_t worker_idx = 0; worker_idx < query_workers; ++worker_idx) {
            futures.push_back(async(launch::async, [&, worker_idx]() {
                const HEEnv &query_worker = query_workers <= 1 ? env : query_pool[worker_idx];
                pair<size_t, PointwiseRunResult> first_result;
                bool have_first = false;
                OpCount total_ops;
                for (size_t query_idx = worker_idx; query_idx < query_batch; query_idx += query_workers) {
                    PointwiseRunResult single = run_single(query_worker);
                    total_ops += single.ops;
                    if (!have_first) {
                        first_result = {query_idx, move(single)};
                        have_first = true;
                    }
                }
                if (!have_first) {
                    return pair<size_t, PointwiseRunResult>{static_cast<size_t>(-1), PointwiseRunResult{}};
                }
                first_result.second.ops = total_ops;
                return first_result;
            }));
        }

        size_t best_idx = static_cast<size_t>(-1);
        for (auto &future_result : futures) {
            auto [query_idx, single] = future_result.get();
            result.ops += single.ops;
            if (query_idx == static_cast<size_t>(-1)) {
                continue;
            }
            if (best_idx == static_cast<size_t>(-1) || query_idx < best_idx) {
                best_idx = query_idx;
                result.values = move(single.values);
                result.block_ciphertexts = move(single.block_ciphertexts);
            }
        }
    }
    const auto query_end = steady_clock::now();
    result.elapsed_ms = duration<double, milli>(query_end - start).count();
    const auto end = steady_clock::now();
    result.decrypt_ms = duration<double, milli>(end - query_end).count();
    result.online_e2e_ms = duration<double, milli>(end - start).count();
    return result;
}

inline PointwiseRunResult run_pointwise_mask(
    const PackedTable &table,
    const vector<size_t> &predicate_cols,
    FilterKind kind,
    int weight_col,
    EvalMode mode,
    const HEEnv &env,
    size_t query_batch = 1)
{
    const EncryptedTable encrypted = encrypt_table(env, table);
    return run_pointwise_mask_preencrypted(encrypted, predicate_cols, kind, weight_col, mode, env, query_batch);
}

inline string matches_string(bool match)
{
    return match ? "match" : "MISMATCH";
}

} // namespace hulk
