// Implementation of the SEAL-compatible facade (include/seal/seal.h) on HEonGPU BFV.
// See the header for the semantics that keep the CPU (SEAL) and GPU runs comparable.
#include "seal/seal.h"

#include <heongpu/host/bfv/ciphertext.cuh>
#include <heongpu/host/bfv/context.cuh>
#include <heongpu/host/bfv/decryptor.cuh>
#include <heongpu/host/bfv/encoder.cuh>
#include <heongpu/host/bfv/encryptor.cuh>
#include <heongpu/host/bfv/evaluationkey.cuh>
#include <heongpu/host/bfv/keygenerator.cuh>
#include <heongpu/host/bfv/operator.cuh>
#include <heongpu/host/bfv/plaintext.cuh>
#include <heongpu/host/bfv/publickey.cuh>
#include <heongpu/host/bfv/secretkey.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>

namespace seal {
namespace gpu_detail {

constexpr auto BFV = heongpu::Scheme::BFV;
using HCtx = heongpu::HEContext<BFV>;
using HCt  = heongpu::Ciphertext<BFV>;
using HPt  = heongpu::Plaintext<BFV>;

inline void sync()
{
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
    }
}

struct ContextImpl {
    HCtx ctx;
    std::unique_ptr<heongpu::HEEncoder<BFV>> encoder;
    std::unique_ptr<heongpu::HEArithmeticOperator<BFV>> op;
    std::size_t n = 0;
    std::uint64_t t = 0;
    std::size_t q_size = 0;   // data primes (all but the special prime)
};

struct CiphertextImpl {
    HCt ct;
    std::shared_ptr<ContextImpl> ctx;
};

struct PlaintextImpl {
    HPt pt;
    std::shared_ptr<ContextImpl> ctx;
    bool is_constant = false;          // every slot holds `constant`
    std::uint64_t constant = 0;
};

struct SecretKeyImpl  { heongpu::Secretkey<BFV> key; };
struct PublicKeyImpl  { heongpu::Publickey<BFV> key; };
struct RelinKeysImpl  { heongpu::Relinkey<BFV> key; };
struct GaloisKeysImpl {
    std::shared_ptr<ContextImpl> ctx;
    std::shared_ptr<SecretKeyImpl> sk;
    std::vector<int> shifts;
    std::unique_ptr<heongpu::Galoiskey<BFV>> key;   // generated on first use
    std::mutex mu;
    heongpu::Galoiskey<BFV> &get()
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!key) {
            heongpu::HEKeyGenerator<BFV> kg(ctx->ctx);
            key = std::make_unique<heongpu::Galoiskey<BFV>>(ctx->ctx, shifts);
            kg.generate_galois_key(*key, sk->key);
            sync();
            std::fprintf(stderr, "[gpu-shim] Galois keys generated lazily (%zu shifts)\n", shifts.size());
        }
        return *key;
    }
};

// ---------------------------------------------------------------- helpers
inline std::shared_ptr<CiphertextImpl> fresh(const std::shared_ptr<ContextImpl> &ctx)
{
    auto p = std::make_shared<CiphertextImpl>();
    p->ctx = ctx;
    p->ct = HCt(ctx->ctx);
    return p;
}
inline HCt &in(const Ciphertext &c)
{
    if (!c.impl) throw std::invalid_argument("gpu-shim: use of an empty Ciphertext");
    return const_cast<HCt &>(c.impl->ct);
}
inline HPt &in(const Plaintext &p)
{
    if (!p.impl) throw std::invalid_argument("gpu-shim: use of an empty Plaintext");
    return const_cast<HPt &>(p.impl->pt);
}
// copy-on-write: make `c` the unique owner of its device buffer before an in-place op
inline HCt &own(Ciphertext &c)
{
    if (!c.impl) throw std::invalid_argument("gpu-shim: in-place op on an empty Ciphertext");
    if (c.impl.use_count() > 1) {
        auto p = std::make_shared<CiphertextImpl>();
        p->ctx = c.impl->ctx;
        p->ct = c.impl->ct;          // deep device copy
        c.impl = std::move(p);
    }
    return c.impl->ct;
}

// ---------------------------------------------------------------- SEAL prime search
inline std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m)
{
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) % m);
}
inline std::uint64_t powmod(std::uint64_t a, std::uint64_t e, std::uint64_t m)
{
    std::uint64_t r = 1 % m; a %= m;
    while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; }
    return r;
}
inline bool is_prime(std::uint64_t n)
{
    if (n < 2) return false;
    for (std::uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        if (n % p == 0) return n == p;
    }
    std::uint64_t d = n - 1; int s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (std::uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        std::uint64_t x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int r = 1; r < s; ++r) { x = mulmod(x, x, n); if (x == n - 1) { comp = false; break; } }
        if (comp) return false;
    }
    return true;
}
// SEAL util::get_primes: the `count` largest primes p = 1 (mod factor) with bit_size bits
inline std::vector<std::uint64_t> get_primes(std::uint64_t factor, int bit_size, std::size_t count)
{
    std::vector<std::uint64_t> out;
    std::uint64_t value = ((std::uint64_t(1) << bit_size) - factor) + 1;
    const std::uint64_t lower = std::uint64_t(1) << (bit_size - 1);
    while (count > 0 && value > lower) {
        if (is_prime(value)) { out.push_back(value); --count; }
        value -= factor;
    }
    if (count > 0) throw std::logic_error("gpu-shim: failed to find enough qualifying primes");
    return out;
}

}  // namespace gpu_detail

using namespace gpu_detail;

// ---------------------------------------------------------------- moduli
std::vector<Modulus> CoeffModulus::Create(std::size_t n, std::vector<int> bit_sizes)
{
    std::map<int, std::size_t> count;
    for (int b : bit_sizes) count[b]++;
    std::map<int, std::vector<std::uint64_t>> primes;
    for (auto &[b, c] : count) primes[b] = get_primes(2 * n, b, c);
    std::vector<Modulus> out;
    for (int b : bit_sizes) { out.emplace_back(primes[b].front()); primes[b].erase(primes[b].begin()); }
    return out;
}

Modulus PlainModulus::Batching(std::size_t n, int bit_size)
{
    return Modulus(get_primes(2 * n, bit_size, 1).front());
}

// ---------------------------------------------------------------- context
SEALContext::SEALContext(const EncryptionParameters &parms, bool, sec_level_type)
    : data_(std::make_shared<ContextData>(parms))
{
    try {
        const auto &q = parms.coeff_modulus();
        if (q.size() < 2) throw std::invalid_argument("need >= 2 coefficient primes (data + special)");
        std::vector<std::uint64_t> Q, P;
        for (std::size_t i = 0; i + 1 < q.size(); ++i) Q.push_back(q[i].value());
        P.push_back(q.back().value());                    // SEAL: last prime = special prime
        auto c = std::make_shared<ContextImpl>();
        c->ctx = heongpu::GenHEContext<BFV>();
        c->ctx->set_poly_modulus_degree(parms.poly_modulus_degree());
        c->ctx->set_coeff_modulus_values(Q, P);
        c->ctx->set_plain_modulus(static_cast<int>(parms.plain_modulus().value()));
        c->ctx->generate();
        c->encoder = std::make_unique<heongpu::HEEncoder<BFV>>(c->ctx);
        c->op = std::make_unique<heongpu::HEArithmeticOperator<BFV>>(c->ctx, *c->encoder);
        c->n = parms.poly_modulus_degree();
        c->t = parms.plain_modulus().value();
        c->q_size = Q.size();
        sync();
        impl = std::move(c);
    } catch (const std::exception &e) {
        error_ = e.what();
    }
}

std::size_t Ciphertext::size() const noexcept { return impl ? 2 : 0; }

std::streamoff Ciphertext::save_size(compr_mode_type) const
{
    if (!impl) return 0;
    return static_cast<std::streamoff>(2 * impl->ctx->n * impl->ctx->q_size * sizeof(std::uint64_t) + 16);
}

// ---------------------------------------------------------------- keys
KeyGenerator::KeyGenerator(const SEALContext &context) : ctx_(context.impl)
{
    sk_.impl = std::make_shared<SecretKeyImpl>();
    sk_.impl->key = heongpu::Secretkey<BFV>(ctx_->ctx);
    heongpu::HEKeyGenerator<BFV> kg(ctx_->ctx);
    kg.generate_secret_key(sk_.impl->key);
    sync();
}

void KeyGenerator::create_public_key(PublicKey &pk) const
{
    pk.impl = std::make_shared<PublicKeyImpl>();
    pk.impl->key = heongpu::Publickey<BFV>(ctx_->ctx);
    heongpu::HEKeyGenerator<BFV> kg(ctx_->ctx);
    kg.generate_public_key(pk.impl->key, sk_.impl->key);
    sync();
}

void KeyGenerator::create_relin_keys(RelinKeys &rk) const
{
    rk.impl = std::make_shared<RelinKeysImpl>();
    rk.impl->key = heongpu::Relinkey<BFV>(ctx_->ctx);
    heongpu::HEKeyGenerator<BFV> kg(ctx_->ctx);
    kg.generate_relin_key(rk.impl->key, sk_.impl->key);
    sync();
}

void KeyGenerator::create_galois_keys(const std::vector<int> &steps, GaloisKeys &gk) const
{
    gk.impl = std::make_shared<GaloisKeysImpl>();
    gk.impl->ctx = ctx_;
    gk.impl->sk = sk_.impl;
    gk.impl->shifts.clear();
    for (int s : steps) if (s != 0) gk.impl->shifts.push_back(s);   // 0 = column rotation (always present)
}

void KeyGenerator::create_galois_keys(GaloisKeys &gk) const
{
    std::vector<int> steps;                      // SEAL default: +/- powers of two
    for (std::size_t s = 1; s < ctx_->n / 2; s <<= 1) { steps.push_back(int(s)); steps.push_back(-int(s)); }
    create_galois_keys(steps, gk);
}

// ---------------------------------------------------------------- encrypt / decrypt / encode
Encryptor::Encryptor(const SEALContext &context, const PublicKey &pk) : ctx_(context.impl), pk_(pk) {}

void Encryptor::encrypt(const Plaintext &plain, Ciphertext &destination) const
{
    heongpu::HEEncryptor<BFV> enc(ctx_->ctx, const_cast<heongpu::Publickey<BFV> &>(pk_.impl->key));
    auto out = fresh(ctx_);
    enc.encrypt(out->ct, in(plain));
    sync();
    destination.impl = std::move(out);
}

Decryptor::Decryptor(const SEALContext &context, const SecretKey &sk) : ctx_(context.impl), sk_(sk) {}

void Decryptor::decrypt(const Ciphertext &encrypted, Plaintext &destination)
{
    heongpu::HEDecryptor<BFV> dec(ctx_->ctx, sk_.impl->key);
    auto p = std::make_shared<PlaintextImpl>();
    p->ctx = ctx_;
    p->pt = HPt(ctx_->ctx);
    dec.decrypt(p->pt, in(encrypted));
    sync();
    destination.impl = std::move(p);
}

int Decryptor::invariant_noise_budget(const Ciphertext &encrypted)
{
    heongpu::HEDecryptor<BFV> dec(ctx_->ctx, sk_.impl->key);
    const int b = dec.remainder_noise_budget(in(encrypted));
    sync();
    return std::max(0, b);
}

BatchEncoder::BatchEncoder(const SEALContext &context) : ctx_(context.impl) {}

std::size_t BatchEncoder::slot_count() const noexcept { return ctx_->n; }

void BatchEncoder::encode(const std::vector<std::uint64_t> &values, Plaintext &destination) const
{
    std::vector<std::uint64_t> v(values);
    v.resize(ctx_->n, 0);
    auto p = std::make_shared<PlaintextImpl>();
    p->ctx = ctx_;
    p->pt = HPt(ctx_->ctx);
    ctx_->encoder->encode(p->pt, v);
    p->is_constant = std::all_of(v.begin(), v.end(), [&](std::uint64_t x) { return x == v[0]; });
    p->constant = v[0] % ctx_->t;
    sync();
    destination.impl = std::move(p);
}

void BatchEncoder::decode(const Plaintext &plain, std::vector<std::uint64_t> &destination) const
{
    destination.clear();
    ctx_->encoder->decode(destination, in(plain));
    sync();
}

// ---------------------------------------------------------------- evaluator
Evaluator::Evaluator(const SEALContext &context) : ctx_(context.impl) {}

#define OP (*ctx_->op)

void Evaluator::add(const Ciphertext &a, const Ciphertext &b, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.add(in(a), in(b), o->ct); sync(); d.impl = std::move(o); }
void Evaluator::add_inplace(Ciphertext &a, const Ciphertext &b) const
{ HCt &x = own(a); OP.add_inplace(x, in(b)); sync(); }
void Evaluator::sub(const Ciphertext &a, const Ciphertext &b, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.sub(in(a), in(b), o->ct); sync(); d.impl = std::move(o); }
void Evaluator::sub_inplace(Ciphertext &a, const Ciphertext &b) const
{ HCt &x = own(a); OP.sub_inplace(x, in(b)); sync(); }
void Evaluator::negate(const Ciphertext &a, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.negate(in(a), o->ct); sync(); d.impl = std::move(o); }
void Evaluator::negate_inplace(Ciphertext &a) const
{ HCt &x = own(a); OP.negate_inplace(x); sync(); }
void Evaluator::multiply(const Ciphertext &a, const Ciphertext &b, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.multiply(in(a), in(b), o->ct); sync(); d.impl = std::move(o); }
void Evaluator::multiply_inplace(Ciphertext &a, const Ciphertext &b) const
{ Ciphertext r; multiply(a, b, r); a = std::move(r); }
void Evaluator::square(const Ciphertext &a, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.multiply(in(a), in(a), o->ct); sync(); d.impl = std::move(o); }
void Evaluator::square_inplace(Ciphertext &a) const
{ Ciphertext r; square(a, r); a = std::move(r); }
void Evaluator::relinearize_inplace(Ciphertext &a, const RelinKeys &rk) const
{ HCt &x = own(a); OP.relinearize_inplace(x, const_cast<heongpu::Relinkey<BFV> &>(rk.impl->key)); sync(); }
void Evaluator::relinearize(const Ciphertext &a, const RelinKeys &rk, Ciphertext &d) const
{ d = a; relinearize_inplace(d, rk); }
void Evaluator::add_plain(const Ciphertext &a, const Plaintext &p, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.add_plain(in(a), in(p), o->ct); sync(); d.impl = std::move(o); }
void Evaluator::add_plain_inplace(Ciphertext &a, const Plaintext &p) const
{ HCt &x = own(a); OP.add_plain_inplace(x, in(p)); sync(); }
void Evaluator::sub_plain(const Ciphertext &a, const Plaintext &p, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.sub_plain(in(a), in(p), o->ct); sync(); d.impl = std::move(o); }
void Evaluator::sub_plain_inplace(Ciphertext &a, const Plaintext &p) const
{ HCt &x = own(a); OP.sub_plain_inplace(x, in(p)); sync(); }

void Evaluator::multiply_plain(const Ciphertext &a, const Plaintext &p, Ciphertext &d) const
{
    auto o = fresh(ctx_);
    if (p.impl && p.impl->is_constant) OP.multiply_plain_scalar(in(a), p.impl->constant, o->ct);
    else OP.multiply_plain(in(a), in(p), o->ct);
    sync();
    d.impl = std::move(o);
}
void Evaluator::multiply_plain_inplace(Ciphertext &a, const Plaintext &p) const
{ Ciphertext r; multiply_plain(a, p, r); a = std::move(r); }

void Evaluator::rotate_rows(const Ciphertext &a, int steps, const GaloisKeys &gk, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.rotate_rows(in(a), o->ct, gk.impl->get(), steps); sync(); d.impl = std::move(o); }
void Evaluator::rotate_rows_inplace(Ciphertext &a, int steps, const GaloisKeys &gk) const
{ Ciphertext r; rotate_rows(a, steps, gk, r); a = std::move(r); }
void Evaluator::rotate_columns(const Ciphertext &a, const GaloisKeys &gk, Ciphertext &d) const
{ auto o = fresh(ctx_); OP.rotate_columns(in(a), o->ct, gk.impl->get()); sync(); d.impl = std::move(o); }
void Evaluator::rotate_columns_inplace(Ciphertext &a, const GaloisKeys &gk) const
{ Ciphertext r; rotate_columns(a, gk, r); a = std::move(r); }

#undef OP

}  // namespace seal
