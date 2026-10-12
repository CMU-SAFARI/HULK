// SEAL-compatible facade over HEonGPU (BFV), for running the HULK harness
// (TCHES/HE_workload_common.h and every driver that includes it) unmodified on a GPU.
//
// Only the subset of the Microsoft SEAL 4.1 API that the harness uses is provided.
// Every class is a thin pimpl handle; the implementation (seal_heongpu.cu) is compiled
// with nvcc against HEonGPU, so the drivers themselves compile with the host compiler.
//
// Semantics that matter for a fair CPU/GPU comparison:
//  * Coefficient and plaintext moduli are the SAME primes SEAL would pick
//    (CoeffModulus::Create / PlainModulus::Batching are re-implemented with SEAL's
//    prime search), and the last prime is the key-switching special prime, as in SEAL.
//  * Every Evaluator call ends with cudaDeviceSynchronize(), so the host-side timers of
//    the harness measure completed GPU work, per operation and per stage.
//  * Ciphertext/Plaintext have value semantics (copy-on-write): a copy is shared until
//    one side is modified in place, then it is deep-copied on the device.
//  * multiply_plain with a constant plaintext (all slots equal) uses HEonGPU's
//    multiply_plain_scalar, the counterpart of SEAL's single-coefficient fast path.
//  * square() is multiply(x, x): HEonGPU has no dedicated squaring kernel.
//  * GaloisKeys are generated lazily on the first rotation (the harness rotates only
//    when E2E_SLOT_AGGREGATE / TPC_SLOT_AGGREGATE are set; bench_ops warms up first).
#pragma once

#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <string>
#include <vector>

namespace seal {

enum class scheme_type : std::uint8_t { none = 0, bfv = 1 };
enum class sec_level_type : int { none = 0, tc128 = 128, tc192 = 192, tc256 = 256 };
enum class compr_mode_type : std::uint8_t { none = 0, zlib = 1, zstd = 2 };

class Modulus {
public:
    Modulus() = default;
    Modulus(std::uint64_t value) : value_(value) {}
    std::uint64_t value() const noexcept { return value_; }
    int bit_count() const noexcept
    {
        int b = 0;
        for (std::uint64_t v = value_; v; v >>= 1) ++b;
        return b;
    }
    bool operator==(const Modulus &o) const noexcept { return value_ == o.value_; }
private:
    std::uint64_t value_ = 0;
};

struct CoeffModulus {
    // Same prime selection as SEAL: for each bit size, the largest primes
    // p = 1 (mod 2N) below 2^bits, in decreasing order.
    static std::vector<Modulus> Create(std::size_t poly_modulus_degree, std::vector<int> bit_sizes);
};

struct PlainModulus {
    static Modulus Batching(std::size_t poly_modulus_degree, int bit_size);
};

class EncryptionParameters {
public:
    EncryptionParameters() = default;
    explicit EncryptionParameters(scheme_type s) : scheme_(s) {}
    void set_poly_modulus_degree(std::size_t n) { n_ = n; }
    void set_coeff_modulus(const std::vector<Modulus> &q) { q_ = q; }
    void set_plain_modulus(const Modulus &t) { t_ = t; }
    void set_plain_modulus(std::uint64_t t) { t_ = Modulus(t); }
    std::size_t poly_modulus_degree() const noexcept { return n_; }
    const std::vector<Modulus> &coeff_modulus() const noexcept { return q_; }
    const Modulus &plain_modulus() const noexcept { return t_; }
    scheme_type scheme() const noexcept { return scheme_; }
private:
    scheme_type scheme_ = scheme_type::bfv;
    std::size_t n_ = 0;
    std::vector<Modulus> q_;
    Modulus t_;
};

namespace gpu_detail {
struct ContextImpl;
struct CiphertextImpl;
struct PlaintextImpl;
struct SecretKeyImpl;
struct PublicKeyImpl;
struct RelinKeysImpl;
struct GaloisKeysImpl;
}

class SEALContext {
public:
    class ContextData {
    public:
        struct Qualifiers { bool parameters_set = true; bool using_batching = true; };
        explicit ContextData(const EncryptionParameters &p) : parms_(p)
        {
            const std::uint64_t t = p.plain_modulus().value(), two_n = 2 * p.poly_modulus_degree();
            q_.using_batching = two_n != 0 && t % two_n == 1;
        }
        const EncryptionParameters &parms() const noexcept { return parms_; }
        Qualifiers qualifiers() const noexcept { return q_; }
    private:
        EncryptionParameters parms_;
        Qualifiers q_;
    };

    SEALContext(const EncryptionParameters &parms, bool expand_mod_chain = true,
                sec_level_type sec_level = sec_level_type::tc128);
    bool parameters_set() const noexcept { return error_.empty(); }
    std::string parameter_error_message() const { return error_.empty() ? "valid" : error_; }
    std::shared_ptr<const ContextData> key_context_data() const { return data_; }
    std::shared_ptr<const ContextData> first_context_data() const { return data_; }

    std::shared_ptr<gpu_detail::ContextImpl> impl;   // internal
private:
    std::shared_ptr<const ContextData> data_;
    std::string error_;
};

class Plaintext {
public:
    Plaintext() = default;
    std::shared_ptr<const gpu_detail::PlaintextImpl> impl;   // internal, immutable once written
};

class Ciphertext {
public:
    Ciphertext() = default;
    std::size_t size() const noexcept;
    std::streamoff save_size(compr_mode_type = compr_mode_type::none) const;
    std::shared_ptr<gpu_detail::CiphertextImpl> impl;        // internal, copy-on-write
};

class SecretKey  { public: std::shared_ptr<gpu_detail::SecretKeyImpl>  impl; };
class PublicKey  { public: std::shared_ptr<gpu_detail::PublicKeyImpl>  impl; };
class RelinKeys  { public: std::shared_ptr<gpu_detail::RelinKeysImpl>  impl; };
class GaloisKeys { public: std::shared_ptr<gpu_detail::GaloisKeysImpl> impl; };

class KeyGenerator {
public:
    explicit KeyGenerator(const SEALContext &context);
    const SecretKey &secret_key() const { return sk_; }
    void create_public_key(PublicKey &pk) const;
    void create_relin_keys(RelinKeys &rk) const;
    void create_galois_keys(GaloisKeys &gk) const;
    void create_galois_keys(const std::vector<int> &steps, GaloisKeys &gk) const;
private:
    std::shared_ptr<gpu_detail::ContextImpl> ctx_;
    SecretKey sk_;
};

class Encryptor {
public:
    Encryptor(const SEALContext &context, const PublicKey &pk);
    void encrypt(const Plaintext &plain, Ciphertext &destination) const;
private:
    std::shared_ptr<gpu_detail::ContextImpl> ctx_;
    PublicKey pk_;
};

class Decryptor {
public:
    Decryptor(const SEALContext &context, const SecretKey &sk);
    void decrypt(const Ciphertext &encrypted, Plaintext &destination);
    int invariant_noise_budget(const Ciphertext &encrypted);
private:
    std::shared_ptr<gpu_detail::ContextImpl> ctx_;
    SecretKey sk_;
};

class BatchEncoder {
public:
    explicit BatchEncoder(const SEALContext &context);
    void encode(const std::vector<std::uint64_t> &values, Plaintext &destination) const;
    void decode(const Plaintext &plain, std::vector<std::uint64_t> &destination) const;
    std::size_t slot_count() const noexcept;
private:
    std::shared_ptr<gpu_detail::ContextImpl> ctx_;
};

class Evaluator {
public:
    explicit Evaluator(const SEALContext &context);

    void add(const Ciphertext &a, const Ciphertext &b, Ciphertext &destination) const;
    void add_inplace(Ciphertext &a, const Ciphertext &b) const;
    void sub(const Ciphertext &a, const Ciphertext &b, Ciphertext &destination) const;
    void sub_inplace(Ciphertext &a, const Ciphertext &b) const;
    void negate(const Ciphertext &a, Ciphertext &destination) const;
    void negate_inplace(Ciphertext &a) const;
    void multiply(const Ciphertext &a, const Ciphertext &b, Ciphertext &destination) const;
    void multiply_inplace(Ciphertext &a, const Ciphertext &b) const;
    void square(const Ciphertext &a, Ciphertext &destination) const;
    void square_inplace(Ciphertext &a) const;
    void relinearize_inplace(Ciphertext &a, const RelinKeys &rk) const;
    void relinearize(const Ciphertext &a, const RelinKeys &rk, Ciphertext &destination) const;
    void add_plain(const Ciphertext &a, const Plaintext &p, Ciphertext &destination) const;
    void add_plain_inplace(Ciphertext &a, const Plaintext &p) const;
    void sub_plain(const Ciphertext &a, const Plaintext &p, Ciphertext &destination) const;
    void sub_plain_inplace(Ciphertext &a, const Plaintext &p) const;
    void multiply_plain(const Ciphertext &a, const Plaintext &p, Ciphertext &destination) const;
    void multiply_plain_inplace(Ciphertext &a, const Plaintext &p) const;
    void rotate_rows(const Ciphertext &a, int steps, const GaloisKeys &gk, Ciphertext &destination) const;
    void rotate_rows_inplace(Ciphertext &a, int steps, const GaloisKeys &gk) const;
    void rotate_columns(const Ciphertext &a, const GaloisKeys &gk, Ciphertext &destination) const;
    void rotate_columns_inplace(Ciphertext &a, const GaloisKeys &gk) const;
private:
    std::shared_ptr<gpu_detail::ContextImpl> ctx_;
};

}  // namespace seal
