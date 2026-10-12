/*
 * Approximate nearest-neighbour (ANN) search over an encrypted database of binary codes,
 * end to end under encryption: LSH candidate generation, Hamming re-ranking, radius selection.
 *
 * Setting: outsourced vector database. The data owner encrypts the database and the client
 * encrypts its query, so both operands of every bitwise operation are ciphertexts. Embeddings
 * (d = 32) are hashed with SimHash (signs of random projections) before encryption:
 *   L = 8 LSH tables, each with a k = 12-bit bucket key (96 bits), and a 64-bit re-ranking code.
 *
 * Layout (item-major, one database item per slot, 32,768 items per ciphertext block),
 * bit-sliced: one ciphertext per bit plane (96 key planes + 64 code planes). The query sends
 * the same 160 bits, each as one ciphertext with the bit in every slot.
 *
 * Pipeline (arithmetic approach):
 *   and_test   bulk bitwise   LSH candidates: per table, key equality = AND over 12 bits of
 *                               XNOR(q, d) = 1 - q - d + 2qd (1 HMul per bit, then an HMul tree);
 *                               candidate = OR over the 8 tables (1 - prod(1 - e_t))
 *   xor        bulk bitwise   XOR(q, d) = q + d - 2qd on the 64 code bits (1 HMul each);
 *                               Hamming distance h = sum of the XOR bits (HAdd)
 *   select     comparison     radius check [h <= R], bounded comparison polynomial over {0..64}
 *   weight     arithmetic     returned score = candidate * [h <= R] * (64 - h) (2 HMul);
 *                             the client decrypts the scores and keeps the top-k (non-zero = hit)
 *
 * Result retrieval (second round, as in Tiptoe): every item carries an encrypted payload of
 * W = 8 words of 16 bits (one ciphertext per word, the data owner's record: document id, URL
 * hash, ...). The client sends k encrypted one-hot selectors, one per chosen item, and the server
 * returns all k payloads packed in one ciphertext:
 *   retrieve_select     arithmetic   selector * payload word (W HMul per result)
 *   retrieve_aggregate  arithmetic   shift word w by w slots (W - 1 rotations), fold every slot onto
 *                                    its residue mod W (rotations by W, 2W, ..., plus the row swap),
 *                                    mask the W slots of result r into slots [rW, rW + W) and add.
 *                                    Word w of result r lands in slot rW + (i_r + w) mod W; the
 *                                    client knows i_r and undoes the cyclic shift.
 *
 * Arithmetic vs HULK (only the bulk bitwise stages differ; select, weight and retrieval are identical):
 *   arithmetic   XOR = q + d - 2qd, XNOR = 1 - q - d + 2qd (1 HMul per bit); AND = HMul tree;
 *                OR = 1 - prod(1 - x)
 *   HULK         sum then indicator: S = q - d (HAdd), XOR = [S != 0] = S^2, XNOR = 1 - S^2 (1 squaring
 *                per bit); table match = [sum of the 12 XNOR bits = 12]; candidate = [sum of the 8 matches
 *                >= 1] with its final 1/K! deferred to the client (it only feeds products).
 *
 * Timing as in the other e2e drivers (enc_ms / eval_ms / dec_ms); eval_ms is the server time of both
 * rounds, the client step in between (decrypt, choose top-k, encrypt selectors) is client time.
 * Every returned slot and payload word is checked against a plaintext oracle.
 */
#include "HE_workload_common.h"
#include <array>
#include <random>
using namespace hulk;

static constexpr size_t kDim = 32, kTables = 8, kKeyBits = 12, kCodeBits = 64, kWords = 8;

struct AnnCfg : RunConfig { size_t radius = 10; size_t planted = 64; size_t k = 10; };

static AnnCfg parse_cfg(int argc, char **argv){
    AnnCfg c; c.threads = 1;
    for(int i=1;i<argc;++i){ string a=argv[i];
        auto nx=[&]{ return stoull(argv[++i]); };
        if(a=="--radius") c.radius=nx();
        else if(a=="--planted") c.planted=nx();
        else if(a=="--k") c.k=max<size_t>(1,nx());
        else if(a=="--seed") c.seed=nx();
        else if(a=="--threads") c.threads=max<size_t>(1,nx());
        else if(a=="--mode") c.mode=parse_mode_value(argv[++i]);
        else if(a=="--limbs"||a=="--arithmetic-limbs") c.arithmetic_limbs=max<size_t>(1,nx());
        else if(a=="--hulk-limbs") c.hulk_limbs=max<size_t>(1,nx());
        else throw invalid_argument("unknown arg: "+a);
    }
    return c;
}

// SimHash bits of a vector for a set of hyperplanes
static vector<uint64_t> simhash(const vector<double> &v, const vector<vector<double>> &H){
    vector<uint64_t> bits(H.size());
    for(size_t j=0;j<H.size();++j){ double s=0; for(size_t t=0;t<kDim;++t) s+=v[t]*H[j][t]; bits[j]= s>=0; }
    return bits;
}

int main(int argc, char **argv){
    const AnnCfg cfg = parse_cfg(argc, argv);
    const BenchmarkEnvs envs = make_benchmark_envs(cfg);
    const size_t N = envs.slot_count, R = cfg.radius;

    // ---- data: embeddings, SimHash hyperplanes, the query and planted near neighbours
    mt19937_64 rng(cfg.seed); normal_distribution<double> G(0,1);
    auto randvec=[&]{ vector<double> v(kDim); for(auto &x: v) x=G(rng); return v; };
    vector<vector<double>> Hkey(kTables*kKeyBits), Hcode(kCodeBits);
    for(auto &h: Hkey) h=randvec(); for(auto &h: Hcode) h=randvec();
    const vector<double> query = randvec();
    vector<vector<uint64_t>> key(N), code(N);                // per item
    for(size_t i=0;i<N;++i){
        vector<double> v = randvec();
        if(i % (N / cfg.planted) == 0){ const double eps = 0.05 + 0.25*double(i % 7)/6.0;   // near neighbours
            for(size_t t=0;t<kDim;++t) v[t] = query[t] + eps*v[t]; }
        key[i] = simhash(v, Hkey); code[i] = simhash(v, Hcode);
    }
    const auto qkey = simhash(query, Hkey), qcode = simhash(query, Hcode);
    vector<array<uint64_t, kWords>> payload(N);              // per-item record, 16-bit words
    uniform_int_distribution<uint64_t> word(0, 65535);
    for(auto &rec: payload) for(auto &w: rec) w = word(rng);

    // plaintext oracle
    vector<uint64_t> expect(N,0); size_t ncand=0, nhit=0;
    for(size_t i=0;i<N;++i){
        bool cand=false;
        for(size_t t=0;t<kTables && !cand;++t){ bool eq=true;
            for(size_t b=0;b<kKeyBits;++b) eq = eq && key[i][t*kKeyBits+b]==qkey[t*kKeyBits+b];
            cand = eq; }
        size_t h=0; for(size_t b=0;b<kCodeBits;++b) h += code[i][b]!=qcode[b];
        ncand += cand; if(cand && h<=R){ expect[i] = kCodeBits - h; nhit++; }
    }
    cout << "ANN e2e: " << N << " encrypted items, L=" << kTables << " x " << kKeyBits << "-bit LSH keys, "
         << kCodeBits << "-bit re-rank code, radius " << R << "; " << ncand << " LSH candidates, "
         << nhit << " returned neighbours\n" << fixed << setprecision(3);

    bool all_ok = true;
    for(int pass=0; pass<2; ++pass){
    const EvalMode mode = pass ? EvalMode::hulk : EvalMode::arithmetic;
    if(mode==EvalMode::arithmetic && !wants_arithmetic(cfg.mode)) continue;
    if(mode==EvalMode::hulk && !wants_hulk(cfg.mode)) continue;
    const HEEnv &env = envs.for_mode(mode); const uint64_t p = env.plain_modulus;
    const char *mn = mode==EvalMode::hulk ? "hulk" : "arithmetic";

    // ---- client side: data owner encrypts the bit planes, querier encrypts the query bits (enc_ms)
    Stopwatch enc_sw;
    vector<Ciphertext> DK(kTables*kKeyBits), DC(kCodeBits), QK(kTables*kKeyBits), QC(kCodeBits);
    vector<uint64_t> plane(N);
    for(size_t b=0;b<kTables*kKeyBits;++b){ for(size_t i=0;i<N;++i) plane[i]=key[i][b]; DK[b]=encrypt_slots(env,plane);
                                            QK[b]=encrypt_slots(env, vector<uint64_t>(N, qkey[b])); }
    for(size_t b=0;b<kCodeBits;++b){ for(size_t i=0;i<N;++i) plane[i]=code[i][b]; DC[b]=encrypt_slots(env,plane);
                                     QC[b]=encrypt_slots(env, vector<uint64_t>(N, qcode[b])); }
    vector<Ciphertext> DP(kWords);                             // encrypted payload, one ciphertext per word
    for(size_t w=0;w<kWords;++w){ for(size_t i=0;i<N;++i) plane[i]=payload[i][w]; DP[w]=encrypt_slots(env,plane); }
    double enc_ms = enc_sw.ms();
    size_t upload_bytes = ciphertext_upload_bytes(DK)+ciphertext_upload_bytes(DC)+ciphertext_upload_bytes(QK)
                        + ciphertext_upload_bytes(QC)+ciphertext_upload_bytes(DP);

    // ---- server: homomorphic evaluation (eval_ms); per-stage op counts are instrumentation only
    OpCount ops_lsh, ops_xor, ops_sel, ops_w;
    const auto t0 = steady_clock::now();
    // and_test: LSH candidate generation
    vector<Ciphertext> eq_tables; eq_tables.reserve(kTables);
    for(size_t t=0;t<kTables;++t){
        vector<Ciphertext> xn; xn.reserve(kKeyBits);
        for(size_t b=0;b<kKeyBits;++b){
            const size_t j=t*kKeyBits+b;
            Ciphertext e;
            if(mode==EvalMode::arithmetic){                                       // XNOR = 1 - q - d + 2qd
                Ciphertext qd = multiply_ciphertexts(env, QK[j], DK[j], ops_lsh);
                e = multiply_plain_const(env, qd, 2, ops_lsh);
                e = subtract_ciphertexts(env, e, QK[j], ops_lsh); e = subtract_ciphertexts(env, e, DK[j], ops_lsh);
                add_plain_inplace(env, e, 1, ops_lsh);
            } else {                                                              // HULK: S = q - d, XNOR = 1 - S^2
                Ciphertext sd = subtract_ciphertexts(env, QK[j], DK[j], ops_lsh);
                e = one_minus_bit(env, square_ciphertext(env, sd, ops_lsh), ops_lsh);
            }
            xn.push_back(move(e));
        }
        auto a = mode==EvalMode::hulk ? hulk_all_true(xn, env) : arithmetic_all_true(xn, env);   // [sum = 12] vs HMul tree
        ops_lsh += a.second; eq_tables.push_back(move(a.first));
    }
    // HULK: the candidate mask only feeds products (x [h <= R] x (64 - h)), so the OR's final 1/K!
    // multiply is deferred to the client (HulkDeferScaleScope), which rescales the decrypted scores.
    uint64_t score_scale = 1;
    pair<Ciphertext, OpCount> cand;
    if(mode==EvalMode::hulk){ HulkDeferScaleScope defer(true); cand = hulk_any_true(eq_tables, env); score_scale = defer.skipped(); }
    else cand = arithmetic_any_true(eq_tables, env);
    ops_lsh += cand.second;
    // xor: Hamming distance on the re-ranking code
    Ciphertext h; bool init=false;
    for(size_t b=0;b<kCodeBits;++b){
        Ciphertext x;
        if(mode==EvalMode::arithmetic){                                           // XOR = q + d - 2qd
            Ciphertext qd = multiply_ciphertexts(env, QC[b], DC[b], ops_xor);
            x = multiply_plain_const(env, qd, mod_sub(0, 2, p), ops_xor);
            add_inplace(env, x, QC[b], ops_xor); add_inplace(env, x, DC[b], ops_xor);
        } else {                                                                  // HULK: S = q - d, XOR = S^2
            x = square_ciphertext(env, subtract_ciphertexts(env, QC[b], DC[b], ops_xor), ops_xor);
        }
        if(!init){ h=move(x); init=true; } else add_inplace(env, h, x, ops_xor);
    }
    // select: radius check
    auto inr = he_le_plain_bounded(env, h, kCodeBits, R); ops_sel += inr.second;
    // weight: score = candidate * [h <= R] * (64 - h)
    Ciphertext sim = multiply_plain_const(env, h, mod_sub(0, 1, p), ops_w); add_plain_inplace(env, sim, kCodeBits, ops_w);
    Ciphertext m = multiply_ciphertexts(env, inr.first, sim, ops_w);               // [h <= R] * (64 - h)
    Ciphertext score = multiply_ciphertexts(env, cand.first, m, ops_w);           // candidate mask last (deepest)
    const double round1_ms = duration<double, milli>(steady_clock::now() - t0).count();

    // ---- client: decrypt the scores, keep the top-k, encrypt one one-hot selector per result
    Stopwatch dec_sw; auto got = decrypt_slots(env, score);
    if(score_scale != 1) for(auto &v: got) v = mod_mul(v % p, score_scale, p);      // undo the deferred 1/K!
    double dec_ms = dec_sw.ms();
    bool ok_scores = true; for(size_t i=0;i<N && ok_scores;++i) ok_scores = (got[i] % p) == expect[i];
    vector<pair<uint64_t,size_t>> top; for(size_t i=0;i<N;++i) if(got[i]%p) top.push_back({got[i]%p,i});
    sort(top.begin(), top.end(), [](auto &a, auto &b){ return a.first != b.first ? a.first > b.first : a.second < b.second; });
    const size_t K = cfg.k;
    vector<long long> chosen(K, -1);                           // -1 = dummy query (fewer hits than k)
    for(size_t r=0;r<K && r<top.size();++r) chosen[r] = (long long)top[r].second;
    Stopwatch sel_sw;
    vector<Ciphertext> SEL(K);
    for(size_t r=0;r<K;++r){ vector<uint64_t> onehot(N,0); if(chosen[r]>=0) onehot[chosen[r]]=1; SEL[r]=encrypt_slots(env,onehot); }
    enc_ms += sel_sw.ms(); upload_bytes += ciphertext_upload_bytes(SEL);
    if(K*kWords > N/2) throw runtime_error("k * W exceeds one row of slots");

    // ---- server, round 2: private retrieval of the k payloads into one packed ciphertext
    OpCount ops_rsel, ops_ragg;
    const auto t1 = steady_clock::now();
    const size_t row = N/2;
    Ciphertext out; bool out_init=false;
    for(size_t r=0;r<K;++r){
        vector<Ciphertext> sw(kWords);
        for(size_t w=0;w<kWords;++w) sw[w] = multiply_ciphertexts(env, SEL[r], DP[w], ops_rsel);   // selector * word
        // shift word w by w slots (Horner: W - 1 rotations by one slot)
        Ciphertext acc = sw[kWords-1];
        for(size_t w=kWords-1; w-- > 0; ){
            Ciphertext rot; env.eval->rotate_rows(acc, -1, env.gk, rot); ops_ragg.rotations++;
            acc = move(rot); add_inplace(env, acc, sw[w], ops_ragg);
        }
        // fold every slot onto its residue mod W within the row, then across the two rows
        for(size_t s=kWords; s<row; s<<=1){
            Ciphertext rot; env.eval->rotate_rows(acc, (int)s, env.gk, rot); ops_ragg.rotations++;
            add_inplace(env, acc, rot, ops_ragg);
        }
        { Ciphertext sw2; env.eval->rotate_columns(acc, env.gk, sw2); ops_ragg.rotations++; add_inplace(env, acc, sw2, ops_ragg); }
        // keep slots [rW, rW + W) and pack into the response
        vector<uint64_t> mask(N,0); for(size_t t=0;t<kWords;++t) mask[r*kWords+t]=1;
        Plaintext pm; env.batch->encode(mask, pm);
        Ciphertext part; env.eval->multiply_plain(acc, pm, part); ops_ragg.plain_mul++;
        if(!out_init){ out=move(part); out_init=true; } else add_inplace(env, out, part, ops_ragg);
    }
    const double round2_ms = duration<double, milli>(steady_clock::now() - t1).count();
    const double eval_ms = round1_ms + round2_ms;

    // ---- client: decrypt the response and undo each result's cyclic shift
    Stopwatch dec2_sw; auto resp = decrypt_slots(env, out); dec_ms += dec2_sw.ms();
    bool ok_payload = true;
    for(size_t r=0;r<K;++r) for(size_t w=0;w<kWords;++w){
        const size_t ir = chosen[r] >= 0 ? size_t(chosen[r]) : 0;
        const uint64_t v = resp[r*kWords + (ir + w) % kWords] % p;
        ok_payload = ok_payload && v == (chosen[r] >= 0 ? payload[ir][w] : 0);
    }
    const bool ok = ok_scores && ok_payload;

    OpCount ops; ops += ops_lsh; ops += ops_xor; ops += ops_sel; ops += ops_w; ops += ops_rsel; ops += ops_ragg;
    cout << "\n[" << mn << "] limbs=" << env.coeff_modulus_primes << "\n"
         << "  encrypt (client) : " << enc_ms << " ms   upload " << upload_bytes << " B\n"
         << "  HE eval (server) : " << eval_ms << " ms (round 1 " << round1_ms << ", round 2 " << round2_ms
         << ")   HMul=" << ops.total_ct_mul() << "   rotations=" << ops.rotations << "\n"
         << "  decrypt (client) : " << dec_ms << " ms   [scores " << (ok_scores? "match":"MISMATCH")
         << ", payloads " << (ok_payload? "match":"MISMATCH") << "]\n"
         << "  noise budget left: " << min(env.dec->invariant_noise_budget(score), env.dec->invariant_noise_budget(out))
         << " bits\n  top-" << K << " (item:64-h):";
    for(size_t k=0;k<min<size_t>(K,top.size());++k) cout << " " << top[k].second << ":" << top[k].first;
    cout << "\n";
    print_ops(mn, ops);
    print_stage_line(mn, "and_test", ops_lsh);
    print_stage_line(mn, "xor", ops_xor);
    print_stage_line(mn, "select", ops_sel);
    print_stage_line(mn, "weight", ops_w);
    print_stage_line(mn, "retrieve_select", ops_rsel);
    print_stage_line(mn, "retrieve_aggregate", ops_ragg);
    E2EReport rep; rep.mode=mode; rep.limbs=env.coeff_modulus_primes; rep.enc_ms=enc_ms; rep.eval_ms=eval_ms;
    rep.dec_ms=dec_ms; rep.upload_bytes=upload_bytes; rep.hmul=ops.total_ct_mul(); rep.rot=ops.rotations; rep.ok=ok;
    print_e2e_line(rep);
    all_ok = all_ok && ok;
    }
    return all_ok? 0 : 1;
}
