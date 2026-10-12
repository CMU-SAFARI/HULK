/*
 * HE_comparison.h
 *
 * Engorgio-style homomorphic comparison over bounded integer domains.
 *
 * Each predicate is a polynomial f : [0, D) -> {0,1} computed via
 * Lagrange interpolation over the plaintext ring, then evaluated with
 * Paterson–Stockmeyer on an encrypted integer column.
 *
 * The output is a vector<Ciphertext> column (one per SIMD block) encoding
 * 0/1 per slot — structurally identical to a pre-computed bitmap index, and
 * directly consumable by run_packed_aggregates_preencrypted.
 *
 * Limitation: only plaintext-constant thresholds are supported.
 */
#pragma once
#include <map>
#include <limits>
#include <set>
#include <mutex>
#include <cstdlib>
#include <iterator>
#include <algorithm>
#include "HE_workload_common.h"
#include <chrono>

using namespace hulk;
using namespace std::chrono;

// ─── comparison operator ────────────────────────────────────────────────────

enum class CompOp { EQ, LT, LE, GT, GE, RANGE, IN, PASS };   // PASS: precomputed bit (join / column-vs-column), no comparison

struct CompSpec {
    string          name;
    size_t          col_idx;
    CompOp          op;
    size_t          domain_size;   // x in [0, domain_size)
    size_t          lo = 0;
    size_t          hi = 0;
    vector<size_t>  in_set;
};

// ─── integer data table (plaintext) ─────────────────────────────────────────

struct IntTable {
    vector<string>            col_names;
    vector<vector<uint64_t>>  cols;
    vector<size_t>            domains;

    size_t rows()  const { return cols.empty() ? 0 : cols[0].size(); }
    size_t ncols() const { return cols.size(); }

    void add(const string& name, vector<uint64_t> v, size_t domain) {
        col_names.push_back(name);
        cols.push_back(move(v));
        domains.push_back(domain);
    }

    // Convert to PackedTable for encryption
    PackedTable to_packed() const {
        PackedTable p;
        p.column_names = col_names;
        p.columns      = cols;
        return p;
    }
};

// ─── per-comparison timing record ───────────────────────────────────────────

struct ComparisonRecord {
    string   name;
    CompOp   op;
    size_t   domain_size;
    size_t   ps_hmul  = 0;
    size_t   ps_sq    = 0;
    size_t   ps_pmul  = 0;
    size_t   ps_depth = 0;
    size_t   ps_add   = 0;   // instrumentation (stage decomposition): ct-adds, plain-adds,
    size_t   ps_padd  = 0;   // relinearizations and rotations of the BSGS evaluation
    size_t   ps_relin = 0;
    size_t   ps_rot   = 0;
    size_t   n_blocks = 0;
    double   runtime_ms = 0.0;
};

// ─── polynomial helpers ───────────────────────────────────────────────────────

static vector<uint64_t> comparison_coeffs(const CompSpec& spec, uint64_t mod)
{
    const size_t D = spec.domain_size;
    vector<uint64_t> vals(D, 0);
    for (size_t x = 0; x < D; x++) {
        bool pass = false;
        switch (spec.op) {
            case CompOp::EQ:    pass = (x == spec.lo); break;
            case CompOp::LT:    pass = (x <  spec.lo); break;
            case CompOp::LE:    pass = (x <= spec.lo); break;
            case CompOp::GT:    pass = (x >  spec.lo); break;
            case CompOp::GE:    pass = (x >= spec.lo); break;
            case CompOp::RANGE: pass = (x >= spec.lo && x <= spec.hi); break;
            case CompOp::IN:
                for (auto v : spec.in_set) if (x == v) { pass = true; break; }
                break;
            case CompOp::PASS:  break;   // precomputed bit: never interpolated
        }
        vals[x] = pass ? 1 : 0;
    }
    return lagrange_coefficients(vals, mod);
}

// Cache key: every field that determines the interpolated polynomial (incl. the IN set).
static string comparison_key(const CompSpec& spec, uint64_t mod)
{
    string key = spec.name + "|" + to_string(spec.domain_size) + "|"
               + to_string(static_cast<int>(spec.op)) + "|"
               + to_string(spec.lo) + "|" + to_string(spec.hi) + "|in:";
    for (size_t v : spec.in_set) { key += to_string(v); key.push_back(','); }
    key += "|t=" + to_string(mod);
    return key;
}

// Lagrange coefficients of the comparison polynomial, computed once per spec.
// This is plaintext setup (O(D^2) per spec); drivers call
// precompute_comparison_coeffs() before any timer so that the timed comparison
// stage contains homomorphic evaluation only.  Identical for both modes.
static const vector<uint64_t>& comparison_coeffs_cached(const CompSpec& spec, uint64_t mod)
{
    static map<string, vector<uint64_t>> cache;       // node-based: references stay valid
    static mutex mu;
    const string key = comparison_key(spec, mod);
    lock_guard<mutex> lk(mu);
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, comparison_coeffs(spec, mod)).first;
    return it->second;
}

inline void precompute_comparison_coeffs(const vector<CompSpec>& specs, const HEEnv& env)
{
    for (const auto& spec : specs)
        if (spec.op != CompOp::PASS) (void)comparison_coeffs_cached(spec, env.plain_modulus);
}

// ─── per-block comparison ────────────────────────────────────────────────────

// Evaluate one comparison predicate on a single ciphertext block.
static pair<Ciphertext, OpCount>
eval_comparison_block(const HEEnv& env,
                      const Ciphertext& block,
                      const CompSpec& spec)
{
    // Depth-optimal BSGS evaluation of the comparison interpolant (depth ~ ceil(log2 D) + 1
    // instead of ~ sqrt(D) for PS); identical in both modes.  Coefficients come from the
    // precomputed cache, so no plaintext interpolation happens inside the timed region.
    return eval_poly_bsgs(env, block, comparison_coeffs_cached(spec, env.plain_modulus));
}

// ─── full-column comparison ──────────────────────────────────────────────────

// Evaluate comparison on all blocks of an encrypted integer column.
// Returns (result_column_blocks, record).
inline pair<vector<Ciphertext>, ComparisonRecord>
eval_comparison(const HEEnv& env,
                const vector<Ciphertext>& col_blocks,
                const CompSpec& spec)
{
    auto t0 = steady_clock::now();
    vector<Ciphertext> result;
    result.reserve(col_blocks.size());
    OpCount total_ops;

    if (spec.op == CompOp::PASS) {                       // precomputed bit: fed straight to composition
        result = col_blocks;
        ComparisonRecord rec; rec.name=spec.name; rec.op=spec.op; rec.domain_size=spec.domain_size;
        rec.n_blocks=col_blocks.size(); rec.runtime_ms=duration<double,milli>(steady_clock::now()-t0).count();
        return {move(result), rec};
    }
    for (const auto& block : col_blocks) {
        auto [out, ops] = eval_comparison_block(env, block, spec);
        result.push_back(move(out));
        total_ops.ct_ct_mul += ops.ct_ct_mul;
        total_ops.ct_sq     += ops.ct_sq;
        total_ops.plain_mul += ops.plain_mul;
        total_ops.ct_add    += ops.ct_add;      // instrumentation only (stage decomposition)
        total_ops.plain_add += ops.plain_add;
        total_ops.relin     += ops.relin;
        total_ops.rotations += ops.rotations;
        total_ops.depth      = max(total_ops.depth, ops.depth);
    }
    double ms = duration<double, milli>(steady_clock::now() - t0).count();

    ComparisonRecord rec;
    rec.name        = spec.name;
    rec.op          = spec.op;
    rec.domain_size = spec.domain_size;
    rec.ps_hmul     = total_ops.ct_ct_mul;
    rec.ps_sq       = total_ops.ct_sq;
    rec.ps_pmul     = total_ops.plain_mul;
    rec.ps_depth    = total_ops.depth;
    rec.ps_add      = total_ops.ct_add;
    rec.ps_padd     = total_ops.plain_add;
    rec.ps_relin    = total_ops.relin;
    rec.ps_rot      = total_ops.rotations;
    rec.n_blocks    = col_blocks.size();
    rec.runtime_ms  = ms;

    return {move(result), rec};
}

// ─── comparison stage ────────────────────────────────────────────────────────

struct ComparisonStage {
    // One vector<Ciphertext> per predicate (matches EncryptedTable column format)
    vector<vector<Ciphertext>> pred_cols;
    vector<string>             pred_names;
    vector<ComparisonRecord>   records;
    OpCount                    total_ops;
    double                     total_ms = 0.0;
};

// TPC_SHARE_POWERS (default on; =0 disables): predicates on the SAME column share one set
// of BSGS baby/giant powers of that column's ciphertext per block (bsgs_powers); each
// predicate then costs only its own giant-step combination. Identical in both modes.
inline bool share_column_powers()
{
    const char* raw = getenv("TPC_SHARE_POWERS");
    return !(raw && string(raw) == "0");
}

inline void add_record_ops(OpCount& total, const ComparisonRecord& rec)
{
    total.ct_ct_mul += rec.ps_hmul;
    total.ct_sq     += rec.ps_sq;
    total.plain_mul += rec.ps_pmul;
    total.ct_add    += rec.ps_add;     // instrumentation only (stage decomposition)
    total.plain_add += rec.ps_padd;
    total.relin     += rec.ps_relin;
    total.rotations += rec.ps_rot;
    total.depth      = max(total.depth, (int)rec.ps_depth);
}

inline void fill_record_ops(ComparisonRecord& rec, const OpCount& o)
{
    rec.ps_hmul = o.ct_ct_mul; rec.ps_sq = o.ct_sq; rec.ps_pmul = o.plain_mul; rec.ps_depth = o.depth;
    rec.ps_add = o.ct_add; rec.ps_padd = o.plain_add; rec.ps_relin = o.relin; rec.ps_rot = o.rotations;
}

inline ComparisonStage
run_comparison_stage(const HEEnv& env,
                     const EncryptedTable& enc,
                     const vector<CompSpec>& specs)
{
    ComparisonStage stage;
    precompute_comparison_coeffs(specs, env);   // plaintext setup, before any per-spec timer
    stage.pred_cols.resize(specs.size());
    stage.pred_names.reserve(specs.size());
    stage.records.resize(specs.size());
    for (const auto& spec : specs) stage.pred_names.push_back(spec.name);

    // Columns with >= 2 non-PASS predicates share their BSGS powers.
    map<size_t, vector<size_t>> by_col;
    for (size_t i = 0; i < specs.size(); i++)
        if (specs[i].op != CompOp::PASS) by_col[specs[i].col_idx].push_back(i);
    vector<bool> done(specs.size(), false);
    vector<ComparisonRecord> power_records;
    if (share_column_powers()) {
        for (auto& [col_idx, idxs] : by_col) {
            if (idxs.size() < 2) continue;
            size_t dmax = 0;
            for (size_t i : idxs) {
                const auto c = bsgs_normalise(comparison_coeffs_cached(specs[i], env.plain_modulus), env.plain_modulus);
                if (c.size() > 1) dmax = max(dmax, c.size() - 1);
            }
            if (dmax == 0) continue;
            const auto& col = enc.columns.at(col_idx);
            vector<OpCount> spec_ops(idxs.size());
            vector<double> spec_ms(idxs.size(), 0.0);
            OpCount pow_ops; double pow_ms = 0.0;
            for (size_t i : idxs) stage.pred_cols[i].reserve(col.size());
            for (const auto& block : col) {
                auto t0 = steady_clock::now();
                const BsgsPowers P = bsgs_powers(env, block, dmax, pow_ops);
                pow_ms += duration<double, milli>(steady_clock::now() - t0).count();
                for (size_t g = 0; g < idxs.size(); g++) {
                    auto t1 = steady_clock::now();
                    auto [out, ops] = eval_poly_bsgs_with(env, P, comparison_coeffs_cached(specs[idxs[g]], env.plain_modulus));
                    spec_ms[g] += duration<double, milli>(steady_clock::now() - t1).count();
                    stage.pred_cols[idxs[g]].push_back(move(out));
                    spec_ops[g] += ops;
                }
            }
            ComparisonRecord pr; pr.name = "powers(col " + to_string(col_idx) + ", shared by " + to_string(idxs.size()) + ")";
            pr.op = specs[idxs[0]].op; pr.domain_size = specs[idxs[0]].domain_size; pr.n_blocks = col.size();
            fill_record_ops(pr, pow_ops); pr.runtime_ms = pow_ms;
            power_records.push_back(pr);
            for (size_t g = 0; g < idxs.size(); g++) {
                ComparisonRecord& r = stage.records[idxs[g]];
                r.name = specs[idxs[g]].name; r.op = specs[idxs[g]].op; r.domain_size = specs[idxs[g]].domain_size;
                r.n_blocks = col.size(); fill_record_ops(r, spec_ops[g]); r.runtime_ms = spec_ms[g];
                done[idxs[g]] = true;
            }
        }
    }
    for (size_t i = 0; i < specs.size(); i++) {
        if (done[i]) continue;
        auto [bits, rec] = eval_comparison(env, enc.columns.at(specs[i].col_idx), specs[i]);
        stage.pred_cols[i] = move(bits);
        stage.records[i] = rec;
    }
    for (auto& pr : power_records) stage.records.push_back(pr);   // printed after the predicates
    for (const auto& rec : stage.records) {
        add_record_ops(stage.total_ops, rec);
        stage.total_ms += rec.runtime_ms;
    }
    return stage;
}

// ─── EncryptedTable builders ─────────────────────────────────────────────────

// Build EncryptedTable from predicate bit columns + one weight column.
// pred_cols: output of run_comparison_stage
// weight_col: one column from an EncryptedTable (enc.columns[wc])
inline EncryptedTable
build_composition_table(const vector<vector<Ciphertext>>& pred_cols,
                        const vector<Ciphertext>&          weight_col,
                        size_t rows)
{
    EncryptedTable enc;
    enc.rows = rows;
    enc.columns.reserve(pred_cols.size() + 1);
    for (const auto& c : pred_cols) enc.columns.push_back(c);
    enc.columns.push_back(weight_col);
    return enc;
}

// COUNT-only version (no weight column)
inline EncryptedTable
build_composition_table_count(const vector<vector<Ciphertext>>& pred_cols,
                               size_t rows)
{
    EncryptedTable enc;
    enc.rows = rows;
    enc.columns.reserve(pred_cols.size());
    for (const auto& c : pred_cols) enc.columns.push_back(c);
    return enc;
}

// Build EncryptedTable from pred_cols + multiple weight columns from enc.
inline EncryptedTable
build_composition_table_multi(const vector<vector<Ciphertext>>& pred_cols,
                               const EncryptedTable& enc,
                               const vector<int>& weight_col_indices,
                               size_t rows)
{
    EncryptedTable out;
    out.rows = rows;
    out.columns.reserve(pred_cols.size() + weight_col_indices.size());
    for (const auto& c : pred_cols)          out.columns.push_back(c);
    for (int wc : weight_col_indices)        out.columns.push_back(enc.columns.at(wc));
    return out;
}

// ─── composition with filter reuse + common-subexpression sharing ───────────
// 1. Dedup: outputs that share the same predicate set (e.g. the several metrics
//    of one group) share one filter mask (one SharedFilter), so both approaches
//    compute it once and reuse it, as a real system would.
// 2. CSE: predicates common to many unique filters (the query-wide conjuncts,
//    or a common prefix of a group family) are factored into one root
//    SharedFilter (parent_filter = -1); each unique filter becomes a child with
//    parent_filter = root and only its remaining, group-specific predicates.
//    Arithmetic multiplies the shared parent bit into its product tree; HULK
//    feeds the parent indicator into its sum.  The DAG is built from the
//    predicate structure only, so it is identical for both modes.
//    Root choice per unique filter: among candidate sets (driver-order prefixes
//    of length >= 2, and the set common to all filters of that kind) contained in
//    the filter, pick the one adding the least AND depth, then maximising the
//    saved AND work (|root|-1)*(uses-1).  Sharing usually costs one AND level
//    (the parent enters the child's AND as a leaf), so the no-CSE variant should
//    also be measured and each mode reported at its own best.
//    Set TPC_DISABLE_CSE=1 for the dedup-only (no-CSE) variant, and
//    TPC_CSE_MAX_DEPTH_PENALTY=0 to allow only sharing that adds no AND level.
struct DedupOutputs {
    vector<AggregateOutput> outputs;
    vector<SharedFilter>    shared;
    size_t unique_output_filters = 0;   // distinct (predicate set, kind) among outputs
    size_t cse_roots = 0;               // shared parents introduced by CSE
};

inline bool cse_disabled()
{
    const char* raw = getenv("TPC_DISABLE_CSE");
    if (!raw) return false;
    const string v(raw);
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "YES";
}

// Maximum extra AND levels a CSE root may add (default 1; 0 = depth-free sharing only).
inline int cse_max_depth_penalty()
{
    const char* raw = getenv("TPC_CSE_MAX_DEPTH_PENALTY");
    return raw ? max(0, atoi(raw)) : 1;
}

inline DedupOutputs dedup_filters(const vector<AggregateOutput>& outs, bool cse = !cse_disabled())
{
    using Key = pair<vector<size_t>, int>;               // (sorted predicate set, filter kind)
    struct UF { string name; vector<size_t> ordered, sorted; FilterKind kind; };
    vector<UF> uf; map<Key, int> uf_index; vector<int> out_uf(outs.size());
    for (size_t i = 0; i < outs.size(); i++) {
        vector<size_t> srt = outs[i].predicate_cols; sort(srt.begin(), srt.end());
        srt.erase(unique(srt.begin(), srt.end()), srt.end());
        const Key key{srt, (int)outs[i].filter_kind};
        auto it = uf_index.find(key);
        if (it == uf_index.end()) {
            it = uf_index.emplace(key, (int)uf.size()).first;
            uf.push_back({outs[i].name, outs[i].predicate_cols, srt, outs[i].filter_kind});
        }
        out_uf[i] = it->second;
    }

    const auto subset = [](const vector<size_t>& a, const vector<size_t>& b) {   // a ⊆ b (both sorted)
        return includes(b.begin(), b.end(), a.begin(), a.end());
    };

    // Choose a CSE root (possibly none) for every unique filter.
    vector<int> root_of(uf.size(), -1);                  // index into roots
    vector<Key> roots;
    if (cse && uf.size() >= 2) {
        set<Key> cands;
        map<int, vector<size_t>> common; map<int, bool> has_common;
        for (const auto& u : uf) {
            for (size_t L = 2; L < u.ordered.size(); L++) {
                vector<size_t> pre(u.ordered.begin(), u.ordered.begin() + L);
                sort(pre.begin(), pre.end()); pre.erase(unique(pre.begin(), pre.end()), pre.end());
                if (pre.size() >= 2 && pre.size() < u.sorted.size()) cands.insert({pre, (int)u.kind});
            }
            const int k = (int)u.kind;
            if (!has_common[k]) { common[k] = u.sorted; has_common[k] = true; }
            else {
                vector<size_t> inter;
                set_intersection(common[k].begin(), common[k].end(), u.sorted.begin(), u.sorted.end(),
                                 back_inserter(inter));
                common[k] = move(inter);
            }
        }
        for (auto& [k, c] : common) if (c.size() >= 2) cands.insert({c, k});

        map<Key, long> saving;
        for (const auto& c : cands) {
            long uses = 0;
            for (const auto& u : uf) if ((int)u.kind == c.second && subset(c.first, u.sorted)) uses++;
            saving[c] = (long)(c.first.size() - 1) * (uses - 1);
        }
        // AND-depth model shared by both modes: a w-input AND costs ceil(log2 w) levels
        // (arithmetic product tree; HULK indicator of a width-w sum, with HMul for w = 2).
        // The harness feeds the parent in as one leaf of the child's AND, so a child with
        // root size c and r local predicates has depth ceil(log2 c) + ceil(log2 (r+1)).
        const auto clog2 = [](size_t w) { int d = 0; while ((size_t(1) << d) < w) d++; return d; };
        const auto penalty = [&](size_t c, size_t n) {
            if (c >= n) return 0;
            return clog2(c) + clog2(n - c + 1) - clog2(n);
        };
        const int max_pen = cse_max_depth_penalty();
        map<Key, int> root_index;
        for (size_t i = 0; i < uf.size(); i++) {
            // lexicographic: least extra AND depth (limbs), then most saved AND work, then larger root
            const Key* best = nullptr; long best_s = 0; int best_p = 0;
            const size_t n = uf[i].sorted.size();
            for (const auto& c : cands) {
                if (c.second != (int)uf[i].kind || !subset(c.first, uf[i].sorted)) continue;
                const long sv = saving[c];
                if (sv <= 0) continue;
                const int pn = penalty(c.first.size(), n);
                if (pn > max_pen) continue;
                if (!best || pn < best_p || (pn == best_p && (sv > best_s ||
                    (sv == best_s && c.first.size() > best->first.size())))) {
                    best = &c; best_s = sv; best_p = pn;
                }
            }
            if (!best) continue;
            auto it = root_index.find(*best);
            if (it == root_index.end()) { it = root_index.emplace(*best, (int)roots.size()).first; roots.push_back(*best); }
            root_of[i] = it->second;
        }
    }

    DedupOutputs d;
    d.unique_output_filters = uf.size();
    d.cse_roots = roots.size();
    for (size_t r = 0; r < roots.size(); r++)
        d.shared.push_back({"cse_root" + to_string(r), roots[r].first, (FilterKind)roots[r].second, -1});
    vector<int> uf_shared(uf.size());
    for (size_t i = 0; i < uf.size(); i++) {
        const auto& u = uf[i];
        if (root_of[i] < 0) {
            uf_shared[i] = (int)d.shared.size();
            d.shared.push_back({u.name + "_filter", u.sorted, u.kind, -1});
            continue;
        }
        const auto& root = roots[(size_t)root_of[i]].first;
        vector<size_t> rest;
        set_difference(u.sorted.begin(), u.sorted.end(), root.begin(), root.end(), back_inserter(rest));
        if (rest.empty()) { uf_shared[i] = root_of[i]; continue; }       // filter == root
        uf_shared[i] = (int)d.shared.size();
        d.shared.push_back({u.name + "_filter", move(rest), u.kind, root_of[i]});
    }
    for (size_t i = 0; i < outs.size(); i++) {
        AggregateOutput o = outs[i];
        o.shared_filter = uf_shared[(size_t)out_uf[i]];
        o.predicate_cols.clear();
        d.outputs.push_back(move(o));
    }
    return d;
}

// Padding slots of a partial last block hold 0 in every column; predicates that
// are true at x = 0 would let COUNT outputs (weight_col = -1) count them.
// Weighted outputs are safe (weight 0 in padding).  Refuse such runs.
inline void require_no_count_padding(const vector<AggregateOutput>& outs, size_t rows, size_t slot_count)
{
    if (slot_count == 0 || rows % slot_count == 0) return;
    for (const auto& o : outs)
        if (o.weight_col < 0)
            throw invalid_argument("COUNT output '" + o.name + "' with rows=" + to_string(rows) +
                                   " not a multiple of slot_count=" + to_string(slot_count) +
                                   ": padding slots would be counted");
}

inline AggregateRunResult run_composition(const EncryptedTable& tbl, const vector<AggregateOutput>& outs,
                                          EvalMode mode, const HEEnv& env, bool check_padding = true,
                                          const AggregateOptions& opts = {})
{
    if (check_padding) require_no_count_padding(outs, tbl.rows, env.slot_count);
    auto d = dedup_filters(outs);
    return run_packed_aggregates_preencrypted(tbl, d.outputs, mode, env, d.shared, 0, opts);
}
inline size_t unique_filters(const vector<AggregateOutput>& outs) { return dedup_filters(outs).unique_output_filters; }
inline size_t cse_roots(const vector<AggregateOutput>& outs)      { return dedup_filters(outs).cse_roots; }

// ─── AND-only timing ─────────────────────────────────────────────────────────
// Same outputs and shared masks with the weight columns removed: this evaluates
// exactly the filter ANDs (plus the cheap mask sums), so AND time = this.
// Slot aggregation and the cache warm-up are forced OFF here regardless of
// HE_SLOT_AGGREGATE / HE_WARM_RUN, so the rotate-and-sum cost is never charged
// to AND.  Breakdown of a composition result `res`:
//   AND = and_only_ms, aggregation = res.aggregate_ms,
//   weighting = res.elapsed_ms - res.aggregate_ms - AND   (see print_composition_breakdown).
// Uses the harness's evaluation-only timer (excludes decryption, printing).
inline double and_only_ms(const EncryptedTable& tbl, vector<AggregateOutput> outs, EvalMode mode, const HEEnv& env)
{
    for (auto& o : outs) o.weight_col = -1;
    AggregateOptions opts;
    opts.slot_aggregate = false;
    opts.warm_first_block = false;
    opts.quiet = true;
    auto r = run_composition(tbl, outs, mode, env, /*check_padding=*/false, opts);   // counts unused: timing only
    return r.elapsed_ms;
}

inline void print_composition_breakdown(const AggregateRunResult& res, double and_ms)
{
    const double weighting = res.elapsed_ms - res.aggregate_ms - and_ms;
    cout << "  composition breakdown: and ms=" << and_ms
         << "  weighting ms=" << weighting
         << "  aggregation ms=" << res.aggregate_ms
         << (res.slot_aggregated ? "" : " (slot aggregation off)") << "\n";
}

// ─── diagnostic helpers ───────────────────────────────────────────────────────

inline void print_comparison_stage(const ComparisonStage& s)
{
    const char* op_str[] = {"EQ","LT","LE","GT","GE","RANGE","IN","PASS"};
    cout << "  comparison stage: " << s.records.size()
         << " predicates, " << fixed << setprecision(2) << s.total_ms << " ms\n";
    for (const auto& r : s.records) {
        cout << "    [" << r.name << "] op=" << op_str[(int)r.op]
             << " D=" << r.domain_size
             << " hmul=" << r.ps_hmul << " sq=" << r.ps_sq
             << " d=" << r.ps_depth
             << " t=" << r.runtime_ms << " ms\n";
    }
    cout << "  cmp totals: hmul=" << s.total_ops.ct_ct_mul
         << " sq=" << s.total_ops.ct_sq
         << " pmul=" << s.total_ops.plain_mul
         << " max_depth=" << s.total_ops.depth << "\n";
}

// ═══ End-to-end TPC driver (round 2) ═════════════════════════════════════════
// One declarative query description drives BOTH the homomorphic run (either
// mode, identical pipeline) and a plaintext oracle-only check.
//
// Pipeline per mode (every stage between client encryption and the returned
// aggregates is homomorphic):
//   client:  encrypt IntTable (timed: enc_ms; upload_bytes = serialized size)
//   untimed: interpolate comparison polynomials + pre-encode their coefficient
//            plaintexts (identical for both modes)
//   server:  comparison stage (BSGS)            -> cmp_ms
//            derived OR-of-AND columns (if any) -> derive_ms (AND work)
//            composition: filter ANDs, mask x weight, HAdd over blocks,
//            rotate-and-sum (slot aggregation)  -> comp_ms (aggregation_ms inside)
//   client:  decrypt one ciphertext per output  -> dec_ms
//   eval_ms = cmp_ms + derive_ms + comp_ms.
//
// Composition-table layout: [one bit column per CompSpec, in spec order]
//   + [one column per DerivedColumn] + [IntTable columns listed in weight_src].
// AggregateOutput::predicate_cols / weight_col index into that layout.

// A derived predicate column: sum over branches of AND(branch predicate bits).
// Branches MUST be mutually exclusive (checked by the oracle), so the sum is the
// exact OR and stays 0/1.  Each branch AND uses the mode's own filter evaluator
// (arithmetic product tree / HULK indicator).
struct DerivedColumn {
    string name;
    vector<vector<size_t>> branches;   // indices into the comparison bit columns
};

struct TpcQuery {
    string title;
    const IntTable* table = nullptr;
    vector<CompSpec> specs;
    vector<DerivedColumn> derived;
    vector<size_t> weight_src;                        // IntTable columns appended as weights
    vector<uint64_t> weight_max;                      // declared max value per weight_src entry
    vector<AggregateOutput> outputs;
    vector<uint64_t> reference;                       // oracle, one value per output (output order)
};

struct TpcModeResult {
    AggregateRunResult res;
    bool correct = false;
    double enc_ms = 0, cmp_ms = 0, derive_ms = 0, comp_ms = 0, and_ms = 0;
    OpCount derive_ops;
    size_t upload_bytes = 0;
};

// Strip driver-only flags, then hand the rest to the harness parser.
inline RunConfig tpc_parse_args(int argc, char** argv, bool& oracle_only)
{
    oracle_only = false;
    vector<char*> rest; rest.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--oracle-only") { oracle_only = true; continue; }
        rest.push_back(argv[i]);
    }
    return parse_basic_args((int)rest.size(), rest.data());
}

inline uint64_t tpc_plain_modulus()
{
    return PlainModulus::Batching(kPolyDegree, kPlainModulusBits).value();
}

inline size_t tpc_weight_base(const TpcQuery& q) { return q.specs.size() + q.derived.size(); }

// Every aggregate is computed homomorphically and reduced mod t, so the worst
// case (every row in one output, every weight at its declared max) must stay < t.
inline void require_aggregate_headroom(const TpcQuery& q, uint64_t t)
{
    if (q.weight_max.size() != q.weight_src.size())
        throw invalid_argument(q.title + ": weight_max must list one bound per weight_src column");
    const size_t rows = q.table->rows();
    uint64_t wmax = 1;                                  // COUNT outputs weigh 1
    for (size_t i = 0; i < q.weight_src.size(); i++) {
        const auto& col = q.table->cols.at(q.weight_src[i]);
        const uint64_t sample_max = col.empty() ? 0 : *max_element(col.begin(), col.end());
        if (sample_max > q.weight_max[i])
            throw logic_error(q.title + ": weight column '" + q.table->col_names[q.weight_src[i]] + "' max " +
                              to_string(sample_max) + " exceeds its declared bound " + to_string(q.weight_max[i]));
        wmax = max(wmax, q.weight_max[i]);
    }
    const bool overflow = rows != 0 && wmax > numeric_limits<uint64_t>::max() / rows;
    const uint64_t bound = overflow ? numeric_limits<uint64_t>::max() : wmax * (uint64_t)rows;
    cout << "  aggregate headroom: max_weight=" << wmax << " x rows=" << rows << " = " << bound
         << " < t=" << t << (bound < t ? "" : "  VIOLATED") << "\n";
    if (bound >= t)
        throw runtime_error(q.title + ": max_weight x rows >= t (aggregate >= t possible); rescale the weights");
    for (size_t i = 0; i < q.reference.size(); i++)
        if (q.reference[i] >= t) throw runtime_error(q.title + ": oracle aggregate >= t at output " + to_string(i));
}

// Plaintext predicate semantics (the interpolated polynomial must agree with it).
inline bool tpc_pred_plain(const CompSpec& s, uint64_t x)
{
    switch (s.op) {
        case CompOp::EQ:    return x == s.lo;
        case CompOp::LT:    return x <  s.lo;
        case CompOp::LE:    return x <= s.lo;
        case CompOp::GT:    return x >  s.lo;
        case CompOp::GE:    return x >= s.lo;
        case CompOp::RANGE: return x >= s.lo && x <= s.hi;
        case CompOp::IN:    return find(s.in_set.begin(), s.in_set.end(), (size_t)x) != s.in_set.end();
        case CompOp::PASS:  return x != 0;
    }
    return false;
}

inline uint64_t tpc_poly_eval_plain(const vector<uint64_t>& c, uint64_t x, uint64_t t)
{
    uint64_t acc = 0;                                   // acc < t < 2^25, x < 2^20: no overflow
    for (size_t i = c.size(); i-- > 0;) acc = (acc * (x % t) + c[i]) % t;
    return acc;
}

// Plaintext oracle of the full declared pipeline: checks the domain of every
// compared column, the interpolated polynomial on its whole domain, branch
// disjointness of derived columns, and every output aggregate (mod t) against
// the driver's independently computed reference.
inline bool tpc_oracle_check(const TpcQuery& q)
{
    const uint64_t t = tpc_plain_modulus();
    const IntTable& T = *q.table;
    const size_t rows = T.rows();
    bool ok = true;
    cout << q.title << " --oracle-only (plaintext, t=" << t << ")\nrows=" << rows << "\n";
    require_aggregate_headroom(q, t);

    vector<vector<uint8_t>> bits;                                   // composition-table bit columns
    for (const auto& s : q.specs) {
        const auto& col = T.cols.at(s.col_idx);
        for (uint64_t v : col) if (v >= s.domain_size) {
            cout << "  [" << s.name << "] value " << v << " outside domain D=" << s.domain_size << "\n"; ok = false; break;
        }
        if (s.op != CompOp::PASS) {
            const auto& c = comparison_coeffs_cached(s, t);
            for (size_t x = 0; x < s.domain_size; x++)
                if (tpc_poly_eval_plain(c, x, t) != (tpc_pred_plain(s, x) ? 1u : 0u)) {
                    cout << "  [" << s.name << "] interpolant wrong at x=" << x << "\n"; ok = false; break;
                }
            const PolyCost pc = bsgs_cost(c, t);
            cout << "  [" << s.name << "] D=" << s.domain_size << " bsgs depth=" << pc.depth << " hmul=" << pc.muls << "\n";
        }
        vector<uint8_t> b(rows);
        for (size_t r = 0; r < rows; r++) b[r] = tpc_pred_plain(s, col[r]) ? 1 : 0;
        bits.push_back(move(b));
    }
    for (const auto& d : q.derived) {
        vector<uint8_t> b(rows, 0);
        size_t overlap = 0;
        for (size_t r = 0; r < rows; r++) {
            unsigned hits = 0;
            for (const auto& br : d.branches) {
                bool all = true; for (size_t p : br) all = all && bits.at(p)[r];
                hits += all;
            }
            if (hits > 1) overlap++;
            b[r] = hits ? 1 : 0;
        }
        if (overlap) { cout << "  [" << d.name << "] branches overlap on " << overlap << " rows (sum is not an OR)\n"; ok = false; }
        bits.push_back(move(b));
    }
    const size_t W0 = tpc_weight_base(q);
    vector<uint64_t> got;
    for (const auto& o : q.outputs) {
        uint64_t acc = 0;                               // <= max_weight x rows (headroom-checked)
        for (size_t r = 0; r < rows; r++) {
            bool pass = (o.filter_kind == FilterKind::all_true);
            for (size_t p : o.predicate_cols) {
                const bool bit = bits.at(p)[r];
                pass = (o.filter_kind == FilterKind::all_true) ? (pass && bit) : (pass || bit);
            }
            if (!pass) continue;
            if (o.weight_col < 0) acc += 1;
            else {
                const size_t wi = (size_t)o.weight_col - W0;
                acc += T.cols.at(q.weight_src.at(wi))[r];
            }
        }
        got.push_back(acc % t);
    }
    const bool match = (got == q.reference);
    ok = ok && match;
    size_t nz = 0; for (auto v : got) nz += (v != 0);
    const auto d = dedup_filters(q.outputs);
    cout << "  outputs=" << q.outputs.size() << " unique_filters=" << d.unique_output_filters
         << " cse_roots=" << d.cse_roots << " nonzero_outputs=" << nz << "\n";
    cout << "  oracle-consistent=" << (match ? "yes" : "NO") << (ok ? "" : "  [FAILED CHECKS]") << "\n";
    return ok;
}

// Identical in both modes, before any timer: interpolate every comparison
// polynomial and encode each distinct nonzero coefficient as a plaintext in the
// env cache, so the timed comparison stage contains no plaintext encoding.
inline double prewarm_comparison_plaintexts(const vector<CompSpec>& specs, const HEEnv& env, size_t* distinct = nullptr)
{
    Stopwatch sw;
    precompute_comparison_coeffs(specs, env);
    for (const auto& s : specs) {
        if (s.op == CompOp::PASS) continue;
        warm_coefficients(env, bsgs_normalise(comparison_coeffs_cached(s, env.plain_modulus), env.plain_modulus));
    }
    if (distinct && env.cache) { lock_guard<mutex> g(env.cache->plain_mutex); *distinct = env.cache->plain_cache.size(); }
    return sw.ms();
}

// Derived OR-of-disjoint-AND columns, homomorphic, per block.
inline vector<vector<Ciphertext>> eval_derived_columns(const TpcQuery& q, const ComparisonStage& cs,
                                                        EvalMode mode, const HEEnv& env, OpCount& ops, double& ms)
{
    Stopwatch sw;
    vector<vector<Ciphertext>> out;
    for (const auto& d : q.derived) {
        const size_t blocks = cs.pred_cols.empty() ? 0 : cs.pred_cols[0].size();
        vector<Ciphertext> col; col.reserve(blocks);
        for (size_t b = 0; b < blocks; b++) {
            bool have = false; Ciphertext acc;
            for (const auto& br : d.branches) {
                vector<Ciphertext> leaves;
                for (size_t p : br) leaves.push_back(cs.pred_cols.at(p).at(b));
                auto [bit, o] = apply_filter(mode, FilterKind::all_true, leaves, env);
                ops += o;
                if (!have) { acc = move(bit); have = true; } else add_inplace(env, acc, bit, ops);
            }
            col.push_back(move(acc));
        }
        out.push_back(move(col));
    }
    ms = sw.ms();
    return out;
}

// Per-stage operation counts of one mode (instrumentation only; no HE work).
//   filter    = comparison stage (BSGS interpolants), all op types
//   and       = derived OR-of-AND columns + composition filter ANDs incl. CSE masks
//               = composition ops - weight - aggregate
//   weight    = mask x weight HMul (+ its relinearization): OpCount::weight_muls
//   aggregate = zero-guard HAdd + HAdd over blocks (OpCount::agg_adds)
//               + rotate-and-sum (AggregateRunResult::aggregate_ops)
// The four lines sum exactly to cmp + derive + composition ops, i.e. to the E2E
// hmul (= hmul + sq) and rot of this mode. Sorting: none -- ORDER BY / LIMIT act on
// the decrypted aggregates (client side), so there is no homomorphic sorting stage.
inline void print_tpc_stage_lines(const TpcQuery& q, EvalMode mode, const ComparisonStage& cs, const TpcModeResult& r)
{
    const string m = (mode == EvalMode::hulk) ? "hulk" : "arithmetic";
    OpCount weight;
    weight.ct_ct_mul = r.res.ops.weight_muls;
    weight.relin     = r.res.ops.weight_muls;
    OpCount aggregate = r.res.aggregate_ops;
    aggregate.ct_add += r.res.ops.agg_adds;
    OpCount and_ops = op_diff(op_diff(r.res.ops, weight), aggregate);
    and_ops += r.derive_ops;
    print_stage_line(m, "filter", cs.total_ops);
    print_stage_line(m, "and", and_ops);
    print_stage_line(m, "weight", weight);
    print_stage_line(m, "aggregate", aggregate);
    size_t weighted = 0;
    for (const auto& o : q.outputs) weighted += (o.weight_col >= 0);
    cout << "STAGEINFO outputs=" << q.outputs.size() << " weighted_outputs=" << weighted << " sorting=none\n";
}

inline TpcModeResult run_tpc_mode(const TpcQuery& q, EvalMode mode, const HEEnv& env)
{
    const string lbl = (mode == EvalMode::arithmetic) ? "arithmetic" : "HULK";
    TpcModeResult r;

    auto enc = encrypt_table_timed(env, q.table->to_packed(), r.enc_ms);        // client
    r.upload_bytes = ciphertext_upload_bytes(enc);

    size_t distinct = 0;
    const double warm_ms = prewarm_comparison_plaintexts(q.specs, env, &distinct);   // untimed, both modes

    auto cs = run_comparison_stage(env, enc, q.specs);
    r.cmp_ms = cs.total_ms;                                                       // HE evaluation only

    auto derived = eval_derived_columns(q, cs, mode, env, r.derive_ops, r.derive_ms);

    EncryptedTable comp_tbl;
    comp_tbl.rows = q.table->rows();
    for (auto& c : cs.pred_cols) comp_tbl.columns.push_back(move(c));   // cs.records still printed below
    for (auto& c : derived)      comp_tbl.columns.push_back(move(c));
    for (size_t w : q.weight_src) comp_tbl.columns.push_back(enc.columns.at(w));

    AggregateOptions opts;
    // SUM/COUNT: default = client-side slot reduction. The server returns one HAdd-folded
    // ciphertext per output (no rotations); the client sums its slots after decryption.
    // TPC_SLOT_AGGREGATE=1 restores the homomorphic rotate-and-sum (sum_slots).
    opts.slot_aggregate = env_flag_enabled("TPC_SLOT_AGGREGATE");
    r.res = run_composition(comp_tbl, q.outputs, mode, env, /*check_padding=*/true, opts);
    r.comp_ms = r.res.elapsed_ms;                 // evaluation only (excl. decrypt)
    const double and_comp = and_only_ms(comp_tbl, q.outputs, mode, env);
    r.and_ms = r.derive_ms + and_comp;            // derived OR-of-AND columns are filter work too

    r.correct = verify_aggregates(r.res, q.reference, env, q.title + " " + lbl);

    const auto d = dedup_filters(q.outputs);
    cout << "\n--- " << lbl << " ---\n" << fixed << setprecision(3);
    cout << "  comparison ms=" << r.cmp_ms;
    if (!q.derived.empty()) cout << "  derived-or ms=" << r.derive_ms;
    cout << "  composition ms=" << r.comp_ms << "  and ms=" << r.and_ms
         << "  aggregation ms=" << r.res.aggregate_ms << "  decrypt ms=" << r.res.decrypt_ms
         << "  encrypt ms=" << r.enc_ms << "\n";
    print_composition_breakdown(r.res, and_comp);
    cout << "  outputs=" << q.outputs.size() << " unique_filters=" << d.unique_output_filters
         << " cse_roots=" << d.cse_roots << "\n";
    cout << "  prewarm (untimed): " << distinct << " cached plaintexts, " << warm_ms << " ms\n";
    print_comparison_stage(cs);
    if (!q.derived.empty()) print_ops("  " + lbl + " derived-or", r.derive_ops);
    print_noise_budget(lbl, r.res);

    E2EReport e = make_e2e_report(mode, env, r.enc_ms, r.upload_bytes, r.res, r.correct);
    e.eval_ms = r.cmp_ms + r.derive_ms + r.comp_ms;
    e.hmul = cs.total_ops.total_ct_mul() + r.derive_ops.total_ct_mul() + r.res.ops.total_ct_mul();
    print_e2e_line(e);
    cout << "E2E_STAGES mode=" << (mode == EvalMode::hulk ? "hulk" : "arithmetic")
         << " cmp_ms=" << r.cmp_ms << " derive_ms=" << r.derive_ms << " comp_ms=" << r.comp_ms
         << " and_ms=" << r.and_ms << " agg_ms=" << r.res.aggregate_ms
         << " cmp_hmul=" << cs.total_ops.total_ct_mul() << " cmp_depth=" << cs.total_ops.depth
         << " comp_hmul=" << r.res.ops.total_ct_mul() << " outputs=" << q.outputs.size()
         << " unique_filters=" << d.unique_output_filters << " cse_roots=" << d.cse_roots << "\n";
    print_tpc_stage_lines(q, mode, cs, r);
    return r;
}

// Standard main body: oracle-only, or both modes on the same data.
// `report` prints the query-specific result line for a finished mode.
inline int run_tpc_query(const TpcQuery& q, const RunConfig& cfg, bool oracle_only,
                         const function<void(const TpcModeResult&)>& report = {})
{
    if (oracle_only) return tpc_oracle_check(q) ? 0 : 1;
    const BenchmarkEnvs envs = make_benchmark_envs(cfg);
    cout << q.title << "\nrows=" << q.table->rows() << " mode=" << mode_name(cfg.mode)
         << " threads=" << cfg.threads << "\n";
    require_aggregate_headroom(q, wants_arithmetic(cfg.mode) ? envs.arithmetic->plain_modulus
                                                             : envs.hulk->plain_modulus);
    bool ok = true;
    for (EvalMode mode : {EvalMode::arithmetic, EvalMode::hulk}) {
        if (mode == EvalMode::arithmetic && !wants_arithmetic(cfg.mode)) continue;
        if (mode == EvalMode::hulk       && !wants_hulk(cfg.mode))       continue;
        const TpcModeResult r = run_tpc_mode(q, mode, envs.for_mode(mode));
        ok = ok && r.correct;
        if (report) report(r);
        cout << "  result [" << (r.correct ? "match" : "MISMATCH") << "]\n";
        print_ops("  " + string(mode == EvalMode::arithmetic ? "arithmetic" : "HULK"), r.res.ops);
    }
    return ok ? 0 : 1;
}
