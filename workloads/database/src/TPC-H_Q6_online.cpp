/*
 * TPC-H Q6 -- end-to-end homomorphic pipeline (comparison -> AND -> weight -> SUM).
 *
 * col 0: l_shipdate  day index [0,2526) D=2526 (1992-01-02 .. 1998-12-01)
 * col 1: discount_bp {1..10}  D=11   (percent)
 * col 2: l_quantity  {1..50}  D=51
 * col 3: revenue     weight = l_extendedprice * l_discount (client-encoded row value)
 *
 * Comparisons:
 *   P0: RANGE l_shipdate in [1994-01-01, 1995-01-01) = days [730,1094]  (D=2526)
 *   P1: RANGE discount in [5,7]                                         (D=11)
 *   P2: LT    l_quantity < 24                                            (D=51)
 * Output: 1 aggregate sum(revenue) (paper: 1 output), computed homomorphically.
 * Headroom: extprice scaled to 5..100, revenue <= 100*10 = 1000; 1000 x 32768 < t.
 */
#include "HE_comparison.h"
#include <random>
using namespace hulk;

constexpr uint64_t kRevMax6 = 1000;

static pair<IntTable, uint64_t> build_q6_int(size_t rows, uint64_t seed) {
    mt19937_64 rng(seed);
    uniform_int_distribution<int> day_d(0,2525), disc_d(1,10), qty_d(1,50), price_d(5,100);
    vector<uint64_t> sm(rows), disc(rows), ql(rows), rev(rows);
    uint64_t ref = 0;
    for (size_t i=0; i<rows; i++) {
        int d=day_d(rng), di=disc_d(rng), q=qty_d(rng), p=price_d(rng);
        sm[i]=d; disc[i]=di; ql[i]=q; rev[i]=(uint64_t)p*di;
        if (d>=730&&d<=1094 && di>=5&&di<=7 && q<24) ref += (uint64_t)p*di;
    }
    IntTable t;
    t.add("shipdate",    move(sm),  2526);
    t.add("discount_bp", move(disc),11);
    t.add("quantity",    move(ql),  51);
    t.add("revenue",     move(rev),  0);
    return {move(t), ref};
}

int main(int argc, char** argv) {
    bool oracle_only = false;
    const RunConfig cfg = tpc_parse_args(argc, argv, oracle_only);
    auto [int_table, ref] = build_q6_int(cfg.rows, cfg.seed);

    TpcQuery q;
    q.title = "TPC-H Q6 online-comparison benchmark (end-to-end)";
    q.table = &int_table;
    q.specs = {
        CompSpec{"ship_range", 0, CompOp::RANGE, 2526, 730, 1094},
        CompSpec{"disc_range", 1, CompOp::RANGE, 11, 5, 7},
        CompSpec{"qty_lt24",   2, CompOp::LT,    51, 24, 0},
    };
    q.weight_src = {3}; q.weight_max = {kRevMax6};
    const int W = (int)tpc_weight_base(q);
    q.outputs = {{"revenue",{0,1,2},FilterKind::all_true,W}};
    q.reference = {ref};
    cout << "reference revenue: " << ref << "\n";
    return run_tpc_query(q, cfg, oracle_only, [](const TpcModeResult& r) {
        cout << "  revenue=" << (r.res.values.empty() ? 0 : r.res.values[0]) << "\n";
    });
}
