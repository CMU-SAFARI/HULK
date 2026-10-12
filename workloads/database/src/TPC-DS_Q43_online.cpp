/*
 * TPC-DS Q43 -- online homomorphic-comparison pipeline, real TPC domains.
 *
 *   where d_year = [YEAR]                -> EQ year_code  (D=6, 1998..2003), target 3
 *     and s_gmt_offset = [GMT]           -> EQ gmt_offset (D=6, -5..-10),    target 2
 *   group by s_store (3 of 12 stores)    -> EQ store_id   (D=12), stores 0..2
 *   sum per day name (7 columns)         -> EQ day_of_week (D=7), one bit per day
 *   sum(ss_sales_price)                  -> weight column
 *
 * Per (store, day) output: F[s,d] = year AND gmt AND store_s AND day_d.
 * IntTable columns: 0 year_code, 1 gmt_offset, 2 store_id, 3 day_of_week, 4 sales.
 * 12 filters, 21 AND-4 outputs (paper: 12 / 21); SUMs homomorphic.
 * comp_tbl columns: 0 year, 1 gmt, 2..4 store bits, 5..11 day bits, 12 sales.
 */
#include "HE_comparison.h"
#include <random>
using namespace hulk;

constexpr uint64_t kSalMax43 = 350;   // 350 x 32768 < t

static pair<IntTable, vector<uint64_t>>
build_q43_int(size_t rows, uint64_t seed)
{
    mt19937_64 rng(seed);
    // d_year 1998..2003 (6), s_gmt_offset -5..-10 (6), s_store over 12 stores (outputs for 3), day name (7)
    uniform_int_distribution<int> yr_d(0,5), gmt_d(0,5), st_d(0,11), day_d(0,6), sal_d(20,(int)kSalMax43);

    vector<uint64_t> yr(rows), gmt(rows), st(rows), day(rows), sal(rows);
    vector<uint64_t> ref(3*7, 0);

    for (size_t i=0; i<rows; i++) {
        int y=yr_d(rng), g=gmt_d(rng), s=st_d(rng), d=day_d(rng), sl=sal_d(rng);
        yr[i]=y; gmt[i]=g; st[i]=s; day[i]=d; sal[i]=sl;
        if (y==3 && g==2 && s<3) ref[s*7+d]+=sl;
    }

    IntTable t;
    t.add("year_code",   move(yr),  6);
    t.add("gmt_offset",  move(gmt), 6);
    t.add("store_id",    move(st), 12);
    t.add("day_of_week", move(day), 7);
    t.add("sales",       move(sal), 0);
    return {move(t), move(ref)};
}

int main(int argc, char** argv)
{
    bool oracle_only = false;
    const RunConfig cfg = tpc_parse_args(argc, argv, oracle_only);
    auto [int_table, reference] = build_q43_int(cfg.rows, cfg.seed);

    TpcQuery q;
    q.title = "TPC-DS Q43 online-comparison benchmark (end-to-end)";
    q.table = &int_table;
    q.specs.push_back({"year_tgt",  0, CompOp::EQ, 6, 3, 0});
    q.specs.push_back({"gmt_tgt",   1, CompOp::EQ, 6, 2, 0});
    for (size_t s=0; s<3;  s++) q.specs.push_back({"store_"+to_string(s), 2, CompOp::EQ, 12, s, 0});
    for (size_t d=0; d<7;  d++) q.specs.push_back({"day_"+to_string(d),   3, CompOp::EQ, 7, d, 0});
    // pred: 0=year, 1=gmt, 2..4=stores, 5..11=days; weight after them
    q.weight_src = {4}; q.weight_max = {kSalMax43};
    const int W = (int)tpc_weight_base(q);
    for (size_t s=0; s<3; s++) for (size_t d=0; d<7; d++)
        q.outputs.push_back({"s"+to_string(s)+"d"+to_string(d), {0,1,2+s,5+d}, FilterKind::all_true, W});
    q.reference = reference;
    return run_tpc_query(q, cfg, oracle_only);
}
