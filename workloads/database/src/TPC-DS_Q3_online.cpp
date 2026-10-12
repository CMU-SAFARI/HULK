/*
 * TPC-DS Q3 -- online homomorphic-comparison pipeline, real TPC domains.
 *
 *   where i_manufact_id = [MANUFACT]      -> EQ i_manufact_id (D=1001, 1..1000), target kManuf
 *     and d_moy = [MONTH]                  -> EQ d_moy (D=13, 1..12), target kMonth
 *   group by d_year, i_brand_id            -> EQ d_year (D=6, 1998..2003) for the first kYears years,
 *                                             EQ brand_group (D=kBrands); one output per (year, brand)
 *   sum(ss_ext_sales_price)                -> weight column
 *
 * d_year data spans the full domain 0..5; rows in years >= kYears belong to no output group.
 * IntTable columns: 0 i_manufact_id, 1 d_moy, 2 d_year, 3 brand_group, 4 ss_ext_sales_price.
 * comp_tbl columns: 0 manufact, 1 moy, 2..2+kYears-1 year bits, then kBrands brand bits, then sales.
 * 10 filters (2 + 4 years + 4 brands), 16 AND-4 outputs (paper: 10 filters, 16 outputs); SUMs homomorphic.
 */
#include "HE_comparison.h"
#include <random>
using namespace hulk;

constexpr size_t kBrands = 4, kYears = 4, kManuf = 128, kMonth = 11;   // 4 x 4 = 16 outputs (paper: 16)
constexpr uint64_t kSalesMax3 = 350;                                         // 350 x 32768 < t

// select d_year, i_brand_id, sum(ss_ext_sales_price) from date_dim, store_sales, item
// where i_manufact_id = [MANUFACT] (1..1000) and d_moy = [MONTH] group by d_year, i_brand_id
static pair<IntTable, vector<uint64_t>>
build_q3_int(size_t rows, uint64_t seed)
{
    mt19937_64 rng(seed);
    uniform_int_distribution<int> manuf_d(1,1000), moy_d(1,12), year_d(0,5),                   // d_year over the full D=6 domain
                                   bg_d(0,(int)kBrands-1), sales_d(20,(int)kSalesMax3), pick(0,19);
    vector<uint64_t> man(rows), moy(rows), yr(rows), bgrp(rows), sales(rows);
    vector<uint64_t> ref(kYears*kBrands, 0);
    for (size_t i=0; i<rows; i++) {
        int m=manuf_d(rng), mo=moy_d(rng), y=year_d(rng), bg=bg_d(rng), sl=sales_d(rng);
        if (pick(rng)==0) m=(int)kManuf;                 // ~5% of rows from the target manufacturer
        man[i]=m; moy[i]=mo; yr[i]=y; bgrp[i]=bg; sales[i]=sl;
        if (m==(int)kManuf && mo==(int)kMonth && y<(int)kYears) ref[y*kBrands+bg]+=sl;
    }
    IntTable t;
    t.add("i_manufact_id", move(man),  1001);
    t.add("d_moy",         move(moy),    13);
    t.add("d_year",        move(yr),     6);    // sales years 1998..2003
    t.add("brand_group",   move(bgrp),  kBrands);
    t.add("ss_ext_sales_price", move(sales), 0);
    return {move(t), move(ref)};
}

int main(int argc, char** argv)
{
    bool oracle_only = false;
    const RunConfig cfg = tpc_parse_args(argc, argv, oracle_only);
    auto [int_table, reference] = build_q3_int(cfg.rows, cfg.seed);

    TpcQuery q;
    q.title = "TPC-DS Q3 online-comparison benchmark (real domains, end-to-end)";
    q.table = &int_table;
    q.specs = {
        {"manufact", 0, CompOp::EQ, 1001, kManuf, 0},
        {"moy",      1, CompOp::EQ,   13, kMonth, 0},
    };
    for (size_t y=0; y<kYears;  y++) q.specs.push_back({"year_"+to_string(y), 2, CompOp::EQ, 6, y, 0});
    for (size_t b=0; b<kBrands; b++) q.specs.push_back({"brand_"+to_string(b), 3, CompOp::EQ, kBrands, b, 0});
    // pred bits: 0 manufact, 1 moy, 2..2+kYears-1 years, then kBrands brands; then sales
    q.weight_src = {4}; q.weight_max = {kSalesMax3};
    const int wsales = (int)tpc_weight_base(q);
    for (size_t y=0; y<kYears; y++) for (size_t b=0; b<kBrands; b++)
        q.outputs.push_back({"y"+to_string(y)+"_b"+to_string(b), {0,1,2+y,2+kYears+b}, FilterKind::all_true, wsales});
    q.reference = reference;
    return run_tpc_query(q, cfg, oracle_only);
}
