/*
 * TPC-DS Q7 -- online homomorphic-comparison pipeline, real TPC domains.
 *
 *   where cd_gender = 'F'                     -> EQ gender  (D=2),  target 1
 *     and cd_marital_status = 'M'             -> EQ marital (D=5),  target 1
 *     and cd_education_status = 'College'    -> EQ educ    (D=7),  target 1
 *     and (p_channel_email = 'N' or p_channel_event = 'N')
 *                                             -> IN promo code (D=4) in {0,1,2}; code 3 = (Y,Y) fails
 *     and d_year = [YEAR]                     -> EQ year    (D=6, 1998..2003), target 2
 *   group by i_item_id (kItems7 items)        -> EQ item    (D=kItems7), one bit per item
 *   outputs per item: count (weight -1), avg numerators qty, list, coupon, sales (weights)
 *
 * IntTable columns: 0 year, 1 gender, 2 marital, 3 educ, 4 promo, 5 item, 6 qty, 7 list, 8 coupon, 9 sales.
 * 69 filters (5 shared + 64 item bits), 64 unique AND-6 filters, 320 outputs (paper: 69 / 64 / 320).
 * comp_tbl columns: 0..4 shared bits, 5..5+kItems7-1 item bits, W0=5+kItems7 .. W0+3 = qty, list, coupon, sales.
 */
#include "HE_comparison.h"
#include <random>
using namespace hulk;

constexpr size_t kItems7 = 64;  // modelled i_item_id groups: 64 x 5 metrics = 320 outputs (paper: 320)
constexpr uint64_t kQtyMax7 = 12, kListMax7 = 200, kCpnMax7 = 20, kSalMax7 = 180;   // all x 32768 < t

static pair<IntTable, vector<uint64_t>>
build_q7_int(size_t rows, uint64_t seed)
{
    mt19937_64 rng(seed);
    // real domains: d_year 1998..2003 (6), cd_gender (2), cd_marital_status (5), cd_education_status (7),
    // channel code = (p_channel_email, p_channel_event) in {0..3}, code 3 = (Y,Y) fails the OR
    uniform_int_distribution<int> yr_d(0,5), gen_d(0,1), mar_d(0,4), edu_d(0,6),
                                   pro_d(0,3), itm_d(0,(int)kItems7-1),
                                   qty_d(1,(int)kQtyMax7), lst_d(20,(int)kListMax7), cpn_d(0,(int)kCpnMax7), sal_d(10,(int)kSalMax7);

    size_t nc = 6 + 4;
    vector<vector<uint64_t>> cols(nc, vector<uint64_t>(rows,0));
    vector<uint64_t> ref(kItems7*5, 0);

    for (size_t i=0; i<rows; i++) {
        int y=yr_d(rng),g=gen_d(rng),m=mar_d(rng),e=edu_d(rng),p=pro_d(rng);
        int it=itm_d(rng),q=qty_d(rng),l=lst_d(rng),c=cpn_d(rng),s=sal_d(rng);
        if (rng()%8==0) { y=2; g=1; m=1; e=1; p=int(rng()%3); }   // plant ~1/8 of rows into the qualifying slice (~64/item)
        cols[0][i]=y; cols[1][i]=g; cols[2][i]=m; cols[3][i]=e; cols[4][i]=p;
        cols[5][i]=it; cols[6][i]=q; cols[7][i]=l; cols[8][i]=c; cols[9][i]=s;

        bool pass = (y==2 && g==1 && m==1 && e==1 && p!=3);
        if (pass) {
            ref[it*5+0]++; ref[it*5+1]+=q; ref[it*5+2]+=l; ref[it*5+3]+=c; ref[it*5+4]+=s;
        }
    }

    IntTable t;
    vector<string> names = {"year","gender","marital","educ","promo","item",
                            "qty","list","coupon","sales"};
    vector<size_t> domains = {6,2,5,7,4,(size_t)kItems7, 0,0,0,0};
    for (size_t c=0; c<nc; c++) t.add(names[c], move(cols[c]), domains[c]);
    return {move(t), move(ref)};
}

int main(int argc, char** argv)
{
    bool oracle_only = false;
    const RunConfig cfg = tpc_parse_args(argc, argv, oracle_only);
    auto [int_table, reference] = build_q7_int(cfg.rows, cfg.seed);

    TpcQuery q;
    q.title = "TPC-DS Q7 online-comparison benchmark (end-to-end, items=" + to_string(kItems7) + ")";
    q.table = &int_table;
    q.specs = {
        {"year",   0, CompOp::EQ, 6, 2, 0},
        {"gender", 1, CompOp::EQ, 2, 1, 0},
        {"marital",2, CompOp::EQ, 5, 1, 0},
        {"educ",   3, CompOp::EQ, 7, 1, 0},
    };
    { CompSpec pr{"promo_email_or_event_N", 4, CompOp::IN, 4, 0, 0}; pr.in_set={0,1,2}; q.specs.push_back(pr); }
    for (size_t it=0; it<kItems7; it++)
        q.specs.push_back({"item_"+to_string(it), 5, CompOp::EQ, kItems7, it, 0});
    // pred: 0..4=shared, 5..5+kItems7-1=item bits; weights qty,list,coupon,sales after them
    q.weight_src = {6,7,8,9}; q.weight_max = {kQtyMax7, kListMax7, kCpnMax7, kSalMax7};
    const int W0 = (int)tpc_weight_base(q);
    for (size_t it=0; it<kItems7; it++) {
        const size_t ib = 5+it;
        const vector<size_t> f = {0,1,2,3,4,ib};
        q.outputs.push_back({"i"+to_string(it)+"_cnt", f, FilterKind::all_true, -1});    // COUNT: no weight
        q.outputs.push_back({"i"+to_string(it)+"_qty", f, FilterKind::all_true, W0+0});
        q.outputs.push_back({"i"+to_string(it)+"_lst", f, FilterKind::all_true, W0+1});
        q.outputs.push_back({"i"+to_string(it)+"_cpn", f, FilterKind::all_true, W0+2});
        q.outputs.push_back({"i"+to_string(it)+"_sal", f, FilterKind::all_true, W0+3});
    }
    q.reference = reference;   // generator order per item: cnt, qty, lst, cpn, sal (== output order)
    return run_tpc_query(q, cfg, oracle_only);   // avg_* = sum/cnt on the client
}
