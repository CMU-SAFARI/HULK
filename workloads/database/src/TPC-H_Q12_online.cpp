/*
 * TPC-H Q12 -- online homomorphic-comparison pipeline, real TPC domains.
 *
 *   where l_shipmode in (':1',':2')            -> EQ shipmode (D=7), one per mode (group by)
 *     and l_commitdate < l_receiptdate          -> PASS (column-vs-column, precomputed bit)
 *     and l_shipdate   < l_commitdate           -> PASS (column-vs-column, precomputed bit)
 *     and l_receiptdate >= date ':3'            -> RANGE l_receiptdate in 1994 = days [729,1093],
 *     and l_receiptdate <  date ':3' + 1 year      day index 0 = 1992-01-03 .. 2554 = 1998-12-31 (D=2555);
 *                                                  1994-01-01 = 729, 1995-01-01 = 1094 (excluded)
 *   4 COUNT outputs (paper: 4), computed homomorphically (slot aggregation).
 *   The receiptdate window is one RANGE predicate (== GE AND LT), so AND-5 per output.
 *   group by l_shipmode; high_line_count = orderpriority in {1-URGENT,2-HIGH} (IN over D=5),
 *                        low_line_count  = the other three priorities.
 */
#include "HE_comparison.h"
#include <random>
using namespace hulk;

constexpr size_t kRecvD = 2555, kY0 = 729, kY1 = 1093;

static pair<IntTable, vector<uint64_t>> build_q12_int(size_t rows, uint64_t seed) {
    mt19937_64 rng(seed);
    uniform_int_distribution<int> mode_d(0,6), recv_d(0,(int)kRecvD-1), pri_d(0,4);
    bernoulli_distribution cr_d(0.6), sc_d(0.5);
    vector<uint64_t> mode(rows), recv(rows), pri(rows), cr(rows), sc(rows), ref(4,0);
    for (size_t i=0;i<rows;i++) {
        int m=mode_d(rng), r=recv_d(rng), p=pri_d(rng); bool a=cr_d(rng), b=sc_d(rng);
        mode[i]=m; recv[i]=r; pri[i]=p; cr[i]=a; sc[i]=b;
        const bool date_ok = a && b && r>=(int)kY0 && r<=(int)kY1, hi = p<=1;
        if (m==0 && date_ok &&  hi) ref[0]++;
        if (m==0 && date_ok && !hi) ref[1]++;
        if (m==1 && date_ok &&  hi) ref[2]++;
        if (m==1 && date_ok && !hi) ref[3]++;
    }
    IntTable t;
    t.add("shipmode",move(mode),7); t.add("receiptdate",move(recv),kRecvD); t.add("orderpriority",move(pri),5);
    t.add("commit_lt_receipt",move(cr),2); t.add("ship_lt_commit",move(sc),2);
    return {move(t),move(ref)};
}
int main(int argc, char** argv) {
    bool oracle_only=false;
    const RunConfig cfg=tpc_parse_args(argc,argv,oracle_only);
    auto [int_table,reference]=build_q12_int(cfg.rows,cfg.seed);
    TpcQuery q;
    q.title="TPC-H Q12 online-comparison benchmark (real domains, end-to-end)";
    q.table=&int_table;
    CompSpec hi{"pri_hi",2,CompOp::IN,5,0,0}; hi.in_set={0,1};
    CompSpec lo{"pri_lo",2,CompOp::IN,5,0,0}; lo.in_set={2,3,4};
    q.specs={{"is_mail",0,CompOp::EQ,7,0,0},{"is_ship",0,CompOp::EQ,7,1,0},
        {"recv_1994",1,CompOp::RANGE,kRecvD,kY0,kY1}, hi, lo,
        {"commit_lt_receipt",3,CompOp::PASS,2,0,0},{"ship_lt_commit",4,CompOp::PASS,2,0,0}};
    // pred indices: 0 mail, 1 ship, 2 recv, 3 hi, 4 lo, 5 cr, 6 sc;  COUNT outputs, no weight column
    q.outputs={{"mail_hi",{0,2,3,5,6},FilterKind::all_true,-1},
        {"mail_lo",{0,2,4,5,6},FilterKind::all_true,-1},{"ship_hi",{1,2,3,5,6},FilterKind::all_true,-1},
        {"ship_lo",{1,2,4,5,6},FilterKind::all_true,-1}};
    q.reference=reference;
    return run_tpc_query(q,cfg,oracle_only,[](const TpcModeResult& r){
        const auto& v=r.res.values;
        cout<<"  MAIL(hi="<<v[0]<<" lo="<<v[1]<<") SHIP(hi="<<v[2]<<" lo="<<v[3]<<")\n";
    });
}
