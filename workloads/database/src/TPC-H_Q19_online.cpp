/*
 * TPC-H Q19 -- online homomorphic-comparison pipeline, real TPC domains.
 * Three disjoint branches j=1..3, each an AND of six predicates:
 *   p_brand = Brand#j                 -> EQ    brand     (D=25)
 *   p_container in (4 containers)      -> IN    container (D=40)   SM* / MED* / LG*
 *   l_quantity between q_j and q_j+10  -> RANGE quantity  (D=51)   q = 5, 15, 25
 *   p_size between 1 and s_j           -> RANGE size      (D=51)   s = 5, 10, 15
 *   l_shipmode in ('AIR','AIR REG')    -> IN    shipmode  (D=7)    shared by all branches
 *   l_shipinstruct = 'DELIVER IN PERSON' -> EQ  shipinstruct (D=4) shared by all branches
 * Output: 3 revenue SUMs, one per branch (paper: 3), computed homomorphically; total = sum.
 */
#include "HE_comparison.h"
#include <random>
#include <array>
using namespace hulk;

constexpr uint64_t kRevMax19 = 700;    // 700 x 32768 < t
constexpr array<int,3> kBrand{11, 12, 13};
constexpr array<int,3> kQty{5, 15, 25};
constexpr array<int,3> kSize{5, 10, 15};
constexpr array<int,3> kCont0{0, 10, 20};               // containers kCont0[j]..kCont0[j]+3

static pair<IntTable, vector<uint64_t>> build_q19_int(size_t rows, uint64_t seed) {
    mt19937_64 rng(seed);
    uniform_int_distribution<int> brand_d(0,24), cont_d(0,39), qty_d(1,50), size_d(1,50),
                                  mode_d(0,6), instr_d(0,3), rev_d(80,(int)kRevMax19), pick(0,9);
    vector<uint64_t> br(rows), ct(rows), qt(rows), sz(rows), sm(rows), si(rows), rev(rows), ref(3,0);
    for (size_t i=0;i<rows;i++) {
        int b=brand_d(rng), c=cont_d(rng), q=qty_d(rng), z=size_d(rng), m=mode_d(rng), s=instr_d(rng);
        const int k = pick(rng);                         // plant ~30% of rows into one branch
        if (k < 3) { b=kBrand[k]; c=kCont0[k]+int(rng()%4); q=kQty[k]+int(rng()%11); z=1+int(rng()%kSize[k]);
                     m=int(rng()%2); s=0;
                     if (rng()%4==0) { switch(rng()%3){ case 0: q=(q+20)%50+1; break; case 1: m=2+int(rng()%5); break; default: s=1; } } }
        br[i]=b; ct[i]=c; qt[i]=q; sz[i]=z; sm[i]=m; si[i]=s; rev[i]=rev_d(rng);
        for (int j=0;j<3;j++)
            if (b==kBrand[j] && c>=kCont0[j] && c<=kCont0[j]+3 && q>=kQty[j] && q<=kQty[j]+10 &&
                z>=1 && z<=kSize[j] && m<=1 && s==0) ref[j]+=rev[i];
    }
    IntTable t;
    t.add("brand",move(br),25); t.add("container",move(ct),40); t.add("quantity",move(qt),51);
    t.add("size",move(sz),51); t.add("shipmode",move(sm),7); t.add("shipinstruct",move(si),4);
    t.add("revenue",move(rev),0);
    return {move(t),move(ref)};
}

int main(int argc, char** argv) {
    bool oracle_only=false;
    const RunConfig cfg=tpc_parse_args(argc,argv,oracle_only);
    auto [int_table,ref3]=build_q19_int(cfg.rows,cfg.seed);
    cout<<"reference total="<<(ref3[0]+ref3[1]+ref3[2])<<"\n";
    TpcQuery q;
    q.title="TPC-H Q19 online-comparison benchmark (real domains, end-to-end)";
    q.table=&int_table;
    CompSpec air{"air",4,CompOp::IN,7,0,0}; air.in_set={0,1};
    q.specs.push_back(air);                                                // 0
    q.specs.push_back({"dip",5,CompOp::EQ,4,0,0});                         // 1
    for (int j=0;j<3;j++) {                                                // 2+4j .. 5+4j
        q.specs.push_back({"brand"+to_string(j),0,CompOp::EQ,25,(size_t)kBrand[j],0});
        CompSpec c{"cont"+to_string(j),1,CompOp::IN,40,0,0};
        for (int d=0;d<4;d++) c.in_set.push_back((size_t)(kCont0[j]+d));
        q.specs.push_back(c);
        q.specs.push_back({"qty"+to_string(j),2,CompOp::RANGE,51,(size_t)kQty[j],(size_t)kQty[j]+10});
        q.specs.push_back({"size"+to_string(j),3,CompOp::RANGE,51,1,(size_t)kSize[j]});
    }
    q.weight_src={6}; q.weight_max={kRevMax19};
    const int wcol=(int)tpc_weight_base(q);                                // revenue after the 14 predicate bits
    for (size_t j=0;j<3;j++)
        q.outputs.push_back({"branch"+to_string(j+1),{0,1,2+4*j,3+4*j,4+4*j,5+4*j},FilterKind::all_true,wcol});
    q.reference=ref3;
    return run_tpc_query(q,cfg,oracle_only,[](const TpcModeResult& r){
        uint64_t tot=0; for(auto v:r.res.values) tot+=v;          // disjoint branches: total = sum (client)
        cout<<"  total_revenue="<<tot<<"\n";
    });
}
