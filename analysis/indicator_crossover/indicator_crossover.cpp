/* Paired-root vs Paterson-Stockmeyer for the degree-n AND indicator [S == n].
 * Reports ct-ct ops, correctness and latency at a given limb count, so the
 * crossover fan-in can be located empirically. */
#include "HE_workload_common.h"
using namespace hulk;
int main(int argc,char**argv){
  const size_t n = argc>1? strtoul(argv[1],nullptr,10) : 16;
  const size_t L = argc>2? strtoul(argv[2],nullptr,10) : 8;
  const int    M = argc>3? atoi(argv[3]) : 0;          // 0 = paired-root, 1 = PS
  const HEEnv env = make_env(EvalMode::hulk, 1, L);
  // S takes every reachable value 0..n across the slots; [S==n] must hold only at S=n
  vector<uint64_t> slots(env.slot_count);
  for(size_t i=0;i<env.slot_count;++i) slots[i] = i % (n+1);
  Ciphertext S = encrypt_slots(env, slots);
  const auto t0=steady_clock::now();
  auto r = M ? eval_poly_ps_streaming(env, S, *indicator_poly_plan(env,n,n))
             : hulk_indicator_from_sum(S,n,n,env);
  const double ms = duration<double,milli>(steady_clock::now()-t0).count();
  vector<uint64_t> got = decrypt_slots(env,r.first);
  bool ok=true;
  for(size_t i=0;i<env.slot_count;++i){
    uint64_t want = ((i%(n+1))==n)?1:0;
    if((got[i]%env.plain_modulus)!=want){ ok=false; break; }
  }
  cout<<(M?"PS ":"PR ")<<"n="<<n<<" limbs="<<L
      <<" ops="<<r.second.total_ct_mul()<<" depth="<<r.second.depth
      <<" ms="<<fixed<<setprecision(1)<<ms
      <<" budget="<<env.dec->invariant_noise_budget(r.first)
      <<" "<<(ok?"OK":"WRONG")<<"\n";
  return ok?0:1;
}
