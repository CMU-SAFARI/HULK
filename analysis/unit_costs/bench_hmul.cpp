// Single-thread B/FV op timing at the online-filter parameters (N=32768, 12 x 55-bit, 25-bit batching p).
#include "seal/seal.h"
#include <chrono>
#include <iostream>
using namespace seal; using namespace std; using namespace std::chrono;
int main(){
    EncryptionParameters p(scheme_type::bfv); p.set_poly_modulus_degree(32768);
    p.set_coeff_modulus(CoeffModulus::Create(32768, vector<int>(12,55))); p.set_plain_modulus(PlainModulus::Batching(32768,25));
    SEALContext ctx(p); KeyGenerator kg(ctx); PublicKey pk; kg.create_public_key(pk); RelinKeys rk; kg.create_relin_keys(rk);
    Encryptor enc(ctx,pk); Evaluator ev(ctx); BatchEncoder be(ctx);
    Plaintext pt; be.encode(vector<uint64_t>(be.slot_count(),1),pt); Ciphertext a,b,c; enc.encrypt(pt,a); enc.encrypt(pt,b);
    auto t=[&](auto f,int n){ auto s=steady_clock::now(); for(int i=0;i<n;i++) f(); return duration<double,milli>(steady_clock::now()-s).count()/n; };
    double m=t([&]{ ev.multiply(a,b,c); ev.relinearize_inplace(c,rk); },20);
    double sq=t([&]{ ev.square(a,c); ev.relinearize_inplace(c,rk); },20);
    double ad=t([&]{ ev.add(a,b,c); },200);
    cout<<"HMul+relin "<<m<<" ms | square+relin "<<sq<<" ms | HAdd "<<ad<<" ms\n";
}
