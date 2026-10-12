//! AND-K per record with TFHE-rs 1.7.0, sampled and extrapolated to a 32,768-record block.
//!
//! A5  shortint multi-input PBS: per record, the K encrypted bits are added (unchecked) in
//!     groups of g <= (message*carry space - 1) and one PBS applies the LUT [s == group size];
//!     group outputs feed the next tree level until one bit remains.
//!     MESSAGE_2_CARRY_2 (16 values) and MESSAGE_3_CARRY_3 (64 values); g is also capped by the
//!     parameter set's max_noise_level (sums of more fresh ciphertexts void the p_fail guarantee).
//! A6  boolean gate-by-gate: K-1 homomorphic AND gates per record (balanced tree), one PBS per gate.
//!
//! Every sampled record is decrypted and checked against the plaintext AND.
//! Output: one line per (approach, params, K): pbs_per_record, measured sample times at 1 thread
//! and with the rayon pool (RAYON_NUM_THREADS), and the extrapolation to 32,768 records.
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;
use tfhe::boolean::prelude as b;
use tfhe::boolean::prelude::BinaryBooleanGates;
use tfhe::shortint::parameters::{PARAM_MESSAGE_2_CARRY_2_KS_PBS, PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128};
use tfhe::shortint::{gen_keys, ClassicPBSParameters};

const BLOCK: usize = 32768;

fn records(k: usize, n: usize, seed: u64) -> Vec<Vec<bool>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|i| {
        let mut r = vec![true; k];
        if i % 8 != 0 { for _ in 0..rng.gen_range(1..=3) { let j = rng.gen_range(0..k); r[j] = false; } }
        r
    }).collect()
}

fn pbs_count(k: usize, g: usize) -> usize {
    let (mut n, mut c) = (k, 0);
    loop { let groups = (n + g - 1) / g; c += groups; if groups == 1 { return c; } n = groups; }
}

fn shortint_run(name: &str, params: ClassicPBSParameters, g_cap: usize, k: usize, sample: usize) {
    let (ck, sk) = gen_keys(params);
    // group size: bounded by the message+carry space AND by the parameter set's noise budget for
    // unchecked additions (max_noise_level), so every PBS input stays within the p_fail guarantee
    let g = g_cap.min(sk.max_noise_level.get() as usize);
    let data = records(k, sample, 7);
    let enc: Vec<Vec<_>> = data.iter().map(|r| r.iter().map(|&x| ck.encrypt(x as u64)).collect()).collect();
    let and_record = |cts: &Vec<tfhe::shortint::Ciphertext>| -> tfhe::shortint::Ciphertext {
        let mut level: Vec<tfhe::shortint::Ciphertext> = cts.clone();
        loop {
            let mut next = Vec::new();
            for chunk in level.chunks(g) {
                let mut acc = chunk[0].clone();
                for c in &chunk[1..] { sk.unchecked_add_assign(&mut acc, c); }
                let need = chunk.len() as u64;
                let lut = sk.generate_lookup_table(move |s| (s == need) as u64);
                next.push(sk.apply_lookup_table(&acc, &lut));
            }
            if next.len() == 1 { return next.pop().unwrap(); }
            level = next;
        }
    };
    // 1 thread
    let t1 = Instant::now();
    let out1: Vec<_> = enc.iter().map(|r| and_record(r)).collect();
    let ms1 = t1.elapsed().as_secs_f64() * 1e3;
    // rayon pool
    let tp = Instant::now();
    let outp: Vec<_> = enc.par_iter().map(|r| and_record(r)).collect();
    let msp = tp.elapsed().as_secs_f64() * 1e3;
    let ok = data.iter().zip(out1.iter().zip(outp.iter())).all(|(r, (a, bq))| {
        let want = r.iter().all(|&x| x) as u64; ck.decrypt(a) == want && ck.decrypt(bq) == want });
    let p = pbs_count(k, g);
    let scale = BLOCK as f64 / sample as f64;
    println!("TFHE approach=A5 params={name} g={g} K={k} pbs_per_record={p} sample={sample} threads={} \
              sample_ms_1t={ms1:.1} sample_ms_par={msp:.1} per_pbs_ms_1t={:.3} block_ms_1t_extrap={:.1} block_ms_par_extrap={:.1} \
              per_record_ms_1t={:.3} per_record_ms_par={:.4} ok={}",
             rayon::current_num_threads(), ms1 / (p * sample) as f64, ms1 * scale, msp * scale,
             ms1 / sample as f64, msp / sample as f64, if ok { "match" } else { "MISMATCH" });
}

fn boolean_run(k: usize, sample: usize) {
    let (ck, sk) = b::gen_keys();
    let data = records(k, sample, 7);
    let enc: Vec<Vec<b::Ciphertext>> = data.iter().map(|r| r.iter().map(|&x| ck.encrypt(x)).collect()).collect();
    let and_record = |cts: &Vec<b::Ciphertext>| -> b::Ciphertext {
        let mut level = cts.clone();
        while level.len() > 1 {
            let mut next: Vec<b::Ciphertext> = level.chunks(2).map(|c| if c.len() == 2 { sk.and(&c[0], &c[1]) } else { c[0].clone() }).collect();
            std::mem::swap(&mut level, &mut next);
        }
        level.pop().unwrap()
    };
    let t1 = Instant::now();
    let out1: Vec<_> = enc.iter().map(|r| and_record(r)).collect();
    let ms1 = t1.elapsed().as_secs_f64() * 1e3;
    let tp = Instant::now();
    let outp: Vec<_> = enc.par_iter().map(|r| and_record(r)).collect();
    let msp = tp.elapsed().as_secs_f64() * 1e3;
    let ok = data.iter().zip(out1.iter().zip(outp.iter())).all(|(r, (a, bq))| {
        let want = r.iter().all(|&x| x); ck.decrypt(a) == want && ck.decrypt(bq) == want });
    let p = k - 1;
    let scale = BLOCK as f64 / sample as f64;
    println!("TFHE approach=A6 params=boolean_default K={k} pbs_per_record={p} sample={sample} threads={} \
              sample_ms_1t={ms1:.1} sample_ms_par={msp:.1} per_pbs_ms_1t={:.3} block_ms_1t_extrap={:.1} block_ms_par_extrap={:.1} \
              per_record_ms_1t={:.3} per_record_ms_par={:.4} ok={}",
             rayon::current_num_threads(), ms1 / (p * sample) as f64, ms1 * scale, msp * scale,
             ms1 / sample as f64, msp / sample as f64, if ok { "match" } else { "MISMATCH" });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sample: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let ks: Vec<usize> = vec![2, 4, 8, 16, 32, 64, 128, 256];
    for &k in &ks {
        shortint_run("MESSAGE_2_CARRY_2", PARAM_MESSAGE_2_CARRY_2_KS_PBS, 15, k, sample);
        shortint_run("MESSAGE_3_CARRY_3", PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128, 63, k, sample);
        boolean_run(k, sample);
    }
}
