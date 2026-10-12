# AND-K across HE approaches (Sec. 4.2, Table 1)

One bulk AND-K per record over 32,768 bit-sliced records.

- `andk_e2e.cpp` (SEAL B/FV): `A1` arithmetic product tree (our baseline), `A2` product tree with
  lazy relinearization, `A3` sum + Fermat zero test (p = 65537), `A7s` HULK with server-side
  decoding, `A7c` HULK with client-side decoding, and THRESHOLD-31/73 with a counting tree (`TT`),
  Fermat tests (`TF`), and HULK (`TH`). Every output is checked against a plaintext oracle.
- `tfhe/` (TFHE-rs 1.7.0): `A5` multi-input programmable bootstrapping and `A6` gate-by-gate AND.
- `andk_csv.py`: builds `output/HULK_andK_comparison.csv` from the logs in `$HULK_ROOT/results_andk`, plus the
  analytical cost of a single equality test with a small plaintext prime (`A4`).
- `run.sh`: runs all configurations (limb search, four timed rounds).

The Fermat rows give the crossover with HULK's indicator (K = 128: 188 us vs. 192 us per record).
