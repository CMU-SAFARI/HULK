# Unit costs

`bench_ops.cpp` measures the single-thread latency of every homomorphic operation (HAdd, HMul +
relinearization, squaring, plaintext multiplication, rotations, ...) for each limb count in
`--lmin..--lmax` (default 4..16): two warm-up calls, then `--reps` timed calls (default 15).
It prints `UNIT L= op= ms=` lines. The parsers multiply these unit costs with the per-stage
operation counts of each run to split the measured latency into stages (Fig. 2, Table 3).
`run.sh cpu` and `run.sh gpu` write the logs that `scripts/parse/units.py` reads.
`bench_hmul.cpp` is a minimal HMul timing check.

