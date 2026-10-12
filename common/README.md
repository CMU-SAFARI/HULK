# common

Shared headers included by every driver (CPU and GPU).

- `HE_workload_common.h`: B/FV context set-up (N = 32768, 25-bit batching plaintext modulus,
  57-bit primes, 55-bit at 16 limbs), bit-sliced encoding, the arithmetic product tree (AND-K with K-1 HMuls,
  OR-K = 1 - prod(1 - x_i)), HULK's sum-then-indicate operations (root-paired factorization
  for small K, Paterson-Stockmeyer otherwise; `HULK_INDICATOR_PS_MIN` sets the switch point),
  deferred normalization of the indicator, client-side slot reduction, and per-stage
  operation counters (`STAGE` lines).
- `HE_comparison.h`: online predicate generation for the TPC queries with the comparison
  polynomial of Iliashenko and Zucca (degree D-1 over a column domain of D values), shared
  baby-step/giant-step powers per column, and the filter/AND/weight/aggregate stages.

Environment switches used by the runs: `TPC_DISABLE_CSE=1` (both modes, TPC queries).
