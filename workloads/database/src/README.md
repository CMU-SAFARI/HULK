# Driver sources

Each driver implements the arithmetic approach and HULK on the same data and checks every
output against a plaintext oracle (`ok=match`). It prints `E2E ... eval_ms=...` (homomorphic
evaluation time) and one `STAGE mode= stage= hmul= sq= ...` line per stage. See the README
of the parent folder for the workloads and flags.
