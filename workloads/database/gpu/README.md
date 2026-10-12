# GPU runs

`run.sh [workload ...]` builds the same drivers in `../src` for the GPU through `gpu_shim`
(`HEONGPU_DIR`) and measures both modes with the protocol of `scripts/protocol.sh`.
Without arguments it runs every workload of this domain. Parse the logs with
`scripts/parse/gpu_parse.py` and `scripts/parse/make_results_csv.py`.
