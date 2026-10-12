# CPU runs

`run.sh [workload ...]` builds the drivers in `../src` with Microsoft SEAL 4.1 (`SEAL_DIR`) and
measures both modes with the protocol of `scripts/protocol.sh`: limb search (pass twice), then
four A-B-B-A timed rounds, one thread pinned to one core. Without arguments it runs every
workload of this domain. Parse the logs with `scripts/parse/make_results_csv.py`.
