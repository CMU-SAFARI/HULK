"""Per-operation unit costs from a bench_ops log ("UNIT L=<limbs> op=<op> ms=<ms>" lines).
CPU log: $HULK_ROOT/results_bench/bench_ops_cpu.log; GPU log: $HULK_ROOT/results_gpu_fair/bench_ops.log
(both written by analysis/unit_costs/run.sh)."""
import os, re
HULK_ROOT = os.environ.get('HULK_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
CPU_LOG = os.path.join(HULK_ROOT, 'results_bench', 'bench_ops_cpu.log')
GPU_LOG = os.path.join(HULK_ROOT, 'results_gpu_fair', 'bench_ops.log')


def load(path):
    """{limbs: {op: ms}}"""
    out = {}
    for line in open(path):
        m = re.match(r'UNIT L=(\d+) op=(\S+) ms=([\d.]+)', line)
        if m: out.setdefault(int(m.group(1)), {})[m.group(2)] = float(m.group(3))
    return out
