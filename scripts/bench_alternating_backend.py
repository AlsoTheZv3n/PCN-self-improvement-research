"""Wall-clock + correctness of the fused CUDA settling kernel ON THE ACTUAL §4g WORKLOAD.

This is the Hook A x Hook B tie-in: the kernel's reason-for-being is that PC settling at SMALL
batch is launch-overhead-bound (docs/12 §4d) — which is exactly the regime of the Song-exact
alternating study (batch 32, sigmoid [32,32], §4g). Here we run the §4g PC arm on both backends,
time each (kernel compile warmed up first, not timed), and compare per-seed mean_both so the
speedup claim comes with a correctness check on the end-to-end study, not just a microbenchmark.

Run in a vcvars64 shell:
    "<...>\\vcvars64.bat" && set "PCN_CUDA_KERNEL=1" && uv run python scripts\\bench_alternating_backend.py
"""
from __future__ import annotations

import os
import time

import torch

from pcn.experiments.continual import run_alternating

assert torch.cuda.is_available(), "need a CUDA device"
assert os.environ.get("PCN_CUDA_KERNEL") == "1", "set PCN_CUDA_KERNEL=1 to enable the kernel"

BUDGET = 800              # the regime where the §4g study actually trains (docs/12 §4g)
SEEDS = [0, 1, 2, 3, 4]
base = {"hidden": [32, 32], "activation": "sigmoid", "weight_init": "xavier", "T": 20,
        "eta_x": 0.05, "eta_w": 0.1, "batch_size": 32, "limit_train": 5000, "tol": None,
        "eval_noise": False}


def run_all(backend):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mb = [run_alternating("pc", {**base, "backend": backend, "seed": s}, total_iters=BUDGET,
                          swap_every=4, eval_stride=max(1, BUDGET // (4 * 20)), split_seed=s)["mean_both"]
          for s in SEEDS]
    torch.cuda.synchronize()
    return time.perf_counter() - t0, mb


print(f"device {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
print(f"§4g PC arm | budget={BUDGET}, batch=32, sigmoid [32,32], n={len(SEEDS)} seeds\n")

tp, mp = run_all("pytorch")
# warm up the kernel JIT compile so it is NOT counted in the cuda timing
run_alternating("pc", {**base, "backend": "cuda", "seed": 0}, total_iters=8, swap_every=4, split_seed=0)
tc, mc = run_all("cuda")

maxdiff = max(abs(a - b) for a, b in zip(mp, mc)) * 100
print(f"  pytorch : {tp:7.1f}s   mean_both/seed = {[round(v * 100, 1) for v in mp]}")
print(f"  cuda    : {tc:7.1f}s   mean_both/seed = {[round(v * 100, 1) for v in mc]}")
print(f"\n  speedup : {tp / tc:.2f}x   | max per-seed mean_both diff = {maxdiff:.2f} pp "
      f"({'statistically identical' if maxdiff < 3 else 'DIVERGED — investigate'})")
