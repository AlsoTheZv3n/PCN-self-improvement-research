"""M6 benchmark: fused CUDA settling kernel vs the PyTorch backend (docs/09, docs/13 M6).

Times a full T-step settle of a batch on the default MNIST topology [784,256,256,10], with
correct CUDA timing (warmup + CUDA events). The scientific question (docs/09): the PyTorch
backend launches many tiny ops per step (launch-overhead-bound at MNIST sizes); does fusing
the whole T-loop into one launch win?

Run in a vcvars64 shell with the kernel enabled:
    set PCN_CUDA_KERNEL=1   (cmd)   /   $env:PCN_CUDA_KERNEL=1  (pwsh)
    uv run python scripts/bench_kernel.py
"""
from __future__ import annotations

import json
import os

import torch

from pcn.model import PCN
from pcn.settling import feedforward_init, settle

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _time(fn, warmup: int = 3, iters: int = 30) -> float:
    """Mean ms per call on CUDA (events + synchronize)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    assert torch.cuda.is_available(), "need a CUDA device"
    assert os.environ.get("PCN_CUDA_KERNEL") == "1", "set PCN_CUDA_KERNEL=1 to enable the kernel"
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}\n")
    print(f"{'B':>4} {'T':>4} | {'PyTorch-SO ms':>14} {'fused-CUDA ms':>14} {'speedup':>8} "
          f"| {'pt/step':>8} {'cu/step':>8}")
    rows = []
    for B in (64, 256, 1024, 4096):
        for T in (20, 40):
            model = PCN([784, 256, 256, 10], device=dev, seed=0)
            x = torch.randn(B, 784, device=dev)
            states = feedforward_init(model, x)
            pt = _time(lambda: settle(model, states, clamp_output=True, T=T, lr_state=0.1,
                                      backend="pytorch"))
            cu = _time(lambda: settle(model, states, clamp_output=True, T=T, lr_state=0.1,
                                      backend="cuda"))
            print(f"{B:>4} {T:>4} | {pt:>14.3f} {cu:>14.3f} {pt / cu:>7.2f}x "
                  f"| {pt / T:>8.4f} {cu / T:>8.4f}")
            rows.append({"B": B, "T": T, "pytorch_ms": pt, "cuda_ms": cu, "speedup": pt / cu})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "m6_kernel_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "topology": [784, 256, 256, 10], "rows": rows}, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
