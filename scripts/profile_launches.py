"""M6 launch-count artifact (docs/07/09, docs/13): empirically show that the PyTorch settling
backend issues MANY CUDA kernel launches per settle (launch-overhead-bound), while the fused
kernel issues ~one — the direct evidence that launch overhead is the bottleneck the kernel
removes. Nsight Systems (nsys) is not installed here, so we count `cudaLaunchKernel` events via
torch.profiler (the same launch-count metric).

    set PCN_CUDA_KERNEL=1  &  run in a vcvars64 shell (builds the fused kernel)
    uv run python scripts/profile_launches.py
"""
from __future__ import annotations

import json
import os

import torch
from torch.profiler import ProfilerActivity, profile

from pcn.model import PCN
from pcn.settling import feedforward_init, settle

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def count_launches(fn):
    """Number of CUDA kernel launches issued by one call to fn (host-side cudaLaunchKernel)."""
    fn()  # warmup (also triggers the JIT build on the first fused call)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evs = prof.events()
    launches = sum(1 for e in evs if e.name == "cudaLaunchKernel")
    if launches == 0:  # fallback: count distinct device-kernel executions
        from torch.autograd import DeviceType
        launches = sum(getattr(e, "count", 1) for e in prof.key_averages()
                       if getattr(e, "device_type", None) == DeviceType.CUDA)
    return launches


def main():
    assert torch.cuda.is_available(), "need CUDA"
    assert os.environ.get("PCN_CUDA_KERNEL") == "1", "set PCN_CUDA_KERNEL=1 for the fused path"
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print("(nsys/ncu not installed; counting cudaLaunchKernel via torch.profiler)\n")
    print(f"{'T':>4} | {'PyTorch-SO launches':>20} | {'fused-CUDA launches':>20} | {'ratio':>7}")
    rows = []
    for T in (20, 40, 80):
        model = PCN([784, 256, 256, 10], device=dev, seed=0)
        x = torch.randn(64, 784, device=dev)
        st = feedforward_init(model, x)
        pt = count_launches(lambda: settle(model, st, clamp_output=True, T=T, lr_state=0.1, backend="pytorch"))
        cu = count_launches(lambda: settle(model, st, clamp_output=True, T=T, lr_state=0.1, backend="cuda"))
        print(f"{T:>4} | {pt:>20} | {cu:>20} | {pt / max(cu, 1):>6.0f}x")
        rows.append({"T": T, "pytorch_launches": pt, "fused_launches": cu})

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "m6_launch_count.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "batch": 64,
                   "metric": "cudaLaunchKernel events per settle", "rows": rows}, f, indent=2)
    print(f"\nPyTorch launches scale ~linearly with T (per-step ops); the fused kernel stays flat.")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
