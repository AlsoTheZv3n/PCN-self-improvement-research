"""Build + correctness gate for the fused CUDA settling kernel (docs/12 §4d/§4g).

Run in a vcvars64 shell with the kernel enabled:
    "<...>\\vcvars64.bat" && set PCN_CUDA_KERNEL=1 && uv run python scripts\\verify_kernel.py

Asserts the fused kernel reproduces the PyTorch settling EQUILIBRIUM (allclose) on the §4g
topology [784,32,32,5] (model.n==3), per activation, on BOTH dispatch paths (B<=128 -> v1
per-sample; B>128 -> v2 tiled). Stable regime (orthogonal init, lr_state=0.05) so float drift
does not mask a real bug (docs/12 §4d lesson: an UNstable settling config amplifies tiny float
diffs into a false mismatch). This is the gate before wiring the kernel into the experiments.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from pcn.model import PCN
from pcn.settling import feedforward_init, settle

assert torch.cuda.is_available(), "need a CUDA device"
assert os.environ.get("PCN_CUDA_KERNEL") == "1", "set PCN_CUDA_KERNEL=1 to enable the kernel"

dev = "cuda"
print(f"device {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
ACTS = ("tanh", "sigmoid")     # sigmoid requires kernel act==2 (added in §4g kernel work)
fail = 0
for act in ACTS:
    for B in (32, 256):        # 32 -> v1 (per-sample); 256 -> v2 (tiled)
        torch.manual_seed(0)
        model = PCN([784, 32, 32, 5], activation=act, weight_init="orthogonal", device=dev, seed=0)
        x = torch.randn(B, 784, device=dev)
        s = feedforward_init(model, x)
        s[-1] = F.one_hot(torch.randint(0, 5, (B,), device=dev), 5).float()  # clamp output (train-style)
        try:
            spt, _, _ = settle(model, [t.clone() for t in s], clamp_output=True, T=20,
                               lr_state=0.05, backend="pytorch")
            scu, _, _ = settle(model, [t.clone() for t in s], clamp_output=True, T=20,
                               lr_state=0.05, backend="cuda")
        except Exception as e:
            print(f"  act={act:7} B={B:3}: ERROR {type(e).__name__}: {e}")
            fail += 1
            continue
        md = max(float((a - b).abs().max()) for a, b in zip(spt, scu))
        ok = all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(spt, scu))
        print(f"  act={act:7} B={B:3}: maxdiff={md:.2e} allclose={ok}  [{'v1' if B <= 128 else 'v2'}]")
        if not ok:
            fail += 1

print("RESULT:", "ALL MATCH ✓" if fail == 0 else f"{fail} MISMATCH/ERROR ✗")
raise SystemExit(1 if fail else 0)
