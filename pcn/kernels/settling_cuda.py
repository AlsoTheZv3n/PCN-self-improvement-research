"""OPTIONAL Phase-3 / M6 — fused CUDA settling kernel: the Python wrapper + dispatch.

The CUDA SOURCE lives in ``settling_kernel.cu`` (the single source of truth). This module just
compiles it (torch.utils.cpp_extension.load, JIT + cached) and dispatches the batch-size hybrid:

  v1 ``pcn_settle_so``       — ONE BLOCK PER SAMPLE -> wins at SMALL batch (launch overhead gone),
                               but re-reads the weights every step -> bandwidth-bound at large batch.
  v2 ``pcn_settle_so_tiled`` — ONE BLOCK PER TILE OF TB SAMPLES -> each weight element is read once
                               and reused across the tile -> wins at LARGE batch. (phi(input) is
                               precomputed on the host; the batch is zero-padded to a multiple of TB.)

``settle()`` dispatches v1 vs v2 by batch size (TILED_THRESHOLD). Correctness for BOTH is proven
vs ``_settle_pytorch`` (allclose ~1e-6; tanh/identity/sigmoid; scripts/verify_kernel.py). Scope:
2-hidden-layer topology (model.n==3), fixed T, float32/CUDA.
"""
from __future__ import annotations

import os

import torch

_KERNEL = None
TB = 8                  # samples per block for the tiled (v2) kernel (must match #define TB in .cu)
TILED_THRESHOLD = 128   # batch sizes above this use v2 (tiled); at/below use v1 (per-sample)


def build():
    """Compile + cache the fused settling kernel from ``settling_kernel.cu`` (load JIT). Needs
    nvcc + MSVC (a vcvars64 shell on Windows) + a toolkit-matched CUDA torch (cu126)."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    if not torch.cuda.is_available():
        raise NotImplementedError("CUDA settling kernel needs a CUDA device; none available.")
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settling_kernel.cu")
    _KERNEL = load(
        name="pcn_settling_kernel_v3",
        sources=[src],
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _KERNEL


def _act_id(model) -> int:
    """Map model.phi -> kernel act code (0=tanh, 1=identity, 2=sigmoid), detected by signature
    values: phi(0)=0.5 uniquely identifies sigmoid; phi(1)=1 with phi(0)=0 is identity; tanh
    otherwise. Raises on anything else so the kernel never SILENTLY computes a wrong activation
    (the pre-§4g bug: sigmoid fell through to tanh because phi(1)!=1)."""
    z = float(model.phi(torch.zeros(1)).item())
    o = float(model.phi(torch.ones(1)).item())
    if abs(z - 0.5) < 1e-4:                        # sigmoid(0)=0.5
        return 2
    if abs(o - 1.0) < 1e-6 and abs(z) < 1e-6:      # identity
        return 1
    if abs(z) < 1e-6:                              # tanh(0)=0 (and not identity)
        return 0
    raise NotImplementedError(
        "CUDA settling kernel supports tanh/identity/sigmoid only; unrecognised activation.")


@torch.no_grad()
def settle(model, states, clamp_output: bool, T: int, lr_state: float,
           record_energy: bool = False, tol: float | None = None):
    """Fused-CUDA settling. For the common 2-hidden-layer net (model.n == 3) a batch-size hybrid
    of the optimised per-sample (v1) and batch-tiled (v2) kernels is used; for any other depth a
    general per-sample kernel (pcn_settle_so_deep) settles arbitrary L. Same (states, energies,
    steps) contract as _settle_pytorch; fixed T; energies=[]."""
    kernel = build()
    dev = states[0].device
    act = _act_id(model)

    def cu(x):
        return x.to(device=dev, dtype=torch.float32).contiguous()

    if model.n != 3:
        # general-depth path (arbitrary number of layers), per-sample kernel
        s = [cu(states[i]) for i in range(model.n + 1)]
        W = [cu(model.W[i]) for i in range(model.n)]
        b = [cu(model.b[i]) for i in range(model.n)]
        outs = kernel.pcn_settle_so_deep(s, W, b, int(T), float(lr_state),
                                         int(bool(clamp_output)), int(act))
        return [s[0], *outs], [], int(T)

    s = [cu(states[i]) for i in range(4)]
    W = [cu(model.W[i]) for i in range(3)]
    b = [cu(model.b[i]) for i in range(3)]
    B = s[0].shape[0]

    if B <= TILED_THRESHOLD:
        s1o, s2o, s3o = kernel.pcn_settle_so(
            s[0], s[1], s[2], s[3], W[0], W[1], W[2], b[0], b[1], b[2],
            int(T), float(lr_state), int(bool(clamp_output)), int(act))
        return [s[0], s1o, s2o, s3o], [], int(T)

    # v2 tiled: precompute phi(input) on host, pad the batch to a multiple of TB.
    p0 = model.phi(s[0]).contiguous()
    pad = (-B) % TB
    if pad:
        p0 = torch.cat([p0, p0.new_zeros(pad, p0.shape[1])], 0)
        s1p = torch.cat([s[1], s[1].new_zeros(pad, s[1].shape[1])], 0)
        s2p = torch.cat([s[2], s[2].new_zeros(pad, s[2].shape[1])], 0)
        s3p = torch.cat([s[3], s[3].new_zeros(pad, s[3].shape[1])], 0)
    else:
        s1p, s2p, s3p = s[1], s[2], s[3]
    s1o, s2o, s3o = kernel.pcn_settle_so_tiled(
        p0, s1p, s2p, s3p, W[0], W[1], W[2], b[0], b[1], b[2],
        int(T), float(lr_state), int(bool(clamp_output)), int(act))
    return [s[0], s1o[:B], s2o[:B], s3o[:B]], [], int(T)
