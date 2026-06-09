"""OPTIONAL Phase-3 CUDA backend for the settling loop.

The PyTorch backend (pcn/settling.py) is the default and fully functional path. This
package becomes active only once the fused CUDA settling kernel is built; see
docs/07_cuda_kernel_build.md (build path) and docs/09_kernel_pc_dynamics_deepdive.md
(fusion design + benchmark matrix).
"""
from __future__ import annotations

import os


def is_available() -> bool:
    """True only when the fused CUDA settling kernel should be used (M6).

    Opt-in via the env flag PCN_CUDA_KERNEL=1 AND a CUDA device, so the default/test path
    stays on the PyTorch backend (backend='cuda' raises a clear NotImplementedError) and the
    JIT build is only attempted when explicitly requested. Building also needs nvcc + MSVC
    (a vcvars64 shell on Windows) — see pcn/kernels/settling_cuda.py.
    """
    if os.environ.get("PCN_CUDA_KERNEL") != "1":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
