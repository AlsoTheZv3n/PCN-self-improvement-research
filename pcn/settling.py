"""State-Optimization settling (the PC inference phase) — docs/09.

Per settling step (Jacobi / all layers in parallel):
    eps_k   = s_k - pred_k                              for k = 1..n
    grad_sk = eps_k - phi'(s_k) ⊙ (W_k^T eps_{k+1})     for free layers k < n
            = eps_k                                     for free output layer k = n
    s_k    <- s_k - lr_state * grad_sk

The layer loops are parallel within a step; only the T-step loop is sequential.
This is exactly the structure the optional CUDA kernel (docs/07, docs/09) fuses.
"""
from __future__ import annotations

import torch

from . import kernels


@torch.no_grad()
def feedforward_init(model, x: torch.Tensor):
    """Initialize states via a feedforward pass; copying predictions sets eps_k = 0."""
    states = [x]
    for i in range(model.n):
        states.append(model.predict(i, states[i]))
    return states


@torch.no_grad()
def energy(model, states) -> float:
    """Total PC energy E = 1/2 sum_k ||eps_k||^2 (mean over batch)."""
    e = states[0].new_zeros(())
    for k in range(1, model.n + 1):
        eps = states[k] - model.predict(k - 1, states[k - 1])
        e = e + 0.5 * (eps ** 2).sum(dim=1).mean()
    return float(e)


@torch.no_grad()
def energy_per_sample(model, states):
    """Per-sample residual energy [B] = 1/2 sum_k ||eps_k||^2 — the anomaly score (Hook C,
    docs/13 M5): a settled input the model predicts poorly leaves high residual energy."""
    e = states[0].new_zeros(states[0].shape[0])
    for k in range(1, model.n + 1):
        eps = states[k] - model.predict(k - 1, states[k - 1])
        e = e + 0.5 * (eps ** 2).sum(dim=1)
    return e


def settle(model, states, clamp_output: bool, T: int, lr_state: float,
           backend: str = "pytorch", record_energy: bool = False,
           tol: float | None = None, clamp_input: bool = True,
           input_mask=None):
    """Relax free states to (approximate) equilibrium.

    backend="pytorch" is the default, fully functional path.
    backend="cuda" is the OPTIONAL Phase-3 kernel; it raises until built (docs/07, docs/09).

    T is the maximum number of settling steps. If ``tol`` is given, the loop stops early
    once the relative change in PC energy between consecutive steps drops below ``tol``
    (the convergence criterion that makes ``settling_steps_to_converge`` meaningful, docs/13
    M1). ``tol=None`` reproduces the original fixed-T behaviour exactly.

    Hook C (docs/13 M5): by default the INPUT (states[0]) is clamped. Set ``clamp_input=False``
    to let the input settle too (generation: clamp the label, settle the image), or pass an
    ``input_mask`` ([B,n0] or [n0], 1=clamped/known, 0=free) to settle only some input pixels
    (occlusion/inpainting: keep visible pixels, fill the rest).

    Returns ``(states, energies, steps)``.
    """
    if backend == "pytorch":
        return _settle_pytorch(model, states, clamp_output, T, lr_state, record_energy, tol,
                               clamp_input, input_mask)
    if backend == "cuda":
        if input_mask is not None or not clamp_input:
            raise NotImplementedError(
                "backend='cuda' kernel v1 clamps the input; input-free / masked settling "
                "(Hook C generation/inpainting) needs backend='pytorch'."
            )
        if not kernels.is_available():
            raise NotImplementedError(
                "backend='cuda' requested but the Phase-3 CUDA settling kernel is not built. "
                "Use backend='pytorch' (default). See docs/07_cuda_kernel_build.md and "
                "docs/09_kernel_pc_dynamics_deepdive.md."
            )
        from .kernels import settling_cuda
        return settling_cuda.settle(model, states, clamp_output, T, lr_state,
                                    record_energy, tol)
    raise ValueError(f"unknown backend {backend!r}; use 'pytorch' or 'cuda'")


@torch.no_grad()
def _settle_pytorch(model, states, clamp_output: bool, T: int, lr_state: float,
                    record_energy: bool = False, tol: float | None = None,
                    clamp_input: bool = True, input_mask=None):
    n = model.n
    phi_deriv = model.phi_deriv
    states = [s.clone() for s in states]
    energies: list[float] = []
    last_free = n - 1 if clamp_output else n  # output is clamped in training, free at test
    need_energy = record_energy or tol is not None
    settle_input = (input_mask is not None) or (not clamp_input)  # Hook C (M5)
    prev_e: float | None = None
    steps = 0

    for t in range(T):
        # 1) all errors first (parallel over layers)
        eps = [None] * (n + 1)
        for i in range(n):
            eps[i + 1] = states[i + 1] - model.predict(i, states[i])

        e = None
        if need_energy:
            acc = states[0].new_zeros(())
            for k in range(1, n + 1):
                acc = acc + 0.5 * (eps[k] ** 2).sum(dim=1).mean()
            e = float(acc)
            if record_energy:
                energies.append(e)

        # 2) then all free-state updates (parallel over layers, Jacobi using current eps)
        new_states = list(states)
        # input layer (k=0) — only settles when not (fully) clamped: grad_s0 = -phi'(s0)*(eps1 @ W0)
        if settle_input:
            grad0 = -phi_deriv(states[0]) * (eps[1] @ model.W[0])
            upd0 = states[0] - lr_state * grad0
            new_states[0] = (upd0 if input_mask is None
                             else input_mask * states[0] + (1.0 - input_mask) * upd0)
        for k in range(1, last_free + 1):
            grad = eps[k]
            if k < n:
                # eps[k+1] @ W[k]  ==  W[k]^T eps[k+1]  (row-vector convention)
                grad = grad - phi_deriv(states[k]) * (eps[k + 1] @ model.W[k])
            new_states[k] = states[k] - lr_state * grad
        states = new_states
        steps = t + 1

        # 3) convergence check on the relative energy change (only if tol requested)
        if tol is not None and prev_e is not None:
            if abs(e - prev_e) / (abs(prev_e) + 1e-12) < tol:
                break
        prev_e = e

    return states, energies, steps
