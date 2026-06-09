"""Local (Hebbian) weight update and training epoch — docs/01, docs/09.

Weight update at equilibrium (purely local, no global backward pass):
    grad_Wi = -eps_{i+1} @ phi(s_i)^T   ->   W_i <- W_i + lr_weight * eps_{i+1} phi(s_i)^T
    b_i    <- b_i + lr_weight * mean(eps_{i+1})
"""
from __future__ import annotations

import torch

from .settling import energy, feedforward_init, settle


@torch.no_grad()
def weight_update(model, states, lr_weight: float) -> None:
    n = model.n
    batch = states[0].shape[0]
    for i in range(n):
        eps = states[i + 1] - model.predict(i, states[i])   # [B, size_{i+1}]
        pre = model.phi(states[i])                          # [B, size_i]
        dW = eps.t() @ pre / batch                          # [size_{i+1}, size_i]
        db = eps.mean(dim=0)
        model.W[i].add_(lr_weight * dW)
        model.b[i].add_(lr_weight * db)


@torch.no_grad()
def train_epoch(model, loader, T: int, lr_state: float, lr_weight: float,
                num_classes: int = 10, backend: str = "pytorch",
                tol: float | None = None, track_energy: bool = False) -> dict:
    """One pass over the data: clamp both ends, settle hidden states, update weights.

    Returns a small metrics dict so the caller can build the per-epoch energy curve and the
    mean settling-step count (docs/13 M1):
        {"final_energy": <mean equilibrium energy over batches | None>,
         "mean_steps":   <mean settling steps over batches>}
    ``track_energy`` adds one extra forward per batch (the equilibrium energy, measured
    before the weight update); leave it off when the curve is not needed.
    """
    e_sum = 0.0
    steps_sum = 0
    n_batches = 0
    for x, y in loader:
        x = x.reshape(x.size(0), -1).to(model.device, model.dtype)
        y_oh = torch.nn.functional.one_hot(y, num_classes).to(model.device, model.dtype)
        states = feedforward_init(model, x)
        states[-1] = y_oh  # clamp output to the target
        states, _, steps = settle(model, states, clamp_output=True, T=T, lr_state=lr_state,
                                  backend=backend, tol=tol)
        if track_energy:
            e_sum += energy(model, states)  # equilibrium energy, pre weight-update
        weight_update(model, states, lr_weight)
        steps_sum += steps
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {
        "final_energy": (e_sum / n_batches) if track_energy else None,
        "mean_steps": steps_sum / n_batches,
    }
