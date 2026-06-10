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


@torch.no_grad()
def _ipc_step(model, states, lr_state: float, lr_weight: float, clamp_output: bool = True):
    """One INCREMENTAL-PC step (Salvatori et al. 2024): from the SAME errors, update the free
    states AND the weights. The only difference to the standard path is *when* weights move —
    here at every inference step, not only at the settled equilibrium. Weights are updated in
    place (still purely local Hebbian); the new states are returned.

    Simultaneous update: the state step uses the current weights and the weight step uses the
    pre-step states, both from the one error vector eps (so neither sees the other's change)."""
    n = model.n
    last_free = n - 1 if clamp_output else n
    eps = [None] * (n + 1)
    for i in range(n):
        eps[i + 1] = states[i + 1] - model.predict(i, states[i])
    # state updates use the current weights -> compute into new_states before touching W
    new_states = list(states)
    for k in range(1, last_free + 1):
        grad = eps[k]
        if k < n:
            grad = grad - model.phi_deriv(states[k]) * (eps[k + 1] @ model.W[k])
        new_states[k] = states[k] - lr_state * grad
    # weight updates use the SAME eps and the pre-step states (identical rule to weight_update)
    batch = states[0].shape[0]
    for i in range(n):
        pre = model.phi(states[i])
        model.W[i].add_(lr_weight * (eps[i + 1].t() @ pre / batch))
        model.b[i].add_(lr_weight * eps[i + 1].mean(dim=0))
    return new_states


@torch.no_grad()
def train_epoch_ipc(model, loader, T: int, lr_state: float, lr_weight: float,
                    num_classes: int = 10, track_energy: bool = False) -> dict:
    """One pass, INCREMENTAL PC (``update_variant="ipc"``, docs/04): clamp both ends, then run
    T interleaved (state-step + weight-step) iterations per batch instead of settling fully and
    updating once. Same return contract as ``train_epoch`` ({final_energy, mean_steps}).

    NB iPC applies T weight updates per batch (vs 1 for the standard path), so the EFFECTIVE
    weight learning rate is ~Tx larger — use a correspondingly smaller ``lr_weight`` (~lr/T) or
    it diverges to NaN (measured: standard works at eta_w=0.02, iPC needs ~0.005; docs/12 §4h).

    Always PyTorch backend: the fused CUDA kernel settles WITHOUT touching weights, so it cannot
    express the interleaved iPC update (it would need a weight write every step)."""
    e_sum, n_batches = 0.0, 0
    for x, y in loader:
        x = x.reshape(x.size(0), -1).to(model.device, model.dtype)
        y_oh = torch.nn.functional.one_hot(y, num_classes).to(model.device, model.dtype)
        states = feedforward_init(model, x)
        states[-1] = y_oh  # clamp output to the target
        for _ in range(T):
            states = _ipc_step(model, states, lr_state, lr_weight)
        if track_energy:
            e_sum += energy(model, states)
        n_batches += 1
    return {"final_energy": (e_sum / max(n_batches, 1)) if track_energy else None,
            "mean_steps": float(T)}
