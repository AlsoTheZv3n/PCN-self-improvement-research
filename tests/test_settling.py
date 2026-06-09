"""Offline correctness tests for the SO settling math (no MNIST download required).

Mirrors the analytic NumPy validation: (1) settling with clamped output reduces energy
monotonically, (2) a Hebbian weight update at equilibrium further reduces energy, (3) a
short training loop reduces the (output-free) prediction error, (4) the hand-derived state
and weight gradients match autograd, (5) the convergence tolerance stops settling early.

Autograd (torch.autograd.grad) appears ONLY in the gradient-check tests as an external
reference — it is never used in the production PC path (docs/13 M1).
"""
from __future__ import annotations

import torch

from pcn.model import PCN
from pcn.settling import feedforward_init, settle, energy
from pcn.learning import weight_update


def _toy(seed: int = 0):
    torch.manual_seed(seed)
    model = PCN([8, 6, 5, 3], activation="tanh", weight_init="normal",
                weight_scale=0.5, device="cpu", seed=seed)
    x = torch.randn(4, 8)
    y = torch.zeros(4, 3)
    y[torch.arange(4), torch.randint(0, 3, (4,))] = 1.0
    return model, x, y


def test_settling_reduces_energy_monotonically():
    model, x, y = _toy()
    states = feedforward_init(model, x)
    states[-1] = y  # clamp output
    states, energies, _ = settle(model, states, clamp_output=True, T=300, lr_state=0.1,
                                 record_energy=True)
    assert energies[-1] < energies[0]
    assert all(energies[t + 1] <= energies[t] + 1e-6 for t in range(len(energies) - 1))


def test_weight_update_reduces_energy():
    model, x, y = _toy()
    states = feedforward_init(model, x)
    states[-1] = y
    states, _, _ = settle(model, states, clamp_output=True, T=200, lr_state=0.1)
    e_before = energy(model, states)
    weight_update(model, states, lr_weight=0.05)
    e_after = energy(model, states)
    assert e_after < e_before


def test_training_loop_reduces_output_error():
    """Hardened: measure the output-free prediction error BEFORE and AFTER the training
    loop and require a real reduction (the previous version only captured `first` on
    iteration 0, so the assertion was nearly trivial)."""
    model, x, y = _toy()

    def output_err() -> float:
        s = feedforward_init(model, x)
        s, _, _ = settle(model, s, clamp_output=False, T=50, lr_state=0.1)
        return float(((s[-1] - y) ** 2).mean())

    err_before = output_err()
    for _ in range(50):
        states = feedforward_init(model, x)
        states[-1] = y
        states, _, _ = settle(model, states, clamp_output=True, T=50, lr_state=0.1)
        weight_update(model, states, lr_weight=0.05)
    err_after = output_err()

    assert err_after < 0.5 * err_before


def test_settling_gradient_matches_autograd():
    """The hand-derived state gradient in _settle_pytorch
        grad_sk = eps[k] - phi'(s_k) * (eps[k+1] @ W[k])   (eps[n] for the output layer)
    must equal autograd's dE/ds_k for the unnormalised energy E = 0.5 * sum(eps^2)."""
    torch.manual_seed(0)
    for act in ("tanh", "sigmoid"):   # sigmoid added for Song's Fig-4e replication
        model = PCN([4, 3, 2], activation=act, weight_init="normal", weight_scale=0.7, seed=0)
        x = torch.randn(2, 4)
        states = feedforward_init(model, x)
        # perturb the free states so the errors (and the gradient) are non-trivial
        for k in (1, 2):
            states[k] = states[k] + 0.3 * torch.randn_like(states[k])
        n = model.n

        # errors with the current states (matches _settle_pytorch step 1)
        eps = [None] * (n + 1)
        for i in range(n):
            eps[i + 1] = states[i + 1] - model.predict(i, states[i])

        # autograd reference: dE/ds_k for E = 0.5 * sum over all elements of eps^2
        s = [t.clone().requires_grad_(True) for t in states]
        with torch.enable_grad():
            E = s[0].new_zeros(())
            for i in range(n):
                pred = model.phi(s[i]) @ model.W[i].t() + model.b[i]
                E = E + 0.5 * ((s[i + 1] - pred) ** 2).sum()
            grads = torch.autograd.grad(E, [s[k] for k in range(1, n + 1)])

        for idx, k in enumerate(range(1, n + 1)):
            manual = eps[k].clone()
            if k < n:
                manual = manual - model.phi_deriv(states[k]) * (eps[k + 1] @ model.W[k])
            assert torch.allclose(grads[idx], manual, atol=1e-5), f"{act} state grad layer {k}"


def test_weight_gradient_matches_autograd():
    """The local Hebbian update dW = eps.t() @ phi(pre) / batch must equal gradient DESCENT
    on the unnormalised energy, i.e. -(dE/dW)/batch (autograd as external reference)."""
    torch.manual_seed(1)
    model = PCN([4, 3, 2], activation="tanh", weight_init="normal", weight_scale=0.7, seed=1)
    x = torch.randn(5, 4)
    states = feedforward_init(model, x)
    for k in (1, 2):
        states[k] = states[k] + 0.3 * torch.randn_like(states[k])
    n = model.n
    batch = x.shape[0]

    W = [w.clone().requires_grad_(True) for w in model.W]
    b = [bb.clone().requires_grad_(True) for bb in model.b]
    with torch.enable_grad():
        E = states[0].new_zeros(())
        for i in range(n):
            pred = model.phi(states[i]) @ W[i].t() + b[i]
            E = E + 0.5 * ((states[i + 1] - pred) ** 2).sum()
        gW = torch.autograd.grad(E, W)

    for i in range(n):
        eps = states[i + 1] - model.predict(i, states[i])
        pre = model.phi(states[i])
        manual_dW = eps.t() @ pre / batch
        assert torch.allclose(manual_dW, -gW[i] / batch, atol=1e-5), f"weight grad mismatch at W[{i}]"


def test_settling_converges_with_tol():
    """With a tolerance, settling stops before max_T and reports the actual step count;
    without it, the loop runs the full fixed T (docs/13 M1)."""
    model, x, y = _toy()
    s1 = feedforward_init(model, x)
    s1[-1] = y
    _, _, steps_full = settle(model, s1, clamp_output=True, T=500, lr_state=0.1)
    s2 = feedforward_init(model, x)
    s2[-1] = y
    _, _, steps_tol = settle(model, s2, clamp_output=True, T=500, lr_state=0.1, tol=1e-4)
    assert steps_full == 500           # tol=None -> fixed T
    assert steps_tol < steps_full      # tol stops early once energy plateaus


def test_cuda_backend_raises_until_built():
    import pytest
    model, x, y = _toy()
    states = feedforward_init(model, x)
    with pytest.raises(NotImplementedError):
        settle(model, states, clamp_output=False, T=5, lr_state=0.1, backend="cuda")
