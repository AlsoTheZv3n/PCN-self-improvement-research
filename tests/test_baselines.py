"""Tests for the BP-MLP baseline and benchmark harness (docs/13 M3).

Hermetic: no MNIST download. Fairness check (identical init + forward to the PCN), a toy
BP training-loss-decrease, and a smoke test of the timing harness.
"""
from __future__ import annotations

import torch

from pcn.model import PCN
from pcn.settling import feedforward_init
from pcn.baselines import BPMLPRef
from pcn.benchmark import time_settle


def test_bp_init_and_forward_match_pcn():
    """The BP baseline must start from identical weights and compute the identical forward
    function as the PCN's feedforward pass (the fairness invariant)."""
    sizes = [8, 6, 5, 3]
    pcn = PCN(sizes, activation="tanh", weight_init="orthogonal", seed=0)
    mlp = BPMLPRef(sizes, activation="tanh", weight_init="orthogonal", seed=0)

    for i in range(pcn.n):
        assert torch.allclose(pcn.W[i], mlp.W[i].detach())
        assert torch.allclose(pcn.b[i], mlp.b[i].detach())

    x = torch.randn(4, 8)
    ff_logits = feedforward_init(pcn, x)[-1]   # PCN feedforward output (states[n])
    with torch.no_grad():
        mlp_logits = mlp(x)
    assert torch.allclose(ff_logits, mlp_logits, atol=1e-6)


def test_bp_training_reduces_loss():
    """A few SGD steps on a fixed toy batch must reduce the cross-entropy loss."""
    torch.manual_seed(0)
    mlp = BPMLPRef([8, 6, 5, 3], activation="tanh", weight_init="normal", weight_scale=0.5, seed=0)
    x = torch.randn(16, 8)
    y = torch.randint(0, 3, (16,))
    opt = torch.optim.SGD(mlp.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        loss0 = loss_fn(mlp(x), y).item()
    for _ in range(50):
        opt.zero_grad()
        loss_fn(mlp(x), y).backward()
        opt.step()
    with torch.no_grad():
        loss1 = loss_fn(mlp(x), y).item()
    assert loss1 < loss0


def test_time_settle_returns_positive_times():
    """The harness returns positive, self-consistent timings (ms_per_step == ms_per_settle/T)."""
    model = PCN([8, 6, 5, 3], activation="tanh", weight_init="normal", weight_scale=0.5, seed=0)
    x = torch.randn(4, 8)
    r = time_settle(model, x, T=10, lr_state=0.1, warmup=1, iters=5)
    assert r["ms_per_settle"] > 0.0
    assert r["ms_per_step"] > 0.0
    assert abs(r["ms_per_step"] - r["ms_per_settle"] / 10) < 1e-9
    assert r["backend"] == "pytorch"
