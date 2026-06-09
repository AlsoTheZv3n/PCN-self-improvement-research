"""Hook C (M5) tests: input-free / masked settling, generation, inpainting, anomaly energy.
Hermetic — toy tensors, no MNIST."""
from __future__ import annotations

import torch

from pcn.model import PCN
from pcn.settling import feedforward_init, settle, energy_per_sample
from pcn.generate import generate, inpaint, anomaly_scores


def _toy(seed: int = 0):
    torch.manual_seed(seed)
    return PCN([8, 6, 5, 3], activation="tanh", weight_init="normal", weight_scale=0.5, seed=seed)


def test_input_free_settle_reduces_energy_monotonically():
    """Generation-style settle (input free, output clamped) must reduce energy monotonically —
    the input-update gradient sign (grad_s0 = -phi'(s0)*(eps1@W0)) is correct."""
    model = _toy()
    x = torch.randn(4, 8)
    states = feedforward_init(model, x)
    y = torch.zeros(4, 3)
    y[torch.arange(4), torch.randint(0, 3, (4,))] = 1.0
    states[-1] = y
    states, energies, _ = settle(model, states, clamp_output=True, T=150, lr_state=0.1,
                                 clamp_input=False, record_energy=True)
    assert energies[-1] < energies[0]
    assert all(energies[t + 1] <= energies[t] + 1e-6 for t in range(len(energies) - 1))


def test_generate_returns_input_shape():
    model = _toy()
    y = torch.zeros(3, 3)
    y[torch.arange(3), torch.tensor([0, 1, 2])] = 1.0
    img = generate(model, y, T=30, lr_state=0.1)
    assert img.shape == (3, 8)
    assert torch.isfinite(img).all()


def test_inpaint_preserves_visible_and_fills_occluded():
    model = _toy()
    x = torch.randn(4, 8)
    mask = torch.zeros(4, 8)
    mask[:, :4] = 1.0  # first 4 pixels visible (clamped), last 4 occluded (free)
    out = inpaint(model, x, mask, T=50, lr_state=0.1)
    assert out.shape == x.shape
    # visible pixels are preserved exactly
    assert torch.allclose(out[:, :4], x[:, :4], atol=1e-6)
    # occluded pixels were updated (changed from their initial value)
    assert (out[:, 4:] - x[:, 4:]).abs().max() > 1e-4


def test_anomaly_scores_per_sample_shape():
    model = _toy()
    x = torch.randn(5, 8)
    s = anomaly_scores(model, x, T=30, lr_state=0.1)
    assert s.shape == (5,)
    assert (s >= 0).all()  # energy is a sum of squares


def test_energy_per_sample_matches_total():
    model = _toy()
    x = torch.randn(4, 8)
    states = feedforward_init(model, x)
    per = energy_per_sample(model, states)
    assert per.shape == (4,)
    # mean of per-sample equals the scalar energy() (which means over batch)
    from pcn.settling import energy
    assert abs(per.mean().item() - energy(model, states)) < 1e-5
