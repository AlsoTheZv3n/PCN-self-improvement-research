"""M5-v2 tests: a generative PC (label -> image) actually learns to generate. Hermetic."""
from __future__ import annotations

import torch

from pcn.model import PCN
from pcn.settling import feedforward_init, settle
from pcn.learning import weight_update
from pcn.generative import generate_images, generate_class_grid, _settle_output_masked


def test_generative_training_reduces_image_error():
    """Train a toy generative PCN [3 -> 5 -> 4] to map 3 one-hot labels to 3 fixed targets;
    the forward-generated output must approach the targets (label->image is learnable)."""
    torch.manual_seed(0)
    model = PCN([3, 5, 4], activation="tanh", weight_init="normal", weight_scale=0.5, seed=0)
    labels = torch.eye(3)
    targets = torch.randn(3, 4)

    def gen_err():
        out = feedforward_init(model, labels)[-1]
        return ((out - targets) ** 2).mean().item()

    e0 = gen_err()
    for _ in range(300):
        states = feedforward_init(model, labels)
        states[-1] = targets
        states, _, _ = settle(model, states, clamp_output=True, T=30, lr_state=0.1)
        weight_update(model, states, lr_weight=0.05)
    assert gen_err() < 0.5 * e0


def test_generate_class_grid_shape():
    model = PCN([10, 8, 16], activation="tanh", weight_init="orthogonal", seed=0)
    grid = generate_class_grid(model, n_classes=10)
    assert grid.shape == (10, 16)
    assert torch.isfinite(grid).all()


def test_output_masked_settle_preserves_visible():
    """Generative inpainting: clamped (visible) output pixels stay fixed; free ones change."""
    torch.manual_seed(1)
    model = PCN([3, 6, 8], activation="tanh", weight_init="normal", weight_scale=0.5, seed=1)
    labels = torch.eye(3)
    states = feedforward_init(model, labels)
    img = torch.randn(3, 8)
    states[-1] = img
    mask = torch.zeros(3, 8)
    mask[:, :4] = 1.0  # first 4 output pixels visible/clamped, last 4 free
    out_states, _, _ = _settle_output_masked(model, states, mask, T=40, lr_state=0.1)
    out = out_states[-1]
    assert torch.allclose(out[:, :4], img[:, :4], atol=1e-6)      # visible preserved
    assert (out[:, 4:] - img[:, 4:]).abs().max() > 1e-4           # occluded changed
