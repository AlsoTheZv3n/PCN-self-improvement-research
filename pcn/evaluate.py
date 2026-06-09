"""Evaluation: classification accuracy (output free) and a noise-robustness sweep.

At test time only the input is clamped; hidden AND output states settle, then we read
argmax of the settled output state (docs/01, docs/03).
"""
from __future__ import annotations

import torch

from .settling import feedforward_init, settle


@torch.no_grad()
def evaluate(model, loader, T: int, lr_state: float, backend: str = "pytorch") -> float:
    correct = total = 0
    for x, y in loader:
        x = x.reshape(x.size(0), -1).to(model.device, model.dtype)
        states = feedforward_init(model, x)
        states, _, _ = settle(model, states, clamp_output=False, T=T, lr_state=lr_state,
                              backend=backend)
        pred = states[-1].argmax(dim=1).cpu()
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def noise_robustness(model, loader, T: int, lr_state: float,
                     sigmas=(0.0, 0.25, 0.5, 1.0), backend: str = "pytorch") -> dict:
    """Accuracy under additive Gaussian input noise — the PC-vs-BP robustness probe (docs/10)."""
    out: dict[float, float] = {}
    for sigma in sigmas:
        correct = total = 0
        for x, y in loader:
            x = x.reshape(x.size(0), -1).to(model.device, model.dtype)
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)
            states = feedforward_init(model, x)
            states, _, _ = settle(model, states, clamp_output=False, T=T, lr_state=lr_state,
                                  backend=backend)
            pred = states[-1].argmax(dim=1).cpu()
            correct += int((pred == y).sum())
            total += int(y.numel())
        out[float(sigma)] = correct / max(total, 1)
    return out
