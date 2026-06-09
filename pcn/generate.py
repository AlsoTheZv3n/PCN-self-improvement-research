"""Hook C (docs/13 M5): generative / inpainting / anomaly via PC settling.

The PCN's distinctive trait over a feedforward BP net: the SAME weights and the SAME settling
mechanism do four jobs — only WHAT is clamped changes:
  - classify : clamp input,            settle output        (pcn.evaluate.evaluate)
  - generate : clamp label,            settle input+hidden  (generate)
  - inpaint  : clamp visible pixels,   settle occluded ones (inpaint)
  - anomaly  : clamp input+pred-label, settle, read energy  (anomaly_scores)

NOTE on initialisation (important for this architecture): `feedforward_init` drives every
prediction error to zero (each layer exactly predicts the next), i.e. a TRIVIAL zero-energy
equilibrium where nothing settles. So generation/inpainting init the FREE states to zero —
this creates a driving error that the settling then relaxes. Likewise a plain "residual
energy after settling all non-input states" is ~0 for any clamped input (the free states can
always relax to the feedforward solution), so the anomaly score clamps the predicted label
too, forcing the network to reconcile the input with a definite class.
"""
from __future__ import annotations

import torch

from .settling import energy_per_sample, feedforward_init, settle


def _zeros_like_layers(model, B):
    return [torch.zeros(B, sz, device=model.device, dtype=model.dtype)
            for sz in model.layer_sizes]


@torch.no_grad()
def generate(model, y_onehot, T: int = 200, lr_state: float = 0.1):
    """Clamp a one-hot label at the output and settle the input (+hidden, both zero-init) into
    a generated image. Returns the settled input [B, n0]."""
    y = y_onehot.to(model.device, model.dtype)
    states = _zeros_like_layers(model, y.shape[0])
    states[-1] = y
    states, _, _ = settle(model, states, clamp_output=True, T=T, lr_state=lr_state,
                          clamp_input=False)
    return states[0]


@torch.no_grad()
def inpaint(model, x, visible_mask, T: int = 200, lr_state: float = 0.1, clamp_label=None):
    """Reconstruct occluded pixels: clamp the visible pixels (mask==1), settle the occluded
    ones (mask==0). Hidden is zero-init so the visible pixels actually drive the fill-in;
    optionally clamp a known label at the output. Returns the settled input."""
    x = x.to(model.device, model.dtype)
    mask = visible_mask.to(model.device, model.dtype)
    states = _zeros_like_layers(model, x.shape[0])
    states[0] = x.clone()  # visible pixels meaningful; occluded ones are free (settle)
    clamp_out = clamp_label is not None
    if clamp_out:
        states[-1] = clamp_label.to(model.device, model.dtype)
    states, _, _ = settle(model, states, clamp_output=clamp_out, T=T, lr_state=lr_state,
                          clamp_input=True, input_mask=mask)
    return states[0]


@torch.no_grad()
def anomaly_scores(model, x, T: int = 50, lr_state: float = 0.1):
    """Per-sample anomaly score [B]: clamp the input AND the model's own predicted label, settle
    the hidden states, and read the residual PC energy. In-distribution inputs reconcile with
    a class at low energy; OOD inputs cannot, leaving high energy. Higher = more anomalous."""
    x = x.to(model.device, model.dtype)
    pred = feedforward_init(model, x)[-1].argmax(dim=1)
    y = torch.zeros(x.shape[0], model.layer_sizes[-1], device=model.device, dtype=model.dtype)
    y[torch.arange(x.shape[0]), pred] = 1.0
    states = _zeros_like_layers(model, x.shape[0])
    states[0] = x
    states[-1] = y
    states, _, _ = settle(model, states, clamp_output=True, T=T, lr_state=lr_state)
    return energy_per_sample(model, states)
