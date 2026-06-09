"""M5-v2: a GENERATIVE PC — label -> image (docs/12 §4e follow-up).

The discriminative PCN (input=image -> output=label) is no generative model, so clamping a
label and settling the image yields noise (docs/12 §4e). Here we flip the orientation: a PCN
with `layer_sizes = [10, *hidden, 784]` maps a one-hot LABEL (bottom) to an IMAGE (top). It is
trained GENERATIVELY — clamp the label at the input AND the image at the output, settle the
hidden states, local Hebbian weight update — so it learns label -> image. Generation is then
just the forward pass (label -> image); inpainting/anomaly use the same clamp-variant settling.

This reuses the entire core unchanged (PCN, settle, weight_update); only the data roles swap
(input := one-hot label, output := image). MSE-to-image training learns roughly the per-class
prototype, so generated digits are recognisable (blurry), not noise.
"""
from __future__ import annotations

import time

import torch

from .api import _mnist_loaders, _resolve_config
from .learning import weight_update
from .model import PCN
from .settling import energy_per_sample, feedforward_init, settle


def _onehot(y, k, device, dtype):
    return torch.nn.functional.one_hot(y, k).to(device, dtype)


def train_generative(config: dict | None = None):
    """Train a generative PCN [10, *hidden, 784] on MNIST (label->image). Returns (model, cfg).

    Clamps the one-hot label at the input and the image at the output, settles the hidden
    states, applies the local Hebbian update — the generative analogue of `train_epoch`.
    """
    cfg = _resolve_config(config)
    torch.manual_seed(int(cfg["seed"]))
    device, dtype = cfg["device"], torch.float32
    train_loader, _, _ = _mnist_loaders(cfg)
    model = PCN([10, *cfg["hidden"], 28 * 28], activation=cfg["activation"],
                weight_init=cfg["weight_init"], device=device, seed=int(cfg["seed"]))

    t0 = time.time()
    for _ in range(int(cfg["epochs"])):
        for x, y in train_loader:
            img = x.reshape(x.size(0), -1).to(device, dtype)
            label = _onehot(y, 10, device, dtype)
            states = feedforward_init(model, label)
            states[-1] = img                       # clamp the image at the output
            states, _, _ = settle(model, states, clamp_output=True, T=cfg["T"],
                                  lr_state=cfg["lr_state"], tol=cfg["tol"])
            weight_update(model, states, cfg["lr_weight"])
    cfg = {**cfg, "train_time_s": time.time() - t0}
    return model, cfg


@torch.no_grad()
def generate_images(model, labels_onehot):
    """Generate images from one-hot labels — the forward pass label->image. Returns [B, 784]."""
    return feedforward_init(model, labels_onehot.to(model.device, model.dtype))[-1]


@torch.no_grad()
def generate_class_grid(model, n_classes: int = 10):
    """One generated image per class (identity one-hot matrix in). Returns [n_classes, 784]."""
    y = torch.eye(n_classes, device=model.device, dtype=model.dtype)
    return generate_images(model, y)


@torch.no_grad()
def inpaint_generative(model, labels_onehot, x_partial, visible_mask, T: int = 100,
                       lr_state: float = 0.1):
    """Generative inpainting: clamp the label at the input and the VISIBLE image pixels at the
    output (mask==1), settle the occluded output pixels (mask==0) + hidden. Returns the image."""
    label = labels_onehot.to(model.device, model.dtype)
    img = x_partial.to(model.device, model.dtype)
    mask = visible_mask.to(model.device, model.dtype)
    states = feedforward_init(model, label)
    states[-1] = img                                # start from the (partial) image at output
    # output is partially clamped: free where mask==0. We run a manual settle on hidden+output.
    states, _, _ = _settle_output_masked(model, states, mask, T, lr_state)
    return states[-1]


@torch.no_grad()
def anomaly_scores_generative(model, x, T: int = 100, lr_state: float = 0.1):
    """Generative anomaly: clamp the image at the output, settle the label (input) + hidden,
    residual energy = how poorly the generative model explains the image. Higher = anomalous."""
    img = x.to(model.device, model.dtype)
    B = img.shape[0]
    states = [torch.zeros(B, sz, device=model.device, dtype=model.dtype)
              for sz in model.layer_sizes]
    states[-1] = img
    states, _, _ = settle(model, states, clamp_output=True, T=T, lr_state=lr_state,
                          clamp_input=False)               # label (input) + hidden settle
    return energy_per_sample(model, states)


@torch.no_grad()
def _settle_output_masked(model, states, output_mask, T, lr_state):
    """Settle with the OUTPUT partially clamped (output_mask==1 fixed). Mirrors the input-mask
    path for the generative orientation (image at the output). Hidden always free; input free."""
    n = model.n
    phi_deriv = model.phi_deriv
    states = [s.clone() for s in states]
    out_fixed = states[n].clone()
    for _ in range(T):
        eps = [None] * (n + 1)
        for i in range(n):
            eps[i + 1] = states[i + 1] - model.predict(i, states[i])
        new_states = list(states)
        # input (label latent) free
        new_states[0] = states[0] - lr_state * (-phi_deriv(states[0]) * (eps[1] @ model.W[0]))
        for k in range(1, n):  # hidden free
            grad = eps[k] - phi_deriv(states[k]) * (eps[k + 1] @ model.W[k])
            new_states[k] = states[k] - lr_state * grad
        # output: free where mask==0, clamped (to the visible pixels) where mask==1
        upd_out = states[n] - lr_state * eps[n]
        new_states[n] = output_mask * out_fixed + (1.0 - output_mask) * upd_out
        states = new_states
    return states, [], T
