"""Backprop MLP reference — the fair PC-vs-BP baseline (docs/13 M3, docs/10).

Fairness rule (Song et al. 2024): identical architecture, initialisation and data pipeline;
the ONLY independent variable is the learning rule. So the MLP here

  * reuses the SAME weights as `pcn.model.PCN` at the same seed (it builds a PCN and clones
    its W/b into nn.Parameters — guaranteed-identical init, no duplicated logic), and
  * uses the SAME forward function as `pcn.settling.feedforward_init`
    (``h <- phi(h) @ W[i].T + b[i]`` chained over layers, phi=tanh),
  * trains over the SAME `_mnist_loaders` pipeline as `train_and_eval`.

Unlike the PC path this DOES use autograd (`loss.backward()`) — that is the point: it is the
backprop baseline, not the PC learner. `train_and_eval_bp` returns the same dict schema as
`pcn.api.train_and_eval` so the search loop (M7) and comparison harness treat them uniformly;
for BP, ``energy_curve`` holds the per-epoch cross-entropy loss and
``settling_steps_to_converge`` is 0 (BP has no settling phase).
"""
from __future__ import annotations

import time

import torch
from torch import nn

from .api import _mnist_loaders, _resolve_config
from .model import PCN


def bp_loss_fn(kind: str = "ce", num_classes: int = 10):
    """A loss callable ``loss(logits, y)`` for the BP arm (M4 B1, docs/12).

    "ce"  = CrossEntropy on logits (the practical BP baseline).
    "mse" = squared error to the one-hot target — the SAME objective the PC arm clamps
            (`states[-1] = y_oh`), giving the like-for-like comparison that isolates the
            learning rule from the loss function.
    """
    if kind == "ce":
        ce = nn.CrossEntropyLoss()
        return lambda logits, y: ce(logits, y)
    if kind == "mse":
        mse = nn.MSELoss()
        return lambda logits, y: mse(
            logits, torch.nn.functional.one_hot(y, num_classes).to(logits.dtype))
    raise ValueError(f"bp_loss must be 'ce' or 'mse', got {kind!r}")


class BPMLPRef(nn.Module):
    """An MLP whose function and initial weights match the PCN exactly (docs/13 M3)."""

    def __init__(self, layer_sizes, activation: str = "tanh",
                 weight_init: str = "orthogonal", weight_scale: float = 0.5,
                 device: str = "cpu", dtype: torch.dtype = torch.float32, seed: int = 0):
        super().__init__()
        ref = PCN(layer_sizes, activation=activation, weight_init=weight_init,
                  weight_scale=weight_scale, device=device, dtype=dtype, seed=seed)
        self.n = ref.n
        self.phi = ref.phi  # same activation callable as the PCN (tanh / identity)
        self.dtype = dtype
        self.device = device
        # Clone the PCN's init so BP and PC provably start from identical weights.
        self.W = nn.ParameterList([nn.Parameter(w.clone()) for w in ref.W])
        self.b = nn.ParameterList([nn.Parameter(b.clone()) for b in ref.b])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits via the same chain as pcn.settling.feedforward_init."""
        h = x
        for i in range(self.n):
            h = self.phi(h) @ self.W[i].t() + self.b[i]
        return h


@torch.no_grad()
def _eval_acc(model: BPMLPRef, loader, device, dtype) -> float:
    """Classification accuracy of the BP-MLP over ``loader`` (used for test and validation)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.reshape(x.size(0), -1).to(device, dtype)
        correct += int((model(x).argmax(dim=1).cpu() == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def _noise_robustness_bp(model: BPMLPRef, loader, device, dtype,
                         sigmas=(0.0, 0.25, 0.5, 1.0)) -> dict:
    """Accuracy under additive Gaussian input noise — mirror of evaluate.noise_robustness."""
    model.eval()
    out: dict[float, float] = {}
    for sigma in sigmas:
        correct = total = 0
        for x, y in loader:
            x = x.reshape(x.size(0), -1).to(device, dtype)
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)
            pred = model(x).argmax(dim=1).cpu()
            correct += int((pred == y).sum())
            total += int(y.numel())
        out[float(sigma)] = correct / max(total, 1)
    return out


def train_and_eval_bp(config: dict | None = None) -> dict:
    """Train the matched BP-MLP baseline on MNIST; return the train_and_eval dict schema.

    Uses `lr_weight` as the SGD learning rate (tune it independently from the PC rate — the
    fair protocol tunes each method's LR separately, Song et al. 2024 / van Zwol et al.).
    """
    cfg = _resolve_config(config)
    torch.manual_seed(int(cfg["seed"]))
    device, dtype = cfg["device"], torch.float32

    train_loader, val_loader, test_loader = _mnist_loaders(cfg)
    sizes = [28 * 28, *cfg["hidden"], 10]
    model = BPMLPRef(sizes, activation=cfg["activation"], weight_init=cfg["weight_init"],
                     device=device, dtype=dtype, seed=int(cfg["seed"])).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=float(cfg["lr_weight"]))
    # Loss choice (M4 B1, docs/12): "ce" = CrossEntropy on logits (the practical BP baseline);
    # "mse" = squared error to the SAME one-hot target the PC arm clamps -> the like-for-like
    # comparison that isolates the learning rule from the loss function.
    loss_fn = bp_loss_fn(cfg["bp_loss"])

    loss_curve: list[float] = []
    t0 = time.time()
    for _ in range(int(cfg["epochs"])):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.reshape(x.size(0), -1).to(device, dtype)
            y = y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        loss_curve.append(running / max(n_batches, 1))
    train_time = time.time() - t0

    acc = _eval_acc(model, test_loader, device, dtype)
    val_acc = _eval_acc(model, val_loader, device, dtype) if val_loader is not None else None
    nr = _noise_robustness_bp(model, test_loader, device, dtype) if cfg["eval_noise"] else {}

    return {
        "test_acc": acc,
        "val_acc": val_acc,
        "energy_curve": loss_curve,        # for BP: per-epoch cross-entropy loss (no PC energy)
        "noise_robustness": nr,
        "train_time_s": train_time,
        "settling_steps_to_converge": 0.0,  # BP has no settling phase
        "method": "bp",
        "config": cfg,
    }
