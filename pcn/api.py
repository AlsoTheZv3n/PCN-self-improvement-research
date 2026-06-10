"""The single Phase-4 interface: train_and_eval(config) -> metrics (docs/02, docs/04).

The search loop (docs/04) and the bio-regime experiments (docs/10) call ONLY this function,
so they remain a thin layer rather than a rewrite. Keep this signature stable.
"""
from __future__ import annotations

import time
import warnings

import torch

from .evaluate import evaluate, noise_robustness
from .learning import train_epoch, train_epoch_ipc
from .model import PCN
from .settling import feedforward_init, settle

DEFAULT_CONFIG = {
    "hidden": [256, 256],      # hidden layer sizes (input=784, output=10 are implicit)
    "T": 20,                   # MAX settling steps (floor T >= L, Qi et al. 2025 arXiv:2506.23800;
                               # upper bound is signal-decay dependent, NOT a fixed 2L — docs/13)
    "lr_state": 0.1,           # state learning rate (lambda / eta_x)
    "lr_weight": 0.01,         # weight learning rate (eta / eta_w); 1e-3 underfits on MNIST (docs/12)
    "activation": "tanh",
    "weight_init": "orthogonal",
    "precision_schedule": "isotropic",  # Pi=I default; "spiking"/"decaying" reserved for M2 (docs/13)
    "update_variant": "standard",        # vs. "ipc" (reserved for M7, docs/04)
    "tol": None,               # settling convergence tol (relative energy delta); None = fixed-T
    "eval_noise": True,        # compute noise_robustness in train_and_eval (off speeds up LR sweeps)
    "val_split": 0.0,          # fraction of train held out as validation for fair LR selection (M4 hardening)
    "bp_loss": "ce",           # BP baseline loss: "ce" (CrossEntropy) or "mse" (to one-hot, matches PC's objective) — M4 B1
    "epochs": 5,
    "batch_size": 64,
    "seed": 0,
    "backend": "pytorch",      # "cuda" is the OPTIONAL Phase-3 path (docs/07, docs/09)
    "data_root": "./data",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "limit_train": None,       # e.g. 1000 for the small-data regime (docs/10)
}

# Documented spec names (docs/02) -> internal config names. Spec names take priority.
_CONFIG_ALIASES = {"eta_x": "lr_state", "eta_w": "lr_weight"}


def _resolve_config(config: dict | None) -> dict:
    """Merge a user config onto DEFAULT_CONFIG, honouring the docs/02 aliases and warning
    on unknown keys instead of silently ignoring them (docs/13 M1)."""
    cfg = dict(DEFAULT_CONFIG)
    for key, value in (config or {}).items():
        canonical = _CONFIG_ALIASES.get(key, key)
        if canonical not in DEFAULT_CONFIG:
            warnings.warn(
                f"unknown config key {key!r}; ignored "
                f"(known keys: {sorted(DEFAULT_CONFIG)})",
                stacklevel=2,
            )
            continue
        cfg[canonical] = value

    # Reserved features fail loudly instead of silently no-op'ing (same philosophy as the
    # CUDA backend stub): only the implemented values work until their milestone lands.
    if cfg["precision_schedule"] != "isotropic":
        raise NotImplementedError(
            f"precision_schedule={cfg['precision_schedule']!r} is reserved for M2 and not yet "
            "implemented — only 'isotropic' (Pi=I) works today. See docs/13 M2 and docs/01."
        )
    if cfg["update_variant"] not in ("standard", "ipc"):
        raise NotImplementedError(
            f"update_variant={cfg['update_variant']!r} unknown — only 'standard' (settle then "
            "update once) and 'ipc' (incremental PC: update weights every inference step, "
            "Salvatori et al. 2024) are implemented. See docs/04."
        )
    return cfg


def _mnist_loaders(cfg):
    """Return (train_loader, val_loader, test_loader). ``val_loader`` is None unless
    ``val_split`` > 0, in which case the first ``val_split`` fraction of the (optionally
    limited) train set is held out for fair LR selection (docs/13 M4 hardening)."""
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # same normalization as the PC benchmarks
    ])
    train_full = datasets.MNIST(cfg["data_root"], train=True, download=True, transform=tf)
    test = datasets.MNIST(cfg["data_root"], train=False, download=True, transform=tf)

    idx = list(range(len(train_full)))
    if cfg.get("limit_train"):
        idx = idx[:int(cfg["limit_train"])]

    val_loader = None
    val_split = float(cfg.get("val_split") or 0.0)
    if val_split > 0.0:
        n_val = max(1, int(len(idx) * val_split))
        val_ds = torch.utils.data.Subset(train_full, idx[:n_val])
        train_ds = torch.utils.data.Subset(train_full, idx[n_val:])
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False)
    elif cfg.get("limit_train"):
        train_ds = torch.utils.data.Subset(train_full, idx)
    else:
        train_ds = train_full

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, val_loader, test_loader


def _measure_settling_steps(model, train_loader, cfg) -> float:
    """Settling steps to reach the convergence tolerance on one training batch.

    Measured with a GENEROUS cap (5*T, min 100) — not the training T — so the value is the
    true number of steps the dynamics need to plateau, independent of the configured T. This
    makes the metric a real diagnostic: if it exceeds the training T, the network is
    under-settling (the T-vs-depth question of docs/09 / Qi et al. 2025). Uses a reference
    tol of 1e-3 when the config's tol is None. (docs/13 M1)
    """
    ref_tol = cfg["tol"] if cfg["tol"] is not None else 1e-3
    max_steps = max(int(cfg["T"]) * 5, 100)
    x, y = next(iter(train_loader))
    x = x.reshape(x.size(0), -1).to(model.device, model.dtype)
    y_oh = torch.nn.functional.one_hot(y, 10).to(model.device, model.dtype)
    states = feedforward_init(model, x)
    states[-1] = y_oh  # clamp output, like training
    _, _, steps = settle(model, states, clamp_output=True, T=max_steps,
                         lr_state=cfg["lr_state"], backend=cfg["backend"], tol=ref_tol)
    return float(steps)


def train_and_eval(config: dict | None = None) -> dict:
    """Train a PCN on MNIST and return metrics — the docs/02 interface contract.

    Config keys mirror DEFAULT_CONFIG; the documented aliases ``eta_x``/``eta_w`` map to
    ``lr_state``/``lr_weight``. Unknown keys raise a warning and are ignored.

    Returns a dict with: ``test_acc``, ``energy_curve`` (per-epoch mean equilibrium energy),
    ``noise_robustness`` (accuracy vs. input-noise sigma), ``train_time_s``,
    ``settling_steps_to_converge``, and the resolved ``config``.
    """
    cfg = _resolve_config(config)
    torch.manual_seed(int(cfg["seed"]))

    train_loader, val_loader, test_loader = _mnist_loaders(cfg)
    sizes = [28 * 28, *cfg["hidden"], 10]
    model = PCN(
        sizes,
        activation=cfg["activation"],
        weight_init=cfg["weight_init"],
        device=cfg["device"],
        seed=int(cfg["seed"]),
    )

    energy_curve: list[float] = []
    ipc = cfg["update_variant"] == "ipc"
    t0 = time.time()
    for _ in range(int(cfg["epochs"])):
        if ipc:   # incremental PC: weights move every inference step (PyTorch backend only)
            info = train_epoch_ipc(model, train_loader, cfg["T"], cfg["lr_state"],
                                   cfg["lr_weight"], track_energy=True)
        else:
            info = train_epoch(model, train_loader, cfg["T"], cfg["lr_state"], cfg["lr_weight"],
                               backend=cfg["backend"], tol=cfg["tol"], track_energy=True)
        energy_curve.append(info["final_energy"])
    train_time = time.time() - t0

    acc = evaluate(model, test_loader, cfg["T"], cfg["lr_state"], backend=cfg["backend"])
    val_acc = (evaluate(model, val_loader, cfg["T"], cfg["lr_state"], backend=cfg["backend"])
               if val_loader is not None else None)
    nr = (noise_robustness(model, test_loader, cfg["T"], cfg["lr_state"], backend=cfg["backend"])
          if cfg["eval_noise"] else {})
    steps = _measure_settling_steps(model, train_loader, cfg)

    return {
        "test_acc": acc,
        "val_acc": val_acc,
        "energy_curve": energy_curve,
        "noise_robustness": nr,
        "train_time_s": train_time,
        "settling_steps_to_converge": steps,
        "config": cfg,
    }
