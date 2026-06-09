"""Hook C (M5) demo: one trained PCN does classification, generation, inpainting and anomaly
detection — via the same settling mechanism (docs/13 M5). Saves PNG artifacts + a JSON summary
to results/.

    uv run python scripts/demo_generative.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pcn.api import _mnist_loaders, _resolve_config
from pcn.evaluate import evaluate
from pcn.generate import anomaly_scores, generate, inpaint
from pcn.learning import train_epoch
from pcn.model import PCN

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _auc(pos, neg) -> float:
    """P(score_pos > score_neg) — separability of OOD (pos) from in-dist (neg)."""
    return (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()


def _train(cfg):
    train_loader, _, test_loader = _mnist_loaders(cfg)
    model = PCN([784, *cfg["hidden"], 10], activation=cfg["activation"],
                weight_init=cfg["weight_init"], device=cfg["device"], seed=cfg["seed"])
    for _ in range(int(cfg["epochs"])):
        train_epoch(model, train_loader, cfg["T"], cfg["lr_state"], cfg["lr_weight"], tol=cfg["tol"])
    acc = evaluate(model, test_loader, cfg["T"], cfg["lr_state"])
    return model, test_loader, acc


def _grid(imgs, path, title, rows=1):
    n = imgs.shape[0]
    cols = (n + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows * 1.1))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        axes[i].imshow(imgs[i].reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    cfg = _resolve_config({"hidden": [256, 256], "T": 40, "eta_x": 0.05, "eta_w": 0.05,
                           "epochs": 5, "limit_train": 20000, "tol": None})
    dev = cfg["device"]
    print(f"device: {dev} | training PCN ({cfg['limit_train']} ex, {cfg['epochs']} ep)...")
    model, test_loader, acc = _train(cfg)
    print(f"classification test_acc: {acc:.4f}")

    # (1) Generation: one image per class from a clamped label
    Y = torch.eye(10, device=dev)
    gen = generate(model, Y, T=300, lr_state=0.05).cpu()
    _grid(gen, os.path.join(RESULTS, "m5_generate.png"), "generated per class (0-9)", rows=1)

    # (2) Inpainting: occlude the bottom half of 8 test images, reconstruct (true label clamped)
    xb, yb = next(iter(test_loader))
    xb = xb[:8].reshape(8, 784).to(dev)
    mask = torch.ones(8, 784, device=dev)
    mask[:, 392:] = 0.0                      # bottom half occluded
    x_occ = xb.clone()
    x_occ[:, 392:] = 0.0
    y_oh = torch.zeros(8, 10, device=dev)
    y_oh[torch.arange(8), yb[:8].to(dev)] = 1.0
    recon = inpaint(model, x_occ, mask, T=300, lr_state=0.05, clamp_label=y_oh).cpu()
    panel = torch.cat([xb.cpu(), x_occ.cpu(), recon], dim=0)  # 3 rows of 8: orig / occluded / recon
    _grid(panel, os.path.join(RESULTS, "m5_occlusion.png"),
          "inpainting: original | occluded | reconstructed", rows=3)

    # (3) Anomaly: in-distribution MNIST test vs uniform-noise OOD, via residual energy
    xt, _ = next(iter(test_loader))
    xt = xt.reshape(xt.size(0), -1).to(dev)
    x_ood = torch.rand_like(xt) * 2.0 - 1.0   # uniform noise in the normalized [-1,1] range
    s_in = anomaly_scores(model, xt, T=50, lr_state=0.05)
    s_ood = anomaly_scores(model, x_ood, T=50, lr_state=0.05)
    auc = _auc(s_ood, s_in)
    print(f"anomaly: in-dist energy={s_in.mean():.3f}  OOD energy={s_ood.mean():.3f}  AUC={auc:.3f}")

    summary = {"test_acc": acc, "anomaly_auc_ood_vs_indist": auc,
               "anomaly_energy_indist_mean": float(s_in.mean()),
               "anomaly_energy_ood_mean": float(s_ood.mean()),
               "artifacts": ["m5_generate.png", "m5_occlusion.png"]}
    with open(os.path.join(RESULTS, "m5_demo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: results/m5_generate.png, results/m5_occlusion.png, results/m5_demo_summary.json")


if __name__ == "__main__":
    main()
