"""M5-v2 demo: a GENERATIVE PC (label -> image) generates / inpaints / scores anomalies —
the capabilities that produced noise on the discriminative PCN (docs/12 §4e). Saves PNGs +
JSON to results/.

    uv run python scripts/demo_generative_v2.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pcn.api import _mnist_loaders, _resolve_config
from pcn.generative import (anomaly_scores_generative, generate_class_grid,
                            inpaint_generative, train_generative)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _auc(pos, neg):
    return (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()


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
    # NB: the 784-dim output makes the hidden settling gradient ~sqrt(784) larger than the
    # 10-dim discriminative case, so eta_x must be ~10x smaller for stability (docs/12 §4f).
    cfg = _resolve_config({"hidden": [256, 256], "T": 30, "eta_x": 0.01, "eta_w": 0.02,
                           "epochs": 10, "limit_train": 30000, "tol": None})
    dev = cfg["device"]
    print(f"device: {dev} | training GENERATIVE PCN [10,256,256,784] "
          f"({cfg['limit_train']} ex, {cfg['epochs']} ep)...")
    model, info = train_generative(cfg)
    print(f"trained in {info['train_time_s']:.0f}s")

    # (1) Generation: one prototype image per class via the forward pass label->image
    gen = generate_class_grid(model, 10).cpu()
    _grid(gen, os.path.join(RESULTS, "m5v2_generate.png"), "GENERATIVE PC: generated per class (0-9)")

    # (2) Inpainting: occlude bottom half of 8 test images, clamp label+visible, settle the rest
    _, _, test_loader = _mnist_loaders(cfg)
    xb, yb = next(iter(test_loader))
    xb = xb[:8].reshape(8, 784).to(dev)
    mask = torch.ones(8, 784, device=dev)
    mask[:, 392:] = 0.0
    x_occ = xb.clone()
    x_occ[:, 392:] = 0.0
    y_oh = torch.zeros(8, 10, device=dev)
    y_oh[torch.arange(8), yb[:8].to(dev)] = 1.0
    recon = inpaint_generative(model, y_oh, x_occ, mask, T=200, lr_state=0.01).cpu()
    panel = torch.cat([xb.cpu(), x_occ.cpu(), recon], dim=0)
    _grid(panel, os.path.join(RESULTS, "m5v2_occlusion.png"),
          "GENERATIVE inpainting: original | occluded | reconstructed", rows=3)

    # (3) Anomaly: in-dist MNIST vs uniform-noise OOD via generative residual energy
    xt, _ = next(iter(test_loader))
    xt = xt.reshape(xt.size(0), -1).to(dev)
    x_ood = torch.rand_like(xt) * 2.0 - 1.0
    s_in = anomaly_scores_generative(model, xt, T=100, lr_state=0.01)
    s_ood = anomaly_scores_generative(model, x_ood, T=100, lr_state=0.01)
    auc = _auc(s_ood, s_in)
    print(f"anomaly: in-dist energy={s_in.mean():.3f}  OOD energy={s_ood.mean():.3f}  AUC={auc:.3f}")

    summary = {"train_time_s": info["train_time_s"], "anomaly_auc_ood_vs_indist": auc,
               "anomaly_energy_indist_mean": float(s_in.mean()),
               "anomaly_energy_ood_mean": float(s_ood.mean()),
               "artifacts": ["m5v2_generate.png", "m5v2_occlusion.png"]}
    with open(os.path.join(RESULTS, "m5v2_demo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("saved: results/m5v2_generate.png, results/m5v2_occlusion.png, results/m5v2_demo_summary.json")


if __name__ == "__main__":
    main()
