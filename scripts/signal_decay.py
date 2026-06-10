"""Phase 0 (innovation path, systems angle): demonstrate the exponential signal-decay of
standard State-Optimization PC at depth, which is what an Error-Optimization (EO/ePC) settling
kernel would fix. This REPRODUCES the Goemaere et al. 2025 phenomenon (docs/09 sec 2) at our
scale; it is the motivation, not the novelty (the novelty would be a fused CUDA EO kernel).

We sweep network depth and measure, for standard SO-PC vs a matched BP baseline (same init):
  * test accuracy vs depth  -> does SO-PC degrade faster than BP as depth grows? ("deeper=worse")
  * per-layer weight-update magnitude ||dW_i|| after settling (output clamped) for the DEEPEST
    net -> the decay signature: early (input-side) layers should receive exponentially less
    learning signal than late (output-side) layers, while BP's gradient reaches all layers.

GATE: if SO-PC tracks BP across the depths we can reach (no clear decay), there is no problem for
an EO kernel to solve at this scale -> we stop the systems-novelty path honestly.

    uv run python scripts/signal_decay.py --epochs 5 --limit-train 5000
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from pcn.api import _mnist_loaders, _resolve_config
from pcn.baselines import BPMLPRef, bp_loss_fn, _eval_acc
from pcn.evaluate import evaluate
from pcn.learning import train_epoch
from pcn.model import PCN
from pcn.settling import feedforward_init, settle

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _train_bp(sizes, lr, epochs, loader, device, dtype):
    m = BPMLPRef(sizes, activation="tanh", weight_init="orthogonal",
                 device=device, dtype=dtype, seed=0).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    loss_fn = bp_loss_fn("mse", num_classes=sizes[-1])
    for _ in range(epochs):
        m.train()
        for x, y in loader:
            x = x.reshape(x.size(0), -1).to(device, dtype)
            y = y.to(device)
            opt.zero_grad(); loss_fn(m(x), y).backward(); opt.step()
    return m


def _train_pc(sizes, T, lr_state, lr_weight, epochs, loader, device):
    m = PCN(sizes, activation="tanh", weight_init="orthogonal", device=device, seed=0)
    for _ in range(epochs):
        train_epoch(m, loader, T, lr_state, lr_weight, num_classes=sizes[-1])
    return m


@torch.no_grad()
def _layer_update_profile(model, loader, T, lr_state):
    """Mean ||dW_i|| per layer at the settled (output-clamped) state — the per-layer learning
    signal. Decay = early layers << late layers. Returns a list over i=0..n-1 (input->output)."""
    x, y = next(iter(loader))
    x = x.reshape(x.size(0), -1).to(model.device, model.dtype)
    y_oh = torch.nn.functional.one_hot(y, model.layer_sizes[-1]).to(model.device, model.dtype)
    s = feedforward_init(model, x)
    s[-1] = y_oh
    s, _, _ = settle(model, s, clamp_output=True, T=T, lr_state=lr_state)
    batch = x.shape[0]
    norms = []
    for i in range(model.n):
        eps = s[i + 1] - model.predict(i, s[i])
        dW = eps.t() @ model.phi(s[i]) / batch
        norms.append(float(dW.norm()))
    return norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--limit-train", dest="limit_train", type=int, default=5000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 8])  # hidden layers
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--lr-state", dest="lr_state", type=float, default=0.1)   # lambda~0.1: decay in 4-8 steps
    ap.add_argument("--lr-weight", dest="lr_weight", type=float, default=0.02)
    ap.add_argument("--bp-lr", dest="bp_lr", type=float, default=0.05)
    args = ap.parse_args()

    cfg = _resolve_config({"hidden": [args.width], "batch_size": 256,
                           "limit_train": args.limit_train, "val_split": 0.0})
    device, dtype = cfg["device"], torch.float32
    train, _, test = _mnist_loaders(cfg)
    print(f"device {device} | width {args.width} | T {args.T} | eta_x {args.lr_state} | epochs {args.epochs}")
    print(f"\n{'hidden':>7} {'depth(n)':>9} {'BP-acc':>7} {'PC-acc':>7} {'gap':>7}")

    rows = []
    deepest_profile = None
    for d in args.depths:
        sizes = [784] + [args.width] * d + [10]
        bp = _train_bp(sizes, args.bp_lr, args.epochs, train, device, dtype)
        pc = _train_pc(sizes, args.T, args.lr_state, args.lr_weight, args.epochs, train, device)
        acc_bp = _eval_acc(bp, test, device, dtype)
        acc_pc = evaluate(pc, test, args.T, args.lr_state)
        rows.append({"hidden": d, "n": d + 1, "bp_acc": acc_bp, "pc_acc": acc_pc})
        print(f"{d:>7} {d + 1:>9} {acc_bp * 100:6.1f}% {acc_pc * 100:6.1f}% {(acc_bp - acc_pc) * 100:+6.1f}")
        if d == max(args.depths):
            deepest_profile = _layer_update_profile(pc, train, args.T, args.lr_state)

    print(f"\nper-layer ||dW|| (input-side -> output-side) for the deepest net "
          f"({max(args.depths)} hidden, T={args.T}):")
    if deepest_profile:
        mx = max(deepest_profile) + 1e-12
        for i, nrm in enumerate(deepest_profile):
            bar = "#" * int(40 * nrm / mx)
            print(f"  layer {i:>2}: {nrm:.2e}  {bar}")
        ratio = deepest_profile[-1] / (deepest_profile[0] + 1e-12)
        print(f"  output/input update-magnitude ratio: {ratio:.1f}x  "
              f"({'DECAY present' if ratio > 5 else 'no strong decay'})")

    summary = {"width": args.width, "T": args.T, "lr_state": args.lr_state, "epochs": args.epochs,
               "rows": rows, "deepest_layer_update_profile": deepest_profile}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"signal_decay_w{args.width}_T{args.T}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
