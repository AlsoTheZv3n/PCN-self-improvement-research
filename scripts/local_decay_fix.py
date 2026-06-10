"""Phase A (innovation long-shot): can a strictly LOCAL (no global backprop) modification of SO
settling rescue deep PC from the signal-decay collapse demonstrated in scripts/signal_decay.py?

The decay is a magnitude problem: the top-down error signal attenuates as (1-lambda)^(t-i) as it
propagates one layer per settling step, so input-side layers receive an exponentially weaker
learning signal. We test two principled, per-layer-LOCAL counter-measures (both keep the
"no backprop, parallel-over-layers" property the fused kernel relies on):

  (1) precision  Pi_k = gamma^(n-k):  a per-layer precision (inverse-variance) gain, larger for
      input-side layers, so their decayed error is amplified during settling. Enters both the
      state update (feedback term) and the weight update. gamma=1 recovers standard SO.
      Energy E = 1/2 sum_k Pi_k ||eps_k||^2; grad_s_k = Pi_k eps_k - phi'(s_k) (.) (W_k^T (Pi_{k+1} eps_{k+1})).

  (2) errnorm:    in the local Hebbian update, normalise each layer's error by its own RMS (with a
      floor), so every layer's weight update is O(1) regardless of how decayed its error is.

GATE: against the deep baseline (standard SO, gamma=1), does any local fix close the gap to BP
while staying local? If not -> honest null (explains why the field went to backprop-based EO).

    uv run python scripts/local_decay_fix.py --hidden 6 --epochs 5 --limit-train 5000
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from pcn.api import _mnist_loaders, _resolve_config
from pcn.baselines import BPMLPRef, bp_loss_fn, _eval_acc
from pcn.model import PCN

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@torch.no_grad()
def _settle(model, states, clamp_output, T, lr_state, Pi):
    """SO settling with per-layer precision Pi (list length n+1, Pi[0] unused). Pi=ones -> standard."""
    n = model.n
    s = [t.clone() for t in states]
    last_free = n - 1 if clamp_output else n
    for _ in range(T):
        eps = [None] * (n + 1)
        for i in range(n):
            eps[i + 1] = s[i + 1] - model.predict(i, s[i])
        new = list(s)
        for k in range(1, last_free + 1):
            grad = Pi[k] * eps[k]
            if k < n:
                grad = grad - model.phi_deriv(s[k]) * ((Pi[k + 1] * eps[k + 1]) @ model.W[k])
            new[k] = s[k] - lr_state * grad
        s = new
    return s


@torch.no_grad()
def _update(model, s, lr_weight, Pi, errnorm):
    n = model.n
    batch = s[0].shape[0]
    for i in range(n):
        eps = s[i + 1] - model.predict(i, s[i])
        if errnorm:                                  # per-layer local RMS normalisation
            eps = eps / (eps.pow(2).mean().sqrt() + 1e-6)
        else:
            eps = Pi[i + 1] * eps                    # precision-weighted error
        model.W[i].add_(lr_weight * (eps.t() @ model.phi(s[i]) / batch))
        model.b[i].add_(lr_weight * eps.mean(dim=0))


@torch.no_grad()
def _train_pc_fix(sizes, T, lr_state, lr_weight, epochs, loader, device, gamma, errnorm, seed=0):
    from pcn.settling import feedforward_init
    m = PCN(sizes, activation="tanh", weight_init="orthogonal", device=device, seed=seed)
    n = m.n
    # Pi_k = gamma^(n-k): output layer k=n -> 1, input-side k=1 -> gamma^(n-1)
    Pi = [0.0] + [float(gamma) ** (n - k) for k in range(1, n + 1)]
    for _ in range(epochs):
        for x, y in loader:
            x = x.reshape(x.size(0), -1).to(device, m.dtype)
            y_oh = torch.nn.functional.one_hot(y, sizes[-1]).to(device, m.dtype)
            s = feedforward_init(m, x)
            s[-1] = y_oh
            s = _settle(m, s, True, T, lr_state, Pi)
            _update(m, s, lr_weight, Pi, errnorm)
    return m, Pi


@torch.no_grad()
def _eval_pc(model, loader, T, lr_state, Pi, device):
    from pcn.settling import feedforward_init
    correct = total = 0
    for x, y in loader:
        x = x.reshape(x.size(0), -1).to(device, model.dtype)
        s = feedforward_init(model, x)
        s = _settle(model, s, False, T, lr_state, Pi)
        correct += int((s[-1].argmax(1).cpu() == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def _layer_profile(model, loader, T, lr_state, Pi):
    """Per-layer ||dW_i|| after settling (output clamped) — the per-layer learning signal."""
    from pcn.settling import feedforward_init
    x, y = next(iter(loader))
    x = x.reshape(x.size(0), -1).to(model.device, model.dtype)
    y_oh = torch.nn.functional.one_hot(y, model.layer_sizes[-1]).to(model.device, model.dtype)
    s = feedforward_init(model, x); s[-1] = y_oh
    s = _settle(model, s, True, T, lr_state, Pi)
    batch = x.shape[0]
    return [float((( s[i + 1] - model.predict(i, s[i])).t() @ model.phi(s[i]) / batch).norm())
            for i in range(model.n)]


def _ci(vals):
    n = len(vals); m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8])  # hidden layers
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--limit-train", dest="limit_train", type=int, default=5000)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--lr-state", dest="lr_state", type=float, default=0.1)
    ap.add_argument("--lr-weight", dest="lr_weight", type=float, default=0.02)
    ap.add_argument("--bp-lr", dest="bp_lr", type=float, default=0.05)
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 1.5])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = _resolve_config({"hidden": [args.width], "batch_size": 256,
                           "limit_train": args.limit_train, "val_split": 0.0})
    device, dtype = cfg["device"], torch.float32
    train, _, test = _mnist_loaders(cfg)
    seeds = list(range(args.seeds))
    print(f"device {device} | width {args.width} | T {args.T} | epochs {args.epochs} | seeds {seeds}")
    print(f"VERIFICATION: does the local precision fix robustly beat SO across depth and seed?\n")

    out = {"width": args.width, "T": args.T, "seeds": seeds, "by_depth": {}}
    for d in args.depths:
        sizes = [784] + [args.width] * d + [10]
        bp = BPMLPRef(sizes, activation="tanh", weight_init="orthogonal", device=device, dtype=dtype, seed=0).to(device)
        opt = torch.optim.SGD(bp.parameters(), lr=args.bp_lr)
        loss_fn = bp_loss_fn("mse", num_classes=10)
        for _ in range(args.epochs):
            bp.train()
            for x, y in train:
                x = x.reshape(x.size(0), -1).to(device, dtype); y = y.to(device)
                opt.zero_grad(); loss_fn(bp(x), y).backward(); opt.step()
        acc_bp = _eval_acc(bp, test, device, dtype)
        print(f"=== depth {d} (n={d + 1}) | BP {acc_bp * 100:.1f}% ===")
        rec = {"bp_acc": acc_bp, "gamma": {}}
        for g in args.gammas:
            accs = []
            for sd in seeds:
                m, Pi = _train_pc_fix(sizes, args.T, args.lr_state, args.lr_weight, args.epochs,
                                      train, device, g, False, seed=sd)
                accs.append(_eval_pc(m, test, args.T, args.lr_state, Pi, device))
            mean, std = _ci(accs)
            rec["gamma"][str(g)] = {"mean": mean, "std": std, "seeds": accs}
            tag = "SO baseline" if g == 1.0 else f"precision g={g}"
            print(f"  {tag:>16}: {mean * 100:5.1f}% +/- {std * 100:.1f}  (seeds {[round(a*100) for a in accs]})")
        out["by_depth"][str(d)] = rec

    # mechanism check at the deepest net: does the best gamma FLATTEN the per-layer signal profile?
    dmax = max(args.depths)
    sizes = [784] + [args.width] * dmax + [10]
    g_best = max((g for g in args.gammas if g > 1.0), default=1.5)
    prof = {}
    for g in (1.0, g_best):
        m, Pi = _train_pc_fix(sizes, args.T, args.lr_state, args.lr_weight, args.epochs, train, device, g, False, seed=0)
        p = _layer_profile(m, train, args.T, args.lr_state, Pi)
        prof[str(g)] = {"profile": p, "out_in_ratio": p[-1] / (p[0] + 1e-12)}
    out["mechanism_depth"] = dmax
    out["profiles"] = prof
    print(f"\nmechanism (depth {dmax}): output/input ||dW|| ratio "
          f"(lower = flatter = decay fixed):")
    print(f"  SO baseline (g=1.0): {prof['1.0']['out_in_ratio']:8.1f}x")
    print(f"  precision  (g={g_best}): {prof[str(g_best)]['out_in_ratio']:8.1f}x")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "local_decay_fix_verify.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
