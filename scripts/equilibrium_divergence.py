"""Equilibrium-resolved PC-vs-BP SOLUTION divergence (innovation probe — docs/06 follow-up).

Question: as PC inference is driven closer to its energy equilibrium (larger settling budget T),
does the learned SOLUTION diverge from a BP baseline trained from the IDENTICAL init --- even at
accuracy parity? The fused CUDA kernel makes large T cheap, which is what lets us sweep T to
near-equilibrium at all. For a fixed [784,h,h,10] net on MNIST we measure, as a function of T:

  * test accuracy of PC(T) vs BP (same init)             -> parity axis
  * weight-space distance PC(T) vs BP (relative Frobenius, per layer, anchored by shared init)
  * functional agreement: fraction of test samples where PC(T) and BP predict the SAME class
  * a noise floor: BP vs BP(different data order, same init) distance + disagreement, so the
    PC-vs-BP numbers are interpretable relative to pure optimization noise
  * settling quality of PC(T): residual energy after T steps vs after a long reference settle
    (is T actually reaching equilibrium?)

Outcome reading (the real finding is in how these move WITH T):
  - agreement/weight-distance stay at the BP-vs-BP floor for all T  -> PC == BP regardless of
    settling (a strong null);
  - they move away from the floor as T grows                        -> equilibrium PC finds a
    DIFFERENT solution than BP, and under-settling hides it (a genuine, novel finding).

Run (small pilot is the default; PCN_CUDA_KERNEL=1 + a vcvars build speeds the large-T points):
    uv run python scripts/equilibrium_divergence.py --epochs 5 --limit-train 5000
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from pcn.api import _mnist_loaders, _resolve_config
from pcn.baselines import BPMLPRef, bp_loss_fn
from pcn.evaluate import evaluate
from pcn.learning import train_epoch
from pcn.settling import feedforward_init, settle, energy_per_sample

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _mnist_train(cfg, shuffle_seed):
    """Train loader with an EXPLICIT shuffle generator, decoupled from the global RNG (which the
    PCN/BPMLPRef init resets via torch.manual_seed). This makes a real BP-vs-BP noise floor
    possible: same init, genuinely different data order."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(cfg["data_root"], train=True, download=True, transform=tf)
    if cfg.get("limit_train"):
        ds = torch.utils.data.Subset(ds, range(int(cfg["limit_train"])))
    g = torch.Generator().manual_seed(int(shuffle_seed))
    return torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, generator=g)


def _train_pc(sizes, T, lr_state, lr_weight, epochs, loader, seed, device, backend, tol=None):
    m = PCN_(sizes, device, seed)
    for _ in range(epochs):
        # tol lets settling STOP at convergence (up to T): large T -> reaches equilibrium without
        # overshooting; a safe lr_state keeps the iteration stable so PC actually learns at every T
        # (the pilot's T=80 collapse was overshoot, not a finding). backend='cuda' ignores tol.
        train_epoch(m, loader, T, lr_state, lr_weight, num_classes=sizes[-1], backend=backend, tol=tol)
    return m


def PCN_(sizes, device, seed):
    from pcn.model import PCN
    return PCN(sizes, activation="tanh", weight_init="orthogonal", device=device, seed=seed)


def _train_bp(sizes, lr, epochs, loader, seed, device, dtype, bp_loss):
    m = BPMLPRef(sizes, activation="tanh", weight_init="orthogonal",
                 device=device, dtype=dtype, seed=seed).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    loss_fn = bp_loss_fn(bp_loss, num_classes=sizes[-1])
    for _ in range(epochs):
        m.train()
        for x, y in loader:
            x = x.reshape(x.size(0), -1).to(device, dtype)
            y = y.to(device)
            opt.zero_grad()
            loss_fn(m(x), y).backward()
            opt.step()
    return m


@torch.no_grad()
def _weight_dist(a, b):
    """Mean relative per-layer Frobenius distance ||W_a - W_b|| / ||W_b||; both share init."""
    ds = []
    for i in range(a.n):
        wa, wb = a.W[i].detach(), b.W[i].detach()
        ds.append(float((wa - wb).norm() / (wb.norm() + 1e-12)))
    return sum(ds) / len(ds)


@torch.no_grad()
def _preds(model, loader, kind, T, lr_state, device, dtype):
    out = []
    for x, y in loader:
        x = x.reshape(x.size(0), -1).to(device, dtype)
        if kind == "pc":
            s = feedforward_init(model, x)
            s, _, _ = settle(model, s, clamp_output=False, T=T, lr_state=lr_state)
            out.append(s[-1].argmax(1).cpu())
        else:
            model.eval()
            out.append(model(x).argmax(1).cpu())
    return torch.cat(out)


@torch.no_grad()
def _settle_quality(model, loader, T, lr_state, ref_T=400):
    """Residual per-sample energy after T steps vs a long reference settle (output clamped, as in
    training). ratio = E(T)/E(ref) >> 1 means under-settled; ~1 means T reaches equilibrium."""
    x, y = next(iter(loader))
    x = x.reshape(x.size(0), -1).to(model.device, model.dtype)
    y_oh = torch.nn.functional.one_hot(y, model.layer_sizes[-1]).to(model.device, model.dtype)
    s0 = feedforward_init(model, x)
    s0[-1] = y_oh
    sT, _, _ = settle(model, [t.clone() for t in s0], clamp_output=True, T=T, lr_state=lr_state)
    sR, _, _ = settle(model, [t.clone() for t in s0], clamp_output=True, T=ref_T, lr_state=lr_state)
    eT = float(energy_per_sample(model, sT).mean())
    eR = float(energy_per_sample(model, sR).mean())
    return eT, eR, eT / (eR + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--limit-train", dest="limit_train", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr-state", dest="lr_state", type=float, default=0.02)  # safe for large T
    ap.add_argument("--lr-weight", dest="lr_weight", type=float, default=0.02)
    ap.add_argument("--bp-lr", dest="bp_lr", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-3)  # settling stops at convergence (no overshoot)
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "cuda"])
    ap.add_argument("--T", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160])
    args = ap.parse_args()

    cfg = _resolve_config({"hidden": [args.hidden, args.hidden], "batch_size": 256,
                           "limit_train": args.limit_train, "val_split": 0.0})
    device, dtype = cfg["device"], torch.float32
    sizes = [784, args.hidden, args.hidden, 10]
    train = _mnist_train(cfg, shuffle_seed=0)
    train_floor = _mnist_train(cfg, shuffle_seed=1)   # same init, genuinely different data order
    _, _, test = _mnist_loaders(cfg)
    print(f"device {device} | arch {sizes} | epochs {args.epochs} | backend {args.backend} | "
          f"eta_x={args.lr_state} eta_w={args.lr_weight} tol={args.tol}")

    # BP baseline (seed 0) and a floor partner (SAME init, different data order)
    bp = _train_bp(sizes, args.bp_lr, args.epochs, train, 0, device, dtype, "mse")
    bp_floor = _train_bp(sizes, args.bp_lr, args.epochs, train_floor, 0, device, dtype, "mse")
    bp_preds = _preds(bp, test, "bp", 0, 0.0, device, dtype)
    floor_wd = _weight_dist(bp_floor, bp)
    floor_agree = float((_preds(bp_floor, test, "bp", 0, 0.0, device, dtype) == bp_preds).float().mean())
    from pcn.baselines import _eval_acc
    acc_bp = _eval_acc(bp, test, device, dtype)
    print(f"\nBP(MSE) test-acc={acc_bp * 100:.1f}%  | NOISE FLOOR (BP vs BP'): "
          f"weight-dist={floor_wd:.4f}  pred-agreement={floor_agree * 100:.1f}%\n")
    print(f"{'T':>4} {'PC-acc':>7} {'wdist(PC,BP)':>13} {'agree(PC,BP)':>13} {'E(T)/E(eq)':>11}")

    rows = []
    for T in args.T:
        pc = _train_pc(sizes, T, args.lr_state, args.lr_weight, args.epochs, train, 0, device,
                       args.backend, tol=(None if args.backend == "cuda" else args.tol))
        acc_pc = evaluate(pc, test, T, args.lr_state, backend=args.backend)
        wd = _weight_dist(pc, bp)
        agree = float((_preds(pc, test, "pc", T, args.lr_state, device, dtype) == bp_preds).float().mean())
        eT, eR, ratio = _settle_quality(pc, train, T, args.lr_state)
        rows.append({"T": T, "pc_acc": acc_pc, "wdist_pc_bp": wd, "agree_pc_bp": agree,
                     "resid_E_T": eT, "resid_E_eq": eR, "settle_ratio": ratio})
        print(f"{T:>4} {acc_pc * 100:6.1f}% {wd:>13.4f} {agree * 100:12.1f}% {ratio:>11.2f}")

    summary = {"arch": sizes, "epochs": args.epochs, "limit_train": args.limit_train,
               "bp_acc": acc_bp, "floor_wdist": floor_wd, "floor_agree": floor_agree,
               "lr": {"eta_x": args.lr_state, "eta_w": args.lr_weight, "bp": args.bp_lr},
               "rows": rows}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"equilibrium_divergence_h{args.hidden}_e{args.epochs}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # the headline read: does PC's solution move AWAY from BP as T grows, beyond the BP-vs-BP floor?
    if rows:
        trend = rows[-1]["wdist_pc_bp"] - rows[0]["wdist_pc_bp"]
        print(f"\nweight-dist trend (T={rows[-1]['T']} minus T={rows[0]['T']}): {trend:+.4f}  "
              f"(floor BP-vs-BP = {floor_wd:.4f})")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
