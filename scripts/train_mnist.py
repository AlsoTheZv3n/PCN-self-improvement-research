"""Short MNIST training run with the PyTorch backend.

Usage:
    uv run python scripts/train_mnist.py
    uv run python scripts/train_mnist.py --epochs 10 --hidden 256 256 --T 20
    uv run python scripts/train_mnist.py --backend cuda      # OPTIONAL Phase 3 (raises until built)
"""
from __future__ import annotations

import argparse

from pcn.api import DEFAULT_CONFIG, train_and_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Train a from-scratch PCN on MNIST.")
    p.add_argument("--hidden", type=int, nargs="+", default=DEFAULT_CONFIG["hidden"])
    p.add_argument("--T", type=int, default=DEFAULT_CONFIG["T"])
    p.add_argument("--lr-state", type=float, default=DEFAULT_CONFIG["lr_state"])
    p.add_argument("--lr-weight", type=float, default=DEFAULT_CONFIG["lr_weight"])
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--backend", choices=["pytorch", "cuda"], default="pytorch")
    p.add_argument("--limit-train", type=int, default=None,
                   help="limit #training examples (small-data regime, docs/10)")
    args = p.parse_args()

    config = {
        "hidden": args.hidden,
        "T": args.T,
        "lr_state": args.lr_state,
        "lr_weight": args.lr_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "backend": args.backend,
        "limit_train": args.limit_train,
    }
    metrics = train_and_eval(config)
    print(f"test accuracy : {metrics['test_acc']:.4f}")
    print(f"train time (s): {metrics['train_time_s']:.1f}")
    print(f"device        : {metrics['config']['device']}  backend: {metrics['config']['backend']}")


if __name__ == "__main__":
    main()
