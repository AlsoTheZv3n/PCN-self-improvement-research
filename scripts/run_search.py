"""Phase-4 (M7) demo: a bounded autonomous search over the PCN config space, as a thin layer
over train_and_eval (docs/04). Random search by default; pass --bayesian to use Optuna (if
installed). Logs trials to results/m7_search.json.

    uv run python scripts/run_search.py --trials 16
"""
from __future__ import annotations

import argparse
import json
import os

from pcn.search import bayesian_search, random_search

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    p = argparse.ArgumentParser(description="Phase-4 bounded search over train_and_eval")
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--limit-train", dest="limit_train", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bayesian", action="store_true", help="use Optuna TPE if installed")
    args = p.parse_args()

    base = {"hidden": [256, 256], "epochs": args.epochs, "limit_train": args.limit_train,
            "batch_size": 256, "tol": None, "eval_noise": False}
    space = {"T": [20, 40], "eta_x": [0.02, 0.05, 0.1], "eta_w": [0.01, 0.02, 0.05]}
    log = os.path.join(RESULTS, "m7_search.json")

    method = "Bayesian (Optuna TPE)" if args.bayesian else "random"
    print(f"Phase-4 {method} search over {list(space)} | {args.trials} trials "
          f"(limit_train={args.limit_train}, epochs={args.epochs})")
    search = bayesian_search if args.bayesian else random_search
    trials = search(space, n_trials=args.trials, seed=0, base_config=base,
                    metric="test_acc", log_path=log)

    print(f"\nexplored {len(trials)} configs. Top 3:")
    for t in trials[:3]:
        print(f"  {t['config']}  acc={t['test_acc']:.4f}  steps2conv~{t['settling_steps_to_converge']:.0f}")
    print(f"  worst: {trials[-1]['config']}  acc={trials[-1]['test_acc']:.4f}")
    print(f"saved: {log}")


if __name__ == "__main__":
    main()
