"""Fair-comparison protocol primitives (docs/13 M4).

Pure orchestration + statistics — no torch at import time, so the stats are unit-testable
offline. The training functions are passed in (or imported lazily in `compare_pc_vs_bp`).

Protocol (Song et al. 2024 / van Zwol et al.): the LR is tuned PER METHOD over a shared
grid, every metric is reported as mean +/- bootstrap CI over >= 3 seeds, and the two arms
share architecture / init / data (guaranteed by `pcn.baselines.BPMLPRef`, docs/13 M3).

LR selection: the driver (`scripts/run_experiments.py:cmd_compare`) selects each method's LR
by best mean VALIDATION accuracy on a held-out split (`val_split`, api.py) and REPORTS test
accuracy — no test-set model selection. PC is tuned over a JOINT eta_x x eta_w grid via
`sweep_grid` (there is a stability frontier; sweeping eta_w alone under-sells PC, docs/12 M4).
The legacy `compare_pc_vs_bp` helper below is single-axis and kept only for convenience;
prefer the driver's sweep_grid path.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Callable


def bootstrap_ci(values, confidence: float = 0.68, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean and bootstrap confidence interval. Default 68% CI = 1-sigma (Song et al.).

    Deterministic given ``seed``. Returns {mean, lo, hi, std, n}.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "std": 0.0, "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "lo": mean, "hi": mean, "std": 0.0, "n": 1}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    alpha = (1.0 - confidence) / 2.0
    # symmetric percentile indices over [0, n_boot-1] (both endpoints use the same rule)
    lo = boots[int(alpha * (n_boot - 1))]
    hi = boots[int((1.0 - alpha) * (n_boot - 1))]
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return {"mean": mean, "lo": lo, "hi": hi, "std": math.sqrt(var), "n": n}


def run_multiseed(fn: Callable[[dict], dict], config: dict, seeds) -> list:
    """Run ``fn`` once per seed (overriding config['seed']); return the list of result dicts."""
    return [fn({**config, "seed": int(s)}) for s in seeds]


def agg_scalar(results: list, key: str, **ci_kwargs) -> dict:
    """Bootstrap CI of a scalar metric ``key`` across a list of result dicts."""
    return bootstrap_ci([r[key] for r in results], **ci_kwargs)


def sweep_lr(fn: Callable[[dict], dict], config: dict, lr_key: str, lr_grid, seeds,
             metric: str = "test_acc") -> list:
    """For each lr in the grid, run all seeds and summarise ``metric``.

    Returns a list of {lr, summary (bootstrap CI of metric), results}.
    """
    out = []
    for lr in lr_grid:
        results = run_multiseed(fn, {**config, lr_key: lr}, seeds)
        out.append({"lr": lr, "summary": agg_scalar(results, metric), "results": results})
    return out


def sweep_grid(fn: Callable[[dict], dict], config: dict, grid: dict, seeds,
               metric: str = "val_acc") -> list:
    """Cartesian-product hyperparameter sweep over ``grid`` = {key: [values], ...}.

    Needed for PC, whose state LR (eta_x) and weight LR (eta_w) must be tuned JOINTLY — there
    is a stability frontier (high eta_w needs low eta_x), so sweeping eta_w alone under-sells
    PC (docs/12 M4). Returns a list of {params, summary (CI of metric), results}.
    """
    keys = list(grid.keys())
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        results = run_multiseed(fn, {**config, **params}, seeds)
        out.append({"params": params, "summary": agg_scalar(results, metric), "results": results})
    return out


def best_of_sweep(sweep: list) -> dict:
    """The sweep entry with the highest mean metric (the per-method LR selection).
    Works for both sweep_lr (entries have 'lr') and sweep_grid (entries have 'params')."""
    return max(sweep, key=lambda s: s["summary"]["mean"])


def compare_pc_vs_bp(base_config: dict, seeds, pc_lr_grid, bp_lr_grid,
                     metric: str = "val_acc", lr_key: str = "eta_w") -> dict:
    """Fair PC-vs-BP comparison: tune each method's LR over its grid, report best per method.

    PC = pcn.api.train_and_eval, BP = pcn.baselines.train_and_eval_bp. Both accept the
    ``eta_w`` alias as the swept weight/SGD learning rate; the PC state LR ``eta_x`` is taken
    from ``base_config`` (fixed across the sweep). Imported lazily so this module stays
    torch-free at import time.
    """
    from ..api import train_and_eval
    from ..baselines import train_and_eval_bp

    pc_sweep = sweep_lr(train_and_eval, base_config, lr_key, pc_lr_grid, seeds, metric)
    bp_sweep = sweep_lr(train_and_eval_bp, base_config, lr_key, bp_lr_grid, seeds, metric)
    return {
        "pc": {"sweep": pc_sweep, "best": best_of_sweep(pc_sweep)},
        "bp": {"sweep": bp_sweep, "best": best_of_sweep(bp_sweep)},
        "base_config": base_config,
        "seeds": list(seeds),
        "metric": metric,
    }
