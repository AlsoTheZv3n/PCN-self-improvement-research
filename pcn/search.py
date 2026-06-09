"""Phase 4 (docs/04, docs/13 M7): bounded autonomous search over the PCN config space.

A THIN layer over `train_and_eval` — propose a config, run it, log the metrics, pick the next
(exactly the interface M1 was designed for; no rewrite). Two built-in proposers with no extra
dependencies: grid and random. An optional Bayesian (Optuna TPE) stage is used IF optuna is
importable. No W&B (needs a login / headless-unfriendly); trials are logged to JSON.

The space is a dict {config_key: [candidate values]}; e.g. {"T": [20,40], "eta_w": [.01,.02]}.
"""
from __future__ import annotations

import itertools
import json
import os
import random

from .api import train_and_eval


def _run_one(base, overrides, metric):
    cfg = {**base, **overrides}
    m = train_and_eval(cfg)
    return {"config": overrides,
            "metric": m[metric],
            "test_acc": m["test_acc"],
            "settling_steps_to_converge": m.get("settling_steps_to_converge"),
            "train_time_s": m["train_time_s"]}


def _save(trials, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2)


def grid_search(space, base_config=None, metric="test_acc", log_path=None):
    """Exhaustive cartesian-product search over ``space``. Returns trials sorted best-first."""
    base = dict(base_config or {})
    keys = list(space.keys())
    trials = []
    for combo in itertools.product(*[space[k] for k in keys]):
        trials.append(_run_one(base, dict(zip(keys, combo)), metric))
        _save(trials, log_path)
    trials.sort(key=lambda t: t["metric"], reverse=True)
    return trials


def random_search(space, n_trials=20, seed=0, base_config=None, metric="test_acc", log_path=None):
    """Random search: sample ``n_trials`` configs from ``space`` (each key a candidate list).
    Returns trials sorted best-first. Deterministic given ``seed``."""
    rng = random.Random(seed)
    base = dict(base_config or {})
    trials = []
    for _ in range(n_trials):
        overrides = {k: rng.choice(v) for k, v in space.items()}
        trials.append(_run_one(base, overrides, metric))
        _save(trials, log_path)
    trials.sort(key=lambda t: t["metric"], reverse=True)
    return trials


def bayesian_search(space, n_trials=20, seed=0, base_config=None, metric="test_acc", log_path=None):
    """Bayesian (TPE) search via Optuna — used only if optuna is installed (`uv add optuna`).
    Falls back to random_search otherwise. Each space key becomes a categorical over its list."""
    try:
        import optuna
    except ImportError:
        return random_search(space, n_trials, seed, base_config, metric, log_path)
    base = dict(base_config or {})
    trials = []

    def objective(trial):
        overrides = {k: trial.suggest_categorical(k, v) for k, v in space.items()}
        r = _run_one(base, overrides, metric)
        trials.append(r)
        _save(trials, log_path)
        return r["metric"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials.sort(key=lambda t: t["metric"], reverse=True)
    return trials
