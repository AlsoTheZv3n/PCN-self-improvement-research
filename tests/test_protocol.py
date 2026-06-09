"""Offline tests for the fair-comparison protocol primitives (docs/13 M4).

No torch needed: the stats/orchestration take a plain callable.
"""
from __future__ import annotations

from pcn.experiments.protocol import (
    bootstrap_ci,
    run_multiseed,
    sweep_lr,
    sweep_grid,
    best_of_sweep,
    agg_scalar,
)


def test_bootstrap_ci_deterministic_and_bounded():
    vals = [0.80, 0.82, 0.85, 0.83, 0.81]
    a = bootstrap_ci(vals, seed=0)
    b = bootstrap_ci(vals, seed=0)
    assert a == b                                   # deterministic given seed
    assert a["lo"] <= a["mean"] <= a["hi"]          # mean inside the CI
    assert abs(a["mean"] - sum(vals) / len(vals)) < 1e-9
    assert a["n"] == 5


def test_bootstrap_ci_single_value_has_zero_width():
    a = bootstrap_ci([0.9])
    assert a["lo"] == a["mean"] == a["hi"] == 0.9
    assert a["std"] == 0.0


def test_run_multiseed_overrides_seed():
    seen = []

    def fake(cfg):
        seen.append(cfg["seed"])
        return {"test_acc": 0.5 + 0.01 * cfg["seed"], "seed": cfg["seed"]}

    res = run_multiseed(fake, {"hidden": [4]}, [0, 1, 2])
    assert seen == [0, 1, 2]
    assert [r["seed"] for r in res] == [0, 1, 2]


def test_sweep_picks_best_lr():
    # accuracy == lr, so the largest lr must win.
    def fake(cfg):
        return {"test_acc": float(cfg["eta_w"])}

    sweep = sweep_lr(fake, {}, "eta_w", [0.1, 0.3, 0.2], seeds=[0, 1])
    best = best_of_sweep(sweep)
    assert best["lr"] == 0.3
    assert abs(best["summary"]["mean"] - 0.3) < 1e-9


def test_agg_scalar():
    results = [{"acc": 0.8}, {"acc": 0.9}, {"acc": 0.85}]
    ci = agg_scalar(results, "acc")
    assert abs(ci["mean"] - 0.85) < 1e-9
    assert ci["n"] == 3


def test_sweep_grid_cartesian_and_best():
    # val_acc == a + b, so the largest (a, b) combo must win the joint sweep.
    def fake(cfg):
        return {"val_acc": cfg["a"] + cfg["b"]}

    sweep = sweep_grid(fake, {}, {"a": [1, 2], "b": [10, 20]}, seeds=[0], metric="val_acc")
    assert len(sweep) == 4                      # 2 x 2 cartesian product
    best = best_of_sweep(sweep)
    assert best["params"] == {"a": 2, "b": 20}
    assert abs(best["summary"]["mean"] - 22) < 1e-9
