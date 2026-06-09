"""Phase-4 search-loop tests (docs/13 M7). Hermetic: train_and_eval is monkeypatched, so no
MNIST/training — we test the grid/random proposers + sorting."""
from __future__ import annotations

import pcn.search as S


def _fake(cfg):
    # deterministic toy objective: accuracy increases with eta_w and slightly with T
    return {"test_acc": float(cfg["eta_w"]) + 0.001 * float(cfg.get("T", 0)),
            "settling_steps_to_converge": 10.0, "train_time_s": 1.0}


def test_grid_search_explores_all_and_sorts(monkeypatch):
    monkeypatch.setattr(S, "train_and_eval", _fake)
    trials = S.grid_search({"T": [20, 40], "eta_w": [0.01, 0.05]})
    assert len(trials) == 4                      # full 2x2 grid
    assert trials[0]["metric"] >= trials[-1]["metric"]   # sorted best-first
    assert trials[0]["config"]["eta_w"] == 0.05  # higher eta_w wins the toy objective


def test_random_search_deterministic_and_bounded(monkeypatch):
    monkeypatch.setattr(S, "train_and_eval", _fake)
    a = S.random_search({"eta_w": [0.01, 0.02, 0.05], "T": [20, 40]}, n_trials=6, seed=0)
    b = S.random_search({"eta_w": [0.01, 0.02, 0.05], "T": [20, 40]}, n_trials=6, seed=0)
    assert len(a) == 6
    assert [t["config"] for t in a] == [t["config"] for t in b]   # deterministic given seed


def test_bayesian_falls_back_without_optuna(monkeypatch):
    # If optuna is absent, bayesian_search must fall back to random_search (still works).
    monkeypatch.setattr(S, "train_and_eval", _fake)
    trials = S.bayesian_search({"eta_w": [0.01, 0.05]}, n_trials=4, seed=0)
    assert len(trials) == 4
    assert trials[0]["metric"] >= trials[-1]["metric"]
