"""Offline test for the GEM continual-learning metrics (docs/13 M4).

gem_metrics is pure (takes an R-matrix), so no MNIST / training is needed.
"""
from __future__ import annotations

from pcn.experiments.continual import gem_metrics, make_permutations, make_class_split_tasks


def test_gem_metrics_known_matrix():
    # 3 tasks; R[i][j] = accuracy on task j after training task i.
    R = [
        [0.90, 0.10, 0.10],   # after task 0
        [0.80, 0.90, 0.10],   # after task 1
        [0.70, 0.85, 0.90],   # after task 2 (final row)
    ]
    m = gem_metrics(R)
    assert abs(m["acc"] - (0.70 + 0.85 + 0.90) / 3) < 1e-9
    # BWT = mean_{j<2} (R[2][j] - R[j][j]) = ((0.70-0.90) + (0.85-0.90)) / 2 = -0.125
    assert abs(m["bwt"] - (-0.125)) < 1e-9
    # learn_acc = mean diagonal = (0.90 + 0.90 + 0.90) / 3 = 0.90
    assert abs(m["learn_acc"] - 0.90) < 1e-9
    assert m["final_per_task"] == [0.70, 0.85, 0.90]


def test_no_forgetting_gives_zero_bwt():
    R = [[0.9, 0.0], [0.9, 0.9]]   # task 0 retained perfectly after task 1
    assert abs(gem_metrics(R)["bwt"]) < 1e-9


def test_permutations_are_valid_and_seeded():
    a = make_permutations(5, seed=0)
    b = make_permutations(5, seed=0)
    assert len(a) == 5
    assert a[0].tolist() == list(range(784))          # task 0 = identity
    for p in a[1:]:
        assert sorted(p.tolist()) == list(range(784))  # a genuine permutation
    assert all((x == y).all() for x, y in zip(a, b))    # deterministic given seed


def test_class_split_tasks_disjoint_complete_seeded():
    # Song shape: 2 tasks x 5 classes (FashionMNIST)
    tasks = make_class_split_tasks(n_tasks=2, classes_per_task=5, seed=0)
    assert len(tasks) == 2 and all(len(t) == 5 for t in tasks)
    assert sorted(c for t in tasks for c in t) == list(range(10))   # disjoint + complete
    # Split-MNIST shape: 5 tasks x 2 classes
    t5 = make_class_split_tasks(5, 2, seed=0)
    assert len(t5) == 5 and sorted(c for t in t5 for c in t) == list(range(10))
    # deterministic given seed
    assert make_class_split_tasks(2, 5, 1) == make_class_split_tasks(2, 5, 1)
