"""Fair PC-vs-BP experiment harness (docs/13 M4, docs/08, docs/10).

The fair-comparison protocol (Song et al. 2024): identical architecture/init/data, the
learning rule is the only independent variable, the LR is tuned *per method* over a shared
grid, and every number is reported as mean +/- bootstrap CI over >= 3 seeds. The building
blocks live in `protocol.py`; concrete regimes (settling-T, small-data, noise, continual)
are driven from `scripts/run_experiments.py`.
"""
from .protocol import (
    bootstrap_ci,
    run_multiseed,
    agg_scalar,
    sweep_lr,
    sweep_grid,
    best_of_sweep,
    compare_pc_vs_bp,
)
from .continual import (
    gem_metrics,
    make_permutations,
    make_class_split_tasks,
    run_permuted_mnist,
    run_class_il,
    run_alternating,
)

__all__ = [
    "bootstrap_ci",
    "run_multiseed",
    "agg_scalar",
    "sweep_lr",
    "sweep_grid",
    "best_of_sweep",
    "compare_pc_vs_bp",
    "gem_metrics",
    "make_permutations",
    "make_class_split_tasks",
    "run_permuted_mnist",
    "run_class_il",
    "run_alternating",
]
