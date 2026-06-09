"""From-scratch Predictive Coding Network (State Optimization) on MNIST.

Default backend is PyTorch (fully functional). The CUDA settling kernel in pcn.kernels is
an OPTIONAL Phase-3 path (docs/07, docs/09) and is a stub until built.
"""
from .api import DEFAULT_CONFIG, train_and_eval
from .baselines import BPMLPRef, train_and_eval_bp
from .benchmark import benchmark_pc_vs_bp, time_settle
from .evaluate import evaluate, noise_robustness
from .generate import anomaly_scores, generate, inpaint
from .generative import (anomaly_scores_generative, generate_class_grid, generate_images,
                         inpaint_generative, train_generative)
from .learning import train_epoch, weight_update
from .model import PCN
from .settling import energy, energy_per_sample, feedforward_init, settle

__all__ = [
    "PCN",
    "feedforward_init",
    "settle",
    "energy",
    "weight_update",
    "train_epoch",
    "evaluate",
    "noise_robustness",
    "energy_per_sample",
    "generate",
    "inpaint",
    "anomaly_scores",
    "train_generative",
    "generate_images",
    "generate_class_grid",
    "inpaint_generative",
    "anomaly_scores_generative",
    "train_and_eval",
    "DEFAULT_CONFIG",
    "BPMLPRef",
    "train_and_eval_bp",
    "time_settle",
    "benchmark_pc_vs_bp",
]
