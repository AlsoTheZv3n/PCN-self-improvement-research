"""Wall-clock benchmarking for the settling loop and a PC-vs-BP comparison (docs/13 M3).

Correct CUDA timing discipline (PyTorch CUDA semantics, docs/11): GPU ops are asynchronous,
so we (1) warm up, then (2) `torch.cuda.synchronize()` (or CUDA events) around the timed
region. On CPU `time.perf_counter` is exact and no sync is needed.

This is the measurement point the fair PC-vs-BP study (M4) and the optional CUDA kernel (M6)
report against. Keep `tol=None` here so every settle does exactly `T` steps and `ms_per_step`
is meaningful.
"""
from __future__ import annotations

import time

import torch

from .settling import feedforward_init, settle


def time_settle(model, x: torch.Tensor, T: int, lr_state: float,
                backend: str = "pytorch", clamp_output: bool = False,
                warmup: int = 2, iters: int = 20) -> dict:
    """Mean wall-clock of one full settle (T steps) on the batch ``x`` (already [B, D] on
    the model's device). Returns ms per settle-call and ms per settling-step.

    Always uses ``tol=None`` (fixed T) so the per-step number is well defined.
    """
    use_cuda = str(model.device).startswith("cuda") and torch.cuda.is_available()

    def one_settle():
        states = feedforward_init(model, x)
        settle(model, states, clamp_output=clamp_output, T=T, lr_state=lr_state,
               backend=backend, tol=None)

    for _ in range(max(warmup, 0)):
        one_settle()

    if use_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            one_settle()
        end.record()
        torch.cuda.synchronize()
        ms_total = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            one_settle()
        ms_total = (time.perf_counter() - t0) * 1000.0

    ms_per_call = ms_total / max(iters, 1)
    return {
        "ms_per_settle": ms_per_call,
        "ms_per_step": ms_per_call / max(T, 1),
        "iters": iters,
        "T": T,
        "backend": backend,
        "device": str(model.device),
        "batch_size": int(x.shape[0]),
    }


def benchmark_pc_vs_bp(config: dict | None = None) -> dict:
    """Train both the PCN (PC learning) and the matched BP-MLP on MNIST and tabulate
    accuracy + wall-clock — the first PC-vs-BP time/accuracy tableau for docs/12.

    Note the two axes the literature separates (docs/11): *sample/episode efficiency*
    (Song et al.) vs. *compute time per update* (Zahid et al.). This reports the compute axis
    (train_time_s) and final accuracy; the sample-efficiency axis lives in M4.
    """
    from .api import train_and_eval
    from .baselines import train_and_eval_bp

    pc = train_and_eval(config)
    bp = train_and_eval_bp(config)
    return {
        "pc": {"test_acc": pc["test_acc"], "train_time_s": pc["train_time_s"],
               "settling_steps_to_converge": pc["settling_steps_to_converge"],
               "noise_robustness": pc["noise_robustness"]},
        "bp": {"test_acc": bp["test_acc"], "train_time_s": bp["train_time_s"],
               "noise_robustness": bp["noise_robustness"]},
        "config": pc["config"],
    }
