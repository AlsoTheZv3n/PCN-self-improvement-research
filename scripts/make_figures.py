"""Generate the paper figures from the saved results/*.json artifacts (docs/05, §4c/§4d/§4g).

Pure plotting — reads only the committed-numbers JSONs, writes PNG+PDF to figures/. No training.
    uv run python scripts/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
C = {"PC": "#1f77b4", "BP_MSE": "#d62728", "BP_CE": "#ff7f0e"}
LBL = {"PC": "PC (local Hebbian)", "BP_MSE": "BP (MSE)", "BP_CE": "BP (cross-entropy)"}


def _load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{stem}.png + .pdf")


def fig_kernel_speedup():
    d = _load("m6_kernel_benchmark.json")
    rows = d["rows"]
    Ts = sorted({r["T"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for T in Ts:
        sub = sorted([r for r in rows if r["T"] == T], key=lambda r: r["B"])
        ax.plot([r["B"] for r in sub], [r["speedup"] for r in sub],
                marker="o", label=f"T={T} settling steps")
    ax.axhline(1.0, ls="--", color="gray", lw=1, label="break-even (PyTorch)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["B"] for r in rows if r["T"] == Ts[0]])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("batch size")
    ax.set_ylabel("speedup  (PyTorch ms / fused-CUDA ms)")
    ax.set_title(f"Fused CUDA settling kernel vs PyTorch\n{d['device']}, MLP {d['topology']}")
    ax.legend(frameon=False)
    ax.annotate("small-batch win\n(launch-overhead bound)", xy=(64, 2.4), xytext=(70, 1.5),
                fontsize=8, color="#1f77b4")
    ax.annotate("cuBLAS wins\n(compute-bound)", xy=(4096, 0.6), xytext=(1100, 0.45),
                fontsize=8, color="gray")
    _save(fig, "fig1_kernel_speedup")


def fig_budget_sweep():
    d = _load("m4_alternating_song_exact_budgetsweep_s10.json")
    budgets = d["budgets"]
    arms = ["PC", "BP_MSE", "BP_CE"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, key, ttl in ((axes[0], "mean_both", "mean of both tasks  (Song Fig-4e metric)"),
                         (axes[1], "min_both", "worse task  (interference tell)")):
        for arm in arms:
            ms = [d["arms_by_budget"][str(b)][arm][key]["mean"] * 100 for b in budgets]
            sd = [d["arms_by_budget"][str(b)][arm][key]["std"] * 100 for b in budgets]
            ax.errorbar(budgets, ms, yerr=sd, marker="o", capsize=3, color=C[arm], label=LBL[arm])
        ax.axhline(20, ls=":", color="gray", lw=1, label="chance (1/5)")
        ax.axvline(84, ls="--", color="black", lw=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("training budget  (minibatch updates)")
        ax.set_ylabel("test accuracy [%]")
        ax.set_title(ttl)
    axes[0].annotate("Song's exact\nbudget = 84\n(untrainable)", xy=(84, 30), xytext=(95, 33),
                     fontsize=7.5, color="black")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Song-exact alternating Split-FashionMNIST 2x5 (sigmoid [32,32], n=10, ±1σ) — "
                 "no PC-vs-BP(MSE) gap exceeds 1σ at any budget", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig3_alternating_budget_sweep")


def fig_pc_vs_bp():
    d = _load("m4_compare3_T40_e8_s3.json")
    arms = [a for a in ("PC", "BP_MSE", "BP_CE") if a in d["arms"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: bulk test accuracy per arm
    xs = range(len(arms))
    axes[0].bar(xs, [d["arms"][a]["test_acc"]["mean"] * 100 for a in arms],
                yerr=[d["arms"][a]["test_acc"]["std"] * 100 for a in arms], capsize=4,
                color=[C[a] for a in arms])
    axes[0].set_xticks(list(xs))
    axes[0].set_xticklabels([LBL[a] for a in arms], fontsize=8)
    axes[0].set_ylabel("test accuracy [%]")
    axes[0].set_ylim(75, 92)
    fg = d["fair_gap_pc_mse_vs_bp_mse"] * 100
    le = d["loss_effect_bpce_minus_bpmse"] * 100
    axes[0].set_title("Bulk test accuracy (matched arch/init/data, LR per method)", fontsize=9)
    axes[0].text(0.03, 0.97, f"fair gap PC−BP(MSE) = {fg:+.1f} pp  (CIs overlap)\n"
                 f"loss effect BP(CE)−BP(MSE) = {le:+.1f} pp",
                 transform=axes[0].transAxes, va="top", ha="left", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    # right: noise robustness
    for a in arms:
        nz = d["arms"][a]["noise"]
        sig = sorted(float(s) for s in nz)
        axes[1].plot(sig, [nz[str(s) if str(s) in nz else f"{s}"]["mean"] * 100 for s in sig],
                     marker="o", color=C[a], label=LBL[a])
    axes[1].set_xlabel("input noise σ")
    axes[1].set_ylabel("test accuracy [%]")
    axes[1].set_title("Noise robustness", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("PC ≈ BP at matched loss — the apparent gaps are confounds (loss, plasticity, baseline tuning)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig2_pc_vs_bp")


def fig_launch_count():
    d = _load("m6_launch_count.json")
    rows = sorted(d["rows"], key=lambda r: r["T"])
    Ts = [r["T"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ts, [r["pytorch_launches"] for r in rows], marker="o", color="#d62728",
            label="PyTorch SO (≈31 launches/step)")
    ax.plot(Ts, [r["fused_launches"] for r in rows], marker="s", color="#1f77b4",
            label="fused kernel (1 launch, T-independent)")
    ax.set_yscale("log")
    ax.set_xticks(Ts)
    ax.set_xlabel("settling steps T")
    ax.set_ylabel("CUDA kernel launches per settle  (log)")
    ax.set_title(f"Launch-overhead mechanism (batch {d['batch']})\n"
                 f"{Ts[-1]}-step settle: {rows[-1]['pytorch_launches']} launches → 1 "
                 f"({rows[-1]['pytorch_launches']}×)")
    ax.legend(frameon=False)
    _save(fig, "fig4_launch_count")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401  (ScalarFormatter used above)
    print("writing figures to figures/ :")
    fig_kernel_speedup()
    fig_pc_vs_bp()
    fig_budget_sweep()
    fig_launch_count()
    print("done.")
