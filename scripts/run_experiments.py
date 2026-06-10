"""M4 — fair PC-vs-BP experiments (docs/13 M4, docs/08, docs/10).

Runs the fair-comparison protocol (LR tuned per method over a shared grid, >= 3 seeds,
bootstrap CIs) and writes both a readable report and a JSON artifact under results/.

Usage (GPU strongly recommended — set nothing, device auto-selects cuda if available):
    uv run python scripts/run_experiments.py compare   --seeds 3 --epochs 8 --T 40
    uv run python scripts/run_experiments.py settling  --seeds 3 --epochs 8
    uv run python scripts/run_experiments.py smalldata --seeds 3 --epochs 15
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from pcn.api import train_and_eval
from pcn.baselines import train_and_eval_bp
from pcn.experiments.protocol import (
    bootstrap_ci,
    run_multiseed,
    agg_scalar,
    sweep_grid,
    best_of_sweep,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
# PC: joint eta_x x eta_w grid (stability frontier — must tune jointly, docs/12 M4).
PC_GRID = {"eta_x": [0.05, 0.1], "eta_w": [0.02, 0.05]}
BP_GRID = {"eta_w": [0.05, 0.1, 0.3]}


def _agg_noise(results: list) -> dict:
    """Aggregate the per-seed noise_robustness dicts into {sigma: bootstrap_ci}."""
    sigmas = sorted(results[0]["noise_robustness"].keys())
    return {str(s): bootstrap_ci([r["noise_robustness"][s] for r in results]) for s in sigmas}


def _fmt(ci: dict) -> str:
    return f"{ci['mean'] * 100:6.2f}% [{ci['lo'] * 100:.2f}, {ci['hi'] * 100:.2f}]"


def _save(name: str, payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))())
    return path


def _device_banner():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"device: {dev} ({name}) | torch {torch.__version__}\n")
    return dev


def _best_test_and_noise(fn, base, best_params, seeds):
    """Re-run the winning config WITH noise+test to report final test_acc / noise CIs."""
    results = run_multiseed(fn, {**base, **best_params, "eval_noise": True}, seeds)
    return agg_scalar(results, "test_acc"), _agg_noise(results), results


def cmd_compare(args):
    """Three-arm PC vs BP — paper-grade: joint PC grid, LR selected on a VALIDATION split,
    reported on test, bootstrap CIs. Arms: PC(MSE), BP(MSE), BP(CE). The like-for-like
    comparison that isolates the LEARNING RULE is PC(MSE) vs BP(MSE); BP(CE)-BP(MSE) is the
    pure loss-function effect (M4 B1, docs/12)."""
    _device_banner()
    base = {"hidden": [256, 256], "T": args.T, "batch_size": args.batch_size,
            "epochs": args.epochs, "limit_train": args.limit_train, "tol": None,
            "eval_noise": False, "val_split": 0.1}
    seeds = list(range(args.seeds))
    t0 = time.time()

    arm_specs = [
        ("PC", train_and_eval, base, PC_GRID),
        ("BP_MSE", train_and_eval_bp, {**base, "bp_loss": "mse"}, BP_GRID),
        ("BP_CE", train_and_eval_bp, {**base, "bp_loss": "ce"}, BP_GRID),
    ]
    arms = {}
    for name, fn, arm_base, grid in arm_specs:
        sweep = sweep_grid(fn, arm_base, grid, seeds, metric="val_acc")
        best = best_of_sweep(sweep)
        test, noise, res = _best_test_and_noise(fn, arm_base, best["params"], seeds)
        arms[name] = {"best": best["params"], "test_acc": test, "noise": noise,
                      "val_sweep": [{"params": e["params"], "val_acc": e["summary"]} for e in sweep]}
        if name == "PC":
            arms[name]["steps2conv"] = agg_scalar(res, "settling_steps_to_converge")
    dt = time.time() - t0

    print(f"=== M4 compare 3-arm (val-selected) | T={args.T} epochs={args.epochs} "
          f"seeds={seeds} limit_train={args.limit_train} ({dt:.0f}s) ===")
    for name in ("PC", "BP_MSE", "BP_CE"):
        print(f"{name:7} best={str(arms[name]['best']):28} test_acc={_fmt(arms[name]['test_acc'])}")
    loss_effect = arms["BP_CE"]["test_acc"]["mean"] - arms["BP_MSE"]["test_acc"]["mean"]
    fair_gap = arms["BP_MSE"]["test_acc"]["mean"] - arms["PC"]["test_acc"]["mean"]
    print(f"\nFAIR gap  PC(MSE) vs BP(MSE): {fair_gap * 100:+.2f} pp  (learning-rule effect)")
    print(f"LOSS effect BP(CE) - BP(MSE): {loss_effect * 100:+.2f} pp  (pure loss-function effect)")
    print("\nNoise robustness (test acc ± 68% CI):")
    print(f"{'sigma':>6} | {'PC(MSE)':^22} | {'BP(MSE)':^22} | {'BP(CE)':^22}")
    for s in arms["PC"]["noise"]:
        print(f"{s:>6} | {_fmt(arms['PC']['noise'][s]):^22} | "
              f"{_fmt(arms['BP_MSE']['noise'][s]):^22} | {_fmt(arms['BP_CE']['noise'][s]):^22}")

    path = _save(f"m4_compare3_T{args.T}_e{args.epochs}_s{args.seeds}.json", {
        "kind": "compare3", "T": args.T, "epochs": args.epochs, "seeds": seeds,
        "limit_train": args.limit_train, "val_split": 0.1, "duration_s": dt,
        "fair_gap_pc_mse_vs_bp_mse": fair_gap, "loss_effect_bpce_minus_bpmse": loss_effect,
        "arms": arms,
    })
    print(f"\nsaved: {path}")


def cmd_settling(args):
    """Does properly settling the network (higher T) close the PC-vs-BP gap? (M3 caveat)."""
    _device_banner()
    seeds = list(range(args.seeds))
    print(f"=== M4 settling-T sweep | epochs={args.epochs} seeds={seeds} "
          f"limit_train={args.limit_train} ===")
    rows = []
    for T in (20, 40, 80):
        base = {"hidden": [256, 256], "T": T, "eta_x": 0.1, "eta_w": 0.02,
                "batch_size": args.batch_size,
                "epochs": args.epochs, "limit_train": args.limit_train, "tol": None}
        results = run_multiseed(train_and_eval, base, seeds)
        acc = agg_scalar(results, "test_acc")
        steps = agg_scalar(results, "settling_steps_to_converge")
        noise = _agg_noise(results)
        print(f"T={T:>3}  acc={_fmt(acc)}  steps2conv~{steps['mean']:.0f}  "
              f"noise@1.0={_fmt(noise['1.0'])}")
        rows.append({"T": T, "acc": acc, "steps2conv": steps, "noise": noise})
    path = _save(f"m4_settling_e{args.epochs}_s{args.seeds}.json",
                 {"kind": "settling", "epochs": args.epochs, "seeds": seeds, "rows": rows})
    print(f"\nsaved: {path}")


def cmd_smalldata(args):
    """Sample efficiency: PC vs BP as training-set size shrinks (Song et al. regime)."""
    _device_banner()
    seeds = list(range(args.seeds))
    print(f"=== M4 small-data | epochs={args.epochs} seeds={seeds} T={args.T} ===")
    rows = []
    for n in (100, 1000, 10000):
        base = {"hidden": [256, 256], "T": args.T, "eta_x": 0.1,
                "batch_size": args.batch_size,
                "epochs": args.epochs, "limit_train": n, "tol": None}
        base = {**base, "eval_noise": False}  # sample-efficiency focus: accuracy only
        pc = run_multiseed(train_and_eval, {**base, "eta_w": 0.02}, seeds)
        bp = run_multiseed(train_and_eval_bp, {**base, "eta_w": 0.1}, seeds)
        pc_acc, bp_acc = agg_scalar(pc, "test_acc"), agg_scalar(bp, "test_acc")
        print(f"n={n:>6}  PC acc={_fmt(pc_acc)}   BP acc={_fmt(bp_acc)}")
        rows.append({"n": n, "pc_acc": pc_acc, "bp_acc": bp_acc})
    path = _save(f"m4_smalldata_e{args.epochs}_s{args.seeds}.json",
                 {"kind": "smalldata", "epochs": args.epochs, "seeds": seeds, "T": args.T, "rows": rows})
    print(f"\nsaved: {path}")


def cmd_continual(args):
    """Permuted-MNIST continual learning: ACC + BWT (forgetting) for PC vs BP (Song regime)."""
    _device_banner()
    from pcn.experiments.continual import run_permuted_mnist

    seeds = list(range(args.seeds))
    base = {"hidden": [256, 256], "T": args.T, "eta_x": 0.1, "batch_size": args.batch_size,
            "limit_train": args.limit_train, "tol": None, "eval_noise": False}
    print(f"=== M4 continual (Permuted-MNIST) | n_tasks={args.n_tasks} "
          f"epochs/task={args.epochs} seeds={seeds} limit_train={args.limit_train} ===")
    out = {}
    # Three arms isolate learning-rule from loss (B1); learn_acc + retained-task0 deconfound
    # "forgets less" from "learns less" (B2). PC at the jointly-tuned optimum (eta_x=eta_w=0.05).
    arms = [("PC", "pc", 0.05, 0.05, "ce"),     # bp_loss unused for PC
            ("BP_MSE", "bp", 0.1, 0.1, "mse"),  # the like-for-like BP arm (same objective as PC)
            ("BP_CE", "bp", 0.1, 0.1, "ce")]    # practical BP reference
    for name, method, eta_x, eta_w, bp_loss in arms:
        accs, bwts, learns, retained0 = [], [], [], []
        for s in seeds:
            # vary the permutation set per seed too, so the result isn't tied to one permutation
            r = run_permuted_mnist(method, {**base, "eta_x": eta_x, "eta_w": eta_w,
                                            "bp_loss": bp_loss, "seed": s},
                                   n_tasks=args.n_tasks, epochs_per_task=args.epochs, perm_seed=s)
            accs.append(r["acc"]); bwts.append(r["bwt"]); learns.append(r["learn_acc"])
            retained0.append(r["final_per_task"][0])  # retained task-0 accuracy after all tasks
        out[name] = {"final_acc": bootstrap_ci(accs), "learn_acc": bootstrap_ci(learns),
                     "bwt": bootstrap_ci(bwts), "retained_task0": bootstrap_ci(retained0)}
        c = out[name]
        print(f"{name:7} final-ACC={_fmt(c['final_acc'])}  learn-ACC={_fmt(c['learn_acc'])}  "
              f"BWT={c['bwt']['mean'] * 100:+.2f}%  retained-T0={_fmt(c['retained_task0'])}")

    # Decision rule (vs the fair BP(MSE) arm): "PC forgets less" is only defensible if PC also
    # learns each task comparably (learn_acc CI overlaps/exceeds BP) AND retains task-0 better.
    pc, bp = out["PC"], out["BP_MSE"]
    learn_ok = pc["learn_acc"]["mean"] >= bp["learn_acc"]["lo"]
    retain_ok = pc["retained_task0"]["mean"] >= bp["retained_task0"]["mean"]
    print(f"\nHeadline 'PC forgets less' vs BP(MSE) defensible? "
          f"learn_acc_ok={learn_ok}  retained_task0_ok={retain_ok}  "
          f"=> {'YES' if (learn_ok and retain_ok) else 'NOT proven (plasticity confound)'}")

    path = _save(f"m4_continual3_t{args.n_tasks}_e{args.epochs}_s{args.seeds}.json",
                 {"kind": "continual3", "n_tasks": args.n_tasks, "epochs_per_task": args.epochs,
                  "seeds": seeds, "arms": out,
                  "headline_defensible": bool(learn_ok and retain_ok)})
    print(f"\nsaved: {path}")


def cmd_classil(args):
    """Class-incremental CL (Song et al. 2024 regime): disjoint-class tasks sharing a K-output
    head -> the interference regime where PC's advantage is claimed. 3 arms + deconfound."""
    _device_banner()
    from pcn.experiments.continual import run_class_il

    seeds = list(range(args.seeds))
    base = {"hidden": [256, 256], "T": args.T, "batch_size": args.batch_size,
            "limit_train": args.limit_train, "tol": None, "eval_noise": False}

    def run_arm(method, eta_x, eta_w, bp_loss, sd):
        return run_class_il(method, {**base, "eta_x": eta_x, "eta_w": eta_w,
                                     "bp_loss": bp_loss, "seed": sd},
                            dataset=args.dataset, n_tasks=args.n_tasks,
                            classes_per_task=args.classes_per_task,
                            epochs_per_task=args.epochs, split_seed=sd)

    # B3 fix: tune each BP arm's eta_w by best LEARN-accuracy on seed 0, so BP actually learns
    # the tasks (else a stuck BP at chance makes the retention comparison meaningless). PC is
    # fixed at its known-good joint optimum (it already learns well; learn-ACC >> chance).
    BP_W_GRID = [0.01, 0.02, 0.05, 0.1, 0.3]

    def tune_bp_lr(bp_loss):
        scan = [(ew, run_arm("bp", 0.1, ew, bp_loss, 0)["learn_acc"]) for ew in BP_W_GRID]
        return max(scan, key=lambda t: t[1])[0], scan

    print(f"=== M4 class-IL (BP-LR tuned by learn-ACC) | dataset={args.dataset} "
          f"n_tasks={args.n_tasks} K={args.classes_per_task} epochs/task={args.epochs} seeds={seeds} ===")
    bw_mse, scan_mse = tune_bp_lr("mse")
    bw_ce, scan_ce = tune_bp_lr("ce")
    print(f"tuned BP eta_w (by learn-ACC@seed0): MSE={bw_mse} CE={bw_ce}")

    chosen = {"PC": ("pc", 0.05, 0.05, "ce"),
              "BP_MSE": ("bp", 0.1, bw_mse, "mse"),
              "BP_CE": ("bp", 0.1, bw_ce, "ce"),
              "EWC": ("ewc", 0.1, bw_mse, "mse")}  # = BP(MSE) + Fisher anchor (isolates EWC effect)
    out = {}
    for name, (method, eta_x, eta_w, bp_loss) in chosen.items():
        rs = [run_arm(method, eta_x, eta_w, bp_loss, s) for s in seeds]
        out[name] = {"final_acc": bootstrap_ci([r["acc"] for r in rs]),
                     "learn_acc": bootstrap_ci([r["learn_acc"] for r in rs]),
                     "bwt": bootstrap_ci([r["bwt"] for r in rs]),
                     "retained_task0": bootstrap_ci([r["final_per_task"][0] for r in rs]),
                     "eta_w": eta_w}
        c = out[name]
        print(f"{name:7} (eta_w={eta_w}) final-ACC={_fmt(c['final_acc'])}  "
              f"learn-ACC={_fmt(c['learn_acc'])}  BWT={c['bwt']['mean'] * 100:+.2f}%  "
              f"retained-T0={_fmt(c['retained_task0'])}")

    pc, bp = out["PC"], out["BP_MSE"]
    bp_learned = bp["learn_acc"]["mean"] > (1.0 / args.classes_per_task + 0.10)  # clearly above chance
    learn_ok = pc["learn_acc"]["mean"] >= bp["learn_acc"]["lo"]
    retain_ok = pc["retained_task0"]["mean"] > bp["retained_task0"]["hi"]  # strictly, CI-disjoint
    defensible = bool(bp_learned and learn_ok and retain_ok)
    print(f"\n'PC retains more' vs (tuned, learning) BP(MSE)? bp_learned={bp_learned} "
          f"learn_ok={learn_ok} retain_ok(CI-disjoint)={retain_ok} => {'YES' if defensible else 'NO'}")

    path = _save(f"m4_classil_{args.dataset}_t{args.n_tasks}k{args.classes_per_task}_s{args.seeds}.json",
                 {"kind": "classil", "dataset": args.dataset, "n_tasks": args.n_tasks,
                  "classes_per_task": args.classes_per_task, "epochs_per_task": args.epochs,
                  "seeds": seeds, "tuned_bp_eta_w": {"mse": bw_mse, "ce": bw_ce},
                  "bp_lr_scan": {"mse": scan_mse, "ce": scan_ce},
                  "arms": out, "headline_defensible": defensible})
    print(f"\nsaved: {path}")


def cmd_alternating(args):
    """Song et al. 2024 (Fig 4d-e) EXACT continual-learning replication: two disjoint 5-class
    FashionMNIST tasks sharing one 5-output head, MINIBATCH-level alternation (swap every 4 iters,
    ~84 total, batch 32), sigmoid [32,32] Xavier — the verified-faithful architecture.

    Honesty contract (per the §4g adversarial review, docs/14):
      * Fig 4e's claim is LESS CATASTROPHIC INTERFERENCE (avoid forgetting + relearn), NOT
        learning-rate robustness (that is Fig 3). LR is tuned PER METHOD (Song's Methods), so we
        sweep LR and report each method at its OWN best LR.
      * mean_both is Song's Fig-4e metric, but min_both / learn / retain are what actually show
        interference -> all reported.
      * PC vs BP-MSE is the loss-matched (honest) comparison; BP-CE is a context row (loss confound).
      * Intervals are 68% (1-sigma) bootstrap over n seeds; per-seed values are saved. With small n
        these are DIRECTIONAL, not significance — the writeup must say so."""
    _device_banner()
    from pcn.experiments.continual import run_alternating

    seeds = list(range(args.seeds))
    base = {"hidden": [32, 32], "activation": "sigmoid", "weight_init": "xavier", "T": args.T,
            "eta_x": 0.05, "batch_size": 32, "limit_train": 5000, "tol": None, "eval_noise": False,
            "backend": args.backend}    # 'cuda' uses the fused settling kernel (PC arms only)
    BUDGETS = [84, 250, 800, 2500]    # Song's exact 84 (untrainable) + growth -> convergence
    LR = [0.05, 0.1]                  # diagnostic bracket of the per-method optimum
    arms = [("PC", "pc", "ce"), ("BP_MSE", "bp", "mse"), ("BP_CE", "bp", "ce")]
    print(f"=== M4 alternating (Song Fig-4d/e EXACT, BUDGET sweep) | Split-FashionMNIST 2x5, "
          f"sigmoid [32,32] xavier, batch=32, swap_every={args.swap_every}, seeds={seeds} ===")
    print("Fig-4e metric=mean test acc of both tasks; min_both=worse task (the interference tell); "
          f"LR tuned per method; 68% (1σ), n={len(seeds)} -> DIRECTIONAL, not significance.")
    print("Song's exact budget=84 -> expect ~chance (20%); the interference regime is intermediate.\n")

    def agg(runs, key):
        vals = [r[key] for r in runs]
        return {**bootstrap_ci(vals), "seeds": vals}

    out = {}
    for budget in BUDGETS:
        evs = max(1, budget // (args.swap_every * 20))     # ~20 eval checkpoints, cost-bounded
        out[str(budget)] = {}
        row = []
        for name, method, bp_loss in arms:
            best, best_lr = None, None
            for lr in LR:
                runs = [run_alternating(method, {**base, "eta_w": lr, "bp_loss": bp_loss, "seed": s},
                                        dataset="fashion", classes_per_task=5, total_iters=budget,
                                        swap_every=args.swap_every, eval_stride=evs, split_seed=s)
                        for s in seeds]
                rec = {"lr": lr, "mean_both": agg(runs, "mean_both"), "min_both": agg(runs, "min_both"),
                       "learn": agg(runs, "learn_acc"), "retain": agg(runs, "retain_acc")}
                if best is None or rec["mean_both"]["mean"] > best["mean_both"]["mean"]:
                    best, best_lr = rec, lr
            out[str(budget)][name] = best
            row.append(f"{name}(lr{best_lr}): mean{best['mean_both']['mean'] * 100:4.1f} "
                       f"min{best['min_both']['mean'] * 100:4.1f}")
        print(f"  budget={budget:5d} | " + " | ".join(row))

    print("\ninterference tell = PC.min_both - BP-MSE.min_both (positive = PC balances both tasks better):")
    gaps = []
    for budget in BUDGETS:
        pc, bm = out[str(budget)]["PC"], out[str(budget)]["BP_MSE"]
        g = (pc["min_both"]["mean"] - bm["min_both"]["mean"]) * 100
        gaps.append((budget, g))
        print(f"  budget={budget:5d}: Δmin_both={g:+5.1f}pp  (PC {pc['min_both']['mean'] * 100:.1f}% "
              f"vs {bm['min_both']['mean'] * 100:.1f}%; PC per-seed min {[round(v*100) for v in pc['min_both']['seeds']]})")
    def sigsum(b):   # sum of PC + BP-MSE min_both 1σ at budget b
        return (out[str(b)]["PC"]["min_both"]["std"] + out[str(b)]["BP_MSE"]["min_both"]["std"]) * 100
    any_sep = any(abs(g) > sigsum(b) for b, g in gaps)
    conv_b, conv_g = gaps[-1]
    conv_sep = abs(conv_g) > sigsum(conv_b)
    print(f"\nConvergence (budget={conv_b}): PC vs BP-MSE Δmin_both={conv_g:+.1f}pp, σ_sum={sigsum(conv_b):.1f}pp -> "
          f"{'separated >1σ' if conv_sep else 'WITHIN 1σ (directional only, NOT significant)'}.")
    print(f"Across all budgets, any >1σ PC-vs-BP-MSE separation? {any_sep} (n={len(seeds)}).")
    print("BP-CE = different objective (loss confound, docs/12 §4c) — context column, not the rule comparison.")
    path = _save(f"m4_alternating_song_exact_budgetsweep_s{args.seeds}.json",
                 {"kind": "alternating_song_exact_budget", "config": base, "budgets": BUDGETS,
                  "lr_grid": LR, "swap_every": args.swap_every, "seeds": seeds, "arms_by_budget": out,
                  "min_both_gap_pp": gaps})
    print(f"\nsaved: {path}")


def main():
    p = argparse.ArgumentParser(description="M4 fair PC-vs-BP experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "settling", "smalldata", "continual", "classil", "alternating"):
        sp = sub.add_parser(name)
        sp.add_argument("--seeds", type=int, default=3)
        sp.add_argument("--epochs", type=int, default=8)
        sp.add_argument("--limit-train", dest="limit_train", type=int, default=None)
        sp.add_argument("--batch-size", dest="batch_size", type=int, default=256)
        sp.add_argument("--T", type=int, default=40)
        sp.add_argument("--n-tasks", dest="n_tasks", type=int, default=5)
        sp.add_argument("--dataset", default="fashion", choices=["fashion", "mnist"])
        sp.add_argument("--classes-per-task", dest="classes_per_task", type=int, default=5)
        sp.add_argument("--total-iters", dest="total_iters", type=int, default=84,
                        help="alternating: total minibatch updates (Song Fig-4e: 84)")
        sp.add_argument("--swap-every", dest="swap_every", type=int, default=4,
                        help="alternating: minibatch updates per task visit before switching (Song: 4)")
        sp.add_argument("--backend", default="pytorch", choices=["pytorch", "cuda"],
                        help="alternating PC arms: 'cuda' uses the fused settling kernel "
                             "(needs PCN_CUDA_KERNEL=1 + vcvars build); 'pytorch' is the default")
    args = p.parse_args()
    {"compare": cmd_compare, "settling": cmd_settling, "smalldata": cmd_smalldata,
     "continual": cmd_continual, "classil": cmd_classil,
     "alternating": cmd_alternating}[args.cmd](args)


if __name__ == "__main__":
    main()
