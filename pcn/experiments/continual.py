"""Continual-learning regime — Permuted-MNIST, PC vs BP (docs/13 M4, docs/10).

Lean in-house implementation: NO Avalanche dependency (avoids version conflicts with
torch 2.12+cu130 and keeps the project from-scratch). Permuted-MNIST is domain-incremental
— a fixed 10-way head throughout, each task applies a fixed input-pixel permutation, so the
PCN / BP architecture (784 -> ... -> 10) needs no head surgery and PC and BP run through the
identical task stream. Metrics are the canonical GEM definitions (Lopez-Paz & Ranzato 2017,
arXiv:1706.08840) read off the R-matrix R[i][j] = accuracy on task j after training task i:

    ACC = mean_j R[T-1][j]                              (average final accuracy)
    BWT = mean_{j<T-1} (R[T-1][j] - R[j][j])            (backward transfer; <0 = forgetting)
"""
from __future__ import annotations

import torch

from ..api import _resolve_config
from ..baselines import BPMLPRef, bp_loss_fn
from ..evaluate import evaluate
from ..learning import train_epoch
from ..model import PCN


def make_permutations(n_tasks: int, seed: int) -> list:
    """n_tasks input-pixel permutations of length 784; task 0 is the identity (plain MNIST)."""
    g = torch.Generator().manual_seed(int(seed))
    perms = [torch.arange(784)]
    for _ in range(n_tasks - 1):
        perms.append(torch.randperm(784, generator=g))
    return perms


def _permuted_loaders(data_root, perm, batch_size, limit_train, test_limit):
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda t: t.view(-1).index_select(0, perm)),  # flatten 784 then permute
    ])
    train = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    if limit_train:
        train = torch.utils.data.Subset(train, range(int(limit_train)))
    if test_limit:
        test = torch.utils.data.Subset(test, range(int(test_limit)))
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(test, batch_size=512, shuffle=False))


def make_class_split_tasks(n_tasks: int, classes_per_task: int, seed: int) -> list:
    """Partition labels 0..(n_tasks*classes_per_task-1) into n_tasks DISJOINT class sets.

    This is the class-incremental setup Song et al. 2024 use (Fig 4e: FashionMNIST split into
    two disjoint 5-class tasks, shared output head) — the interference regime where PC's
    prospective-configuration advantage is claimed, unlike the milder domain-IL Permuted-MNIST.
    """
    g = torch.Generator().manual_seed(int(seed))
    total = n_tasks * classes_per_task
    perm = torch.randperm(total, generator=g).tolist()
    return [perm[i * classes_per_task:(i + 1) * classes_per_task] for i in range(n_tasks)]


def _class_split_loaders(dataset_name, task_classes, batch_size, limit_train, test_limit, data_root):
    """Loaders for ONE class-split task: only samples whose label is in ``task_classes``,
    RELABELED to 0..K-1 so all tasks share a K-output head (forces interference)."""
    from torchvision import datasets, transforms

    DS = datasets.FashionMNIST if dataset_name == "fashion" else datasets.MNIST
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda t: t.view(-1)),
    ])
    remap = {int(c): i for i, c in enumerate(task_classes)}
    target_tf = lambda y: remap[int(y)]
    train = DS(data_root, train=True, download=True, transform=tf, target_transform=target_tf)
    test = DS(data_root, train=False, download=True, transform=tf, target_transform=target_tf)
    cls = torch.tensor(task_classes)

    def subset(ds, limit):
        idx = torch.isin(ds.targets, cls).nonzero().flatten().tolist()
        if limit:
            idx = idx[:int(limit)]
        return torch.utils.data.Subset(ds, idx)

    return (torch.utils.data.DataLoader(subset(train, limit_train), batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(subset(test, test_limit), batch_size=512, shuffle=False))


def gem_metrics(R: list) -> dict:
    """ACC, BWT and learn-accuracy from the R-matrix (Lopez-Paz & Ranzato 2017).

    ``learn_acc`` = mean of the diagonal R[j][j] = accuracy on task j right after learning it.
    Reporting it alongside BWT distinguishes "PC forgets less" from "PC learns each task less
    well" (the stability-plasticity confound): a BWT advantage is only meaningful if learn_acc
    is comparable across methods (docs/13 M4 hardening).
    """
    T = len(R)
    final = list(R[-1])
    acc = sum(final) / T
    bwt = (sum(R[-1][j] - R[j][j] for j in range(T - 1)) / (T - 1)) if T > 1 else 0.0
    learn_acc = sum(R[j][j] for j in range(T)) / T
    return {"acc": acc, "bwt": bwt, "learn_acc": learn_acc, "final_per_task": final}


@torch.no_grad()
def _eval_bp(model, loader, device, dtype) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.reshape(x.size(0), -1).to(device, dtype)
        correct += int((model(x).argmax(dim=1).cpu() == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


def run_permuted_mnist(method: str, config: dict | None = None, n_tasks: int = 5,
                       epochs_per_task: int = 3, perm_seed: int = 0,
                       test_limit: int = 2000) -> dict:
    """Train ``method`` ('pc' or 'bp') sequentially over n_tasks permuted-MNIST tasks,
    evaluating on every task after each, and return {R, acc, bwt, final_per_task}.

    Same architecture/init/data for both arms (BPMLPRef clones the PCN init) — the learning
    rule is the only variable. ``test_limit`` caps the per-task eval set (PC eval = a settle,
    so the full 10k x n_tasks^2 would be costly).
    """
    cfg = _resolve_config(config)
    torch.manual_seed(int(cfg["seed"]))
    device, dtype = cfg["device"], torch.float32
    sizes = [784, *cfg["hidden"], 10]

    perms = make_permutations(n_tasks, perm_seed)  # CPU tensors (used in the data transform)
    loaders = [_permuted_loaders(cfg["data_root"], p, cfg["batch_size"],
                                 cfg["limit_train"], test_limit) for p in perms]

    if method == "pc":
        model = PCN(sizes, activation=cfg["activation"], weight_init=cfg["weight_init"],
                    device=device, seed=int(cfg["seed"]))
    elif method == "bp":
        model = BPMLPRef(sizes, activation=cfg["activation"], weight_init=cfg["weight_init"],
                         device=device, dtype=dtype, seed=int(cfg["seed"])).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=float(cfg["lr_weight"]))
        loss_fn = bp_loss_fn(cfg["bp_loss"])
    else:
        raise ValueError(f"method must be 'pc' or 'bp', got {method!r}")

    R = [[0.0] * n_tasks for _ in range(n_tasks)]
    for t in range(n_tasks):
        train_loader = loaders[t][0]
        for _ in range(epochs_per_task):
            if method == "pc":
                train_epoch(model, train_loader, cfg["T"], cfg["lr_state"], cfg["lr_weight"],
                            tol=cfg["tol"])
            else:
                model.train()
                for x, y in train_loader:
                    x = x.reshape(x.size(0), -1).to(device, dtype)
                    y = y.to(device)
                    opt.zero_grad()
                    loss_fn(model(x), y).backward()
                    opt.step()
        for j in range(n_tasks):
            test_loader = loaders[j][1]
            R[t][j] = (evaluate(model, test_loader, cfg["T"], cfg["lr_state"])
                       if method == "pc" else _eval_bp(model, test_loader, device, dtype))

    return {"R": R, "method": method, "n_tasks": n_tasks, "epochs_per_task": epochs_per_task,
            **gem_metrics(R)}


def _ewc_fisher(model, loader, loss_fn, device, dtype, max_batches: int = 20):
    """Diagonal empirical Fisher (Kirkpatrick et al. 2017, EWC): mean squared gradient of the
    task loss at the just-converged parameters. Weights how strongly the EWC penalty protects
    each parameter on later tasks. Needs autograd (this is a BP-side baseline, not PC)."""
    params = list(model.parameters())
    fisher = [torch.zeros_like(p) for p in params]
    model.eval()
    nb = 0
    for x, y in loader:
        if nb >= max_batches:
            break
        x = x.reshape(x.size(0), -1).to(device, dtype)
        y = y.to(device)
        model.zero_grad()
        loss_fn(model(x), y).backward()
        for f, p in zip(fisher, params):
            if p.grad is not None:
                f += p.grad.detach() ** 2
        nb += 1
    return [f / max(nb, 1) for f in fisher]


def _ewc_penalty(model, ewc_tasks, lam: float):
    """EWC quadratic anchor (lam/2) * sum_tasks sum_i F_i (theta_i - theta*_i)^2. None if no task
    has been consolidated yet (so the first task trains unpenalised)."""
    if not ewc_tasks:
        return None
    params = list(model.parameters())
    pen = params[0].new_zeros(())
    for star, fisher in ewc_tasks:
        for p, s, f in zip(params, star, fisher):
            pen = pen + (f * (p - s) ** 2).sum()
    return 0.5 * lam * pen


def run_class_il(method: str, config: dict | None = None, dataset: str = "fashion",
                 n_tasks: int = 2, classes_per_task: int = 5, epochs_per_task: int = 5,
                 split_seed: int = 0, test_limit: int = 2000, ewc_lambda: float = 1000.0) -> dict:
    """Class-incremental continual learning (Song et al. 2024 regime): n_tasks DISJOINT class
    sets sharing a ``classes_per_task``-output head, trained sequentially. Returns the same
    {R, acc, bwt, learn_acc, final_per_task} as run_permuted_mnist.

    Defaults reproduce Song's Fig-4e shape: dataset='fashion', n_tasks=2, classes_per_task=5
    (FashionMNIST, two disjoint 5-class tasks, shared 5-way head). dataset='mnist',
    n_tasks=5, classes_per_task=2 gives Split-MNIST (docs/13 M2).
    """
    cfg = _resolve_config(config)
    torch.manual_seed(int(cfg["seed"]))
    device, dtype = cfg["device"], torch.float32
    K = classes_per_task
    sizes = [784, *cfg["hidden"], K]

    tasks = make_class_split_tasks(n_tasks, classes_per_task, split_seed)
    loaders = [_class_split_loaders(dataset, tc, cfg["batch_size"], cfg["limit_train"],
                                    test_limit, cfg["data_root"]) for tc in tasks]

    if method == "pc":
        model = PCN(sizes, activation=cfg["activation"], weight_init=cfg["weight_init"],
                    device=device, seed=int(cfg["seed"]))
    elif method in ("bp", "ewc"):
        model = BPMLPRef(sizes, activation=cfg["activation"], weight_init=cfg["weight_init"],
                         device=device, dtype=dtype, seed=int(cfg["seed"])).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=float(cfg["lr_weight"]))
        loss_fn = bp_loss_fn(cfg["bp_loss"], num_classes=K)
        ewc_tasks = []   # (star_params, fisher_diag) per consolidated task — EWC baseline only
    else:
        raise ValueError(f"method must be 'pc', 'bp' or 'ewc', got {method!r}")

    R = [[0.0] * n_tasks for _ in range(n_tasks)]
    for t in range(n_tasks):
        train_loader = loaders[t][0]
        for _ in range(epochs_per_task):
            if method == "pc":
                train_epoch(model, train_loader, cfg["T"], cfg["lr_state"], cfg["lr_weight"],
                            num_classes=K, tol=cfg["tol"])
            else:
                model.train()
                for x, y in train_loader:
                    x = x.reshape(x.size(0), -1).to(device, dtype)
                    y = y.to(device)
                    opt.zero_grad()
                    loss = loss_fn(model(x), y)
                    if method == "ewc":   # quadratic anchor to earlier tasks' important weights
                        pen = _ewc_penalty(model, ewc_tasks, ewc_lambda)
                        if pen is not None:
                            loss = loss + pen
                    loss.backward()
                    opt.step()
        if method == "ewc":   # consolidate task t: snapshot optimal params + diagonal Fisher
            star = [p.detach().clone() for p in model.parameters()]
            ewc_tasks.append((star, _ewc_fisher(model, train_loader, loss_fn, device, dtype)))
        for j in range(n_tasks):
            test_loader = loaders[j][1]
            R[t][j] = (evaluate(model, test_loader, cfg["T"], cfg["lr_state"])
                       if method == "pc" else _eval_bp(model, test_loader, device, dtype))

    return {"R": R, "method": method, "dataset": dataset, "n_tasks": n_tasks,
            "classes_per_task": K, "epochs_per_task": epochs_per_task,
            "ewc_lambda": ewc_lambda if method == "ewc" else None, **gem_metrics(R)}


def run_alternating(method: str, config: dict | None = None, dataset: str = "fashion",
                    classes_per_task: int = 5, total_iters: int = 84, swap_every: int = 4,
                    eval_stride: int = 1, split_seed: int = 0, test_limit: int = 2000) -> dict:
    """Song et al. 2024 (Fig 4d-e) EXACT continual-learning protocol. Two DISJOINT-class tasks
    share a single K-output head and are trained by alternating at the MINIBATCH level: do
    ``swap_every`` minibatch updates on the current task, then switch, for ``total_iters`` total
    updates. Song's Methods: batch size 32, swap every 4 iterations, 84 iterations total — pass
    config['batch_size']=32 to match. (Our earlier proxy alternated every full EPOCH, ~20x more
    updates; this version is the faithful schedule — see docs/12 §4g for the deviation analysis.)

    Metrics (all measured on held-out test sets, both tasks evaluated after every visit):
      * mean_both  = mean of the two tasks' final test accuracy  -> Song's Fig-4e metric
                     (Fig 4e plots mean test ERROR of both tasks vs LR). Chance = 1/K.
      * min_both   = the WORSE task's final accuracy -> exposes "balances both" vs "sacrifices one"
                     (a high mean_both with low min_both is NOT low-interference learning).
      * learn_acc  = mean accuracy of the just-trained task measured right after its visit
      * retain_acc = mean accuracy of the OTHER (stale) task at the same moments
                     (learn vs retain separates failure-to-learn from forgetting; docs/12 §4c).

    NB Fig 4e tunes the LR independently per method — see scripts driver, which sweeps LR per
    method. Returns {acc_traj=[(iter,cur_task,accA,accB)], final, mean_both, min_both,
    learn_acc, retain_acc, method, total_iters, swap_every}.
    """
    cfg = _resolve_config(config)
    torch.manual_seed(int(cfg["seed"]))
    device, dtype = cfg["device"], torch.float32
    K = classes_per_task
    sizes = [784, *cfg["hidden"], K]

    tasks = make_class_split_tasks(2, K, split_seed)
    loaders = [_class_split_loaders(dataset, tc, cfg["batch_size"], cfg["limit_train"],
                                    test_limit, cfg["data_root"]) for tc in tasks]

    if method == "pc":
        model = PCN(sizes, activation=cfg["activation"], weight_init=cfg["weight_init"],
                    device=device, seed=int(cfg["seed"]))
    elif method == "bp":
        model = BPMLPRef(sizes, activation=cfg["activation"], weight_init=cfg["weight_init"],
                         device=device, dtype=dtype, seed=int(cfg["seed"])).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=float(cfg["lr_weight"]))
        loss_fn = bp_loss_fn(cfg["bp_loss"], num_classes=K)
    else:
        raise ValueError(f"method must be 'pc' or 'bp', got {method!r}")

    def stream(loader):                       # endless minibatch stream over one task
        while True:
            for xb, yb in loader:
                yield xb, yb
    streams = [stream(loaders[0][0]), stream(loaders[1][0])]

    def train_batch(t):
        xb, yb = next(streams[t])
        if method == "pc":                    # reuse the tested per-batch PC step (settle+Hebb)
            train_epoch(model, [(xb, yb)], cfg["T"], cfg["lr_state"], cfg["lr_weight"],
                        num_classes=K, tol=cfg["tol"], backend=cfg["backend"])
        else:
            x = xb.reshape(xb.size(0), -1).to(device, dtype)
            y = yb.to(device)
            model.train()
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    def acc(t):
        return (evaluate(model, loaders[t][1], cfg["T"], cfg["lr_state"], backend=cfg["backend"])
                if method == "pc" else _eval_bp(model, loaders[t][1], device, dtype))

    traj, learn, retain = [], [], []
    done, t, visit = 0, 0, 0
    while done < total_iters:
        step = min(swap_every, total_iters - done)
        for _ in range(step):
            train_batch(t)
        done += step
        visit += 1
        if visit % eval_stride == 0 or done >= total_iters:   # bound eval cost at large budgets
            a0, a1 = acc(0), acc(1)
            traj.append((done, t, a0, a1))
            learn.append(a1 if t == 1 else a0)    # task just trained
            retain.append(a0 if t == 1 else a1)   # other (stale) task
        t = 1 - t                                 # switch task
    fa, fb = traj[-1][2], traj[-1][3]
    return {"acc_traj": traj, "final": [fa, fb], "mean_both": (fa + fb) / 2,
            "min_both": min(fa, fb), "learn_acc": sum(learn) / len(learn),
            "retain_acc": sum(retain) / len(retain), "method": method,
            "total_iters": total_iters, "swap_every": swap_every}
