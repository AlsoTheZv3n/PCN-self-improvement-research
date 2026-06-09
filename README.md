# pcn-from-scratch

From-scratch **Predictive Coding Network** (State-Optimization-Formulierung) auf MNIST in
reinem PyTorch — eine biologisch plausible Backprop-Alternative mit **lokalen** Lernregeln.
Optional: ein **CUDA-Settling-Kernel** (Phase 3) und ein autonomer Such-Loop (Phase 4).

## Schnellstart

```bash
uv sync
uv run python scripts/train_mnist.py        # MNIST-Lauf, PyTorch-Backend
uv sync --extra dev && uv run pytest -q      # Settling-Korrektheit (offline, synthetisch)
```

## Was hier drin ist

- `pcn/` — das PCN: Modell, Settling-Loop, lokale Weight-Updates, `train_and_eval`-Schnittstelle
- `pcn/kernels/` — **optionaler** fused CUDA-Settling-Kernel (`settling_kernel.cu`, gebaut +
  correctness-verifiziert ~1e-6; bewusst via `PCN_CUDA_KERNEL=1` / `backend="cuda"` zuschaltbar)
- `docs/` — vollständige Projekt-Doku (`00_overview.md` ist der Einstieg) inkl. exaktem
  Algorithmus (`09`), Methodik (`10`) und allen Quellen (`11`)
- `CLAUDE.md` — Arbeitsanweisungen, Konventionen und Lese-Reihenfolge für Claude Code

## Wichtigste Regeln

- `uv` ausschließlich (kein `pip`).
- PyTorch nur als Tensor-Engine — **kein** Autograd fürs PC-Lernen; alle Updates manuell.
- Der CUDA-Kernel ist **optional** und Default-`backend` ist `"pytorch"`.

Details und Hintergrund: **`CLAUDE.md`** und **`docs/00_overview.md`**.
