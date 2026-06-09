# CLAUDE.md — Arbeitsanweisungen für Claude Code

From-scratch **Predictive Coding Network** (State-Optimization-Formulierung) auf MNIST,
mit optionalem CUDA-Settling-Kernel. Ziel: eine Backprop-Alternative *verstehen und
demonstrieren* — nicht "noch ein Finetune". Begleitendes Paper im NeurIPS-Stil.

## Harte Konventionen (nicht abweichen)

- **`uv` ausschließlich** — niemals `pip`. (`uv sync`, `uv run …`, `uv add …`.)
- **PyTorch nur als Tensor-Engine.** KEIN `loss.backward()` / Autograd für das PC-Lernen.
  Alle State- und Weight-Updates manuell unter `@torch.no_grad()`. Das ist der ganze Punkt
  des Projekts (lokales Lernen statt globalem Backprop).
- **Dokumentation auf Deutsch (Hochdeutsch)**, Code-Bezeichner/Kommentare auf Englisch.
- Settling-Mathematik exakt nach `docs/09` (Vorzeichen + `phi'`-Ableitung). Sie ist gegen
  eine analytische NumPy-Referenz validiert — nicht "vereinfachen".

## Phasenplan & Status

| Phase | Inhalt | Status |
|-------|--------|--------|
| 1 | PyTorch-SO-PCN, Settling-Loop, `train_and_eval` | **fertig + validiert** (M0–M1; Autograd-Gradient-Checks) |
| 2 | Demos: Klassifikation, generativ, Occlusion, Anomalie, MLP-Baseline | **generativ funktioniert** (M5-v2, `docs/12` §4f): Ziffern-Prototypen + Anomalie-AUC 1,0 (`pcn/generative.py`); Notebooks offen |
| 3 | **CUDA-Settling-Kernel (OPTIONAL)** + Benchmark | **gebaut v1+v2+v3** (`pcn/kernels/`), correctness-verifiziert (tanh/identity/**sigmoid**, ~1e-6): 3,2× @ B=64, ~2,0× @ B=256, break-even @ B=1024; B≥2048 → cuBLAS. **In §4g-Studie integriert** (`backend="cuda"`, 1,45× end-to-end, Per-Seed identisch); M6, `docs/12` §4d/§4g |
| 4 | Autonomer Such-Loop über `train_and_eval` | **umgesetzt** (M7): `pcn/search.py` grid/random (+Optuna optional); Phase-4-Lauf rediscovert η_x↔η_w |
| — | PC-vs-BP-Studie (Hook B) | **fertig** (M4): PC ≈ BP, 3 Confounds entlarvt — `docs/12` §4c, `docs/10` §9. **Song-treue alternierende Replikation** (`docs/12` §4g, n=5): keine ≥1σ-Differenz auf irgendeinem Budget; PC früh langsamer, bei Konvergenz marginal vorn (im Rauschen) |
| — | Paper (NeurIPS-Stil, EN) | **Draft fertig** (M7): `docs/05` synthetisiert A+B+C + Limitations + korrigierte Refs |

**Aktueller Stand & Plan:** `docs/13_umsetzungsplan.md` (M0–M7), Befunde `docs/12`, Methodik-Review `docs/14`.

**Reihenfolge-Regel:** Erst Phase 1 lauffähig + korrekt (Energie fällt, Accuracy
plausibel), dann Phase 2. Phase 3 ist **optional** und wird **bewusst** zugeschaltet
(`backend="cuda"`), niemals als Default. Phase 4 dockt nur an `pcn/api.py:train_and_eval`
an — kein Rewrite.

## Der Kernel ist optional (wichtig)

Default ist `backend="pytorch"` und voll funktionsfähig. Der CUDA-Pfad in `pcn/kernels/`
ist **gebaut (v1+v2+v3)** und correctness-verifiziert, wird aber **bewusst** zugeschaltet:
`kernels.is_available()` ist `True` nur mit Env-Flag **`PCN_CUDA_KERNEL=1`** + CUDA-Device,
sonst `False` → `backend="cuda"` wirft `NotImplementedError` (kein stilles Fallback;
Default-/Test-Pfad bleibt PyTorch). **Die CUDA-Quelle ist `pcn/kernels/settling_kernel.cu`**
(echte `.cu`, von `settling_cuda.py` via `load(sources=[...])` kompiliert — KEIN auskommentierter
Stub, KEIN Inline-String). Build braucht nvcc + MSVC (vcvars-Shell) + toolkit-gematchtes
CUDA-torch (cu126); Korrektheits-Gate `scripts/verify_kernel.py` (tanh/identity/sigmoid, ~1e-6).
Ergebnis: 3–4× @ batch 64 (Launch-Overhead-Win), Crossover zu cuBLAS bei großer Batch. **In die
§4g-Studie integriert** (`run_alternating(..., backend="cuda")`, 1,45× end-to-end, Per-Seed
identisch). Design: `docs/07`/`docs/09`; Befund `docs/12` §4d/§4g. **Ohne** Kernel null Low-Level-Code.

## Wo nachschlagen (docs/)

- `00_overview.md` — Gesamtüberblick, These, Datei-Index
- `01_architecture.md` — PCN-Gleichungen, Clamping
- `02_build_data_training.md` — Stack, Daten, `train_and_eval`-Vertrag
- `03_testing_and_interaction.md` — Jupyter-Demos (Phase 2)
- `04_autonomous_search_loop.md` — Phase-4-Such-Loop (ehrliches Scoping)
- `05_paper_outline.md` — Paper-Struktur, Novelty-Hooks
- `06_open_problems_and_approaches.md` — offene Probleme des Felds
- `07_cuda_kernel_build.md` — CUDA-Build-Pfad + Quellen
- `08_bio_regime_experiments.md` — PC-vs-BP-Experimente
- `09_kernel_pc_dynamics_deepdive.md` — **exakter SO-Algorithmus**, Signal-Decay, EO, Fusion
- `10_pc_vs_bp_deepdive.md` — Prospective Configuration, faire Vergleichsmethodik (§9 = M4-Ergebnis)
- `11_quellen.md` — alle Original-Quellen-URLs (+ verifizierter Quellen-Status)
- `12_projektanalyse_und_befunde.md` — **Befunde + alle Experiment-Ergebnisse** (M0–M6)
- `13_umsetzungsplan.md` — **Umsetzungsplan M0–M7** (Status pro Milestone)
- `14_m4_methodik_review.md` — adversarialer Methodik-Review (Gaps-Report)
- `15_kernel_novelty.md` — **Kernel-Neuheits-Verdikt** (verifiziert): verteidigbare Formulierung für Hook A

## Code-Karte (pcn/)

- `model.py` — `PCN`: Gewichte/Bias/Aktivierung, `predict(i, state)`
- `settling.py` — `feedforward_init`, `settle(...)` (Backend-Dispatch) + PyTorch-Impl
- `learning.py` — `weight_update` (lokal, Hebb), `train_epoch`
- `evaluate.py` — `evaluate`, `noise_robustness`
- `api.py` — **`train_and_eval(config) -> metrics`** (Phase-4-Schnittstelle, lädt MNIST; eta_x/eta_w-Aliase, val_split, bp_loss)
- `baselines.py` — BP-MLP-Referenz (klont PCN-Init, gleiche Forward-Fn); `train_and_eval_bp`, `bp_loss_fn` (ce/mse)
- `benchmark.py` — `time_settle`, `benchmark_pc_vs_bp` (CUDA-Events)
- `generate.py` — Hook C diskriminativ: `generate`/`inpaint`/`anomaly_scores` (liefert Rauschen — docs/12 §4e)
- `generative.py` — **Hook C generativ** [10,h,h,784]: `train_generative`, `generate_*`, `inpaint_generative`, `anomaly_scores_generative` (funktioniert — §4f)
- `search.py` — **Phase 4**: `grid_search`/`random_search`/`bayesian_search` (Optuna optional) über `train_and_eval`
- `experiments/` — `protocol.py` (fair-comparison: bootstrap-CI, sweep_grid), `continual.py` (Permuted/Split-MNIST, GEM-Metriken)
- `kernels/` — **gebaut v1+v2+v3** (`settling_cuda.py`): fused CUDA-Settling, hybrid Dispatch; bewusst per `PCN_CUDA_KERNEL=1`

## Befehle

```bash
uv sync                              # Basis-Umgebung (torch, torchvision)
uv run python scripts/train_mnist.py # kurzer MNIST-Lauf (PyTorch-Backend)
uv run pytest -q                     # Settling-Korrektheit (synthetisch, offline)
uv sync --extra dev                  # + pytest/jupyter/matplotlib
uv sync --extra experiments          # + avalanche/optuna/wandb (Phase 4 / docs/10)
```

## Plugin-/Adapter-Architektur? Nein.

Eine einzelne kohärente Implementierung mit einem Mechanismus, keine austauschbaren Fälle.
Die einzige bewusste Abstraktion ist die `train_and_eval(config)`-Schnittstelle (für Phase 4)
und der `backend`-Switch (für den optionalen Kernel) — sonst keine.
