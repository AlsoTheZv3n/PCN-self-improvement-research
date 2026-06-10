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
  correctness-verifiziert ~1e-6, beliebige Tiefe; via `PCN_CUDA_KERNEL=1` / `backend="cuda"`)
- `scripts/` — alle Experimente (faire PC-vs-BP-Studie, Song-Replikation, Kernel-Benchmark,
  Tiefen-Ablation, generativ, Suche) + `make_figures.py`
- `docs/` — vollständige Projekt-Doku (`00_overview.md` Einstieg; exakter Algorithmus `09`,
  Methodik `10`, Quellen `11`, alle Befunde `12`)
- `thesis/` — die fertige **Masterarbeit als PDF** (`Weidenmann_Predictive-Coding-from-Scratch.pdf`)
- `figures/` — die Paper-Figuren (PNG + PDF), reproduzierbar aus `results/*.json`
- `CLAUDE.md` — Arbeitsanweisungen, Konventionen und Lese-Reihenfolge für Claude Code

## Ergebnisse (ehrlich, mit n-Seed-CIs)

Eine bewusst **nüchterne, reproduzierbare** Studie — die Disziplin ist der Beitrag, nicht ein
Durchbruch:

- **PC ≈ BP** unter fairem Vergleich (gematchter Loss/Init/Arch, LR pro Methode getunt, Bootstrap-
  CIs) über Bulk-Accuracy, Noise, Sample-Effizienz und Continual Learning — inkl. einer
  architektur-treuen Replikation von Songs exaktem alternierenden Protokoll (kein ≥1σ-Gap bei
  keinem Budget, n=10). Drei **Confounds** entlarvt (Loss-Funktion, Stabilität-Plastizität,
  ungetunte Baseline), die scheinbare „PC-Vorteile" erzeugen.
- **Kernel:** der erste hand-geschriebene fused CUDA-Settling-Kernel für PC (nach unserem
  Kenntnisstand); **3,2× @ Batch 64**, launch-overhead-charakterisiert, beliebige Tiefe, in eine
  Studie integriert (1,45× end-to-end, identische Ergebnisse).
- **Generativ:** ein generativ trainiertes PC liefert „eine Maschinerie, viele Aufgaben"
  (Klassifikation, Generierung, Inpainting, Anomalie-AUC 1,0).
- **Tiefe:** der Signal-Decay („tiefer = schlechter") ist reproduziert; ein lokaler Präzisions-
  Schedule mildert ihn (bekannte Technik, Qi 2025/μPC). Drei gezielte Innovations-Versuche endeten
  als saubere, evidenzbasierte Negative — genuine PC-Neuheit ist auf MNIST-Skala nicht erreichbar.

Alle Experimente, Figuren und Tests sind aus dem Repo reproduzierbar (`results/*.json`,
`scripts/`, 40 Offline-Tests); die fertige Masterarbeit liegt als PDF unter `thesis/` bei.

## Wichtigste Regeln

- `uv` ausschließlich (kein `pip`).
- PyTorch nur als Tensor-Engine — **kein** Autograd fürs PC-Lernen; alle Updates manuell.
- Der CUDA-Kernel ist **optional** und Default-`backend` ist `"pytorch"`.

Details und Hintergrund: **`CLAUDE.md`** und **`docs/00_overview.md`**.
