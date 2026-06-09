# Predictive Coding Network (PCN) — From-Scratch Projekt

Ein From-Scratch-PCN auf MNIST in reinem PyTorch, optional beschleunigt durch einen
benchmarkten C++/CUDA-Settling-Kernel, dokumentiert in einem knappen Paper im
NeurIPS-Stil. Ziel: tiefes Verständnis einer **Backprop-Alternative** demonstrieren,
nicht "noch ein Finetune".

## Worum es geht

Predictive Coding (PC) ist eine biologisch plausible Alternative zu Backpropagation.
Statt eines globalen Backward-Pass minimiert ein PCN die **variationelle freie Energie**
(Summe präzisionsgewichteter quadrierter Vorhersagefehler) über **rein lokale** Updates.
Konzeptionell sitzt das Projekt an der Schnittstelle dreier Achsen, die oft verwechselt
werden:

1. **Lernalgorithmus** — Backprop vs. lokales/Free-Energy-Lernen *(das ändert ein PCN)*
2. **Architektur** — Transformer vs. SSM/RNN *(das ändern RWKV, Mamba, RetNet)*
3. **Ziel / Agency** — passive Vorhersage vs. Active Inference *(Hebel 2: die Welt verändern)*

Ein PCN adressiert **Achse 1**. Ein PCN ist weiterhin ein neuronales Netz — was sich
ändert, ist das Credit-Assignment, nicht die Verwendung von Netzen.

## Ehrliche Einordnung (Scope)

- Ein From-Scratch-MNIST-PCN, das bekannte Resultate (Whittington & Bogacz 2017,
  Millidge et al., Song et al., Innocenti et al. 2025) reproduziert, ist primär
  **Verständnis + saubere Reproduktion**. Das ist der Wert fürs Portfolio.
- Das "Neue" kommt aus **einem** fokussierten Winkel (Novelty-Hook), nicht aus einem
  Durchbruch. Siehe `05_paper_outline.md`.
- Die **headline open problems** des Felds (Depth-Scaling, Scale-Gap zu Transformern)
  zu lösen ist Frontier-Lab-Arbeit (Oxford BNDU, Sussex/VERSES, Ghent imec) — nicht
  realistisch für ein Solo-MNIST-Projekt. Was ein kleines Projekt **kann**: definierte
  Teilfragen sauber untersuchen. Siehe `06_open_problems_and_approaches.md`.

## Phasenplan

| Phase | Inhalt | Datei |
|-------|--------|-------|
| 1 | PyTorch-Referenz, Settling-Loop korrekt, Korrektheit validiert | `01`, `02` |
| 2 | Jupyter-Demos: Klassifikation, generativ, Occlusion, Anomalie, Baseline | `03` |
| 3 | C++/CUDA-Settling-Kernel + Benchmark gegen PyTorch | `02` |
| 4 | Autonomer Such-Loop ("Hermes mini") über den Konfigurationsraum | `04` |
| — | Paper schreiben, **parallel ab Phase 2** | `05` |

**Sequenzierungs-Regel:** Phase 4 ist eine Dependency, keine Wahl. Ein Such-Loop sucht
*über* etwas — er braucht ein laufendes PCN mit `train_and_eval(config)`-Schnittstelle
als Substrat. Deshalb wird diese Schnittstelle ab Phase 1 als sauberes Interface
angelegt, damit Phase 4 nur ein dünner Layer obendrauf ist statt eines Rewrites.

## Datei-Index

- `01_architecture.md` — PCN-Gleichungen, Knotenstruktur, Sizing, Clamping
- `02_build_data_training.md` — Stack, C++/Python-Build-Reihenfolge, Daten, Training, Interface
- `03_testing_and_interaction.md` — Jupyter-Tests & interaktive Demos
- `04_autonomous_search_loop.md` — Phase-4-Orchestrator, ehrliches Scoping, Referenzsysteme
- `05_paper_outline.md` — LaTeX-Struktur, Novelty-Hooks, Seitenbudget
- `06_open_problems_and_approaches.md` — aktuelle Schwierigkeiten des Felds + Lösungsansätze
- `07_cuda_kernel_build.md` — Build-Guide CUDA-Settling-Kernel: Lücke, Parallelisierung, Quellen, Benchmark
- `08_bio_regime_experiments.md` — Build-Guide PC-vs-BP-Experimente: Prospective Configuration, Protokolle, Tooling
- `09_kernel_pc_dynamics_deepdive.md` — **Deep-Dive** Kernel & PC-Dynamik: exakter SO-Algorithmus, Signal-Decay, EO, Fusion, Benchmark-Matrix
- `10_pc_vs_bp_deepdive.md` — **Deep-Dive** PC vs. BP: Prospective Configuration, faire Vergleichsmethodik, Protokolle, Caveats
- `11_quellen.md` — alle Original-URLs der verwendeten Web-Quellen, nach Thema geordnet und auf die Dateien gemappt

Die beiden gewählten Beitrags-Winkel (siehe `06`, Probleme 2 + 4) sind in `07`/`08` als
Build-Guides und in `09`/`10` als vertiefte, aus den Primärquellen destillierte Referenzen
ausgearbeitet — letztere sind die "maximales Wissen zum Start"-Dokumente.

## Nächster Schritt

Repo-Skelett aufsetzen: `uv`-Projekt, `pcn/`-Modul mit dem Settling-Loop, die
`train_and_eval`-Schnittstelle, ein Test-Notebook. Die Settling-Mathematik wird beim
Code-Schreiben Schritt für Schritt hergeleitet (Vorzeichen + θ'-Ableitung).
