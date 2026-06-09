# 02 — Build, Daten & Training

## Stack

- **`uv`** für die Umgebung (ausschließlich — kein `pip`)
- **PyTorch** — aber **nur als Tensor-Engine**. KEIN `autograd.backward()`: das würde
  dem ganzen Punkt widersprechen. Alle Updates manuell unter `torch.no_grad()`.
- **Settling-Loop**: ~50 Zeilen, Kern des Repos
- **C++/CUDA** (Phase 3): Settling-Kernel via `pybind11` / `libtorch`
- **Jupyter** zum Testen (siehe `03`)
- **W&B** oder SQLite zum Logging (für Phase 4 und die Paper-Plots)

## Build-Reihenfolge: PyTorch zuerst, C++ als Messpunkt

Für ein MNIST-PCN ist C++-first die **falsche** Reihenfolge. Begründung:

1. **PyTorch-Referenz** → Settling-Dynamik, Präzisions-Schedules, Update-Regeln
   iterieren. Hier ist Geschwindigkeit der Iteration entscheidend.
2. **Korrektheit validieren** → reproduziert MNIST-Accuracy aus der Literatur?
   Konvergiert die Energie? Stimmt das Generative-Verhalten?
3. **DANN C++/CUDA-Settling-Kernel** → als **benchmarkte Contribution**, gemessen gegen
   die PyTorch-Version.

### Warum der CUDA-Kernel eine *echte* Contribution ist

Der Flaschenhals von PCNs ist die **iterative Inference-Phase**: numerische Solver,
viele Iterationen bis zur State-Konvergenz. Auf GPUs, die für das dichte Matmul-Muster
von Backprop gebaut sind, ist das teuer (Hardware-Algorithmus-Mismatch; siehe `06`,
Problem 2). Es gibt **keine DL-Library, die die Inference-Phase voll parallelisiert**.
Ein sauber benchmarkter Settling-Kernel beantwortet damit eine reale offene
Engineering-Frage — und liegt nach dem 355M-LLM mit custom GEMM / Flash-Attention exakt
im eigenen Revier. Das ist Novelty-Hook A (siehe `05`).

## Daten: MNIST — kein Streaming

MNIST sind ~11 MB. Einmal laden:

```python
from torchvision import datasets, transforms   # oder datasets.load_dataset("mnist")
```

**HF-Streaming löst hier ein Problem, das nicht existiert.** Streaming ist für Korpora
gedacht, die nicht in den Speicher/auf die Platte passen.

### Wo Streaming KORREKT einsteigen würde

Falls das Projekt später auf ein **PC-Transformer-Sprachmodell** skaliert (PC-basierte
Transformer erreichen fast die Leistung normaler Transformer bei gleicher Komplexität —
siehe `06`), dann gilt für die Datenpipeline das Muster aus dem LLM-Projekt:
HuggingFace-Datasets (FineWeb-Edu o.ä.) im Streaming-Modus → tokenisieren → `.bin`. Dann
ist `datasets.load_dataset(..., streaming=True)` richtig. Für MNIST nicht.

## Trainingsprozedur

Pro Batch:

```
1. Forward-Init:   x_l einmal forward initialisieren (gute Startwerte fürs Settling)
2. Clamp:          Bild an Input-Knoten, One-Hot-Label an Output-Knoten
3. Inference:      T Schritte Δx_l (Gewichte fix) bis Energie-Gleichgewicht
4. Learning:       ΔW_l am Gleichgewicht (lokal, siehe 01)
```

Details zu Clamping und Auslesen: `01_architecture.md`.

## Die `train_and_eval`-Schnittstelle (kritisch für Phase 4)

Von Anfang an als sauberes Interface anlegen — das macht den späteren Such-Loop fast
gratis:

```python
def train_and_eval(config: dict) -> dict:
    """
    config: {
        "hidden": [256, 256], "T": 20, "eta_x": 0.1, "eta_w": 1e-3,
        "activation": "tanh", "precision_schedule": "isotropic",
        "update_variant": "standard",   # vs. "ipc"
        "epochs": 10, "seed": 0, ...
    }
    returns: {
        "test_acc": ..., "energy_curve": [...], "noise_robustness": ...,
        "train_time_s": ..., "settling_steps_to_converge": ..., ...
    }
    """
```

Phase 4 (siehe `04`) ist dann nur noch: einen `config` vorschlagen → `train_and_eval`
aufrufen → Metriken loggen → nächsten `config` wählen. Kein Rewrite.

### Implementierungs-Stand (2026-06-09, M1)

Der obige Vertrag ist in `pcn/api.py` **umgesetzt** (siehe `docs/13` M1):

- **Config-Aliase:** Die hier dokumentierten Spec-Namen `eta_x`/`eta_w` werden akzeptiert
  und intern auf `lr_state`/`lr_weight` gemappt (`_CONFIG_ALIASES`). Beide Schreibweisen
  funktionieren; die Spec-Namen haben Vorrang.
- **Unbekannte Keys** lösen ein `warnings.warn(...)` aus (kein stilles Ignorieren mehr).
- **Rückgabe:** `train_and_eval` liefert jetzt `test_acc`, `energy_curve` (mittlere
  Gleichgewichts-Energie pro Epoche), `noise_robustness` (Accuracy über Input-Rausch-Sigma),
  `train_time_s`, `settling_steps_to_converge` und die aufgelöste `config`.
- **`settling_steps_to_converge`** braucht ein Konvergenzkriterium: `settle(..., tol=...)`
  stoppt, wenn die relative Energie-Änderung unter `tol` fällt; `tol=None` (Default) behält
  das reine Fixed-`T`-Verhalten. Die Metrik wird mit einer Referenz-`tol` von `1e-3` auf
  einem Trainings-Batch gemessen.
- **Default `eta_w`/`lr_weight`** wurde von `1e-3` auf `0.01` angehoben — `1e-3` underfittet
  auf MNIST (83 % statt 92 % bei 10 Epochen, siehe `docs/12`).
- Die `config` enthält zusätzlich `precision_schedule` (Default `"isotropic"`, Π=I) und
  `update_variant` (Default `"standard"`) als Platzhalter für M2 bzw. M7.
