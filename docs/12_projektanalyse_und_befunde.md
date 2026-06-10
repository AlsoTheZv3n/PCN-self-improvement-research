# 12 — Projektanalyse, Befunde & Build-Einordnung

*Stand: 2026-06-09. Konsolidierte Erkenntnisse aus der vollständigen Analyse des Repos
(Code `pcn/`, `tests/`, `scripts/`, Doku `docs/00`–`docs/11`, `pyproject.toml`) plus
Consistency-Audit Code-vs-Doku. Dieses Dokument ist der Ausgangspunkt für den
Umsetzungsplan in `docs/13`.*

---

## 1. Stand der Implementierung (Was ist real, was Stub)

**Funktionsfähig (Phase 1–2, reiner PyTorch-Pfad):**

- **Settling-Loop** (`pcn/settling.py`): pro Schritt
  `eps[k] = states[k] − W[k-1]·phi(states[k-1]) − b`, dann
  `states[k] -= lr_state · (eps[k] − phi'(states[k]) ⊙ (eps[k+1] @ W[k]))` für freie
  Schichten; Jacobi-Iteration, synchron, exakt `T` Schritte. **Vorzeichen, Transposition
  und das `phi'`-Gating sind korrekt** und decken sich mit `docs/09`.
- **Lern-Update** (`pcn/learning.py`): `dW = eps.t() @ phi(pre) / batch`, `W += lr_weight·dW`
  — rein lokal (Hebb), Vorzeichen korrekt gegen `docs/01`.
- **Energie** (`pcn/settling.py`): `E = ½·mean_batch(Σ_k ‖eps_k‖²)`, fällt monoton (per
  Tests belegt).
- **API** `train_and_eval(config) -> {test_acc, train_time_s, config}` (`pcn/api.py`).
- **Tests** (`tests/test_settling.py`): 4 Stück, hermetisch (Toy-Tensoren, kein
  MNIST-Download), laufen CPU-only durch.

**Stub / TODO:**

- **CUDA-Kernel** — reiner, *ehrlich gekennzeichneter* Stub: `kernels.is_available()` →
  `False`, `build()`/`settle()` werfen `NotImplementedError`, `backend="cuda"` wird sauber
  abgewiesen. Docs (`docs/07`) durchgehend in Zukunftsform — **keine Überzeichnung**.
- **Präzision `Π`** — nirgends im Code (hartkodiert isotrop = Identität).
- **Update-Varianten iPC / EO** — nicht implementiert (nur Standard-SO).
- **Generativ / Occlusion / Anomalie** — kein input-freier / label-geklemmter Pfad; keine
  Jupyter-Notebooks vorhanden.
- **Bio-Regime** — nur `limit_train` (Small-Data) existiert; Split-/Permuted-MNIST,
  Continual-Metriken, BP-MLP-Baseline fehlen (`avalanche-lib` deklariert, ungenutzt).
- **Konvergenzkriterium** — existiert nicht; `settle` läuft immer fix `T` Schritte.

---

## 2. Befunde & Risiken (priorisiert)

1. **`train_and_eval` weicht von `docs/02` ab (höchstes Risiko).** Config nutzt
   `lr_state`/`lr_weight` statt dokumentiert `eta_x`/`eta_w` (`pcn/api.py`) → eine nach
   Spec geschriebene Config wird **stillschweigend ignoriert**. Es fehlen die zugesagten
   Rückgaben `energy_curve`, `noise_robustness`, `settling_steps_to_converge` und die
   Config-Keys `precision_schedule`, `update_variant`. Damit ist der „thin layer / kein
   Rewrite"-Vertrag für Phase 4 untergraben — die scharfen Metriken (Energie-Konvergenz,
   Noise-Robustheit) werden nie emittiert.
2. **Präzision `Π` ist als Kern dokumentiert (`docs/01`, `docs/04`), im Code aber komplett
   abwesend.** Die ganze „precision schedule"-Suchdimension ist derzeit gegenstandslos.
   (Für `Π=I` mathematisch unkritisch — reduziert `F` auf SSE.)
3. **Doku-Falle (kein Code-Bug):** `docs/01` indiziert **top-down** (`ε_{l-1}`/`W_{l-1}`),
   Code/`docs/09` **feedforward** (`ε_{k+1}`). Äquivalent unter Index-Flip, aber Copy-Trap
   für künftige Kernel-Autoren → eine Konventionsnotiz in `docs/01` entschärft das.
4. **Hook C (generativ/Occlusion/Anomalie) und Bio-Regime/Continual sind nur Doku** — kein
   input-freier Clamping-Pfad, keine Notebooks, keine BP-Baseline.
5. **Theorie-Anker arXiv:2602.07697 (PC≈BP, „Infinite Width/Depth Limits of PCNs") wird
   zitiert, ist aber nicht ⭐ „gelesen"** — Behauptung second-hand, vor Übernahme ins Paper
   verifizieren. (Siehe `docs/13` / Recherche-Ergebnis und `docs/11`-Update.)
6. **`test_training_loop_reduces_output_error` hat eine schwache Assertion**
   (`tests/test_settling.py`) — `first` wird nur in Iteration 0 gesetzt, `err` ist der
   letzte Wert → es prüft nur „Epoche N < Epoche 0", nicht den Trend.

**Als korrekt bestätigte Nicht-Probleme:** Settling-Gradient (Vorzeichen + Transposition +
`phi'`-Gating), Fehler-Vorzeichen, Hebb-Update-Vorzeichen, Energieformel, CUDA-Stub-
Ehrlichkeit, Test-Imports/-Verdrahtung. **Kein Vorzeichenfehler in der Kernmathematik.**

---

## 3. Build-Einordnung: Muss eine eigene (kompilierte) Library gebaut werden?

Kurzantwort: **Nein — nicht für ein funktionierendes, publizierbares PCN. Eine eigene
kompilierte Library braucht es nur für Novelty-Hook A (den CUDA-Kernel), und der ist
optional.** Zwei Bedeutungen von „Library bauen" sauber trennen:

### 3.1 Das `pcn/`-Python-Paket — schon da, kein Build
Reines Python/PyTorch, **nichts zu kompilieren**. `uv sync` (zieht `torch`/`torchvision`),
dann `uv run python scripts/train_mnist.py`. Phase 1–2 und der Phase-4-Suchloop laufen
komplett ohne Low-Level-Code (vgl. `CLAUDE.md`: *„Ohne Kernel ist null Low-Level-Code
nötig"*).

### 3.2 Der CUDA-Settling-Kernel — *das* ist „Library bauen", und er ist OPTIONAL
Eine **kompilierte C++/CUDA-Extension** (echte Shared-Library `.pyd`/`.so`):
`settling_kernel.cu` (fused Kernel **+ `PYBIND11_MODULE`-Binding im selben File**), nach Python
exponiert. **Realisiert:** `torch.utils.cpp_extension.load(sources=[settling_kernel.cu])` (JIT aus
der echten `.cu`, cached; `load_inline` mit Inline-String war nur der Prototyp-Schritt). Optional
später `setuptools`/`CUDAExtension` (AOT, ABI-stabile LibTorch-API für PyTorch ≥ 2.10).

**Wichtig:** Auch hier wird *keine* allgemeine PC-Library gebaut — nur *ein* fused Kernel +
Wrapper + Benchmark-Matrix. Der wissenschaftliche Mehrwert ist die Messung („wie groß ist
der Inference-Overhead, wie weit drückt Fusion ihn?"), nicht ein Library-Release. Gerade
weil es **keine** PC-Library mit handoptimiertem CUDA-Inference-Kernel gibt
(`docs/07`), ist das eine echte Lücke.

### 3.3 Übersicht

| Was | Build nötig? | Pflicht? |
|---|---|---|
| PCN lauffähig (Phase 1–2) | Nein (pure PyTorch) | ✅ Kern |
| Hook B (PC-vs-BP-Regime) + Hook C (generativ) | Nein | optionale Paper-Hooks |
| Hook A (CUDA-Kernel) | **Ja — C++/CUDA-Extension** | **optional**, bewusst zugeschaltet |

Man kann ein vollständiges Projekt + Paper über Hooks B/C abliefern, ohne je einen Kernel
zu kompilieren. Der CUDA-Kernel ist der **stärkste, aber optionale** Beitrag — und genau
der Teil, bei dem „eine eigene Library bauen" wörtlich zutrifft. **Reihenfolge-Regel
(strikt):** erst PyTorch-Basis lauffähig + validiert (Energie fällt, Accuracy plausibel),
*dann* der Kernel.

---

## 4. Forschungsinhalt (Kontext für die Umsetzung)

**Zwei gewählte Beitrags-Winkel** (`docs/06`):

- **(A) CUDA-Settling-Kernel** (stärkster Hook, Problem 2 „Compute/HW-Mismatch"): fused
  Single-Launch-Settling-Schritt; layerweise Ops innerhalb eines Schritts parallel,
  Zustände über `T` Schritte resident in Shared Memory/Registern. Correctness-first
  (Gewichte/Accuracy matchen), dann Speed; immer `torch.cuda.synchronize()` vor dem Timing.
  Benchmark-Matrix (`docs/09`): PyTorch-SO vs. Fused-CUDA-SO vs. EO vs. Backprop.
- **(B) PC-vs-BP-Bio-Regime** (Problem 4 „What-For"): testet „Prospective Configuration"
  (Song et al. 2024). Faire Methodik: identische Architektur/Init/Daten-Splits, nur
  Lernregel variiert, ≥3 Seeds mit Konfidenzintervallen. Regime: Online (Batch-Size 1),
  Small-Data, Continual (Split-/Permuted-MNIST via Avalanche), Noise-Robustness,
  Depth-Scaling. Negatives Ergebnis ist explizit als valider Beitrag eingeplant.

**Kern-Mathematik / Deep-Dives:**

- **Signal-Decay** (`docs/09`): `‖ε_{L-i}‖ ∝ C(t,i)·λ^i·(1−λ)^(t−i)` — exponentielle
  Dämpfung pro Schicht; bei `λ≈0.1` fällt das Signal in ~4–8 Iterationen unter
  float32-Epsilon, tiefe Schichten (>8) bleiben praktisch untrainiert. Faustregel
  `L < T < 2L`; orthogonale Init (Eigenwerte ≈1) dämpft den Effekt.
- **EO-Reparametrisierung**: `s_i = ŝ_i + ε_i` wandelt Zustands- in Fehlerupdates; ein
  globaler Graph transportiert das Output-Signal ungedämpft (~100× schneller auf tiefen
  Netzen) — **ändert aber die PC-Formeln**.
- **PC≈BP-Versöhnung** (`docs/10`): Constraint-Regime (kleines `λ`, `T`→ff-Init)
  konvergiert gegen BP; das unbeschränkte Settling-Regime zeigt die distinkten PC-Vorteile.

---

## 4a. Empirische Ausgangsbasis (gemessen 2026-06-09)

Erste Lauffähigkeits- und Korrektheitsmessung (lokal, `torch 2.12.0+cpu`, CPU; GPU = RTX
3080 Ti / CUDA 12.4 vorhanden, aber das `uv`-`.venv` zog die CPU-Variante):

- **`pytest -q`: 4/4 grün** — die Settling-Mathematik ist gegen die analytische Referenz
  validiert.
- **MNIST end-to-end läuft**, aber die **Defaults underfitten**:

  | Lauf | Config | Test-Accuracy | Train-Zeit |
  |---|---|---|---|
  | Smoke | 4k Bsp., 3 Ep., `lr_weight=1e-3` | 26,8 % | 3,3 s |
  | A | volle Daten, 10 Ep., `lr_weight=1e-3` (Default) | **83,1 %** | 157 s |
  | B | volle Daten, 10 Ep., `lr_weight=0.02` | **92,2 %** | 131 s |

- **Befund 1 (Underfitting):** Kein Bug — die Pipeline erreicht plausible MNIST-Accuracy
  (92 %). Aber der **Default `lr_weight=1e-3` ist zu niedrig** (Underfitting). → in M1 auf
  `0.01` angehoben (erledigt); Feintuning via Such-Loop (M7). Siehe `docs/13` (M0/M1).
- **Befund 2 (Under-Settling, neu nach M1):** Die neue Metrik `settling_steps_to_converge`
  zeigt für `[784,256,256,10]`, dass das Settling bei `tol=1e-3` erst nach **≈ 31 Schritten**
  konvergiert — der Default `T=20` **under-settled** also. Da PCs distinkte Vorteile laut
  Song et al. 2024 / Innocenti et al. 2026 nur **am Aktivitäts-Gleichgewicht** zeigen, ist
  das relevant: für die PC-vs-BP-Experimente (M4) sollte `T` erhöht oder `tol`-basiertes
  Settling genutzt werden. Zitierbarer Diagnose-Befund, direkt an der T-vs-Tiefe-Frage
  (`docs/09`, Qi et al. 2025) — ermöglicht durch das M1-Konvergenzkriterium.

## 4b. Erstes PC-vs-BP-Tableau (M3, gemessen 2026-06-09)

Faire Gegenüberstellung über `pcn/benchmark.py:benchmark_pc_vs_bp` — **identische
Architektur `[784,256,256,10]`, identische Initialisierung** (die BP-MLP klont die PCN-
Gewichte beim selben Seed), **identische MNIST-Pipeline**; einzige Variable ist die Lernregel.
5 Epochen, volle Daten, CPU, gemeinsame `lr_weight=0.01`:

| Lernregel | Test-Acc | Train-Zeit | Acc σ=0.25 | Acc σ=0.5 | Acc σ=1.0 |
|---|---|---|---|---|---|
| **PC** (settling + Hebb) | 0,896 | 74,7 s | 0,863 | 0,715 | 0,337 |
| **BP** (SGD + CE) | 0,928 | 51,8 s | 0,925 | 0,911 | 0,828 |

`time_settle` (Micro-Benchmark, T=20, B=64, CPU): **5,96 ms/Settle, 0,30 ms/Schritt.**

**Befund 3 (PC-vs-BP, vorläufig):** Bei *gleicher* Lernrate ist BP hier sowohl **genauer**
(92,8 % vs. 89,6 %) als auch **schneller** (52 s vs. 75 s — die Settling-Phase kostet, vgl.
Zahids Compute-Achse) als auch **deutlich rausch-robuster** (σ=1.0: 0,83 vs. 0,34). Das ist
das *Gegenteil* der naiven „PC ist robuster"-Erwartung.

> **⚠️ Caveats (verhindern eine Überinterpretation — die faire Studie ist M4):**
> 1. **PC ist under-settled** (`settling_steps_to_converge=28 > T=20`) — PC-Vorteile treten
>    laut Song/Innocenti nur am Gleichgewicht auf. M4 muss mit höherem `T` / `tol`-Settling
>    wiederholen.
> 2. **LR nicht pro Methode getunt** — das faire Protokoll (Song et al. 2024) tunt die LR je
>    Lernregel separat über ein gemeinsames Grid. Hier teilen sich beide `0.01`.
> 3. Nur **1 Seed**, keine Konfidenzintervalle (M4: ≥ 3 Seeds + Bootstrap-CI).
>
> Als *erster Datenpunkt* steht: PC ist hier nicht überlegen. Ein sauber abgesichertes
> negatives Ergebnis ist laut `docs/10` ein valider Beitrag — aber erst nach M4 belastbar.

## 4c. M4 — Faire PC-vs-BP-Studie (gemessen 2026-06-09, GPU)

Faire Methodik (Song et al. 2024): identische Architektur `[784,256,256,10]`/Init/Daten (die
BP-MLP klont die PCN-Init), LR **pro Methode** über ein Grid getunt, jede Zahl als
Mittel ± 68%-Bootstrap-CI über 3 Seeds. GPU (RTX 3080 Ti, torch 2.12+cu130). Harness:
`pcn/experiments/`, Treiber `scripts/run_experiments.py`, Artefakte in `results/`.

### Methoden-Lektion (kritisch): `eta_x` und `eta_w` gemeinsam tunen
Der erste Vergleich fixierte `eta_x=0.1` und variierte nur `eta_w` → PC schien um ~10 pp zu
verlieren. Ein `eta_x × eta_w`-Scan deckte eine **Stabilitätsgrenze** auf: bei `eta_x=0.1`
**divergiert** PC für `eta_w=0.05` (→ 9,8% = Zufall), bei *niedrigerem* `eta_x=0.05` ist
`eta_w=0.05` stabil und liefert 84,9%. Mit gemeinsamem Tuning (`eta_x=0.05, eta_w=0.05`)
schrumpft die Accuracy-Lücke von ~10 auf ~4 pp. **Nur `eta_w` zu sweepen benachteiligt PC
systematisch** — die ~10pp waren großteils ein Tuning-Artefakt.

### ⚠️ Erste Ergebnisse waren konfundiert — entkonfundierte Zahlen unten

Eine **adversariale Methodik-Prüfung** (4 Kritiker-Agenten) fand zwei Confounds, die die
ersten „PC-Siege" erklären:
- **B1 — Loss-Mismatch:** PC minimiert MSE-zu-One-hot, die erste BP-Baseline aber
  CrossEntropy. Das verletzt „nur die Lernregel variiert". Fix: ein **BP(MSE)-Kontrollarm**
  mit *identischem* Ziel wie PC. Der faire Vergleich ist **PC(MSE) vs BP(MSE)**;
  BP(CE)−BP(MSE) ist der reine Loss-Effekt.
- **B2 — Plastizitäts-Confound (Continual):** „weniger Vergessen" kann „lernt jeden Task
  schlechter" heißen. Fix: zusätzlich **Learn-Accuracy** (R-Diagonale) und **behaltene
  Task-0-Accuracy** messen.

### Entkonfundierte Ergebnisse (3 Arme, val-selektiert, 3 Seeds)

**Bulk (10k, 8 Ep., T=40, joint-getunt):**

| Arm | Test-Accuracy | Noise σ=1.0 |
|---|---|---|
| PC (MSE) | 83,5% [82,8, 84,3] | 52,6% |
| **BP (MSE)** — fairer Arm | 83,9% [83,6, 84,3] | 60,4% |
| BP (CE) — Referenz | 89,9% [89,7, 90,0] | 81,0% |

→ **FAIR-Gap PC(MSE) vs BP(MSE): +0,40 pp — CIs überlappen, statistisch ununterscheidbar.**
**LOSS-Effekt BP(CE)−BP(MSE): +5,92 pp.** Der zuvor gemeldete „BP +4–10 pp"-Vorsprung war
**fast vollständig der Loss, nicht die Lernregel.** Bei Noise bleibt BP(MSE) ~8 pp robuster
als PC(MSE) (realer, aber viel kleinerer Effekt als die ~28 pp gegen BP(CE)).

**Continual (Permuted-MNIST, 5 Tasks):**

| Arm | Final-ACC | Learn-ACC | BWT | Behaltene Task-0 |
|---|---|---|---|---|
| PC (MSE) | 55,0% | 65,1% | −12,7% | 41,8% [39,8, 43,8] |
| **BP (MSE)** | 62,9% | 69,1% | **−7,8%** | **54,2% [54,1, 54,3]** |
| BP (CE) | 60,7% | 78,7% | −22,5% | 47,2% |

→ Die Headline „PC vergisst weniger" war **doppelt artefaktisch**: (1) sie verglich gegen
BP(CE), das wegen aggressiver Softmax-Updates *mehr* vergisst; gegen den fairen BP(MSE)
vergisst BP **weniger** (−7,8% vs −12,7%). (2) PC lernt jeden Task schlechter (Learn-ACC
65,1% < 69,1%) und **behält Task 0 absolut schlechter** (41,8% < 54,2%, CIs disjunkt) →
Plastizitäts-Confound. **Entscheidungsregel: NICHT bewiesen.**

**Class-incremental (Songs eigentliches Regime, BP-LR pro Arm per learn-ACC getunt):**

| Setup | Arm | Final-ACC | Learn-ACC | BWT | Behaltene T0 | Entscheidung |
|---|---|---|---|---|---|---|
| Split-FashionMNIST (2×5) | PC | 60,9% | 84,3% | −46,8% | 37,8% [34,3, 41,3] | |
| | BP(MSE) | 61,3% | 86,1% | −49,6% | 37,3% [34,0, 40,6] | **NO** (alles überlappt) |
| Split-MNIST (5×2) | PC | 70,8% | 97,4% | −33,3% | 60,1% [58,6, 61,7] | |
| | BP(MSE) | 73,4% | 97,5% | −30,2% | 62,0% [60,4, 63,7] | **NO** (BP minimal besser) |

→ Auch in Songs **class-IL-Interferenz-Regime** — gegen ein korrekt getuntes, *lernendes*
BP(MSE) — **kein PC-Vorteil**: PC und BP(MSE) sind statistisch ununterscheidbar (Split-Fashion)
bzw. BP minimal besser (Split-MNIST). **Drittes Confound entlarvt:** ein ungetuntes BP(MSE)
(`eta_w=0.1`) hängt bei Split-MNIST auf Zufallsniveau (50%) fest → erzeugte einen
falsch-positiven PC-„Sieg", den die learn-ACC-Diagnose fing. Es bleibt ein konsistenter
**Stabilität-Plastizität-Tradeoff** (BP(CE) lernt am meisten/vergisst am meisten → PC am
wenigsten/wenigsten), aber PC liegt **auf derselben Kurve** wie loss-gematchtes BP.

### Synthese (entkonfundiert, vollständig)
Über **alle getesteten Regime** (Bulk-Accuracy, Noise, Sample-Effizienz, Permuted-MNIST
domain-IL, Split-FashionMNIST + Split-MNIST class-IL) gilt mit **gematchtem Loss und fair
getunten Baselines: vanilla SO-PC zeigt KEINEN Vorteil gegenüber BP** — gleichauf bei
Accuracy/Lernen, auf derselben Stabilität-Plastizität-Kurve. **Alle drei vorherigen
PC-„Siege" waren Methoden-Artefakte.** Der wissenschaftliche Wert (`docs/10` §8): ein sauberes
Null-/Negativ-Resultat **plus die systematische Entlarvung dreier Confounds**, die naive
PC-vs-BP-Vergleiche verzerren.

> **WICHTIG — was das NICHT heißt:** Dies **widerlegt Song et al. 2024 nicht global.** Songs
> Fig-4e nutzt **alternierendes** Training, **Sigmoid**, hidden=32 und ggf. andere Metriken/
> Baselines; meine faithful-ish Reproduktion ist **sequenziell**, tanh, `[256,256]`, matched
> loss, mit *sorgfältig getuntem* BP. Die Diskrepanz kann an Songs Spezifika oder an einer
> weniger sorgfältig getunten BP-Baseline bei Song liegen. Belastbare Aussage: **bei
> MNIST/FashionMNIST-Skala mit fairem, getuntem BP verschwindet der PC-Vorteil.**

### Methoden-Lektionen (eigenständig berichtenswert — der Kern-Beitrag)
1. **`eta_x`↔`eta_w` gemeinsam tunen** (Stabilitätsgrenze) — nur `eta_w` zu sweepen
   unterschätzte PC um ~6 pp.
2. **Loss-Funktion dominiert** PC-vs-BP-Vergleiche stärker als die Lernregel (BP(CE)−BP(MSE)
   ≈ +6 pp Bulk) — der häufigste versteckte Confound.
3. **Plastizitäts-Confound:** „vergisst weniger" muss gegen learn-ACC + behaltene Task-0
   geprüft werden, sonst zählt „lernt weniger" als Vorteil.
4. **Baseline-LR-Confound:** eine ungetunte (kaputte) BP-Baseline erzeugt falsch-positive
   PC-Siege — BP muss per learn-ACC tatsächlich lernen, bevor man vergleicht.
5. **Settling-T** (20/40/80) ändert die Bulk-Accuracy nicht → kein Under-Settling-Artefakt.

**Offene Caveats / Weg zur Paper-Reife:** volle Daten + mehr Seeds; **Songs exaktes Protokoll**
(alternierend, Sigmoid, hidden=32) als strikte Replikation; **CL-Baselines** EWC/Replay/
Joint-Upper-Bound (B4, zur Verortung der absoluten Forgetting-Zahlen — jetzt niedrigere
Priorität, da PC-vs-BP-Frage beantwortet). Quelle: adversarialer Gaps-Report (`docs/14`).

## 4d. M6 — Fused CUDA-Settling-Kernel (Hook A, gemessen 2026-06-09, RTX 3080 Ti)

**Der erste echte „besser"-Befund des Projekts — ein Engineering-Win.**

> **Neuheits-Einordnung (verifiziert, `docs/15`):** Verteidigbar ist *„erster handgeschriebener
> fused CUDA-Settling-Kernel **für PC**, to our knowledge"* — **stark belegt** (alle 11 offenen
> PC-Libraries direkt inspiziert: 0 `.cu`-Dateien, alle JAX/JIT oder PyTorch-autograd). **NICHT**
> verteidigbar: dass das *Design* (persistent, states-resident, single-launch) neu sei — das ist
> etablierte Prior Art (cuDNN/Baidu Persistent RNN 2016, LLM-Megakernels). Unser Beitrag = die
> **Übertragung auf PC + Messung**. Die „Launch-Overhead-dominiert"-Beobachtung ist generisch
> („framework tax"), kein eigener Beitrag. ePC (arXiv:2505.20137) als nächste PC-Arbeit
> zitieren+abgrenzen (algorithmisch vs kernel-level). **Doppelt abgesichert:** auch der nächste
> *algorithmische* Verwandte **Equilibrium Propagation** wurde auditiert (18 Repos) — ebenfalls
> *kein* handgeschriebener CUDA-Settling-Kernel (nur `autonull/bioplausible`: per-Step Triton +
> Host-Loop, zu zitieren). Konfidenz hoch. Details: `docs/15`.

Ein handgeschriebener CUDA-Kernel führt die *gesamte* T-Schritt-Settling-Schleife in **einem**
Launch aus (ein
Thread-Block pro Sample, States resident in Shared Memory; docs/09-Fusion-Design) statt der
vielen Mini-Op-Launches des PyTorch-Backends. **Correctness-first:** der Kernel matcht
`_settle_pytorch` auf ~1e-6 (allclose, tanh/identity/sigmoid × clamp on/off; `scripts/verify_kernel.py`). Build: `load(sources=[settling_kernel.cu])`
(nvcc + MSVC + ninja, torch cu126); CUDA-Quelle `pcn/kernels/settling_kernel.cu` (echte `.cu`),
Wrapper/Dispatch `settling_cuda.py`, Benchmark `scripts/bench_kernel.py`, Artefakt `results/m6_kernel_benchmark.json`.

**Benchmark (Topologie `[784,256,256,10]`, ms pro voller T-Schritt-Settle):**

| Batch | T | PyTorch-SO | fused-CUDA | Speedup |
|---|---|---|---|---|
| **64** | 20 / 40 / 80 | 10,6 / 23,0 / 40,8 ms | 3,3 / 6,1 / 12,2 ms | **3,2–3,8×** |
| 256 | 20 / 40 / 80 | 11,1 / 22,0 / 41,5 ms | 10,6 / 21,1 / 42,3 ms | ~1,0× (Break-even) |
| 1024 | 20 / 40 / 80 | 9,9 / 22,5 / 42,3 ms | 34,2 / 69,0 / 137,1 ms | 0,29–0,33× (verliert) |

**Interpretation (bestätigt die Projekt-These docs/09 exakt):**
- **PyTorch-SO ist ~0,5 ms/Schritt — batch-unabhängig** (B=64≈256≈1024): klassisch
  **launch-overhead-bound** (viele winzige Kernel-Launches pro Schritt; erklärt auch die nur
  ~25–49 % GPU-Auslastung in M4). Der fused Kernel eliminiert den Overhead → **3–4× schneller
  bei kleiner Batch.**
  - **Launch-Count-Beleg (`scripts/profile_launches.py`, `results/m6_launch_count.json`):**
    direkt gemessen (torch.profiler `cudaLaunchKernel`-Events; nsys nicht installiert):
    PyTorch-SO setzt **31 Kernel-Launches PRO Schritt** ab (T=20→620, T=40→1240, T=80→2480
    Launches), der fused Kernel **genau 1** — unabhängig von T (**620–2480× weniger Launches**).
    Das ist der *mechanistische* Beweis, dass Launch-Overhead der adressierte Bottleneck ist
    (nicht nur ein Wall-Clock-Indiz) — direkt für den Einwand der „Solver-Sicht" (JPC/ePC).
- **Crossover:** Bei großer Batch (1024) verliert der Kernel 3×, weil der **naive
  In-Kernel-GEMM** (ein Block/Sample, Weights pro Schritt aus Global Memory, kein Tiling)
  bandbreiten-limitiert ist — cuBLAS (PyTorch) gewinnt dort. Exakt die docs/09-Warnung
  „große Breiten/Batches wollen cuBLAS".

**Bedeutung:** Der Win liegt **genau im Small-Batch-/Online-Regime** (B=64) — biologisch und
für PC am relevantesten. Ein ehrlich charakterisierter Beitrag: 3–4× durch
Launch-Overhead-Elimination, mit klarem Crossover. **Scope/Caveats (v1):** spezialisiert auf
2-Hidden-Layer (`model.n==3`), fixes T (kein tol-Early-Stop), keine Energie-Aufzeichnung,
naiver GEMM. Aktivierung bewusst per `PCN_CUDA_KERNEL=1` + `backend="cuda"` (Default PyTorch).

**M6-v2 (Batch-Tiling, gemessen 2026-06-09):** Der Large-Batch-Verlust kam vom naiven GEMM
(v1: ein Block pro Sample → jedes Gewicht wird pro Sample neu aus Global Memory gelesen,
bandbreiten-limitiert). v2 (`pcn_settle_so_tiled`) bearbeitet eine **Kachel von TB=8 Samples
pro Block** und liest jedes Gewicht **einmal, wiederverwendet über die Kachel** (die
GEMM-Schlüssel-Optimierung; Batch auf TB-Vielfaches gepaddet, phi(Input) host-seitig
vorberechnet). `settle()` dispatcht **hybrid** nach Batch-Größe (v1 ≤128, v2 darüber). Beide
Kernel **correctness-verifiziert** (allclose ~1e-6 vs PyTorch, mit *stabiler* Config —
instabiles Settling verstärkt sonst Float-Diffs und täuscht „Fehler" vor).

**M6-v3 (P0-Caching):** v3 cacht zusätzlich das φ(Input)-Tile *einmalig* in Shared Memory
(Input geklemmt → konstant) und eliminiert so ~256× redundante Global-Reads von P0 in der
Input→Hidden-Schicht (Shared-Opt-in für ~75 KB/Block auf sm_86).

| Batch | v1 | v2 (Batch-Tiling) | **v3 (+P0-Cache)** | Pfad |
|---|---|---|---|---|
| 64 | 3,2× | 3,2× | **2,3–2,9×** | v1 |
| 256 | ~1,0× | 1,4× | **~2,0×** | v3 |
| 1024 | 0,33× | ~0,85× | **~1,0× (break-even)** | v3 |
| 2048 | — | — | ~0,46× | v3 |
| 4096 | — | ~0,30× | ~0,33× | v3 |

**Bilanz + architektonischer Befund (ehrlich):** v3 hob B=256 auf ~2,0× und schob den
Crossover auf **~1024–1500**. Das **„Win auch bei B≥2048"-Ziel ist NICHT erreicht** — und der
Grund ist fundamental: bei großer Batch *dominiert Compute* (nicht mehr Memory/Launch), und
die naive In-Kernel-GEMM (skalare FMAs, kein Register-Blocking) erreicht nur einen Bruchteil
des FLOP-Peaks, während cuBLAS register-geblockt nahe ans Optimum kommt. **Der fused-resident
Entwurf** (States über alle T Schritte resident → killt Launch-Overhead) **und ein
cuBLAS-Klasse-GEMM sind architektonisch gegenläufig**: Residenz frisst genau das Register-/
Shared-Budget, das ein schnelles GEMM bräuchte. Konsequenz: **der Kernel gewinnt im kleinen/
mittleren Batch-Bereich (≤~1024, 1–3×, der reale MNIST-PC-Trainingsbereich), bei B≥2048 ist
cuBLAS (`backend="pytorch"`) richtig.** Das ist selbst ein berichtenswerter Systems-Befund
(fused-resident vs tiled-GEMM-Tradeoff), kein bloßes „noch nicht optimiert". Wirklich offen
(v4): voll register-geblocktes Tiled-GEMM = im Kern cuBLAS nachbauen — jenseits sinnvollen
Aufwands; EO-Vergleichsarm + Nsight-Report bleiben die wertvolleren nächsten Schritte.

## 4e. M5 — Hook C: Generativ / Occlusion / Anomalie (gemessen 2026-06-09)

**Mechanismus implementiert + verdrahtet + getestet, aber qualitatives Ergebnis = ehrliches
Negativ.** Settling wurde um input-freies/maskiertes Settling erweitert (`pcn/settling.py`,
`clamp_input`/`input_mask` + `energy_per_sample`); `pcn/generate.py` bietet `generate`,
`inpaint`, `anomaly_scores` — alles über *denselben* Settling-Mechanismus, nur die Klemmung
ändert sich (30 Tests grün). Demo: `scripts/demo_generative.py` (PNG/JSON in `results/`).

**Ergebnis auf dem diskriminativ trainierten PCN (test_acc 90,6%):**

| Fähigkeit | Ergebnis |
|---|---|
| Generierung (Label klemmen → Bild) | graues Rauschen, **keine erkennbaren Ziffern** |
| Inpainting (verdeckte Hälfte füllen) | sichtbare Pixel exakt erhalten, verdeckte → **Rauschen** |
| Anomalie (residuale Energie) | **AUC 0,438** vs Uniform-Noise-OOD — **keine Trennung** |

**Befund (architektonisch, nicht Tuning):** Das Projekt-PCN ist **diskriminativ** (Input+Label
geklemmt, Hidden settlet) mit **feedforward-prädiktiver** Struktur (`pred_{i+1}=phi(s_i)·Wᵀ+b`).
Es lernt eine Input→Label-Abbildung — **kein generatives Modell der Eingaben.** Settling
„rückwärts" findet daher kein gelerntes Bild-Manifold → Rauschen. (Verwandt: `feedforward_init`
erzeugt ein triviales Null-Energie-Gleichgewicht; freie States müssen daher auf Null
initialisiert werden, sonst settlet nichts.) Residuale Energie trennt OOD nicht, weil bei
geklemmtem Input die freien Zustände *immer* zur (energie-0-)Feedforward-Lösung relaxieren
können.

**Konsequenz:** Generative/Occlusion/Anomalie-Fähigkeiten (Hook C) brauchen ein **generativ
trainiertes PC** (Rao-Ballard-Stil: top-down-generativ, Latent oben/Bild unten, unsupervised),
nicht den diskriminativen Supervised-PCN. `docs/01`s „Generativ"-Clamping-Zeile setzt das
implizit voraus. **Der Mechanismus/Code ist korrekt** und würde mit einem generativen PC
funktionieren — die Architektur/das Trainingsziel ist die Grenze, nicht die Implementierung.

**Weg vorwärts (M5-v2, optional):** ein generatives PC-Trainingsziel (Bild am unteren Knoten
klemmen, Latent oben frei settlen, lokales Update) als zweite `train`-Variante; dann liefern
dieselben `generate`/`inpaint`/`anomaly_scores` echte Bilder. Ist ein eigenes Modell-Variant,
kein Bugfix.

## 4f. M5-v2 — Generatives PC: Hook C funktioniert (gemessen 2026-06-09)

**Der §4e-Befund aufgelöst — mit einem *generativ* trainierten PC liefern Generierung/
Inpainting/Anomalie echte Resultate.** Statt Input→Label wird die Orientierung umgedreht:
ein PCN `[10, 256, 256, 784]` bildet **Label → Bild** ab. Trainiert *generativ* (Label am
Input + Bild am Output geklemmt, Hidden settlet, lokaler Hebb-Update — `pcn/generative.py`,
renutzt PCN/settle/weight_update unverändert). Generieren = Forward-Pass Label→Bild.

**Stabilitäts-Befund (eigenständig berichtenswert):** der 784-dim Output macht den
Hidden-Settling-Gradienten (`eps[out] @ W`) **~√784 ≈ 28× größer** als beim 10-dim-Label →
`lr_state=0.1` (gut fürs diskriminative Modell) **divergiert** (NaN). Mit `lr_state≈0.01`
(≈ /√out) stabil. Die State-LR muss mit der Output-Dimension skaliert werden.

**Ergebnisse (PCN [10,256,256,784], 30k Bilder, 10 Ep., GPU; `scripts/demo_generative_v2.py`):**

| Fähigkeit | Diskriminativ (§4e) | **Generativ (M5-v2)** |
|---|---|---|
| Generierung | graues Rauschen | **erkennbare Ziffern-Prototypen 0–9** ✅ |
| Inpainting | Rausch-Füllung | strukturierte (verschwommene) Rekonstruktion ✅ |
| Anomalie (MNIST vs Uniform-OOD) | AUC 0,44 (keine Trennung) | **AUC 1,00** (In-Dist-Energie 5,1 vs OOD 94,5) ✅ |

**Bedeutung:** Erstmals demonstriert das Projekt PCs distinktive „eine Maschinerie, viele
Aufgaben"-Eigenschaft *funktionierend* — dasselbe Settling klassifiziert (diskriminatives
Modell), generiert, füllt und detektiert Anomalien (generatives Modell), je nach Klemmung.
Die perfekte Anomalie-Trennung (AUC 1,0) ist ein echter Win: das generative Energie-Modell
erkennt OOD-Eingaben, weil es sie nicht generieren kann.

**Ehrliche Caveats:** One-hot-Input → nur 10 distinkte Outputs = **Klassen-Prototypen** (das
Per-Klasse-Mittel), keine diverse Sampling-Vielfalt; dafür bräuchte es einen kontinuierlichen
Latent + Prior (VAE-artig). Inpainting ist prototyp-getrieben (verschwommen), nicht
pixel-scharf. Uniform-Noise ist ein leichtes OOD (AUC 1,0 ist daher kein harter Test);
near-OOD (z.B. FashionMNIST, rotierte Ziffern) wäre der nächste, härtere Schritt. 33 Tests grün.

---

## 4g. Song-exakte alternierende Replikation (Fig 4d-e, gemessen 2026-06-09, n=5)

Die bisher **architektur-treueste** Replikation von Song et al. 2024 — und ein Lehrstück darüber,
wie eine adversariale Verifikation einen voreiligen Befund vor dem Eintrag rettet. Drei
unabhängige Prüf-Linsen (Literatur-Treue, Statistik, Alternativerklärungen) haben den ersten
Entwurf zerlegt; alle drei Korrekturen sind unten eingearbeitet.

**Was gebaut wurde (`run_alternating`, `cmd_alternating`):** zwei **disjunkte 5-Klassen-Tasks**
aus FashionMNIST teilen *einen* 5-Output-Kopf (erzwingt Interferenz), trainiert durch
**Minibatch-Alternation** (`swap_every=4` Updates, dann Task-Wechsel; batch=32) — Netz
`[784, 32, 32, 4-Schichten]`, **Sigmoid**, **Xavier-normal**. Gegen Songs Methods-Teil
verifiziert: 4 Schichten, 32 Hidden/Schicht, Sigmoid, Xavier, 2×5 disjunkt, geteilter Kopf,
alternierend — **auf diesen Achsen exakt getroffen** (vs. der frühere §4c-Proxy: sequenziell,
tanh, `[256,256]`).

**Framing-Korrektur (Literatur-Linse):** Songs Fig 4e ist **kein „PC ist robuster über
Lernraten"-Claim** (das ist Fig 3, Target-Alignment). Fig 4e zeigt *weniger katastrophale
Interferenz* (Caption: „mean test error beider Tasks vs. Lernrate"), mit **pro Methode separat
optimierter LR**. Wir testen also Interferenz (nicht LR-Robustheit) und tunen die LR pro Methode.

**Faithfulness-vs-Trainierbarkeits-Spannung:** Songs *exaktes* Budget (~84 Minibatch-Iterationen,
Wechsel alle 4) lässt unser Netz **auf Zufall** (alle Arme ~20–30 % bei Chance 20 %). Statt das
Budget still zu vergrößern, machen wir es zur **kontrollierten Achse**: 84 (exakt) → Konvergenz.

**Ergebnis (mean / min beider Tasks in %, ±1σ Bootstrap, n=5; min_both = der *schwächere* Task =
der eigentliche Interferenz-Indikator; LR pro Methode getunt, hier durchweg 0,1):**

| Budget (Iter) | **PC** mean/min | **BP(MSE)** mean/min | BP(CE) mean/min | Δmin (PC−BP-MSE) |
|---|---|---|---|---|
| **84** (Songs exakt) | 22,5 / 21,2 | 26,1 / 23,9 | 29,9 / 28,6 | −2,8 |
| 250 | 27,9 / 25,7 | 36,1 / 31,4 | 38,8 / 35,6 | −5,7 |
| 800 | 56,2 / 50,1 | 52,9 / 47,6 | 60,6 / 54,9 | +2,5 |
| **2500** (konvergiert) | 71,9 / 68,3 | 68,8 / 64,5 | 73,6 / 70,6 | +3,7 |

**Verdikt (kalibriert):** Bei **n=5 ist KEINE PC-vs-BP(MSE)-Differenz auf irgendeinem Budget um
≥1σ getrennt** (σ_sum 11–21 pp; die Effekte sind **direktional, nicht signifikant**). Richtung:
**PC lernt früh langsamer** (84/250: 3–6 pp hinter BP-MSE — Settling braucht mehr Schritte, um
„prospective configuration" zu erreichen), **holt bis zur Konvergenz auf und liegt dort marginal
vorn** (2500: mean +3,1, min_both +3,7, learn +3,1, retain +2,7 pp). Der min_both-Vorsprung ≥
mean-Vorsprung ist ein *schwaches* „PC balanciert beide Tasks etwas besser"-Signal — aber im
Rauschen. **BP(CE) führt durchgehend** (2500: 73,6/70,6) = derselbe **Loss-Confound** wie §4c,
keine Aussage über die Lernregel.

**Bedeutung:** Dies ist das **erste Regime, in dem vanilla-PC am Konvergenzpunkt nicht *hinter*
loss-gematchtem BP liegt** (in §4c lag PC bei Bulk/Noise/domain-IL gleichauf bis leicht zurück) —
ein hauchdünner, nicht-signifikanter Edge im Interferenz-Regime, konsistent mit Songs *Richtung*,
aber weit von seiner berichteten Stärke. Es **widerlegt Song nicht** (anderes Budget-Semantik,
andere Skala) und **bestätigt ihn nicht** (Effekt nicht von Null trennbar). Ehrlichstes Fazit:
**die treue Replikation reproduziert bei n=5 keinen klaren PC-Interferenz-Vorteil** — im Einklang
mit dem Gesamt-Verdikt „PC ≈ BP bei fairem Tuning" (§9/docs10).

**Offengelegte Abweichungen (Literatur-Linse):** (1) Trainings-*Budget* — Songs 84 Iterationen
sind in unseren Händen untrainierbar; wir berichten die ganze Budget-Achse. (2) **Festes T=20**
(kein Konvergenz-`tol`) → Unter-Settling als mögliche Mit-Ursache der frühen PC-Langsamkeit. (3)
Geteilter Kopf + Relabeling 0–4 wie bei Song. LR pro Methode getunt ✓.

**Statistik-Ehrlichkeit (Statistik-Linse):** n=5, Intervalle sind **68 % (1σ)** — *nicht* 95 %
(das war ein Label-Fehler in einer früheren Zwischenmeldung; `bootstrap_ci` rechnet bewusst 1σ
à la Song, `protocol.py:25`). Bei 800 ist die Streuung riesig (PC min-per-Seed [51,47,47,39,67]),
jede 800-Aussage ist Rauschen. Per-Seed-Werte sind im JSON
(`results/m4_alternating_song_exact_budgetsweep_s5.json`) gespeichert. Ein epoch-basierter
Vorgänger-Proxy (~20× Budget) + ein (η_x,T)-Stabilitäts-Probe wurden durch dieses exakte
Protokoll **abgelöst**.

**Kernel-Integration (Hook A × Hook B — der eigentliche Grund für den Kernel):** §4g ist das
Regime, *für das* der fused Settling-Kernel geplant war — PC-Settling bei **kleinem Batch (32)**
ist launch-overhead-gebunden (§4d), und genau das eliminiert der Kernel. Der Kernel wurde dafür
um **Sigmoid (`act==2`)** erweitert (vorher stille tanh-Falle: `_act_id` gab für Sigmoid
fälschlich tanh zurück → behoben, Guard gegen unbekannte Aktivierungen) und über `backend="cuda"`
(`PCN_CUDA_KERNEL=1`) in den Experiment-Pfad (`run_alternating`, Train **und** Eval) gefädelt.
Korrektheit: Kernel-vs-PyTorch **allclose ~1e-6** für tanh+sigmoid auf beiden Dispatch-Pfaden
(v1/v2, `scripts/verify_kernel.py`). End-to-End auf der **echten §4g-PC-Last** (Budget 800, n=5,
`scripts/bench_alternating_backend.py`): **1,45× Wall-Clock** (103,9 s → 71,6 s) bei **Per-Seed
bit-identischen** Ergebnissen (max. Δ 0,00 pp). Der bescheidene Faktor (nicht die 3,2× vom
Mikrobenchmark @ B=64) ist ehrlich: das Training gewinnt im Small-Batch-Regime, aber die Eval
settled bei Batch 512 — verdünnt. **Bedeutung:** der hand-geschriebene Kernel ist damit kein
Benchmark-Artefakt mehr, sondern der **Motor einer realen Studie** — die konkrete Verbindung von
Beitrag A (Systems) und B (Science).

---

## 4h. Optionale Tiefen (M8, gemessen 2026-06-10): iPC, EWC, Kernel-Tiefe, n=10

Vier eigenständige Erweiterungen — jede implementiert *und* validiert, nicht nur skizziert.

**iPC (Incremental Predictive Coding, Salvatori et al. 2024) — `update_variant="ipc"`.** Statt voll
zu settlen und *einmal* die Gewichte zu aktualisieren, werden bei iPC nach **jedem** Inferenzschritt
Zustände *und* Gewichte aus denselben Fehlern aktualisiert (`pcn/learning.py:_ipc_step`,
`train_epoch_ipc`). Korrektheits-Anker (Test): mit `lr_weight=0` reproduziert ein iPC-Schritt **exakt**
einen Standard-Settling-Schritt (iPCs State-Update *ist* das SO-Update, es bewegt nur zusätzlich die
Gewichte). **Befund:** iPC ist deutlich **lr-sensitiver** — es macht T-mal mehr Weight-Updates pro
Batch, also ist die effektive Lernrate ~T× größer; bei Standard-`eta_w=0.02` divergiert es (NaN), bei
**`eta_w≈0.005` (~lr/T)** läuft es stabil und erreicht im schnellen Regime (MNIST, 3k Samples, 2
Epochen) **74,3 % vs. 63,1 % für Standard-PC** — iPC lernt pro Schritt schneller (konsistent mit
Salvatori et al.). Lehre: iPC braucht eine ~T× kleinere Gewichts-LR, dann ist es kompetitiv/besser.

**EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017) — Continual-Baseline.** Ein dritter
class-IL-Arm (`run_class_il(method="ewc")`): nach jeder Task wird die diagonale empirische **Fisher**
(mittlere quadrierte Gradienten) + die optimalen Gewichte gespeichert, und spätere Tasks erhalten den
quadratischen Anker `(λ/2)·Σ F_i(θ_i−θ*_i)²` (`_ewc_fisher`/`_ewc_penalty`). **Validiert e2e**
(Split-MNIST 5×2, lernendes Regime): EWC **vergisst weniger** als das gematchte BP(MSE) —
**BWT −41,1 % vs. −45,7 %**, **Final-ACC 65,7 % vs. 62,0 %**, bei gleichem Lernen (learn-ACC 98,5 %).
Genau der erwartete EWC-Effekt; im `classil`-Treiber als Arm verfügbar. (Wichtig, §4c-Lektion: im zu
schwachen Regime — BP bei Zufall — ist EWC erwartungsgemäß inert, weil nichts zu vergessen ist.)

**Kernel für BELIEBIGE Tiefe (n≠3).** Der fused Settling-Kernel war auf 2 Hidden Layer
(`model.n==3`) spezialisiert. Neu: ein **general-depth per-sample Kernel** (`settle_kernel_deep` /
`pcn_settle_so_deep` in `settling_kernel.cu`), der Gewichte/Zustände/Biases als **Arrays von
Device-Pointern** (als int64-Adressen) plus per-Layer-Größen und **dynamisch berechnete
Shared-Memory-Offsets** entgegennimmt → settled PCNs *beliebiger* Tiefe mit demselben Jacobi-Update
wie `_settle_pytorch`. Dispatch (`settling_cuda.py`): `n==3` nutzt weiterhin den optimierten
v1/v2-Pfad, jede andere Tiefe den Deep-Kernel. **Validiert** (`scripts/verify_kernel.py`): über die
Tiefen **2/3/4/5** (1–4 Hidden Layer) × tanh+sigmoid × B=32/256 — **alle 16 Konfigurationen
allclose ~1e-6** zu PyTorch. (Fallstrick gefixt: auf Windows ist `long` 32-bit, die Pointer-Arrays
müssen `int64_t` sein — sonst Typ-/Größen-Mismatch.) Per-sample only (das launch-bound
Small-Batch-Regime); ein register-geblocktes Tiled-GEMM für große Batch bei großer Tiefe bleibt
offen. **Bedeutung:** die Tiefen-Spezialisierung (eine der dokumentierten Kernel-Limitationen,
`docs/15`) ist damit aufgehoben — der Kernel ist nicht mehr an die 2-Hidden-Layer-Topologie gebunden.

---

## 5. Verweis

Der konkrete, geordnete Umsetzungsplan (mit verifizierten Quellen und Schritt-für-Schritt-
Reihenfolge) steht in **`docs/13_umsetzungsplan.md`**. Die Quellen-Provenance ist in
**`docs/11_quellen.md`** nachgezogen (verifiziert / korrigiert / als second-hand markiert).
