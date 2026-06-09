# 13 — Umsetzungsplan (Schritt für Schritt)

Dieser Plan ist die geordnete, ausführbare Roadmap für das From-scratch-PCN (State-Optimization, reines PyTorch ohne Autograd, optionaler CUDA-Settling-Kernel). Er setzt direkt auf dem auditierten Stand auf (siehe `docs/12_projektanalyse_und_befunde.md`).

**Harte Reihenfolge-Regel (nicht abweichen):** Erst die PyTorch-Basis lauffähig + validiert (M0–M1), dann Präzision/Konvention klären (M2), dann Baselines (M3), dann die Experiment-Hooks (M4–M5). Der CUDA-Kernel (M6) ist **optional** und wird **bewusst zuletzt** zugeschaltet. Paper + Such-Loop (M7) docken nur als dünner Layer an `train_and_eval` an.

**Konventionen:** `uv` ausschließlich (nie `pip`); kein `loss.backward()`/Autograd im PC-Lernen (Ausnahme: der reine Gradient-Check in M1, der bewusst Autograd nur als externe Referenz nutzt); Doku Deutsch, Code-Bezeichner Englisch.

**Quellen-Kürzel (verifiziert, siehe `docs/11_quellen.md`):**
- *Bogacz 2017* — J. Math. Psychol. 76:198–211, DOI 10.1016/j.jmp.2015.11.003 (Inferenz-/Hebb-Gleichungen).
- *Whittington & Bogacz 2017* — Neural Computation 29(5):1229–1262, DOI 10.1162/NECO_a_00949 (PC≈BP, lokale Hebb-Plastizität).
- *Millidge, Seth & Buckley 2021* — arXiv:2107.12979 (Review; **Autoren: Seth, NICHT Tschantz**; präzisions-gewichtete freie Energie Gl. 11–13).
- *Song et al. 2024* — Nature Neuroscience 27(2):348–358, DOI 10.1038/s41593-023-01514-1 (Prospective Configuration; faire PC-vs-BP-Methodik).
- *Zahid, Guo & Fountas 2023* — Neural Computation 35(12):1881–1909, DOI 10.1162/neco_a_01620 (arXiv:2304.02658; Kritik der BP-äquivalenten PC-Varianten; **peer-reviewed, kein bloßer Preprint**).
- *Goemaere et al. 2025/2026 (ePC)* — arXiv:2505.20137, ICML 2026 (Error-Optimization/ePC; Signal-Decay; T≈5×Tiefe für sPC). **Titel: „ePC … Exponential Signal Decay…", NICHT „EO"/„Digital Simulation".** Code: github.com/cgoemaere/error_based_PC.
- *Salvatori et al. 2024 (iPC)* — arXiv:2212.00720, ICLR 2024 (inkrementelles PC, simultane Weight+State-Updates; per arXiv-ID + Venue zitieren).
- *Qi et al. 2025* — arXiv:2506.23800 (Präzisions-Schedule: isotropic/spiking/decaying; Heuristik T ≥ L).
- *Innocenti, Achour & Bogacz 2026* — arXiv:2602.07697 (Infinite-Width/Depth-Limit; PC=BP **nur** für lineare Residual-Netze, width ≫ depth, equilibrierte Aktivitäten). **Verifiziert real, aber zukunftsdatiert → vor Zitat in voller Länge lesen.**
- *PCX* — arXiv:2407.01163, ICLR 2025 (Benchmark-Lib, JAX; A100: PC ~5.33 s vs BP ~1.61 s/Epoche; Σ=I fixiert; Inference nicht voll parallelisierbar).
- *JPC* — arXiv:2412.03676 (JAX, ODE-Solver für Settling; SSE-Energie, Π=I implizit).
- *PRECO* — github.com/bjornvz/PRECO; van Zwol et al. ACM Comput. Surv., DOI 10.1145/3797870 / arXiv:2407.04117 (PyTorch-PCN-Referenz).
- *Torch2PC* — github.com/RobertRosenbaum/Torch2PC (PyTorch; `PCInfer`, Variante `'Exact'` reproduziert BP-Gradienten → Cross-Check-Baseline).
- *Lopez-Paz & Ranzato 2017 (GEM)* — arXiv:1706.08840 (ACC/BWT/FWT-Definitionen).
- *Avalanche* — Lomonaco et al. 2021, arXiv:2104.00405; `avalanche-lib==0.6.0` (Split/Permuted-MNIST, Metriken).
- *PyTorch Custom-Ops* — docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html, `torch.utils.cpp_extension` (`load_inline`/`CUDAExtension`), `torch.library` (`custom_op`/`register_fake`/`opcheck`), CUDA-Semantik (async → `torch.cuda.synchronize`); github.com/pytorch/extension-cpp (mymuladd-Gerüst, Pfad `extension_cpp/extension_cpp/csrc/...`).

---

## M0 — Umgebung & Lauffähigkeit

**Ziel:** Die existierende PyTorch-Basis startet reproduzierbar, alle 4 Tests grün, ein kurzer MNIST-Lauf zeigt fallende Energie und plausible Accuracy. Erst danach wird irgendetwas verändert.

> **Empirische Ausgangsbasis (gemessen 2026-06-09, CPU, torch 2.12.0+cpu):** `pytest` 4/4 grün. MNIST-Pipeline läuft end-to-end. **Accuracy ist erreichbar, aber die Defaults underfitten:** volle Daten / 10 Epochen → **83,1 %** mit dem Default `lr_weight=1e-3`, **92,2 %** mit `lr_weight=0.02`. Der kleine Smoke-Lauf (4k Bsp., 3 Ep.) liefert nur 26,8 % — wenige Updates × kleine LR. **Konsequenz für M1:** `lr_weight=1e-3` ist als Default zu niedrig; anheben (Kandidat 0.01–0.02) oder dem Such-Loop (M7) überlassen, aber den Default-Kommentar entsprechend ehrlich halten.

**Tasks:**
- [x] `uv sync` (Basis: torch, torchvision, numpy) und `uv sync --extra dev` (pytest, jupyter, matplotlib) ausführen.
- [x] `uv run pytest -q` — die 4 Tests in `tests/test_settling.py` müssen grün sein (offline, kein MNIST-Download nötig).
- [x] `uv run python scripts/train_mnist.py --epochs 2 --hidden 256 256 --T 20` ausführen; Konsolen-Output (`test accuracy`, `train time`, `device`) festhalten.
- [ ] In `scripts/train_mnist.py` temporär `record_energy=True` über einen Mini-Lauf prüfen oder eine Wegwerf-Zelle nutzen, um sicherzustellen, dass `settle(..., record_energy=True)` eine **monoton fallende** Energiekurve liefert (Sanity-Check vor M1).
- [x] Hardware-Status dokumentieren: `torch.cuda.is_available()` lokal `True` (RTX 3080 Ti, CUDA 12.4 global). **Aber:** das `uv`-`.venv` hat `torch 2.12.0+cpu` gezogen → für M6 muss gezielt ein CUDA-Build von torch installiert werden (uv-Index-Config + Re-Sync).

**Akzeptanzkriterium:** `pytest` 4/4 grün; `scripts/train_mnist.py` läuft ohne Fehler durch und erreicht plausible `test_acc` (≳ 0,90 bei genügend Epochen/LR — siehe Ausgangsbasis); die Energiekurve eines Trainings-Settles fällt monoton. Output-Werte in `docs/12` notiert.

**Relevante Quelle(n):** `CLAUDE.md` (Befehle, Reihenfolge-Regel); `docs/02` (Trainingsprozedur); Settling-Mathematik gegen Bogacz 2017 / Millidge et al. 2021 bereits validiert.

---

## M1 — `train_and_eval`-Interface an `docs/02` angleichen + Konvergenzkriterium + Test-Härtung

**Ziel:** `pcn/api.py:train_and_eval` erfüllt den in `docs/02` dokumentierten Vertrag (der „dünne Layer" für Phase 4). Config-Keys `eta_x`/`eta_w` werden akzeptiert, unbekannte Keys gewarnt, die Rückgabe enthält `energy_curve`, `noise_robustness`, `settling_steps_to_converge`. Ein Konvergenzkriterium ersetzt das blinde Fixed-T-Settling. Der schwache Test wird gehärtet und ein numerischer Gradient-Check gegen Autograd abgesichert.

**Tasks:**
- [ ] **Config-Aliasing** in `pcn/api.py`: `eta_x` → `lr_state`, `eta_w` → `lr_weight` akzeptieren (dokumentierte Spec-Namen aus `docs/02` haben Vorrang; intern weiter `lr_state`/`lr_weight`). Neue Default-Keys ergänzen: `precision_schedule: "isotropic"`, `update_variant: "standard"`. Bei unbekannten Config-Keys ein `warnings.warn(...)` statt stiller Ignoranz.
- [ ] **Default `lr_weight` korrigieren** (Befund aus M0): von `1e-3` auf einen nicht-underfittenden Wert (Kandidat `0.01`) anheben, ODER den Kommentar ehrlich machen („1e-3 underfittet; tunen via Such-Loop").
- [ ] **Rückgabe erweitern** in `pcn/api.py`: `energy_curve` (pro Epoche der Mittelwert der finalen Trainings-Settle-Energie, via `settle(..., record_energy=True)` in `train_epoch` durchgereicht), `noise_robustness` (Aufruf von `pcn/evaluate.py:noise_robustness` über den `test_loader`), `settling_steps_to_converge` (mittlere Schrittzahl bis zum Konvergenzkriterium aus dem nächsten Task).
- [ ] **`noise_robustness` verdrahten:** in `pcn/evaluate.py` ist die Funktion bereits vorhanden — in `train_and_eval` einmal am Ende aufrufen und ins Metrik-Dict legen (Default-Sigmas `(0.0, 0.25, 0.5, 1.0)`).
- [ ] **`train_epoch`/`weight_update`-Signatur** in `pcn/learning.py`: optionalen Parameter durchreichen, damit die finale Settle-Energie pro Batch/Epoche zurückgegeben und in `api.py` zu `energy_curve` aggregiert werden kann.
- [ ] **Konvergenzkriterium** in `pcn/settling.py:_settle_pytorch`: optionaler Parameter `tol` (relative Energie-Änderung) + `max_T`. Stoppe, wenn `abs(E_t − E_{t−1}) / (abs(E_{t−1}) + 1e−12) < tol` (oder `max|grad_s| < tol`), und gib die tatsächliche Schrittzahl zurück. `T` bleibt als oberes Limit erhalten (Rückwärtskompatibilität: `tol=None` → reines Fixed-T-Verhalten wie bisher). Heuristik im Docstring: `T ≥ L` als Floor (NICHT die alte „L < T < 2L"-Regel, die so in der Literatur nicht belegt ist).
- [ ] **`DEFAULT_CONFIG`-Kommentar** in `pcn/api.py` korrigieren: die Notiz „rule of thumb: L < T < 2L" durch „T ≥ L (Qi et al. 2025); obere Grenze decay-abhängig" ersetzen.
- [ ] **`docs/02` ↔ Code synchronisieren:** Entweder Code-Keys an `docs/02` (`eta_x`/`eta_w`) anpassen oder `docs/02` um den Alias-Hinweis ergänzen — in beiden Dateien dieselbe Sprache. Rückgabe-Beispiel in `docs/02` so lassen (es ist bereits der Zielzustand).
- [ ] **Schwachen Test fixen** in `tests/test_settling.py:test_training_loop_reduces_output_error`: `first` wird aktuell nur bei Iteration 0 gesetzt — die Assertion ist quasi trivial. Stattdessen Fehler **vor** dem Loop und **nach** dem Loop messen und `err_after < 0.5 * err_before` (oder ähnlich strenge Schranke) prüfen.
- [ ] **Numerischer Gradient-Check** neu in `tests/test_settling.py` (oder `tests/test_gradients.py`): ein winziges Netz (z. B. `[4,3,2]`, `identity`-Aktivierung), den per Hand abgeleiteten State-Gradienten aus `_settle_pytorch` (`eps[k] − phi'(s_k)·(eps[k+1]@W[k])`) gegen `torch.autograd.grad` der Energie `E = 0.5·Σ‖eps_k‖²` vergleichen (`torch.allclose`, `atol=1e-5`). Analog ein Check des Hebb-Weight-Gradienten gegen Autograd-`dE/dW`. **Wichtig:** Autograd hier nur als externe Referenz im Test, nicht im Produktivpfad.
- [ ] **Konvergenz-Test** ergänzen: mit `tol` gesetzt stoppt `settle` vor `max_T` und gibt eine plausible `settling_steps_to_converge` zurück.

**Akzeptanzkriterium:** `train_and_eval({})` liefert ein Dict mit allen Keys aus `docs/02` (`test_acc`, `energy_curve`, `noise_robustness`, `train_time_s`, `settling_steps_to_converge`, `config`); `train_and_eval({"eta_x": 0.1, "eta_w": 1e-3})` funktioniert; unbekannter Key erzeugt eine Warnung. Neuer Gradient-Check-Test grün (`allclose` zwischen handgeleitetem und Autograd-Gradienten). Gehärteter Trainings-Test grün. `pytest` insgesamt grün.

**Relevante Quelle(n):** `docs/02` (Vertrag); `docs/04` (Phase-4-Suchraum benötigt `precision_schedule`/`update_variant`); Bogacz 2017 Gl. 8/9 + Millidge et al. 2021 Gl. 12 (State-Gradient zur Autograd-Verifikation); Qi et al. 2025 (T ≥ L); Innocenti et al. 2026 + Song et al. 2024 (PC-Gradienten sind nur **bei erreichtem Gleichgewicht** aussagekräftig → motiviert das Konvergenzkriterium statt Fixed-T).

---

## M2 — Präzision Π entscheiden + Index-Konventions-Notiz in `docs/01`

**Ziel:** Die Diskrepanz „Π in `docs/01`/`docs/04` als Kern dokumentiert, im Code aber abwesend (hart Π=I)" auflösen — quellenbasiert. **Empfehlung:** Doku-Anspruch herabstufen (Π=I ist der mainstream-Default) und Π als *optionalen, scalar-pro-Layer Präzisions-Schedule* (spiking/decaying) implementieren, der nur bei tiefen Netzen Nutzen bringt. Plus: die Index-Konvention (top-down vs. feedforward) explizit dokumentieren.

> **Status (2026-06-09):** Doku-Teil **erledigt** — `docs/01` enthält jetzt die Π=I-
> Klarstellung (inkl. „früher als Kern dokumentiert, nicht durch Code gedeckt"), die
> Index-Konventions-Notiz (top-down ↔ feedforward) und die korrigierte T≥L-Heuristik.
> `precision_schedule`/`update_variant` sind als Config-Keys vorhanden, werfen aber für
> nicht-implementierte Werte ein `NotImplementedError` (kein stilles No-Op).
> **Offen (bewusst aufgeschoben):** die eigentliche scalar-pro-Layer-Schedule-Mathematik
> (spiking/decaying). Grund: die exakten Profile stammen aus Qi et al. 2025
> (arXiv:2506.23800) und sind noch nicht glyphengenau verifiziert — daher **erst das Paper
> im Volltext lesen, dann implementieren** (statt Formeln zu erfinden). Die Implementierung
> muss Π konsistent in `energy`/`_settle_pytorch`/`weight_update` anwenden (sonst divergieren
> Settling- und Lernziel) und per Autograd-Gradient-Check (analog M1) abgesichert werden.

**Tasks:**
- [ ] **Doku-Downgrade:** In `docs/01_architecture.md` (und ggf. `docs/04`) den Anspruch „Π ist Kern" ersetzen durch „isotrope Präzision Π=I angenommen (Standard moderner MNIST-PC-Implementierungen: PCX fixiert Σ=I, JPC nutzt reine SSE-Energie); nicht-identische Präzision ist ein optionales/fortgeschrittenes Feature, das primär tiefen Netzen (> 7 Layer) nützt."
- [ ] **Π als Schedule implementieren (scalar-pro-Layer, optional):** in `pcn/settling.py` den Energie- und State-Update-Term um einen Faktor `pi_k` (Skalar pro Layer/Zeitschritt) erweitern: `eps[k] → pi_k · eps[k]`. `precision_schedule ∈ {"isotropic","spiking","decaying"}` aus der Config (M1) auf konkrete `pi_k(t,l)` abbilden: isotropic = 1 überall; spiking = `alpha` bei `l = L − t`, sonst 1; decaying = exponentiell-normalisiertes Profil (Hyperparameter `k ∈ {1.0,1.5,2.0}`). Default bleibt `isotropic` → numerisch identisch zum heutigen Verhalten.
- [ ] **`weight_update` konsistent halten** in `pcn/learning.py`: falls `pi_k ≠ 1`, denselben Faktor im Hebb-Update anwenden (sonst divergieren Settling- und Lernziel).
- [ ] **Regression sichern:** Test, dass `precision_schedule="isotropic"` bit-für-bit (bzw. `allclose`) dieselbe Energie/Accuracy liefert wie vor M2 (kein stilles Verhaltensändern des Defaults).
- [ ] **Index-Konventions-Notiz** in `docs/01_architecture.md`: expliziter Absatz, der `docs/01` (top-down-Indizierung `eps_{l−1}`/`W_{l−1}`) ↔ `docs/09`+Code (feedforward `eps_{k+1}`) mappt und festhält, dass beide unter einem Index-Flip äquivalent sind (Millidge et al. 2021 Gl. 12 schreibt beide Terme simultan). Ziel: die Copy-Trap entschärfen.

**Akzeptanzkriterium:** `docs/01` enthält die Π=I-Klarstellung **und** die Index-Konventions-Notiz; kein Doku-Satz behauptet mehr „Π ist implementiert/Kern", der nicht durch Code gedeckt ist. `precision_schedule="spiking"`/`"decaying"` laufen ohne Fehler; `"isotropic"` reproduziert die M1-Metriken (`allclose`). Regressions-Test grün.

**Relevante Quelle(n):** PCX arXiv:2407.01163 (Σ=I fixiert) und JPC arXiv:2412.03676 (Π=I implizit) → Begründung für Downgrade; Qi et al. 2025 arXiv:2506.23800 (Schedule-Definitionen spiking/decaying, scalar-pro-Layer); Millidge et al. 2021 arXiv:2107.12979 Gl. 11/12/21 (präzisions-gewichtete Energie + Lernregel, falls später gelernte Π); ePC arXiv:2505.20137 (Signal-Decay als Motivation für Schedules in tiefen Netzen); `docs/09` (Feedforward-Indizierung).

---

## M3 — Non-CUDA-Baselines: BP-MLP-Referenz + Benchmark-Harness

**Ziel:** Eine BP-trainierte MLP-Referenz mit **identischer Architektur, Initialisierung und Datenpipeline** zum PCN, plus ein Benchmark-Harness, das Wall-Clock pro Settling-Schritt/Epoche korrekt misst (`torch.cuda.synchronize` vor und nach der Messung, Warmup). Dies ist die Voraussetzung für jeden fairen PC-vs-BP-Vergleich (M4) und der Messpunkt, gegen den der optionale Kernel (M6) später antritt.

**Tasks:**
- [ ] **BP-MLP-Referenz** neu in `pcn/baselines.py`: `nn.Module`-MLP mit denselben `layer_sizes` (`[784, *hidden, 10]`), derselben Initialisierung (Xavier/normal, gleicher Seed) und derselben MNIST-Pipeline wie `pcn/api.py:_mnist_loaders`. Standard-SGD, `CrossEntropyLoss`. Funktion `train_and_eval_bp(config) -> dict` mit **gleichem Rückgabe-Schema** wie `train_and_eval` (Lernregel ist die einzige unabhängige Variable). *(Hier ist `loss.backward()` erlaubt — es ist die BP-Baseline, nicht das PC-Lernen.)*
- [ ] **Optionaler Cross-Check gegen Torch2PC:** in einem Test/Notebook die PCN-Updates qualitativ gegen Torch2PC `PCInfer(..., ErrType='Exact')` (reproduziert BP-Gradienten) gegenprüfen — als unabhängige Bestätigung zusätzlich zum Autograd-Check aus M1.
- [ ] **Benchmark-Harness** neu in `pcn/benchmark.py`: Funktion `time_settle(model, ..., backend, warmup=2, iters=20)`, die ≥ 1 Warmup-Iteration macht, dann `torch.cuda.synchronize()` **vor** Start und **vor** Stop ruft. Bevorzugt CUDA-Events (`Event(enable_timing=True)`, `start.record()/end.record()`, `synchronize()`, `elapsed_time`); Fallback `time.perf_counter` mit Sync-Klammer. Misst (a) ms pro Settling-Schritt, (b) ms pro Epoche, für `backend="pytorch"` und (später) `"cuda"` sowie für die BP-MLP-Referenz.
- [ ] **Achsentrennung dokumentieren** (für `docs/08`/`docs/10`): „Episoden/Samples bis Ziel-Accuracy" (Song-Achse) getrennt von „Compute-Zeit pro Update" (Zahid-Achse) — nicht vermischen.
- [ ] **Referenzpunkt notieren:** PCX meldet auf A100 ~3,3× PC/BP-Overhead pro Epoche (PC 5,33 s vs BP 1,61 s) — als Erwartungs-Anker für das eigene Harness-Ergebnis in `docs/12`.
- [ ] **Test** neu: `train_and_eval_bp` läuft auf MNIST und erreicht ≳ 95 % (Sanity, dass die BP-Baseline korrekt ist); Harness gibt monotone, nicht-null Zeiten zurück.

**Akzeptanzkriterium:** `pcn/baselines.py:train_and_eval_bp` liefert ein zu `train_and_eval` schema-kompatibles Dict und erreicht plausible BP-Accuracy; `pcn/benchmark.py:time_settle` produziert reproduzierbare Zeiten mit korrekter Sync-Klammer (CPU- wie GPU-Pfad); ein erstes PC-vs-BP-Zeit/Accuracy-Tableau ist in `docs/12` abgelegt.

**Relevante Quelle(n):** Song et al. 2024 (faire-Vergleichs-Protokoll: identische Arch/Init/Daten, Lernregel = einzige Variable; LR pro Methode separat tunen; ≥ Seeds + CIs); Zahid et al. 2023 (Compute-Zeit-Achse; BP-äquivalente PC-Varianten sind beweisbar nicht schneller); PyTorch CUDA-Semantik (`torch.cuda.synchronize`, Warmup, Events); PCX arXiv:2407.01163 (A100-Timing-Referenz); Torch2PC (`'Exact'`-Cross-Check); PRECO (PyTorch-PCN-Referenzimplementierung).

---

## M4 — Hook B: PC-vs-BP Bio-Regime-Experimente

**Ziel:** Die in der Literatur belegten PC-Vorteile auf MNIST-Klasse-Aufgaben reproduzieren bzw. falsifizieren: Online-/Batch-1-Lernen, Small-Data, Continual Learning (Split/Permuted-MNIST via Avalanche, mit Forgetting/BWT), Rausch-Robustheit, Tiefen-Skalierung — jeweils mit ≥ 3 Seeds und Konfidenzintervallen, fair gegen die BP-MLP-Baseline aus M3.

**Tasks:**
- [ ] **`uv add avalanche-lib`** (Import-Name `avalanche`, Version `0.6.0` pinnen) — die in `pyproject.toml` als „experiments"-Extra deklarierte, bislang ungenutzte Dependency aktivieren; `uv sync --extra experiments`.
- [ ] **Continual-Learning-Harness** neu in `pcn/experiments/continual.py`: `SplitMNIST(n_experiences=5, seed=...)` (Klassen `[0,1],[2,3],…,[8,9]`) und `PermutedMNIST(n_experiences=10, seed=...)`. PCN als Avalanche-Strategie: Subklasse von `avalanche.training.templates.SupervisedTemplate`, Hooks `training_epoch()/forward()/criterion()/backward()/optimizer_step()` überschreiben — `backward()/optimizer_step()` werden No-Ops, das PCN-Update (Settling + Hebb) läuft in `training_epoch()`. BP-Arm = derselbe Template mit `nn.Module`-MLP + SGD über **dasselbe** Benchmark-Objekt und **dieselbe** `EvaluationPlugin`.
- [ ] **Metriken** via `EvaluationPlugin(accuracy_metrics(stream=True), forgetting_metrics(experience=True, stream=True), bwt_metrics(...), forward_transfer_metrics(...))`; ACC/BWT/FWT nach GEM-Definitionen (R-Matrix) plus Per-Task-Accuracy berichten.
- [ ] **Online/Small-Data-Läufe:** über `train_and_eval` (M1) `batch_size=1` (online) bzw. `limit_train ∈ {100, 1000}` (small-data) gegen die BP-Baseline; Samples/Episoden bis Ziel-Accuracy berichten (Song-Achse, getrennt von Compute).
- [ ] **Rausch-Robustheit:** `noise_robustness` (M1) für PCN vs. BP-MLP über die Sigma-Sweep-Achse.
- [ ] **Tiefen-Skalierung / Target-Alignment:** Accuracy bzw. „target alignment" (Cosinus zwischen Zielrichtung und tatsächlicher Lernrichtung) über zunehmende Tiefe; prüfen, ob BP-Alignment mit Tiefe abfällt, PC stabil bleibt (Song-Befund).
- [ ] **Statistik:** jedes Experiment mit ≥ 3 Seeds (für leichte MNIST-Tasks gern 10), Fehlerbalken als 68 %-Bootstrap-CI; LR-Grid pro Methode separat (gleiches Kandidaten-Set, dokumentiert).
- [ ] **`docs/08`/`docs/10` aktualisieren:** Protokoll (identische Arch/Init/Daten, LR-Grid, Seed-Zahl) und Ergebnisse eintragen; Framing-Falle vermeiden (Song = Sample-Effizienz, Zahid = Compute — keine direkte Kontradiktion).

**Akzeptanzkriterium:** `pcn/experiments/continual.py` läuft Split- **und** Permuted-MNIST für beide Arme über dieselbe Benchmark/Plugin durch und gibt ACC/BWT/FWT aus; mindestens eine reproduzierbare, mit CI versehene Aussage steht fest (z. B. „PCN hat geringeres Forgetting / höhere Small-Data-Accuracy / bessere Rausch-Robustheit als BP-MLP bei identischer Architektur") — oder die Gegenthese sauber widerlegt; Ergebnisse in `docs/08`/`docs/10`.

**Relevante Quelle(n):** Song et al. 2024 (Regime-Liste online/continual/changing-env/limited-data/RL; faires Protokoll, Xavier-Init, n=10/3 Seeds, 68 %-Bootstrap-CI, LR-Grid pro Modell); Zahid et al. 2023 (Compute-Achse, Bayes-Interpretation); GEM arXiv:1706.08840 (ACC/BWT/FWT-Formeln); Avalanche arXiv:2104.00405 + `avalanche-lib==0.6.0` (Split/Permuted-MNIST-Signaturen, Metrik-API, `SupervisedTemplate`); van Zwol et al. arXiv:2407.04117 (Caveat: Hyperparameter-Suchraum explodiert → Grid explizit fixieren).

---

## M5 — Hook C: Generativ / Occlusion / Anomalie

**Ziel:** Den fehlenden input-freien / label-geklammerten Settling-Pfad ergänzen, der PCs generative Fähigkeiten zeigt (Klasse → Bild generieren, verdeckte Pixel rekonstruieren, Anomalie über residuale Energie detektieren), und die zugehörigen Jupyter-Demos liefern.

**Tasks:**
- [ ] **Input-freier Clamping-Pfad** in `pcn/settling.py`/`pcn/evaluate.py`: `feedforward_init` und `settle` so erweitern/parametrisieren, dass auch der **Input** (`states[0]`) frei settlen kann, während der **Output** (Label) geklammert ist — symmetrisch zum bestehenden `clamp_output`. Z. B. Parameter `clamp_input: bool` bzw. eine Clamp-Maske, sodass Generation (Label fix, Pixel frei) möglich wird.
- [ ] **Generierung** in `pcn/generate.py`: `generate(model, y_onehot, T, lr_state)` → settelt bei geklammertem Label zu einem Bild; pro Ziffernklasse ein Sample erzeugen.
- [ ] **Occlusion/Inpainting:** Settling mit teilweise geklammertem Input (sichtbare Pixel fix, verdeckte frei) → rekonstruierte Pixel auslesen.
- [ ] **Anomalie:** finale residuale Energie `energy(model, states)` als Score; In-Distribution- vs. Out-of-Distribution-Eingaben (z. B. rotiert/Rauschen) trennen.
- [ ] **Notebooks** in `notebooks/`: `02_classification.ipynb`, `03_generative.ipynb`, `04_occlusion.ipynb`, `05_anomaly.ipynb` (Phase-2-Demos aus `docs/03`), inklusive der MLP-Baseline-Gegenüberstellung aus M3.
- [ ] **Smoke-Test** in `tests/`: input-freier Settle-Pfad läuft und reduziert Energie monoton; Generierung gibt korrekt geformten `[B, 784]`-Tensor zurück.

**Akzeptanzkriterium:** `pcn/generate.py:generate` produziert für jede der 10 Klassen ein settled Bild; Occlusion-Demo rekonstruiert verdeckte Regionen sichtbar; Anomalie-Score trennt In- von Out-of-Distribution; die vier Notebooks laufen end-to-end (`jupyter nbconvert --execute`); input-freier-Pfad-Test grün.

**Relevante Quelle(n):** `docs/03` (Phase-2-Demo-Spezifikation); pypc (github.com/infer-actively/pypc, `scripts.generative` als Template für den generativen/clamping-Pfad); PRECO (PyTorch-Referenz für PCN-Generierung); Rao & Ballard 1999 (generatives Predictive-Coding-Modell als konzeptioneller Anker).

---

## M6 — OPTIONAL Hook A: CUDA-Settling-Kernel

**Ziel:** Den optionalen fusionierten CUDA-Settling-Kernel bauen — **correctness-first** gegen das PyTorch-Backend, dann gegen eine Benchmark-Matrix (PyTorch-SO vs. fused-CUDA-SO vs. EO/ePC vs. BP). Wird **bewusst** über `backend="cuda"` zugeschaltet, **niemals** Default. Nur sinnvoll, wenn eine CUDA-GPU verfügbar ist (siehe M0).

> **Prerequisite (Befund M0):** Das `uv`-`.venv` hat `torch+cpu` gezogen. Vor M6 ein CUDA-torch-Build installieren (uv-Index für die passende CUDA-Version konfigurieren, z. B. cu124 für die lokale RTX 3080 Ti / CUDA 12.4) und `nvcc`/CUDA-Toolkit sicherstellen. Erst dann ist `load_inline`/`CUDAExtension` baubar.

**Tasks:**
- [ ] **JIT-Prototyp** in `pcn/kernels/settling_cuda.py`: `torch.utils.cpp_extension.load_inline(name="pcn_settle", cpp_sources=[...], cuda_sources=[...], functions=["settle"], with_cuda=True, extra_cuda_cflags=["-O3"], verbose=True)`. Op via `TORCH_LIBRARY(pcn, m){ m.def("settle(...) -> ...") }` + `TORCH_LIBRARY_IMPL(pcn, CUDA, m){ m.impl("settle", &settle_cuda); }` registrieren (NICHT pybind11-direkt). Aufruf aus Python als `torch.ops.pcn.settle(...)`. Standard-Launch-Idiom: `stream = at::cuda::getCurrentCUDAStream(); kernel<<<(numel+255)/256, 256, 0, stream>>>(...)`; vor Pointer-Zugriff `a.contiguous()` + `TORCH_CHECK(dtype/device)`.
- [ ] **`is_available()` umstellen** in `pcn/kernels/__init__.py`: von hartem `return False` auf einen echten Build-/Import-Check, sobald der Kernel kompiliert; bis dahin bleibt der `NotImplementedError`-Pfad in `pcn/settling.py:settle` intakt.
- [ ] **Korrektheit zuerst:** `torch.library.opcheck(torch.ops.pcn.settle, args)` (prüft schema/fake/autograd-registration/aot-dispatch) + `register_fake("pcn::settle")` (Meta-Kernel, gibt `torch.empty_like(states[-1])` zurück) für `torch.compile`/Shape-Inference. **Autograd-Registrierung entfällt** (Projekt nutzt kein Autograd). Numerischer Vergleich: CUDA-Settle vs. `_settle_pytorch` mit `torch.allclose` (gleiche `tol`/`T`).
- [ ] **AOT-Build** in `setup.py`: `ext = CUDAExtension if (USE_CUDA and torch.cuda.is_available() and CUDA_HOME) else CppExtension`; `ext_modules=[ext("pcn._C", sources, extra_compile_args={"cxx":["-O3"],"nvcc":["-O3"]})]`; `cmdclass={"build_ext": BuildExtension}`. Build mit `uv`: `uv pip install --no-build-isolation -e .` bzw. `uv run python setup.py build_ext --inplace` (nie `pip`). Gerüst aus `extension_cpp/extension_cpp/csrc/muladd.{cpp,cu}` übernehmen (Pfad ist genestet; flacher `csrc/`-Pfad 404t).
- [ ] **Benchmark-Matrix** über `pcn/benchmark.py` (M3): PyTorch-SO vs. fused-CUDA-SO vs. EO/ePC-Referenz (github.com/cgoemaere/error_based_PC als Perf-Anker) vs. BP-MLP; ms/Settling-Schritt und ms/Epoche, mit `torch.cuda.synchronize`/Events.
- [ ] **Nsight-Artefakt:** `torch.cuda.nvtx.range_push/range_pop` um die Settle-Region; ein Nsight-Systems-Trace (`nsys`) bzw. Nsight-Compute-Kernel-Report (`ncu`) als Paper-Artefakt erzeugen.
- [ ] **`docs/07`/`docs/09` aktualisieren:** Build-Schritte, opcheck-Gate, Benchmark-Zahlen, Nsight-Befund eintragen.

**Akzeptanzkriterium:** `backend="cuda"` läuft (nicht mehr `NotImplementedError`), `opcheck` besteht, CUDA-Settle stimmt mit `_settle_pytorch` per `allclose` überein; die Benchmark-Matrix zeigt eine reproduzierbare Zeitmessung (mit korrekter Sync-Klammer) und der CUDA-Pfad ist messbar gegen PyTorch-SO/EO/BP eingeordnet; mindestens ein Nsight-Trace existiert. CPU-only-Fall: Prototyp-Code + Design abgenommen, Ausführung auf Remote-GPU vermerkt.

**Relevante Quelle(n):** PyTorch Custom-C++/CUDA-Tutorial + `torch.utils.cpp_extension` (`load_inline`/`CUDAExtension`/`BuildExtension`) + `torch.library` (`opcheck`/`register_fake`) + CUDA-Semantik (Async/Sync) + extension-cpp-Gerüst; ePC arXiv:2505.20137 (EO/ePC als Vergleichsarm + Signal-Decay-Motivation; GPU-Referenzcode); PCX arXiv:2407.01163 (Inference nicht voll parallelisierbar — zitierbare Rechtfertigung des Kernels); `docs/07`/`docs/09` (Fusion-Plan, Benchmark-Matrix).

---

## M7 — Paper (`docs/05`) + Phase-4 autonomer Such-Loop (`docs/04`)

**Ziel:** Den autonomen Such-Loop als dünnen Layer über `train_and_eval` (M1) bauen und das NeurIPS-Stil-Paper schreiben — gestützt auf die Befunde aus M3–M6. Der Such-Loop docke **nur** an `pcn/api.py:train_and_eval` an (kein Rewrite).

**Tasks:**
- [ ] **`uv add`/`uv sync --extra experiments`** sicherstellen (optuna, wandb bereits deklariert).
- [ ] **Such-Loop** neu in `pcn/search.py`: `propose(config) → train_and_eval(config) → log → next`. Stufen: (1) Grid/Random über den Raum aus `docs/04` (`T`, `precision_schedule`, Breiten/Tiefe, `activation`, `eta_x`/`eta_w`, `update_variant`); (2) Bayesian Optimization (Optuna); (3) optional LLM-Agent, der aus den Logs den nächsten `config` vorschlägt. Logging nach W&B/SQLite, damit das Paper die Kurven direkt zieht.
- [ ] **`update_variant="ipc"`** (optional) in `pcn/learning.py`: iPC = Weight-Update bei **jedem** Settling-Schritt simultan mit den States (entfernt die separate „erst settlen, dann updaten"-Phase; konvergenzgarantiert, automatisch) — als zweiter Punkt im Suchraum.
- [ ] **Quellenverifikation vor Zitat:** arXiv:2602.07697 (Innocenti et al. 2026) **in voller Länge lesen**, bevor es im Paper zitiert wird (zukunftsdatiert relativ zum Wissensstand; nicht aus dem Gedächtnis paraphrasieren). PC≈BP-Anspruch **nur** im Regime lineare Residual-Netze, width ≫ depth, equilibrierte Aktivitäten formulieren. Für den foundational PC≈BP-Anker stattdessen Whittington & Bogacz 2017 zitieren.
- [ ] **Zitations-Hygiene fixen** in `docs/05`/`docs/11`: Millidge-Review = Millidge, **Seth**, Buckley (nicht Tschantz); Song et al. = **2024**, Nat. Neurosci. 27(2):348–358; Zahid et al. = peer-reviewed Neural Computation 35(12) (kein bloßer Preprint); ePC = „ePC … Exponential Signal Decay" (nicht „EO"/„Digital Simulation"); iPC per arXiv:2212.00720 + ICLR 2024 zitieren.
- [ ] **Paper schreiben** (`docs/05` als Outline): Novelty-Hooks A (CUDA-Kernel, M6), B (Such-Loop/Ablation, dieser Schritt), C (generativ, M5); Plots aus W&B/SQLite-Logs; die scharfe empirische Aussage aus M4 als Kernbefund.

**Akzeptanzkriterium:** `pcn/search.py` führt mindestens Grid/Random + Optuna über `train_and_eval` aus und loggt reproduzierbar nach W&B/SQLite; mindestens ein im Paper berichtbarer Befund (z. B. „`Π`-Schedule × `T`-Kombination schlägt BP-MLP bei Rausch-Robustheit") liegt mit CI vor; alle Zitate in `docs/05`/`docs/11` korrigiert; arXiv:2602.07697 nachweislich gelesen und regime-korrekt zitiert.

**Relevante Quelle(n):** `docs/04` (Suchraum, Metriken, Bounded-Search-Scoping); `docs/05` (Paper-Struktur, Novelty-Hooks); iPC arXiv:2212.00720 (`update_variant="ipc"`); Innocenti et al. 2026 arXiv:2602.07697 (regime-bedingter PC=BP-Anker, vor Zitat lesen); Whittington & Bogacz 2017 (foundational PC≈BP); Korrekturen aus Millidge et al. 2021 / Song et al. 2024 / Zahid et al. 2023 / ePC (Zitations-Hygiene).

---

## Sofort-Startpunkt

Die allerersten 2–3 Schritte (Milestone M0), in dieser Reihenfolge:

1. **Umgebung + Tests:** `uv sync && uv sync --extra dev` ausführen, dann `uv run pytest -q` — die 4 Tests in `tests/test_settling.py` müssen grün sein. ✅ *(erledigt 2026-06-09)*
2. **Erster MNIST-Lauf:** `uv run python scripts/train_mnist.py --epochs 2 --hidden 256 256 --T 20` — `test_acc`, Trainingszeit und Device-Zeile festhalten. ✅ *(erledigt; Defaults underfitten — siehe M0-Ausgangsbasis)*
3. **Erster Edit (Start von M1):** in `pcn/api.py` das Config-Aliasing `eta_x → lr_state` / `eta_w → lr_weight` plus `warnings.warn` bei unbekannten Keys einbauen, `precision_schedule`/`update_variant` zu `DEFAULT_CONFIG` hinzufügen und den `lr_weight`-Default-Befund adressieren — der kleinste Schritt, der `train_and_eval` an den `docs/02`-Vertrag heranführt.
