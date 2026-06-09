# 14 — M4 Adversariale Methodik-Pruefung (Gaps-Report)

*Stand 2026-06-09. Ergebnis der 4-Kritiker-Workflow-Pruefung der PC-vs-BP-Studie. Aktionen 1+2 (B1 Loss-Matching, B2 Entkonfundierung) sind bereits umgesetzt - siehe docs/12 §4c. Dieser Report bewahrt die vollstaendige Kritik fuer Paper-Limitations/Methodik.*

## M4 — Adversariale Methodik-Prüfung (Gaps & Aktionen)

Vier unabhängige adversariale Reviews (Protokoll-Audit, Song-Treue, Forgetting-Refuter, Vollständigkeits-Kritik). Befunde dedupliziert, Widersprüche aufgelöst, gegen den tatsächlichen Code verifiziert. Korrigierte Reviewer-Irrtümer sind explizit markiert.

**Verifikations-Notiz (gegen Code geprüft):** Drei Reviewer-Behauptungen waren falsch und wurden korrigiert: (1) der Treiber `cmd_continual` sammelt `learn_acc` sehr wohl (`scripts/run_experiments.py:166,173-175`) — nur das *gespeicherte* Artefakt ist veraltet; (2) die Per-Seed-Permutation ist bereits implementiert (`run_experiments.py:168-170`, `perm_seed=s`); (3) die LR-Selektion läuft im Treiber korrekt auf `val_acc` (`run_experiments.py:77-78`), nur die Doku ist veraltet. Bestätigt hingegen: das Artefakt `results/m4_continual_t5_e3_s3.json` enthält ausschließlich `acc`+`bwt`, KEIN `learn_acc` — die Headline-Zahlen (-11.6 / -28.0) stammen also aus einem Lauf VOR der learn_acc-Verdrahtung.

---

### 1. Blocker (vor jeder Paper-Aussage zu beheben)

**B1 — Loss/Objektiv ist NICHT identisch über die Arme (PC=MSE-to-one-hot, BP=Cross-Entropy).**
PC klemmt das One-hot-Target auf den linearen Output-State und minimiert quadratischen Prediction-Error (`pcn/learning.py:45,47` `states[-1]=y_oh`; Energie = ½‖eps‖² in `pcn/settling.py`), während BP `nn.CrossEntropyLoss` auf Logits minimiert (`pcn/baselines.py:101,113`; `pcn/experiments/continual.py:104`). Das ist eine andere Verlustfläche, nicht nur eine andere Lernregel. Das verletzt die eigene Fairness-Charta (docs/10 §3: nur die Lernregel darf variieren). MSE-to-one-hot unterperformt Softmax-CE auf Klassifikation bekanntermaßen — das erklärt plausibel einen Großteil des +4.4pp-Bulk- und des Noise-Robustness-Vorsprungs von BP, und potenziell auch einen Teil der BWT-Lücke (CE+Softmax erzeugt große, aggressive Output-Updates → mehr Überschreiben → mehr Forgetting).
**Fix/Experiment:** MSE-BP-Kontrollarm hinzufügen — `CrossEntropyLoss` durch `MSELoss` gegen dasselbe One-hot-Target ersetzen, das der PC-Arm klemmt. Drei Arme melden: PC(MSE), BP(MSE), BP(CE). Der gültige Gleich-für-Gleich-Vergleich ist PC(MSE) vs BP(MSE). Wenn BP(MSE) deutlich weniger vergisst / weniger genau ist als BP(CE), erklärt der Loss die Lücke und die Headlines sind kontaminiert. Betrifft Bulk-Accuracy, Noise UND Continual gleichermaßen.

**B2 — Die „PC vergisst weniger"-Headline ist im gemeldeten Artefakt NICHT entkonfundiert (Stability-Plasticity).**
BWT misst den Abfall *relativ zur frisch gelernten Diagonale* R[j][j] (`continual.py:62`). Schreibt PC schwächere/diffusere Repräsentationen (niedrigere Diagonale), hat es mechanisch weniger zu verlieren → weniger negativer BWT bedeutet dann „lernt jede Task schlechter", nicht „behält besser". Die niedrigere PC-Final-ACC (53.4% < BP 56.0%) macht das real. Der Treiber sammelt `learn_acc` zwar (`run_experiments.py:173-175`), aber das gespeicherte Headline-Artefakt `results/m4_continual_t5_e3_s3.json` enthält für beide Arme NUR `acc`+`bwt` — kein `learn_acc`. **Status: NICHT BEWIESEN.**
**Fix/Experiment (das einzelne Experiment, das die Headline airtight macht — siehe auch §4 Aktion 1):** Continual neu laufen lassen und melden:
- `learn_acc` (mittlere Diagonale) mit CI je Arm — Entscheidungsregel: BWT-Vorsprung ist nur real, wenn PCs `learn_acc`-CI das von BP überlappt oder übertrifft.
- **Schärfste Einzelkontrolle:** die behaltene Absolut-Accuracy auf Task 0 nach allen Tasks, R[T-1][0], direkt (nicht den Abfall). Ist PCs behaltenes R[T-1][0] tatsächlich höher als BPs, ist die Retention real — unabhängig von den Diagonalen.
- Falls `learn_acc` disjunkt zugunsten BP: Retained Fraction (final/peak je Task) statt absolutem BWT melden, oder beide Arme bei gleicher Peak-Accuracy matchen.

**B3 — Continual-LRs sind asymmetrisch und NICHT auf das Forgetting-Objektiv getunt; die Asymmetrie macht BP per Konstruktion plastischer.**
`run_experiments.py:165` hardcodet PC `eta_w=0.05` vs BP `lr=0.1` — beide aus einem Bulk-MNIST-Single-Task-Scan importiert, nie auf BWT getunt; `cmd_continual` hat keinen Val-Sweep (anders als `cmd_compare`). Eine 2×-größere Weight-LR für BP bedeutet größere Per-Task-Gewichtsverschiebung → mehr Überschreiben prior Tasks → negativerer BWT, unabhängig von der Lernregel. „PC unterfittet, BP übertrainiert" bei fixen `epochs_per_task=3` ohne Early-Stop (`continual.py:111-122`) ist eine plausible Voll-Erklärung der Lücke.
**Fix/Experiment:** (a) BP-LR auf dem Continual-Objektiv unabhängig grid-suchen (genauso wie PCs joint `eta_x×eta_w`-Tuning); (b) Per-Task-Gewichtsverschiebung melden: mittlere ‖W_nach − W_vor‖_F je Task für beide Arme — ist BPs viel größer, BP-LR senken bis Verschiebung matcht, dann BWT neu messen; (c) BWT über kleinem LR-Grid melden, um zu zeigen, dass das Ergebnis kein Artefakt der höheren BP-LR ist. Akzeptanz-Gate: BWT muss stabil bleiben, wenn `learn_acc`/Verschiebung gematcht sind.

**B4 — Continual-Ergebnis hat KEINE Standard-CL-Baselines (nur PC vs naive-SGD-BP).**
Der einzige positive Headline-Befund hat keinen Anker außer dem naiven Lower-Bound. Reviewer fragen sofort: PC schlägt naiv — aber wie gegen die billigen Mitigationen, zu denen ein Praktiker greift? Bei EWC=-8% und Replay=-3% wäre PCs -11.6% eine Kuriosität statt Beitrag.
**Fix/Experiment:** Auf identischem [256,256]-Netz/Daten/Seeds vier Referenzpunkte hinzufügen: (1) naive-SGD-BP (Lower-Bound, vorhanden); (2) EWC auf dem BP-Netz (~40 Zeilen: Fisher-Diagonale an jeder Task-Grenze, ein λ auf Held-out-Split gesweept); (3) Experience-Replay mit Mini-Buffer (~200 Exemplare/Task, ~20 Zeilen); (4) Joint-Training-Upper-Bound (alle Tasks gemischt). Dann PCs BWT auf der Forgetting/Compute-Pareto-Front gegen diese verorten. Saubere ehrliche Framing-Option auch bei „PC zwischen naiv und EWC": EWC-artiger Schutz „gratis" aus der Inferenzdynamik, ohne Fisher-Matrix/Task-Grenzen/Buffer.

---

### 2. Major (stärkt das Ergebnis materiell)

**M1 — Benchmark-Mismatch zu Song et al. 2024: Permuted-MNIST ist NICHT Songs Continual-Setup.**
Songs Fig 4d-e ist Split-FashionMNIST mit ZWEI Tasks disjunkter Klassenmengen (5+5), geteiltem 5-Output-Head, alternierendem Training. Das Projekt nutzt 5-Task-Permuted-MNIST domain-IL mit fixem 10-way-Head und Identitäts-Task-0 (`continual.py:1-12,24-30`). Die Protokolle stressen verschiedene Failure-Modes: domain-IL = Input-Distribution-Shift bei stabiler Label-Semantik; Songs Split = Output-/Repräsentations-Interferenz disjunkter Klassen auf geteiltem Head. BWT auf Permuted-MNIST ist ein Apples-to-Oranges-Proxy, keine Replikation.
**Fix/Experiment:** `run_split_fashionmnist(n_classes_per_task=5, alternating=True, shared_head_dim=5)` implementieren — wiederverwendet PCN/BPMLPRef/`train_epoch`/`evaluate`/`gem_metrics`; nur Daten-Pipeline (`_permuted_loaders` → FashionMNIST-Klassenfilter statt Pixel-Permute, output 10→5, Identitäts-Task-0 raus) und Schedule (sequenziell→alternierend) ändern. Permuted-MNIST explizit als „domain-IL-Generalisierung jenseits Songs Protokoll" labeln, NICHT als Replikation.

**M2 — Split-MNIST class-IL fehlt; M4-Akzeptanzkriterium nicht erfüllt.**
Permuted-MNIST ist das mildeste nicht-triviale CL-Regime (geteilter 10-way-Head, nur Input-Drift, nie Head-Repartitionierung). Class-IL (Split-MNIST, 5×2 Klassen, kein Task-ID bei Inferenz) ist das diskriminierende Regime und genau das, was Songs „geteilte-Gewichte-Interferenz"-Argument vorhersagt. docs/13:132 fordert „Split UND Permuted"; nur Permuted existiert.
**Fix/Experiment:** Split-MNIST class-IL (5 Tasks × 2 Klassen, single shared Head, kein Task-ID) mit denselben Baselines (B4) implementieren, ACC/BWT/`learn_acc` neu messen. Überlebt PCs Vorsprung class-IL → Beitrag stärker; verschwindet er → ehrlicher Scope-Befund („Interferenz-Resistenz hilft domain-IL, nicht class-IL"). van-de-Ven&Tolias-Vokabular (task/domain/class-IL) explizit verwenden.

**M3 — Compute-Achse (Zahid et al. 2023) ist Prosa, keine gemessene Zahl in den Tabellen.**
PCs Per-Update-Zeit ist durch Backprop nach unten beschränkt (jede Gewichts-Aktualisierung braucht T Settling-Schritte). M3 maß 5.96 ms/settle und 0.30 ms/step sowie Wall-Clock PC 74.7s vs BP 51.8s — aber keine Compute-Spalte in den M4-Tabellen. PCs Forgetting-Vorteil ist nur interessant, wenn gegen Compute bepreist (EWC/Replay sind billig).
**Fix/Experiment:** Compute-Spalte in jede Ergebnistabelle: Wall-Clock/Epoch (oder /1k Updates) + Total-Train-Time PC vs BP + genutztes T. Per-Update-Kostenverhältnis explizit nennen, Zahid et al. als theoretischen Grund zitieren. Continual-Ergebnis als Forgetting-vs-Compute-Trade-off-Plot (PC, naive-BP, EWC, Replay als Punkte).

**M4 — Statistik unter Publikationsschwelle: n=3 Seeds, 68%-CI, kein Signifikanztest/Effektstärke.**
68%-CI (1σ) ist non-standard (Konvention 95%); disjunkte CIs sind kein Hypothesentest; n=3 bootstrappt aus nur 3 Werten (`run_experiments.py:174`, `protocol.py:23`).
**Fix/Experiment:** (1) Auf 95%-CIs umstellen. (2) ≥5 Seeds Bulk, ≥10 Continual (bei MNIST billig). (3) **Gepaarter** Test über Seeds (paired Bootstrap oder Wilcoxon Signed-Rank) — PC und BP teilen Init/Daten/Seed und im Continual `perm_seed=s` (`run_experiments.py:170`), Pairing ist valide und mächtiger; Per-Seed-(PC−BP)-BWT-Verteilung melden. (4) Effektstärke (pp-Differenz mit CI bzw. Cohen's d). (5) Multiplizität über die 4 Achsen notieren (Lücken groß genug → unkritisch, aber explizit sagen).

**M5 — Veraltete/widersprüchliche Doku zur LR-Selektion (Doku-only, aber irreführend).**
`protocol.py:10-13` Docstring behauptet Test-Set-Selektion („model selection on the test set"); docs/12 §4c:239-240 sagt „LR-Selektion auf Test". Der Treiber selektiert aber korrekt auf `val_acc` (`run_experiments.py:77-78`) und meldet `test_acc`. **Korrektur eines Reviewer-Irrtums:** Test-Leak in die Selektion liegt NICHT vor — die Selektion ist sauber auf Val.
**Fix:** Docstring `protocol.py:10-13` und docs/12 §4c auf die implementierte Val-Selektion aktualisieren. **Code-Falle:** `compare_pc_vs_bp()` (`protocol.py:96-97`) defaultet weiterhin auf `metric='test_acc'` und nutzt `sweep_lr` (Einzelachse) — würde bei Nutzung auf Test selektieren. Default auf `'val_acc'` ändern oder die Funktion löschen, da der Treiber den `sweep_grid`-Pfad nutzt.

---

### 3. Minor / nice-to-have

**Mi1 — Bootstrap-CI: asymmetrische Index-Regel.** `protocol.py:44-45`: `lo = boots[int(alpha*n_boot)]` (ungeclampt) vs `hi = boots[min(int((1-alpha)*n_boot), n_boot-1)]` (geclampt) — zwei verschiedene Regeln. Bei n_boot=2000 winzig, aber inkonsistent. **Fix:** Eine symmetrische Regel: `lo = boots[int(alpha*(n_boot-1))]`, `hi = boots[int((1-alpha)*(n_boot-1))]`. Zusätzlich `std` (`protocol.py:46`) als Sample-Std (n−1), nicht Bootstrap-Std labeln.

**Mi2 — Inferenz-Prozedur unterscheidet sich per Konstruktion.** PC-Eval lässt den Output settlen (`evaluate.py:19`, `clamp_output=False`), BP liest Single-Forward-Logits (`baselines.py:46-51`). Noise-Robustness und Continual vergleichen damit Inferenzdynamiken, nicht nur Plastizität. **Fix:** Explizit dokumentieren (das ist der Punkt von PC). Als Ablation PC-Accuracy am Feedforward-Init (T=0, ohne Settling) melden — dann ist der Forward für mindestens eine Tabellenzeile echt gematcht. Speziell für Continual: schrumpft PCs BWT-Vorteil ohne Test-Settling deutlich, ist Teil des „vergisst weniger" eine Inferenz-Korrektur, keine Gewichts-Retention.

**Mi3 — Architektur-/Init-/Aktivierungs-Divergenz von Songs Fig-4e-Netz.** Song: hidden=32, sigmoid, Xavier-normal. Projekt: [256,256], tanh, orthogonal (`api.py` DEFAULT_CONFIG). **Fix:** Für den treuen Replikations-Arm Songs Architektur spiegeln; Architektur-Sensitivität als Extension.

**Mi4 — BP nutzt vanilla SGD (kein Momentum/Weight-Decay).** Fairness-Entscheidung, korrekt für „nur Lernregel"-Framing und Song-konform (Adam nur für Songs Convnets). **Fix:** Im Paper nennen; optional Adam-BP-Zeile, um zu zeigen, dass die Bulk-Lücke ein Lower-Bound für BP ist.

**Mi5 — 2k-Eval-Subset im Continual.** `continual.py:80` `test_limit=2000` cappt den Eval-Satz; ACC/BWT auf 2k statt 10k geschätzt. Der Seed-Bootstrap-CI fängt diesen Finite-Eval-Sampling-Error nicht. **Fix:** 2k-Subset in der Tabelle notieren; CI unterschätzt Gesamtunsicherheit.

**Mi6 — Val-Split ist deterministischer Präfix, über Seeds identisch.** `api.py:94-96` `idx[:n_val]`, `shuffle=False` — Val-Schätzung hat keine Seed-zu-Seed-Split-Unabhängigkeit; Selektion und Test teilen Seeds. Akzeptabel, aber schwach. **Fix:** Per-Seed-zufälliger Val-Split, oder LR auf Seed-0 selektieren und Test auf den restlichen Seeds melden.

**Mi7 — GEM-ACC/BWT-Mathematik ist KORREKT.** Verifiziert gegen Lopez-Paz & Ranzato 2017: `continual.py:61-62` matcht exakt mit kanonischem T−1-Nenner. Kein Fix nötig. Optional: Ein-Zeilen-Kommentar, dass T−1 (nicht T) Konvention ist; `learn_acc` als in-house-Ergänzung korrekt gelabelt.

**Mi8 — Mechanismus-Treue zu Song IST gegeben (Stärke, keine Lücke).** Vanilla SO-PC (Settle-then-Hebb) IST Prospective Configuration — das ist Songs Name für das Gleichgewichtsverhalten von Standard-PC, keine separate Regel. `settling.py`/`learning.py` matchen Songs Methods. **Aktion:** Im Writeup als Stärke framen („wir implementieren das von Song als biologisch natürlich argumentierte Regime direkt, ohne BP-approximierende Constraints").

**Mi9 — Fehlende Regimes & Ablationen (Limitations-Stärker, keine Core-Fixes).** (a) Online-Learning (batch=1, single-pass) — von Song als PC-Vorteilsregime genannt, geplant, nicht gelaufen; Test-Acc-vs-Examples-Kurve + AUC, ≥3 Seeds. (b) Mechanistische Ablation: Settling-T (0/5/20/40) speziell gegen BWT — wächst der Schutz mit T, bindet das den Vorteil an die Inferenzphase; T=0 sollte mehr vergessen. (c) Target-Alignment-Cosinus (Song) als direkte In-house-Evidenz für den Mechanismus. (d) Depth-Scaling 2/4/8 Layer (Problem 1 der Literatur) — als Erstes zu streichen bei Zeitdruck.

---

### 4. Priorisierte nächste 3 Aktionen (Solo-MNIST-Projekt, in Reihenfolge)

**Aktion 1 — Continual neu laufen mit learn_acc + retained-R[T-1][0], gepaart, ≥5 Seeds, 95%-CI. (Macht die Headline airtight oder kippt sie.)**
Dies ist das einzelne Experiment, das den einzigen positiven Befund verteidigbar macht. Das gespeicherte Artefakt hat kein `learn_acc`; der Treiber sammelt es bereits (`run_experiments.py:173-175`), also primär ein Re-Run plus zwei kleine Code-Änderungen:
- `gem_metrics` (`continual.py:64`) gibt bereits `final_per_task` zurück → in `cmd_continual` zusätzlich R[T-1][0] (behaltene Task-0-Accuracy) je Seed sammeln und mit CI melden.
- `cmd_continual` (`run_experiments.py:165`) so erweitern, dass BPs `eta_w` über ein kleines Grid {0.02, 0.05, 0.1} auf das Continual-Objektiv (mean `learn_acc`) selektiert wird — beseitigt die LR-Asymmetrie aus B3.
- `bootstrap_ci`-Aufrufe (`run_experiments.py:174`) auf `confidence=0.95`, Seeds auf ≥5 (`--seeds 5`), und per-Seed-(PC−BP)-BWT-Differenz mit Wilcoxon Signed-Rank melden (Pairing via `perm_seed=s` ist bereits valide).
- **Entscheidungsregel/Output:** Headline „PC vergisst weniger" ist NUR verteidigbar, wenn (i) PCs `learn_acc`-CI das von BP überlappt/übertrifft UND (ii) PCs behaltenes R[T-1][0] ≥ BPs. Sonst als „PC zahlt Plastizität für Stabilität" framen.

**Aktion 2 — MSE-BP-Kontrollarm hinzufügen. (Entkonfundiert Lernregel von Loss über ALLE Achsen.)**
Behebt B1 und einen Teil von B3. Minimale Änderung: in `baselines.py` `train_and_eval_bp` und in `continual.py:104` `nn.CrossEntropyLoss()` durch eine `loss_fn`-Option ersetzen, die `nn.MSELoss()` gegen `one_hot(y,num_classes).float()` rechnet (dasselbe Target, das PC in `learning.py:45,47` klemmt). Drei Arme über Bulk + Continual melden: PC(MSE), BP(MSE), BP(CE). Der gültige Like-for-Like-Vergleich ist PC(MSE) vs BP(MSE); die Differenz BP(CE)−BP(MSE) quantifiziert, wie viel der bisherigen Lücken reiner Loss-Effekt war. Re-Run von `cmd_compare` und `cmd_continual` mit der neuen Option.

**Aktion 3 — EWC- + Replay-Baselines + Joint-Upper-Bound auf der Continual-Task. (Verankert den einzigen positiven Befund.)**
Behebt B4. Auf identischem [256,256]-Netz/Daten/Seeds in `continual.py`: (a) EWC auf dem BP-Netz (~40 Zeilen: Fisher-Diagonale an jeder Task-Grenze akkumulieren, λ·Σ Fisher·(θ−θ*)² als Penalty, ein λ auf Held-out-Task-Split sweepen); (b) Experience-Replay (~20 Zeilen: 200-Exemplar-Ring-Buffer/Task, in jede Minibatch mischen); (c) Joint-Upper-Bound (alle 5 Tasks gemischt, eine Trainingsschleife). Alle vier Punkte (naive-BP, EWC, Replay, Joint) + PC als Forgetting-vs-Compute-Pareto-Plot melden (liefert zugleich die Compute-Zahl aus M3). Selbst „PC zwischen naiv und EWC" ist eine saubere, ehrliche, publizierbare Framing: EWC-artiger Schutz ohne Fisher-Matrix/Task-Grenzen/Buffer.

**Reihenfolge-Begründung:** Aktion 1 entscheidet, ob es überhaupt eine Headline gibt (sonst sind 2 & 3 verfrüht). Aktion 2 stellt sicher, dass die Headline „Lernregel" und nicht „MSE vs CE" misst. Aktion 3 verankert sie gegen die Baselines, die Reviewer als Tischpflicht erwarten. M1/M2 (Split-FashionMNIST-Replikation, Split-MNIST class-IL) sind die nächst-wertvollsten, übersteigen aber den „nächsten 3"-Horizont; bis dahin die Permuted-MNIST-Headline strikt als „domain-IL, jenseits Songs Protokoll" scopen.