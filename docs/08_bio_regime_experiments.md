# 08 — Build-Guide: PC vs. BP in den "Bio-Regimen" (Problem 4)

Ziel: empirisch testen, ob das PCN das identische Backprop-MLP in den Regimen schlägt, in
denen PC laut Literatur Vorteile haben soll. Das ist Novelty-Hook C / der empirische
Zweit-Befund.

## Die Anker-Quelle: "Prospective Configuration"

**Song et al., Nature Neuroscience 2024** — "Inferring neural activity before plasticity
as a foundation for learning beyond backpropagation" (doi:10.1038/s41593-023-01514-1;
Preprint biorxiv 2022.05.17.492325). Das ist *die* Referenz für Problem 4.

Kernidee: In **Prospective Configuration** infert das Netz erst das Aktivitätsmuster, das
aus dem Lernen resultieren *soll* (die Inference-Phase), und passt dann die Gewichte an,
um diese Aktivitätsänderung zu konsolidieren — die Reihenfolge ist gegenüber Backprop
**umgekehrt**. Energy-based Netze (Hopfield, PCNs) folgen diesem Prinzip implizit.

Das Paper zeigt empirisch Vorteile gegenüber BP in genau den Regimen, die für ein
MNIST-Projekt zugänglich sind:
- **tiefe Strukturen** (Vorteil wächst mit der Tiefe)
- **Online-Learning** (Update nach jedem Beispiel, Batch-Size 1)
- **Lernen mit wenigen Beispielen** (Few-Shot / Small-Data)
- **sich ändernde Umgebungen**
- **Continual Learning** über mehrere Tasks
- Reinforcement Learning (außerhalb des MNIST-Scopes)

Der mechanistische Grund (für die Diskussion im Paper): Die Inference-Phase **verteilt den
Fehler im Netz um**, bevor Gewichte sich ändern — das vermeidet "Weight Interference",
die BP-Updates destabilisieren kann.

## Ehrliches Framing (wichtig — es gibt Gegenstimmen)

- **Kritische Evaluation**: "Predictive Coding as a Neuromorphic Alternative to
  Backpropagation: A Critical Evaluation" (Neural Computation, MIT Press; arXiv:2304.02658)
  und ein Commentary in *Intelligent Computing* (doi:10.34133/icomputing.0244). Lesen, um
  die Trade-offs fair darzustellen.
- Die Vorteile wurden **nicht konsistent** über alle Arbeiten beobachtet (siehe `06`,
  Problem 4). Prospective Configuration nutzte ein **eigenes Setup** — sauber reproduzieren,
  nicht blind übernehmen.
- Verwandt: Alonso et al. 2022 (G-IL / IL-prox) — eine PC-Variante, die SGD/implizites SGD
  approximiert und in **Online-Settings (Batch 1)** robuster gegen hohe Lernraten ist und
  weniger degradiert. Nützlicher Vergleichspunkt für das Online-Experiment.
- Sehr aktuell: arXiv:2512.00619 (Dez 2025) — neuroscience-inspirierter Generative-Replay
  für Continual Learning, PC vs. BP, berichtet ~15 % bessere Task-Retention. Direkt
  vergleichbare Methodik (und zeigt, dass das Thema 2025/26 aktiv ist).

## Konkrete Experimente (auf MNIST, PCN vs. identisches BP-MLP)

| Regime | Protokoll | Metrik |
|--------|-----------|--------|
| **Online** | Batch-Size 1, ein Pass | Test-Accuracy, Lernkurve |
| **Small-Data** | 100 / 1.000 / 10.000 Beispiele | Accuracy vs. Datenmenge |
| **Continual** | **Split-MNIST** (5 Tasks à 2 Klassen) | Avg-Accuracy, Forgetting / Backward Transfer |
| **Continual** | **Permuted-MNIST** | dieselben |
| **Robustheit** | zunehmender Gaussian/Salt-Pepper-Noise | Accuracy über Noise-Level (eigene Ergänzung) |
| **Tiefe** | 2 vs. 4 vs. 8 Hidden-Layer | Accuracy-Differenz PC−BP über Tiefe |

Fairness-Regel: **identische** Architektur, Init und Datensplits für PCN und BP-MLP. Nur
die Lernregel unterscheidet sich. Mehrere Seeds (≥3) mit Konfidenzintervallen — so macht
es auch das Prospective-Configuration-Paper.

## Tooling

- **Continual-Learning-Harness**: **Avalanche** (PyTorch) — Standard-Library mit fertigen
  Split-MNIST/Permuted-MNIST-Benchmarks und Forgetting-/Backward-Transfer-Metriken. Spart
  das Neuschreiben der Eval-Logik; das eigene PCN als Custom-Strategy einhängen.
- **PC-Seite**: das eigene From-Scratch-PCN (über die `train_and_eval`-Schnittstelle aus
  `02`), gegengeprüft an **PRECO** (PyTorch) als Sanity-Check der Implementierung.
- **Logging**: alle Läufe nach W&B/SQLite, damit der Such-Loop (`04`) diese Regime-Sweeps
  gleich mitfahren kann und die Paper-Figuren reproduzierbar sind.

## Erwartetes Ergebnis & Einordnung

Findet man den PC-Vorteil reproduzierbar (z.B. weniger Forgetting in Split-MNIST), ist das
ein legitimer empirischer Befund. Findet man ihn **nicht**, ist das ein berichtenswertes
negatives Resultat, das zur "nicht konsistent beobachtet"-Linie der Literatur beiträgt.
Beides trägt das Paper — der Wert liegt in der sauberen, fairen Methodik, nicht im
Vorzeichen des Ergebnisses.
