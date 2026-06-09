# 06 — Offene Probleme des Felds & Lösungsansätze

Stand 2025/26. Für jedes Problem: **Was** / **Warum es zählt** / **Aktuelle Ansätze** /
**Offene Lücke** / **Was ein Solo-MNIST-Projekt dazu beitragen kann**.

Wichtige Vorab-Ehrlichkeit: Die *headline*-Probleme (1, 3, 8) zu **lösen** ist
Frontier-Lab-Arbeit (Oxford BNDU, Sussex/VERSES, Ghent imec). Ein kleines Projekt kann
sie **reproduzieren und charakterisieren** und definierte Teilfragen untersuchen — das
ist legitim und berichtenswert.

---

## Problem 1 — Depth-Scaling-Failure (das zentrale offene Problem)

**Was.** Tiefere PC-trainierte Modelle sind oft **schlechter** als flachere — sogar in
einfachen supervidierten Settings (Pinchetti et al. 2025). Genau umgekehrt zu Backprop,
wo Tiefe die Repräsentationskraft typischerweise erhöht.

**Warum es zählt.** Solange "mehr Layer → schlechter" gilt, kann PC nicht mit BP in
großen Settings konkurrieren. Das ist der Kern-Blocker für alles Weitere.

**Aktuelle Ansätze.**
- **Ursache identifiziert** (Goemaere et al. 2025, "Error Optimization", arXiv:2505.20137):
  ein inhärentes **Signal-Decay-Problem** — Gradienten attenuieren exponentiell mit der
  Tiefe und werden durch numerische Präzisionsgrenzen vernachlässigbar klein.
- Mehrere Arbeiten beobachten eine **stark ungleiche Energieverteilung** über das Netz
  (Ha et al. 2025; Pinchetti et al. 2025; Qi et al. 2025) → schwächere Gewichts-Gradienten
  in tiefen Schichten. Mechanismus war lange unklar.
- **μPC** (Innocenti et al. 2025, arXiv:2505.13124, NeurIPS): Depth-μP-Parametrisierung,
  trainiert 100+-Layer-ResNets stabil — aber nur auf **einfachen Tasks** und limitiert
  auf dichte/residuale Architekturen.
- **Qi et al. 2025**: modifizieren die Gewichts-Gradienten-Formeln (forward-anchored,
  präzisionsgewichtet), stellen BP-Äquivalenz jenseits ~7 Layer wieder her.
- **EO / Error Optimization** (Goemaere et al. 2025): optimiert über die **Fehler** statt
  über die States → Signale erreichen alle Schichten gleichzeitig ohne Attenuierung,
  konvergiert Größenordnungen schneller, matcht BP auch bei tiefen Modellen.

**Offene Lücke.** Eine **generelle Lösung für Standard-Feedforward-Modelle mit exaktem
PC** bleibt offen (ePC-Paper, arXiv:2505.20137). Die bestehenden Fixes greifen entweder
nur für dichte/residuale Netze (μPC, Ha et al.) oder ändern die PC-Formeln (Qi et al., EO).

**Was ein Solo-Projekt kann.** Den Effekt bei kleiner Tiefe **reproduzieren** (z.B. 2 vs.
4 vs. 8 Hidden-Layer auf MNIST) und die **Energie-/Gradienten-Verteilung pro Schicht
messen** und plotten. Dann eine der Fixes (EO oder μPC-Parametrisierung) auf dem kleinen
Netz nachbauen und zeigen, ob/wie sie das Schicht-Ungleichgewicht glättet. Das ist ein
sauberer empirischer Beitrag — und genau das Material für Limitations + Ablation im Paper.

---

## Problem 2 — Rechenkosten & Hardware-Mismatch

**Was.** PC braucht **iterative Inference** (numerische Solver, viele Iterationen bis zur
State-Konvergenz). Auf digitaler Hardware, die für das Matmul-Muster von Backprop gebaut
ist, entsteht substanzieller Overhead.

**Warum es zählt.** Selbst wenn PC genauso gut lernt, ist es bei gleicher Genauigkeit
langsamer/teurer — ein praktischer Adoptionsblocker.

**Aktuelle Ansätze / Fakten.**
- Optimal: bis zur Konvergenz warten, bevor Gewichte geupdatet werden. Praktisch reichen
  oft `L < T < 2L` Iterationen (L = Tiefe) für kompetitive Resultate.
- Bei `T ≈ L` wird die **Zeit**-Komplexität vergleichbar zu Backprop — aber die
  **Raum**-Komplexität bleibt höher, weil alle Neuronen/Schichten gleichzeitig im Speicher
  gehalten werden müssen.
- **Es fehlen DL-Libraries, die die Inference-Phase voll parallelisieren.**
- PC wäre ideal für **neuromorphe Hardware** (physikalischer Energieminimierungs-Prozess) —
  aber solche Hardware existiert noch nicht; alles läuft auf digitaler Simulation.
- Beschleunigungen: accelerated proximal gradients (Sledge & Principe 2021,
  arXiv:2101.06848); EO konvergiert Größenordnungen schneller (Goemaere et al. 2025).

**Offene Lücke.** Effiziente, voll parallelisierte Inference auf vorhandener
(GPU-)Hardware. Genau die fehlende Library/Kernel-Ebene.

**Was ein Solo-Projekt kann.** **DAS ist der stärkste Beitragswinkel** (Novelty-Hook A):
ein **benchmarkter C++/CUDA-Settling-Kernel** gegen die PyTorch-Referenz. Konkrete
Messung: Wie skaliert die Settling-Zeit mit `T`, Breite, Tiefe? Wie groß ist der reale
Overhead vs. Backprop bei `T ≈ L`? Das adressiert eine reale offene Engineering-Frage
und liegt nach dem 355M-LLM (custom GEMM/Flash-Attention) im eigenen Revier.

---

## Problem 3 — Das PC≈BP-Äquivalenz-Rätsel

**Was.** Neuere Theorie (arXiv:2602.07697, "Infinite Width and Depth Limits of PCNs",
2026): Die Menge der breiten- und tiefen-stabilen Feature-Learning-Parametrisierungen für
PC ist **exakt dieselbe wie für BP**; unter diesen konvergieren PC-Gewichts-Gradienten
gegen die BP-Gradienten, sobald Breite ≫ Tiefe. Und: das einzige Regime, in dem PC
**klar** Vorteile über BP zeigt, wird mit Breite und Tiefe **instabil**.

**Warum es zählt.** Wenn PC im skalierbaren Regime nur BP approximiert, und sein
distinktives Regime bei Skalierung instabil wird — **was ist PC dann eigentlich *für*?**
Das ist die tiefste konzeptionelle Frage des Felds.

**Aktuelle Ansätze.** Prinzipierte PC-spezifische Parametrisierungen (Ishikawa et al.
2024, arXiv:2411.02001; die Infinite-Limits-Arbeit selbst). Die Autoren halten explizit
offen, ob es eine *andere* stabile, "reiche" Parametrisierung gibt, unter der PC **nicht**
gegen BP konvergiert — das wäre der Heilige Gral.

**Offene Lücke.** Ein Regime/eine Parametrisierung finden, in der PC nachweislich etwas
**Nützliches tut, das BP nicht tut**, und das **skaliert**.

**Was ein Solo-Projekt kann.** Hier ehrlich nur **flankieren**: empirisch prüfen, ob auf
dem kleinen Netz PC- und BP-Lösungen messbar divergieren (andere Lösungen? andere
Robustheit?), als Datenpunkt zur Frage. Lösen ist Theorie-Frontier.

---

## Problem 4 — Die "Wofür-Regime"-Frage (Vorteile inkonsistent)

**Was.** PC soll BP überlegen sein bei Problemen, die **biologische Organismen** haben:
**Continual Learning, Online-Learning, Lernen aus wenig Daten** — weil die Inference-Phase
den Fehler so im Netz verteilt, dass **Weight Interference** vermieden wird (Brain-Inspired-
Survey, arXiv:2308.07870). **Aber:** diese Vorteile wurden **nicht konsistent beobachtet**.

**Warum es zählt.** Das ist die plausibelste *praktische* Daseinsberechtigung von PC.
Wenn die Vorteile real und reproduzierbar sind, hat PC eine Nische auch ohne Scale-Sieg.

**Offene Lücke.** Saubere, reproduzierbare Abgrenzung: **unter welchen Bedingungen** schlägt
PC BP bei Continual/Online/Small-Data — und unter welchen nicht?

**Was ein Solo-Projekt kann.** **Direkt testbar** (Novelty-Hook C / Baseline in `03`):
PCN vs. identisches Backprop-MLP auf (a) Rausch-Robustheit, (b) Small-Data (100/1k/10k),
(c) Continual Learning (sequentielle Klassen, Catastrophic Forgetting). Ein positiver
**oder** negativer Befund ist berichtenswert. Das ist der zugänglichste echte Beitrag
neben dem Kernel.

---

## Problem 5 — Generativ-Diskriminativ-Tradeoff

**Was.** Regularisierung, die das **generative** Verhalten verbessert, **verschlechtert**
oft die supervidierte Klassifikation (Orchard et al. 2019). Beide Ziele zugleich gut zu
bedienen erfordert balanciertes Objective-Design.

**Offene Lücke.** Ein Objective/Mechanismus, der generative Qualität und
Klassifikationsgenauigkeit gleichzeitig hält.

**Was ein Solo-Projekt kann.** Den Tradeoff auf MNIST **quantifizieren** (Accuracy vs.
Rekonstruktionsqualität über einen Regularisierungs-Parameter). Kleiner, sauberer
Plot — passt in den Experiments-Teil.

---

## Problem 6 — Nicht-hierarchische / Graph-strukturierte PCNs

**Was.** Lerndynamik, Initialisierung und Konvergenz in **nicht-hierarchischen** PCNs
(beliebige Graphen statt sauberer Schicht-Hierarchie) sind mathematisch noch schlecht
verstanden (Seely 2025). PC ist im Prinzip auf beliebigen Berechnungsgraphen definierbar
(Millidge et al.) — aber die Dynamik dort ist offen.

**Offene Lücke.** Systematisches Verständnis von Konvergenz/Stabilität auf allgemeinen
Graphen.

**Was ein Solo-Projekt kann.** Eher außerhalb des MNIST-Scopes; höchstens ein kleines
Nebenexperiment mit einer einzelnen lateralen Verbindung. Nicht als Hauptbeitrag.

---

## Problem 7 — Hyperparameter-Sensitivität & Initialisierung

**Was.** Stabiles (insb. tiefes) PC-Training hängt kritisch von Inference-Rate,
Weight-Decay, Präzisionsgewichtung, Initialisierung und Batch-Scheduling ab (Qi et al.
2025). Es gibt kein robustes Standard-Rezept wie bei BP.

**Offene Lücke.** Robuste Defaults / automatische Anpassung (vgl. μP-Transfer bei BP, den
μPC für PC zu übertragen versucht).

**Was ein Solo-Projekt kann.** **Direkt** = der Such-Loop aus `04`: systematische Ablation
über genau diese Hyperparameter mit Bayesian Optimization, mit Energie-Dynamik als
Diagnostik. Liefert empirische Defaults für die untersuchte Skala.

---

## Problem 8 — Der Scale-Gap

**Was.** PC funktioniert "noch nicht in den Größenordnungen, die man gerne hätte" (VERSES
2025). Kein PC auf Transformer-Skala. PC-**basierte** Transformer existieren und erreichen
**fast** die Leistung normaler Transformer bei gleicher Komplexität — aber nicht auf
Frontier-Skala hochskaliert.

**Warum es zählt.** Ohne Scale-Beweis bleibt PC akademisch.

**Offene Lücke.** PC stabil auf moderne Architekturen/Größen bringen — hängt direkt an
Problem 1, 2, 3.

**Was ein Solo-Projekt kann.** Nicht lösbar solo. Aber: ein sauberes kleines PC-Baustein
(Kernel + verstandene Dynamik) ist genau der Unterbau, auf dem man später ein
PC-Transformer-Sprachmodell *im Kleinen* aufsetzen könnte (dort käme HF-Streaming
korrekt rein, siehe `02`).

---

## Zusammenfassung: Was solo realistisch ist

| Problem | Solo lösbar? | Solo untersuchbar / Beitrag |
|---------|:---:|---|
| 1 Depth-Scaling | ✗ | ✓ reproduzieren + Schicht-Energie messen + Fix nachbauen |
| 2 Rechenkosten/HW | ✗ (HW) | ✓✓ **CUDA-Kernel-Benchmark = stärkster Hook** |
| 3 PC≈BP-Rätsel | ✗ | ~ flankierend (divergieren die Lösungen?) |
| 4 Wofür-Regime | ✗ | ✓✓ **PC vs. BP bei Noise/Small-Data/Continual** |
| 5 Gen-Disk-Tradeoff | ~ | ✓ quantifizieren |
| 6 Graph-PCNs | ✗ | ~ nur Nebenexperiment |
| 7 Hyperparameter | ~ | ✓ Such-Loop liefert Defaults |
| 8 Scale-Gap | ✗ | ✗ (aber Unterbau für später) |

**Fazit für die Paper-Strategie:** Die beiden Doppel-Häkchen (Problem 2 + 4) sind die
zugänglichen, echten Beiträge — Kernel-Benchmark als Hauptbeitrag, PC-vs-BP-Regime als
empirischer Zweit-Befund. Probleme 1, 5, 7 liefern Ablations- und Limitations-Material.
Die Frontier-Probleme (3, 8) gehören ehrlich in den Limitations/Related-Work-Teil.

---

## Referenzen

- Rao & Ballard 1999 — Predictive coding (Ursprung)
- Friston — Free Energy Principle / Active Inference
- Whittington & Bogacz 2017 — PC approximiert BP
- Song et al. — exakte BP unter fixed-prediction assumption
- Millidge et al. — PC auf beliebigen Berechnungsgraphen
- Orchard et al. 2019 — Generativ-Diskriminativ-Tradeoff
- Sledge & Principe 2021 — DPCN-Inference-Bottleneck, accelerated proximal gradients (arXiv:2101.06848)
- Pinchetti et al. 2025 — PC-Benchmark, Depth-Scaling-Failure
- Ha et al. 2025; Qi et al. 2025 — Energie-Ungleichgewicht, Gradienten-Fixes
- Innocenti, Achour, Buckley 2025 — μPC, Depth-μP (arXiv:2505.13124, NeurIPS 2025)
- Goemaere, Oliviers, Bogacz, Demeester 2025 — Error Optimization / ePC (arXiv:2505.20137)
- Ishikawa, Yokota, Karakida 2024 — stabile Parametrisierung (arXiv:2411.02001)
- "Infinite Width and Depth Limits of PCNs" 2026 (arXiv:2602.07697)
- Seely 2025 — Graph-strukturierte PCN-Dynamik
- Brain-Inspired Computational Intelligence via Predictive Coding (arXiv:2308.07870)
- VERSES AI Research — Benchmarking-Blog (PC funktioniert noch nicht at scale)
