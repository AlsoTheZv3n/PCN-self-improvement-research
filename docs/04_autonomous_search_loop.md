# 04 — Autonomer Such-Loop ("Hermes mini")

Ein kontinuierlich laufendes System, das das PCN selbständig weiterentwickelt — als
**Phase 4**, weil es ein laufendes PCN als Substrat braucht.

## Ehrliches Scoping: "Innovation" vs. "viele Möglichkeiten ausprobieren"

Zwischen "erzeugt eine Innovation" und "probiert viele Möglichkeiten aus" liegt der
ganze Unterschied — und nur das Zweite ist heute zuverlässig baubar. Das ist keine
Bremse, sondern die korrekte Spezifikation.

**Was zuverlässig funktioniert:** gerichtete Suche über einen *definierten Raum* mit
einer *scharfen Metrik*. **Was nicht zuverlässig funktioniert:** offene, autonome
Erzeugung genuiner konzeptioneller Neuheit.

## Referenzsysteme (der ehrliche Stand 2024–2026)

- **AI Scientist** (Sakana, v1 2024 / v2 2025, in Nature 2026): generiert autonom Ideen,
  baut/führt Experimente aus, schreibt ein komplettes LaTeX-Paper. Ernüchternd: ein
  Paper kam durch echtes Blind-Review — beim *"I Can't Believe It's Not Better"*-Workshop
  (Score 6,33). Kernbefund: die explorativere, template-freie v2 produziert **nicht**
  zwangsläufig bessere Paper und hat **niedrigere** Erfolgsraten. Je autonomer/offener,
  desto unzuverlässiger.
- **AlphaEvolve** (DeepMind 2025, arXiv:2506.13131): Coding-Agent, der **tatsächlich**
  neue Algorithmen fand — aber nur, weil Algorithmus-Performance scharf messbar ist, und
  auf DeepMind-Skala.
- **DeepScientist**: rahmt Entdeckung als **Bayesian Optimization** (Exploration/
  Exploitation). Das ist die ehrliche Beschreibung dessen, was geht.
- **Co-Scientist** (Google DeepMind) & **Robin** (FutureHouse), Nature 2026: generieren
  Hypothesen, aber ob genuin neu — offen, und nur auf Open-Access-Daten.

**Lektion für uns:** Bounded search mit scharfer Metrik (= AlphaEvolve/DeepScientist-
Modus), nicht offene Kreativität (= der unzuverlässige AI-Scientist-Modus).

## Die baubare Version = Novelty-Hook B (Ablation)

Ein Experiment-Orchestrator über den PCN-Konfigurationsraum:

### Suchraum
- Settling-Schritte `T`
- Präzisions-Schedules `Π` (isotrop / spiking / decaying)
- Layer-Breiten, Tiefe
- Aktivierung `θ`
- Lernraten `η_x`, `η_w`
- Update-Variante (Standard-PC vs. iPC vs. ggf. EO-Reparametrisierung, siehe `06`)

### Metriken (scharf — das ist der Punkt)
- Accuracy
- Energie-Konvergenz (Schritte bis Gleichgewicht)
- Rausch-Robustheit vs. MLP-Baseline
- Trainingszeit (verbindet zum CUDA-Kernel-Benchmark)

### Loop (stufenweise eskalierend)
```
1. Grid / Random Search        — Baseline-Abdeckung des Raums
2. Bayesian Optimization       — gerichtet (à la DeepScientist)
3. (optional) LLM-Agent        — formuliert aus den Logs die nächste Hypothese
                                  und schlägt den nächsten config vor
```

### Mechanik
`config vorschlagen → train_and_eval(config) → loggen → nächsten config wählen`.
Dank der Schnittstelle aus `02` ist das ein dünner Layer, kein Rewrite. Alles nach
W&B/SQLite, damit das Paper die Kurven direkt zieht.

## Realistischer "Innovation"-Ceiling

Für ein Solo-MNIST-Projekt: Das System findet eine **überraschende, aber reale**
empirische Regularität — z.B. "diese `Π`-Schedule × `T`-Kombination schlägt die
Backprop-MLP-Baseline bei Rausch-Robustheit deutlich". Das ist ein legitimer kleiner
Befund fürs Paper, kein Paradigmenwechsel. Mehr ist solo nicht drin — und mehr braucht
das Portfolio nicht.

## Strategischer Nebeneffekt

Der Orchestrator ist ein **zweites, eigenständiges Portfolio-Stück**: ein agentischer
AutoML-/Experiment-Runner. Zwei distinkte Artefakte aus einem Projekt — das PCN
(Verständnis) und der Such-Loop (Engineering + die Agency-Achse).
