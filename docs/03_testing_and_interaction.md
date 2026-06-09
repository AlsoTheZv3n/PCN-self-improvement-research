# 03 — Testen & Interaktion (Jupyter)

Das eigentlich Schöne: PCNs *können* generativ sein — dasselbe Netz klassifiziert,
generiert, vervollständigt und erkennt Anomalien über dieselbe Energie-Minimierung, nur mit
anderem Clamping (siehe `01`).

> **Status (M5, `docs/12` §4e):** Der Mechanismus ist implementiert + getestet
> (`pcn/generate.py`: `generate`/`inpaint`/`anomaly_scores`; Demo `scripts/demo_generative.py`).
> **ABER:** Der hier trainierte **diskriminative** Supervised-PCN ist *kein generatives Modell*
> — Generierung liefert Rauschen, Anomalie trennt nicht (AUC 0,44). Die generativen Demos
> brauchen ein **generativ trainiertes PC** (Bild unten geklemmt, Latent oben frei). Klassifikation
> + Energie-Dynamik (§1) und die MLP-Baseline (Hook B, `docs/10` §9) sind dagegen voll umgesetzt.
> Jupyter-Notebooks: zurückgestellt zugunsten testbarer Skripte.
>
> **Update (M5-v2, `docs/12` §4f): generativer Pfad umgesetzt + funktioniert.** Ein
> generatives PC `[10,256,256,784]` (Label→Bild, `pcn/generative.py`,
> `scripts/demo_generative_v2.py`) erzeugt **erkennbare Ziffern-Prototypen 0–9**, rekonstruiert
> verdeckte Bildhälften und trennt OOD perfekt (Anomalie-**AUC 1,0**). Damit ist PCs „eine
> Maschinerie, viele Aufgaben"-Eigenschaft *funktionierend* demonstriert (PNGs in `results/`).

## 1. Klassifikation + Energie-Dynamik

- Test-Accuracy auf MNIST (Vergleichswert aus der Literatur als Sanity-Check)
- **Energie-Kurve über die Settling-Schritte** `t = 0..T`: zeigt das "Einrasten" ins
  Minimum. Das ist der charakteristische PC-Plot und gehört ins Paper.

## 2. Generieren (Label → Bild)

- Label-Knoten oben festklemmen (z.B. One-Hot für "7"), Input-Knoten settlen lassen
- Das Netz "malt" eine Ziffer. Qualität ist bei MNIST-Skala bescheiden, aber das
  *Prinzip* (ein diskriminatives Netz, das auch generiert) ist der Punkt.

## 3. Occlusion-Completion

- Halbes Bild festklemmen (z.B. obere Hälfte), Rest rekonstruieren lassen
- Demonstriert die top-down-Vorhersage direkt und visuell

## 4. Anomalie via Restenergie

- Fremd-Input (Nicht-MNIST, Rauschen, FashionMNIST) → das Netz kann ihn nicht gut
  erklären → **hohe Restenergie** nach dem Settling
- Energie als Out-of-Distribution-Score. Einfache ROC-Kurve MNIST vs. Fremd-Input.

## 5. Baseline-Vergleich: PCN vs. Backprop-MLP

Identisches MLP (gleiche Schichten, gleiche Breite) mit Standard-Backprop trainieren,
dann vergleichen:

- **Accuracy** (erwartung: vergleichbar bei dieser Skala)
- **Rausch-Robustheit**: Test-Accuracy über zunehmenden Gaussian/Salt-Pepper-Noise
- ggf. **Small-Data**: Accuracy bei 100 / 1.000 / 10.000 Trainingsbeispielen
- ggf. **Continual Learning**: sequentielles Lernen der Klassen, Catastrophic
  Forgetting messen

Die letzten drei sind genau die Regime, in denen PC laut Literatur Vorteile *haben
soll* (siehe `06`, Problem 4) — die aber **nicht konsistent beobachtet** wurden. Ehrlich
testen: Findet man den Vorteil reproduzierbar, ist das ein legitimer Befund. Findet man
ihn nicht, ist *das* ebenfalls ein berichtenswertes (negatives) Ergebnis.

## Was davon ins Paper geht

Punkte 1 und 5 sind das Rückgrat (quantitativ). Punkte 2–4 als kurzer qualitativer
Abschnitt mit Bildern (Novelty-Hook C, siehe `05`). Alle Plots aus den geloggten
`train_and_eval`-Metriken (siehe `02`), damit die Figuren reproduzierbar sind.
