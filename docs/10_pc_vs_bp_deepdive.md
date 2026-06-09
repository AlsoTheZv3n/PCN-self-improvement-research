# 10 — Deep-Dive: PC vs. BP in den Bio-Regimen (Quellen durchgearbeitet)

Destillat aus Song et al. 2024, "Inferring neural activity before plasticity..."
(Nature Neuroscience, doi:10.1038/s41593-023-01514-1; Preprint biorxiv 2022.05.17.492325),
plus der kritischen Gegenstimme. Ziel: mit maximalem Wissen — inkl. der **fairen
Vergleichsmethodik**, die über die Glaubwürdigkeit der ganzen Experimente entscheidet.

## 1. Das Prinzip: Prospective Configuration

In **Prospective Configuration** ändert das Netz **erst** die neuronale Aktivität quer
durchs Netz, sodass die Output-Neuronen das Ziel besser vorhersagen — **danach** werden
die Gewichte modifiziert, um diese Aktivitätsänderung zu konsolidieren. Bei Backprop ist
die Reihenfolge **umgekehrt**: die Gewichtsänderung führt, die Aktivitätsänderung folgt.

Das ist exakt die **Inference-vor-Learning**-Zweiphasigkeit von PC (siehe `01`/`09`): die
Relaxation der States *ist* das Inferieren der "prospektiven" Aktivität. Energy-based Netze
(Hopfield, PCNs) folgen diesem Prinzip implizit. Autoren: Oxford (Song, Millidge,
Salvatori, Bogacz) + VERSES — dieselbe Gruppe wie EO/μPC/PCX.

## 2. Warum es katastrophale Interferenz vermeidet (das Kern-Argument)

Beispiel aus dem Paper (Bär am Fluss): Anblick → Vorhersage "Wasser hören" + "Lachs
riechen". Der Bär riecht den Lachs, hört aber das Wasser nicht (Ohrverletzung) → nur die
Sound-Erwartung soll sich ändern.
- **Backprop** propagiert den negativen Fehler zurück und **senkt die geteilten Gewichte**
  — schwächt damit auch die *korrekte* Lachs-Erwartung. Das ist katastrophale Interferenz:
  Lernen eines neuen Aspekts beschädigt einen anderen Aspekt derselben Assoziation.
- **Prospective Configuration**: die Hidden-Neuronen settlen erst in ihren prospektiven
  Zustand, "sehen" die Nebenwirkung **voraus** und gleichen sie aus → korrigiert den
  falschen Output, **erhält** den korrekten. In **einer** Iteration, was BP mehrere kostet.

Mess-Metrik dafür: **Target Alignment** = Cosinus-Ähnlichkeit zwischen der tatsächlichen
Output-Änderung und der Zielrichtung. Prospective Configuration alignt besser mit dem Ziel
als Backprop. (Diese Metrik ist direkt für unsere Experimente übernehmbar.)

## 3. Die FAIRE Vergleichsmethodik (das Wichtigste für unser Paper)

Das ist der Teil, der unsere PC-vs-BP-Experimente glaubwürdig macht — sonst vergleicht man
Äpfel mit Birnen:

> Ein PCN erzeugt für gegebenen Input **dieselbe** Vorhersage wie ein Standard-Feedforward-
> ANN, **wenn** die Gewichte einander entsprechen (Input geklemmt, Output frei). Damit ist
> der Loss in **beiden** Modellen **dieselbe Funktion der Gewichte**. Direkte
> Gradientenabstiegs-Minimierung im PCN würde dieselben Gewichtsänderungen wie BP im ANN
> erzeugen. Vergleicht man also PCN (Prospective Configuration) gegen ANN+BP, **isoliert**
> man den Effekt der **Lernregel** — nicht der Architektur.

Konkret heißt das für `08`/dieses Projekt:
- **Identische** Architektur, **identische** Init, **identische** Datensplits.
- Der **einzige** Unterschied ist die Lernregel: Prospective Configuration (PCN, mit
  Inference-Phase) vs. direkte Loss-Minimierung (BP).
- ≥3 Seeds, Konfidenzintervalle (so macht es das Paper).

## 4. Wichtige Versöhnung mit Problem 3 (PC≈BP-Rätsel)

Scheinbarer Widerspruch: `06`/Problem 3 sagt "PC konvergiert gegen BP", dieses Paper sagt
"PC ist *besser* als BP". Die Auflösung steht im Paper:

> Frühere Arbeiten zeigten PC≈BP **nur unter unnatürlichen Bedingungen** — indem die
> neuronale Aktivität daran gehindert wurde, sich vor der Gewichtsänderung substanziell zu
> ändern (infinitesimal kleines Supervisionssignal, wie bei Equilibrium Propagation und
> früheren PCN-Studien) oder es nur infinitesimal kurz wirkte.

**Ohne** diese Constraints folgen Energy-based Netze der **distinkten** Prospective
Configuration — *nicht* Backprop —, und genau dort liegen die Vorteile. Das "PC≈BP"-Resultat
betrifft also das **eingeschränkte** Regime (kleine `λ`, `T→`konvergenz nahe ff-Init); das
**unbeschränkte** Regime ist genuin anders. (Vgl. EO-Theorem C.9: EO kollabiert zu BP bei
`T=1`/kleinem `λT` — derselbe Mechanismus.) → **Für die Bio-Regime-Vorteile braucht man
substanzielles Settling, nicht das BP-nahe Limit.**

## 5. Die Regime — und wie Song et al. sie evaluierten

Das Paper zeigt Vorteile in: **tiefe Strukturen** (Vorteil wächst mit Tiefe), **Online-
Learning** (Update nach jedem Beispiel), **wenige Trainingsbeispiele** (Few-Shot),
**sich ändernde Umgebungen**, **Continual Learning** (mehrere Tasks), **Reinforcement
Learning**. Skalierung bis CNNs auf CIFAR-10 (Fig. 4i). Zusätzlich erklärt es menschliche/
tierische Lerndaten (Sensorimotor, Fear Conditioning, RL).

## 6. Konkrete Experimente für unser Projekt (auf MNIST)

| Regime | Protokoll | Metrik |
|--------|-----------|--------|
| **Online** | Batch-Size 1, ein Pass | Test-Acc, Episoden bis Konvergenz |
| **Small-Data** | 100 / 1.000 / 10.000 Beispiele | Acc vs. Datenmenge |
| **Continual** | Split-MNIST (5 Tasks à 2 Klassen) | Avg-Acc, Forgetting / Backward Transfer |
| **Continual** | Permuted-MNIST | dieselben |
| **Interferenz** | Bär-am-Fluss-Minisetup nachbauen | **Target Alignment** (Cosinus) |
| **Robustheit** | zunehmender Noise | Acc über Noise-Level (eigene Ergänzung) |
| **Tiefe** | 2 / 4 / 8 Hidden-Layer | Acc-Differenz PC−BP über Tiefe |

Tooling: **Avalanche** (PyTorch) für Split-/Permuted-MNIST + Forgetting-Metriken; das
eigene PCN als Custom-Strategy einhängen. PC-Seite gegen **PRECO** sanity-checken.
**Wichtig (aus §4):** `T`/`λ` so wählen, dass echtes Settling passiert — sonst misst man
BP in Verkleidung und sieht keinen Effekt.

## 7. Ehrliche Caveats (gehören ins Paper)

- **Kritische Gegenstimme**: Zahid, Guo, Fountas 2023, "Predictive Coding as a Neuromorphic
  Alternative to Backpropagation: A Critical Evaluation" (Neural Computation, MIT Press;
  arXiv:2304.02658). Trade-offs fair darstellen.
- Vorteile wurden **nicht konsistent** über alle Arbeiten reproduziert; Prospective
  Configuration nutzte **eigene Setups** → sauber, nicht blind übernehmen.
- **Inference-Kosten**: jeder Vorteil wird mit iterativem Settling pro Sample erkauft (Bezug
  zu `09`/Problem 2). Online/Continual-Vorteil muss diesen Mehraufwand rechtfertigen.
- Das EO-Paper selbst (Abschnitt 6.3) nennt das Identifizieren von PCs distinktiven
  Vorteilen (Online/Continual, à la Song et al.) **"underexplored"** → genau hier ist
  offenes, portfolio-taugliches Terrain.
- Verwandt: Alonso et al. 2022/2024 (G-IL / IL-prox) — PC-Variante, robuster gegen hohe
  Lernraten und weniger degradiert bei Batch-Size 1. Guter Online-Vergleichspunkt.

## 8. Bottom Line

Der wissenschaftliche Wert liegt in der **fairen Methodik** (§3) plus echtem Settling (§4),
nicht im Vorzeichen des Ergebnisses. Reproduzierst du den PC-Vorteil (z.B. weniger
Forgetting in Split-MNIST, bessere Target Alignment), ist das ein sauberer Befund.
Reproduzierst du ihn nicht, trägt das negative Resultat zur "nicht konsistent"-Linie bei —
ebenfalls publikationswürdig, weil methodisch sauber.

## 9. Empirische Ergebnisse (M4, 2026-06-09) — entkonfundiert

Die faire Studie ist umgesetzt (`pcn/experiments/`, `scripts/run_experiments.py`) und durch
eine **adversariale Methodik-Prüfung** gehärtet. Erste Ergebnisse zeigten scheinbare PC-Siege
(Continual) und BP-Siege (Bulk) — der Review entlarvte **beide als Confounds**. Detailtabellen
mit CIs in **`docs/12` §4c**. Entkonfundiertes Bild (3 Arme, val-selektiert, 3 Seeds, GPU):

- **Bulk-Accuracy (10k, matched loss):** PC(MSE) 83,5% vs **BP(MSE) 83,9% → FAIR-Gap +0,4 pp,
  CIs überlappen** (Lernregeln äquivalent). BP(CE) 89,9% → der **Loss-Effekt** (BP(CE)−BP(MSE))
  ist +5,9 pp. Der frühere „BP +4–10 pp" war fast vollständig die CrossEntropy-Loss, **nicht
  die Lernregel.**
- **Rausch-Robustheit (σ=1.0, matched loss):** BP(MSE) 60,4% vs PC(MSE) 52,6% — BP ~8 pp
  besser (real, aber viel kleiner als die ~28 pp gegen BP(CE)).
- **Continual domain-IL (Permuted-MNIST, matched loss):** gegen BP(MSE) vergisst **BP weniger**
  (BWT −7,8% vs PC −12,7%), behält Task 0 besser (54,2% vs 41,8%) und PC lernt jeden Task
  schlechter (Learn-ACC 65,1% < 69,1%). Frühere „PC vergisst <½ so viel"-Headline verglich
  gegen BP(CE) + ignorierte den **Plastizitäts-Confound** → **nicht haltbar.**
- **Continual class-IL (Songs Regime: Split-FashionMNIST 2×5 + Split-MNIST 5×2, BP-LR pro Arm
  getunt):** auch hier **kein PC-Vorteil** — PC ≈ BP(MSE) (Split-Fashion: Final-ACC 60,9 vs
  61,3, alles überlappt; Split-MNIST: BP minimal besser, 73,4 vs 70,8). **Drittes Confound:**
  ein ungetuntes BP(MSE) (`eta_w=0.1`) hängt bei Split-MNIST auf Zufall (50%) fest → erzeugte
  einen falsch-positiven PC-Sieg, den die learn-ACC-Diagnose fing.

**Verdikt (vollständig, über ALLE Regime):** vanilla SO-PC zeigt **keinen Vorteil** gegenüber
loss-gematchtem, fair getuntem BP — gleichauf bei Accuracy/Lernen, auf derselben Stabilität-
Plastizität-Kurve, leicht schlechter bei Noise. Ein sauber abgesichertes Negativ-Resultat (§8);
der Beitrag ist die **Methodik** und die **Entlarvung dreier Confounds** (Loss-Funktion,
Plastizität, Baseline-LR).

**Das widerlegt Song et al. NICHT global:** Songs Fig 4e nutzt **alternierendes** Training,
**Sigmoid**, hidden=32; die obige Reproduktion ist sequenziell, tanh, `[256,256]`. Belastbare
Aussage: **bei MNIST/FashionMNIST-Skala mit fairem, getuntem BP verschwindet der PC-Vorteil.**

**Strikte Replikation jetzt umgesetzt (`docs/12` §4g, n=5):** das *architektur-treue* Song-Setup
(Minibatch-Alternation `swap_every=4`, Sigmoid, Xavier, `[32,32]`, 2×5 disjunkt, geteilter Kopf)
mit **Budget als kontrollierter Achse** (Songs exaktes 84-Iter-Budget ist in unseren Händen
untrainierbar → Zufall). Befund: **bei n=5 ist keine PC-vs-BP(MSE)-Differenz auf irgendeinem
Budget um ≥1σ getrennt.** Direktional lernt **PC früh langsamer** und liegt **bei Konvergenz
marginal vorn** (2500 Iter: mean +3,1, min_both +3,7 pp — im Rauschen). Erstes Regime, in dem
vanilla-PC am Ende *nicht hinter* loss-gematchtem BP liegt, aber der Edge ist **nicht von Null
trennbar**. Wichtige Framing-Korrektur (Literatur-Linse): Fig 4e ist ein *Interferenz*-Claim mit
**pro Methode getunter LR**, **kein** LR-Robustheits-Claim (das ist Fig 3). Konsistent mit dem
Gesamt-Verdikt: **PC ≈ BP bei fairem Tuning**, auch im treuen alternierenden Regime.

**Methoden-Lektionen (der eigentliche Beitrag):** (1) `eta_x`↔`eta_w` gemeinsam tunen; (2) die
**Loss-Funktion** dominiert PC-vs-BP stärker als die Lernregel; (3) „vergisst weniger" gegen
**learn-ACC + behaltene Task-0** prüfen (Plastizität); (4) die **BP-Baseline muss tatsächlich
lernen** (LR per learn-ACC getunt), sonst falsch-positive PC-Siege; (5) Settling-T ändert die
Bulk-Accuracy nicht. Continual lean ohne Avalanche (`pcn/experiments/continual.py`).
