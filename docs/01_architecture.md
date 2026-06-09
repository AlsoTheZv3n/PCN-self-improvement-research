# 01 — Architektur & Gleichungen

## Grundidee: zwei Knotensorten pro Schicht

Jede Schicht `l` hat:
- **Value-Nodes** `x_l` — die Repräsentation (analog zur Aktivierung im FFN)
- **Error-Nodes** `ε_l` — der lokale Vorhersagefehler

Vorhersagen laufen **top-down**: die höhere Schicht prädiziert die tiefere.

## Die Gleichungen

```
Vorhersage:    μ_l = W_l · θ(x_{l+1})          (θ = Aktivierung, z.B. tanh)
Fehler:        ε_l = x_l − μ_l
Freie Energie: F   = Σ_l  ½ · ε_lᵀ Π_l ε_l     (Π_l = Präzision, inverse Varianz)
```

Das ist exakt die Gauß-Form der variationellen freien Energie (FEP / Friston), nur
schichtweise im Netz verdrahtet. Bei isotroper Präzision (`Π_l = I`) ist `F` schlicht
die Summe der quadrierten Fehler.

## Zwei Phasen statt Forward + Backward

### Phase A — Inference (Gewichte fix, `x_l` relaxieren bis Gleichgewicht)

```
Δx_l = −η_x · [ Π_l ε_l − θ'(x_l) ⊙ (W_{l-1}ᵀ Π_{l-1} ε_{l-1}) ]
```

Intuition: Jedes `x_l` bewegt sich so, dass es gleichzeitig den Fehler *von oben*
(`Π_l ε_l`, eigene Vorhersage) und den Fehler, den es *unten* verursacht
(`W_{l-1}ᵀ Π_{l-1} ε_{l-1}`, durchgereicht), reduziert. Das Netz "rastet" über mehrere
Schritte `T` in ein Energie-Minimum ein.

> **⚠️ Index-Konvention (Copy-Trap — bitte beachten):** Dieses Dokument indiziert
> **top-down** („die höhere Schicht prädiziert die tiefere"), daher trägt der
> Feedback-Term oben `ε_{l-1}` / `W_{l-1}`. Der **Code** (`pcn/`) und `docs/09` indizieren
> dagegen **feedforward** („Schicht `i` prädiziert `i+1`", siehe `pcn/model.py`), daher
> lautet derselbe Term dort `phi'(s_k) ⊙ (ε_{k+1} @ W[k])` mit `ε_{k+1}` / `W[k]`. Beide
> Formen sind **unter einem Index-Flip identisch** (Millidge et al. 2021, arXiv:2107.12979,
> Gl. 12 schreibt beide Terme simultan). Wer den CUDA-Kernel (`docs/07`) oder neue Layer
> schreibt, muss der **Code-/`docs/09`-Konvention (feedforward, `ε_{k+1}`)** folgen — nicht
> der top-down-Schreibweise hier. Der State-Gradient ist in `tests/test_settling.py`
> numerisch gegen Autograd abgesichert (`test_settling_gradient_matches_autograd`).

### Phase B — Learning (am Gleichgewicht, `x_l` fix)

```
ΔW_l = +η_w · Π_l ε_l · θ(x_{l+1})ᵀ        ← REIN LOKAL
```

Der Kern des ganzen Projekts: Dieses Update braucht **nur** den Fehler dieser Schicht
mal die Aktivität der Nachbarschicht. Kein globaler Backward-Pass, keine transponierten
Gewichts-Ketten über das ganze Netz. Genau das "weight transport problem", das Backprop
biologisch implausibel macht, fällt weg.

## Theoretischer Anker (für die Intro des Papers)

Unter der "fixed prediction assumption" (Song et al.) reproduziert PC die
**exakten Backprop-Gradienten**; allgemeiner approximiert es sie (Whittington & Bogacz
2017, DOI 10.1162/NECO_a_00949; Millidge, Tschantz & Buckley über beliebige
Berechnungsgraphen, arXiv:2006.04182). Ein PCN kann also dasselbe lernen wie ein FFN — über
Free-Energy-Minimierung statt Chain-Rule. Wichtige Nuance (siehe `06`/`11`): neuere Theorie
(Innocenti, Achour & Bogacz 2026, arXiv:2602.07697) zeigt, dass PC-Gradienten gegen
BP-Gradienten konvergieren — **aber nur** für lineare Residual-Netze, im Regime
**Breite ≫ Tiefe** und bei tatsächlich erreichtem Aktivitäts-Äquilibrium; *keine* unbedingte
Äquivalenz. Das wirft die Frage auf, *wann* PC überhaupt etwas anderes tut als BP. **Hinweis:**
arXiv:2602.07697 ist relativ zum Wissensstand zukunftsdatiert — vor dem Zitieren im Volltext
lesen (siehe `docs/11`); für den *fundamentalen* PC≈BP-Anker Whittington & Bogacz 2017 nutzen.

## Sizing für MNIST

```
784  →  256  →  256  →  10
(Pixel) (hidden) (hidden) (One-Hot Label)
```

Bewusst klein. Die Matrizen sind winzig (784-dim Input); die Forschung *ist* das
Austüfteln von Settling-Dynamik, Präzisions-Schedules und Update-Regeln — nicht das
Stemmen großer GEMMs.

## Clamping — die Stelle, die fast alle falsch machen

| Modus | Was festgeklemmt wird | Was settlet | Auslesen |
|-------|----------------------|-------------|----------|
| **Training** | Bild (unten) **und** One-Hot-Label (oben) | nur Hidden-`x_l` | — (dann ΔW) |
| **Test** | nur Bild (unten) | Hidden **und** Label-Knoten | `argmax` am Label-Knoten |
| **Generativ** | nur Label (oben) | Hidden **und** Input-Knoten | das "gemalte" Bild unten |

Beim Test lässt man den Label-Knoten **mitsettlen** und liest das Maximum — man gibt das
Label nicht vor. Das ist der Unterschied zwischen "klassifizieren" und "rekonstruieren".

> **⚠️ Caveat (M5-Befund, `docs/12` §4e):** Die **Generativ**-Zeile setzt ein *generativ*
> trainiertes PC voraus. Der hier gebaute **diskriminative** Supervised-PCN (Input+Label
> geklemmt, feedforward-prädiktiv) lernt **kein** generatives Bild-Modell — Label klemmen +
> Input settlen erzeugt **Rauschen, keine Ziffern** (empirisch belegt). Der Settling-
> Mechanismus für Generierung/Inpainting/Anomalie ist implementiert und getestet
> (`pcn/generate.py`), liefert aber erst mit einem generativen Trainingsziel (Bild unten
> geklemmt, Latent oben frei) sinnvolle Bilder. Auch `feedforward_init` ist ein triviales
> Null-Energie-Gleichgewicht → freie States für Generierung auf Null initialisieren.
>
> **Update (M5-v2, `docs/12` §4f): umgesetzt und funktioniert.** Ein generatives PC
> `[10,256,256,784]` (Label→Bild, `pcn/generative.py`) generiert **erkennbare Ziffern-
> Prototypen** und trennt OOD perfekt (Anomalie-AUC 1,0). Stabilität: `lr_state ≈ 0.01`
> (≈ /√Output-Dim), sonst divergiert das Settling beim 784-dim-Output.

## Bausteine

- `θ` (Aktivierung): tanh als Default; `θ'` wird für die Inference-Dynamik gebraucht
- `Π_l` (Präzision): **isotrop (`Π_l = I`) — das ist der Default und der aktuelle
  Implementierungsstand.** Π ist im Code (`pcn/`) bewusst hartkodiert = Identität; das ist
  der Mainstream moderner MNIST-PC-Libraries (PCX fixiert `Σ=I` explizit, JPC nutzt reine
  SSE-Energie). Nicht-identische Präzision ist ein **optionales/fortgeschrittenes** Feature
  (`precision_schedule` ∈ {isotropic, spiking, decaying}, Qi et al. 2025 arXiv:2506.23800),
  das primär **tiefen Netzen (> 7 Layer)** nützt und in M2 (`docs/13`) als scalar-pro-Layer-
  Schedule nachgerüstet wird — bis dahin gilt `Π = I`. (Frühere Formulierung „Π ist Kern"
  war nicht durch Code gedeckt.)
- `T` (Settling-Schritte): Faustregel **`T ≥ L`** als Untergrenze (Qi et al. 2025); die
  Obergrenze ist signal-decay-abhängig (naives sPC braucht in tiefen Netzen ≈ 5×L Schritte,
  ePC arXiv:2505.20137) — **NICHT** das früher angenommene `L < T < 2L`. Empirisch (`docs/12`):
  bei `[784,256,256,10]` und `T=20` ist das Netz noch *under-settled* (Konvergenz bei
  tol=1e-3 erst nach ≈ 31 Schritten).
- `η_x`, `η_w`: getrennte Lernraten für Inference und Learning (Config-Aliase `eta_x`/`eta_w`)

## Plugin-/Adapter-Architektur? — Nein.

Geprüft (Kriterien: Erweiterung durch Dritte / strukturell ähnliche, inhaltlich diverse
Fälle / separates Versioning). Hier liegt eine **einzelne kohärente Research-Implementierung
mit einem Mechanismus** vor, keine austauschbaren Fälle. Eine Strategy/Adapter-Schicht
wäre Overhead ohne Nutzen. Die *eine* Abstraktion, die sich lohnt, ist die
`train_and_eval(config)`-Schnittstelle (siehe `02`) — nicht wegen Plugins, sondern damit
Phase 4 andocken kann.
