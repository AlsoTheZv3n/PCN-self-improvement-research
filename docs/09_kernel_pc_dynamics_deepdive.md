# 09 — Deep-Dive: Settling-Kernel & PC-Dynamik (Quellen durchgearbeitet)

Destillat aus Goemaere et al. 2025 (EO, arXiv:2505.20137, Code: `cgoemaere/pc_error_optimization`),
der PyTorch-C++/CUDA-Doku und den PC-Libraries. Ziel: mit maximalem Wissen in den
Kernel starten — inkl. der ehrlichen Frage, *welche* Formulierung der Kernel beschleunigt.

## 1. Was der Kernel exakt berechnet — der SO-Algorithmus (präzise)

"State Optimization" (SO) = die Standard-PC-Formulierung. Genauer als in `01`, direkt aus
dem EO-Paper (Algorithmus 3), denn **das** muss die PyTorch-Referenz und der Kernel
implementieren:

```
Feedforward-Init (ff_init):
  s_{-1} = x
  for i = 0..L-1 (SEQUENZIELL):  ŝ_i = f_θi(s_{i-1});  s_i = ŝ_i   # ε_i startet = 0

State-Updates:
  for t = 1..T:
    for i = 0..L-1 (PARALLEL):   ŝ_i = f_θi(s_{i-1});  ε_i = s_i − ŝ_i
    ŷ = f_θL(s_{L-1});  E = ½ Σ‖ε_i‖² + L(ŷ,y)
    ε_L = ∇_ŷ L                                        # Output-Randbedingung
    for j = 0..L-1 (PARALLEL):   ∇_sj E = ε_j − (∂ŝ_{j+1}/∂s_j)ᵀ ε_{j+1}
                                 s_j ← s_j − λ ∇_sj E   # λ = state learning rate
Weight-Update (am Gleichgewicht):
  for j = 0..L-1 (PARALLEL):     ∇_θj E = −(∂ŝ_j/∂θj)ᵀ ε_j
                                 θ_j ← θ_j − η ∇_θj E   # LOKAL
```

**Die Parallelisierungs-Struktur (Kern der Kernel-Contribution):** Innerhalb eines
`t`-Schritts sind die Schleifen über die Layer `i`/`j` **parallel** (jedes ε_i und jedes
Δs_j hängt nur von Größen bei `t` ab). **Sequenziell ist nur** die `T`-Schleife und die
`ff_init`. Genau das nutzt kein bestehendes GPU-Framework aus (PCX gibt zu, dass JIT die
Layer nicht parallelisiert; `vmap` geht nur bei identischen Layer-Dimensionen).

## 2. Der Signal-Decay — warum SO bei flachem MNIST OK ist, bei tiefem nicht

EO identifiziert die Ursache des "tiefer = schlechter"-Problems: ein **exponentieller
Signal-Decay**. Bei der `ff_init` sind alle internen Energien 0, nur die Output-Energie
ist ≠ 0. Das Signal wandert dann rückwärts, **pro Schritt eine Schicht**, und wird bei
jedem Schritt mit `λ` (< 1 für Stabilität) gedämpft → `‖ε_{L-i}‖ ∝ C(t,i) · λ^i ·
(1−λ)^{t−i}` (Pascal'sches Dreieck / Binomialformel). Bei `λ ≈ 0.1` fällt das Signal
binnen **4–8 Schritten** unter die float32-Maschinenepsilon-Schwelle → tiefe Layer
bleiben effektiv **untrainiert**.

Konsequenzen für unser Projekt:
- Bei MNIST mit 2–3 Hidden-Layern ist das **kein** Problem (zu flach). SO funktioniert.
- **Aber**: pro-Schicht-Energie/`ε`-Norm messen und plotten → zeigt Verständnis, und beim
  Hochgehen auf 8/20 Layer reproduziert man das Problem (gutes Limitations-/Ablations-Material).
- Faustregel aus der Literatur: `L < T < 2L` Iterationen sind oft ausreichend; im EO-Paper
  war `T` (#iters) bei MLPs empirisch oft schon **4** optimal.
- Wichtig (EO-Einsicht): Die `ff_init`-Heuristik funktioniert so gut, weil sie **dem
  ersten Schritt von EO entspricht**. Und: orthogonale Gewichts-Init (Eigenwerte ≈ 1)
  mildert den Decay (deshalb war im EO-Paper der 20-Layer-MLP-Baseline überraschend stark).

## 3. EO als Alternative — und die EHRLICHE Value-Prop des Kernels

EO (Error Optimization) reparametrisiert PC: optimiere die **Fehler `ε`** statt der States.
States werden als `s_i = ŝ_i + ε_i` rekonstruiert (= Feedforward-Pass mit Perturbationen
`ε_i` pro Layer — exakt der VAE-Reparametrisierungs-Trick). Energie identisch
(`E = ½Σ‖ε_i‖² + L`), Weight-Update bleibt **lokal**. Der Clou: EO verbindet den
Berechnungsgraphen **global**, sodass `ŷ = func(x, ε_0..ε_{L-1})` — und dann transportiert
**Backprop** das Output-Signal in **einem** Durchgang unattenuiert zu allen `ε_i`.
Resultat: EO konvergiert **~100× schneller** als SO auf tiefen Netzen und matcht BP.

**Damit die ehrliche Spannung zur Kernel-Idee:** Auf GPU ist EO (= Backprop) bereits
hocheffizient. "SO per Kernel schnell machen" konkurriert also teilweise mit "einfach EO
nehmen". Die *niche*, die der Kernel sauber besetzt:
- EO verliert die **parallel/lokale** Eigenschaft (es nutzt globalen Backprop). **SO ist
  die neuromorphik-treue, lokal-parallele Formulierung.** Wer SO-Experimente auf GPU
  tractable machen will (für die Hardware-faithful-Forschung), braucht genau diesen Kernel.
- EO kann **zu reinem Backprop kollabieren**, wenn `T=1` oder `λT` zu klein (Theorem C.9
  im Paper) — man "entdeckt Backprop versehentlich neu". Ein kernel-beschleunigtes SO
  hat dieses Risiko nicht.
- Der stärkste, ehrlichste Beitrag ist daher eine **Benchmark-Matrix**, nicht nur ein Kernel:

| Methode | Lernregel | GPU-Charakter | Was sie misst |
|---------|-----------|---------------|---------------|
| PyTorch-SO | lokal, iterativ | viele Mini-Kernel-Launches | Baseline-Overhead |
| **Fused-CUDA-SO (unser Kernel)** | lokal, iterativ | 1 Launch/Schritt, States resident | wie viel Overhead = Launch-Overhead |
| EO | lokal-Update, globaler Backprop-Transport | cuBLAS-effizient | digitaler "Best Case" für PC |
| Backprop | global | cuBLAS-effizient | absolute Referenz |

Der Kernel ist das **neue Artefakt**; die Matrix (Accuracy + Wall-Clock + pro-Schicht-Energie
über Tiefe) ist der **wissenschaftliche Beitrag**. Das ist sauber, ehrlich und distinkt.

### Gemessenes Ergebnis (M6, 2026-06-09, RTX 3080 Ti, `[784,256,256,10]`)

Kernel implementiert (`pcn/kernels/settling_cuda.py`, correctness-verifiziert: allclose ~1e-7
vs PyTorch). Wall-Clock pro voller T-Schritt-Settle (CUDA-Events + Warmup, `scripts/bench_kernel.py`):

Hybrid v1/v2/v3 (Speedup = PyTorch / fused; >1 = Kernel schneller):

| Batch | v1 | v2 (Batch-Tiling) | **v3 (+P0-Cache)** | Pfad |
|---|---|---|---|---|
| 64 | **3,2×** | 3,2× | 2,3–2,9× | v1 |
| 256 | ~1,0× | 1,4× | **~2,0×** | v3 |
| 1024 | 0,33× | ~0,85× | **~1,0×** | v3 |
| 4096 | — | ~0,30× | ~0,33× | → cuBLAS |

**Befund (These bestätigt + Ceiling):** PyTorch-SO ist **~0,5 ms/Schritt batch-unabhängig** →
launch-overhead-bound. **v1** (ein Block/Sample) gewinnt 3,2× klein, verliert groß (naiver
GEMM, bandbreiten-limitiert). **v2** (TB=8-Kachel, Gewicht-Reuse) und **v3** (+ φ(Input)-Tile
in Shared, eliminiert ~256× redundante P0-Reads) schieben den Crossover auf **~1024–1500**
(B=256: 1,4→2,0×). **Aber B≥2048 bleibt cuBLAS-Land** — dort dominiert Compute, und die naive
In-Kernel-GEMM erreicht nicht cuBLAS' register-geblockten FLOP-Anteil. **Architektonischer
Tradeoff:** der fused-resident Entwurf (States resident → killt Launch-Overhead, gewinnt klein)
und ein cuBLAS-Klasse-GEMM (gewinnt groß) sind gegenläufig — Residenz frisst das Budget fürs
GEMM-Tiling. `settle()` dispatcht hybrid; alle allclose ~1e-6. Details: `docs/12` §4d.

## 4. PyTorch-C++/CUDA-Extension — die Mechanik (code-level)

Zwei Build-Pfade:
- **`torch.utils.cpp_extension.load_inline(...)`** — JIT, kompiliert beim Import. Für
  schnelles Prototyping. Kein `setup.py` nötig.
- **`setuptools` + `setup.py`** mit `CUDAExtension` — AOT, sauberes Modul. Für PyTorch
  **≥ 2.10 die ABI-stabile LibTorch-API** nutzen → läuft ohne Recompile über
  PyTorch-Versionen. Referenz-Repo: **`pytorch/extension-cpp`** (`mymuladd`-Beispiel, zwei
  Varianten: ATen/LibTorch und ABI-stable).

Struktur:
- **`.cu`**: `__global__` (vom Host gestartet) und `__device__` (vom GPU gerufen). Auf
  Tensordaten via `tensor.data_ptr<float>()` oder `packed_accessor32<...>` zugreifen.
  `AT_DISPATCH_FLOATING_TYPES(...)` für dtype-Dispatch. Launch:
  `kernel<<<grid, block, 0, stream>>>(...)`; danach `cudaGetLastError()` prüfen.
- **`.cpp`-Wrapper**: Tensoren entgegennehmen, **device/dtype/contiguity prüfen**, an die
  `.cu`-Funktion forwarden, via `pybind11` (oder `TORCH_LIBRARY`) nach Python exposen.
- **Python**: `@torch.library.register_fake(...)` registrieren → `torch.compile`-kompatibel.
  Lade-Reihenfolge der Registrierungen beachten (falsche Reihenfolge → Fehler).

## 5. Die Fusion-Strategie (das eigentliche Kernel-Design)

Pro Settling-Schritt fallen an: ein Matmul (`ŝ = W·θ(s)`), elementweise Ops (`ε = s−ŝ`),
ein Transpose-Matmul (`Wᵀε`), das elementweise `θ'`-Gating, der State-Update. Bei
MNIST-Größen (≤256-dim) sind diese Ops winzig → **Kernel-Launch-Overhead dominiert**.

- **Fuse einen ganzen Schritt** in einen Launch: ein Block pro Sample/Tile, alle `s_i`
  über die `T` Schritte in **Shared Memory/Registern** resident halten, die `T`-Schleife
  **im Kernel**. Aus `T × (Layer × Ops)` Launches wird **1** Launch.
- **Persistent Kernel**: liest `W` einmal aus Global Memory, hält Aktivierungen lokal über
  alle `T` Schritte → minimiert Memory-Traffic (adressiert SO's höhere Raum-Komplexität).
- Caveat: große Breiten wollen cuBLAS; bei MNIST-Breiten kann ein hand-fused
  GEMM-in-Shared-Memory gewinnen. Exakt dein custom-GEMM-Revier vom 355M-LLM.

## 6. Benchmark-Disziplin (sonst sind die Zahlen wertlos)

- **`torch.cuda.synchronize()` vor jeder Zeitmessung** — CUDA ist asynchron (GPU MODE Lec 1).
- **Korrektheit zuerst**: identisches Gleichgewicht/Accuracy wie PyTorch-SO beweisen, dann Speed.
- **Nsight Compute**-Report (Kernel-Launch-Count vorher/nachher, Occupancy) als Paper-Artefakt.
- ≥3 Seeds, Konfidenzintervalle.

## 7. Referenz-Code zum Studieren
- **`cgoemaere/pc_error_optimization`** — EO, sauberer SO- und EO-Vergleich (PyTorch-style).
- **`bjornvz/PRECO`** — PyTorch, PC-Netze **und** -Graphen, mit Tutorial+Survey (van Zwol et al.).
- **`thebuckleylab/jpc`** — JAX, <1000 LOC, ODE-Solver (Second-Order > Euler) (arXiv:2412.03676).
- **`liukidar/pcx`** — JAX-Benchmark-Library, A100-Messprotokoll (arXiv:2407.01163).
