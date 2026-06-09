# 07 — Build-Guide: CUDA-Settling-Kernel (Problem 2)

Ziel: die **Inference-Phase** (das Settling) in einem handgeschriebenen CUDA-Kernel
ausführen und gegen die PyTorch-Referenz benchmarken. Das ist Novelty-Hook A.

> **Status (2026-06-09): v1+v2+v3 GEBAUT + gebenchmarkt + in §4g-Studie integriert.** Die CUDA-Quelle
> ist die echte `pcn/kernels/settling_kernel.cu` (von `settling_cuda.py` via `load(sources=[...])`
> kompiliert — kein Inline-String/Stub mehr), correctness-verifiziert (allclose ~1e-6 vs PyTorch,
> tanh/identity/sigmoid; `scripts/verify_kernel.py`) und gemessen: **3,2× @ B=64, ~2,0× @ B=256,
> break-even @ B=1024**; B≥2048 → cuBLAS (architekt. Tradeoff). End-to-End auf der §4g-PC-Last:
> **1,45× bei Per-Seed identischen Ergebnissen** (`scripts/bench_alternating_backend.py`). Build:
> nvcc + MSVC (vcvars64-Shell) + torch cu126; `PCN_CUDA_KERNEL=1`, `backend="cuda"`.
> **Neuheit (verifiziert, `docs/15`):** „die unten behauptete Lücke" stimmt — *kein* offenes
> PC-Repo hat einen Custom-CUDA-Settling-Kernel (alle JAX/JIT oder PyTorch-autograd). ABER die
> *Technik* (persistent/fused/single-launch) ist Prior Art (cuDNN/Baidu Persistent RNN 2016,
> LLM-Megakernels) → Beitrag = **erste Anwendung auf PC + Messung**, „to our knowledge". Ergebnis:
> `docs/12` §4d, `docs/09`. Guide unten = AOT-Build + offene v4 (register-geblocktes GEMM).

## Warum das eine echte Lücke ist (nicht nur "Matmul in CUDA neu schreiben")

- Es gibt **keine** PC-Library mit handoptimiertem CUDA-Inference-Kernel. Bestehende
  Libs sind entweder JAX (JPC, PCX, ngc-learn) oder PyTorch-autograd-basiert (PRECO,
  Torch2PC, pypc, pybrid, Song's Library).
- **PCX gibt selbst zu**, dass die Inference-Phase **nicht parallelisiert** ist: JIT
  schafft es nicht, die Layer-Ausführung zu parallelisieren; ihr `vmap`-Workaround
  funktioniert nur, wenn **alle Layer dieselbe Dimension** haben (unpraktisch). Ein
  handgeschriebener Kernel hat diese Einschränkung nicht.
- Damit ist eine saubere Messung ("Wie groß ist der reale Inference-Overhead vs. BP, und
  wie weit lässt er sich durch Fusion drücken?") eine reale, offene Engineering-Frage.

## Die präzise Parallelisierungs-Einsicht (der Kern der Contribution)

Das Settling läuft `T` Schritte. **Innerhalb** eines Schritts `t`:

```
1. ε_l = x_l − W_l·θ(x_{l+1})     für ALLE l   ← parallel (hängt nur von x bei t ab)
2. Δx_l = −η_x·[Π_l ε_l − θ'(x_l)⊙(W_{l-1}ᵀ Π_{l-1} ε_{l-1})]   für ALLE l ← parallel
```

- **Über die Layer** ist ein Schritt parallelisierbar (alle ε_l, dann alle Δx_l).
- **Sequenziell** ist nur die Kette der `T` Schritte.
- Der per-Schritt-Update ist ein Strom **vieler kleiner pointwise- + kleiner Matmul-Ops**
  (ε-Berechnung, Präzisionsgewichtung, das elementweise `θ'`, das Transpose-Matmul).

Bei MNIST-Größen (784-dim) dominiert der **Kernel-Launch-Overhead**: PyTorch startet jede
dieser Mini-Ops als eigenen Kernel. Genau das Szenario, das die offizielle PyTorch-Doku
am LLTM-Beispiel beschreibt — *"viele pointwise-Operationen in Folge, die sich fusionieren
und parallelisieren lassen"*. Die Contribution ist also ein **fused kernel**, der einen
ganzen Settling-Schritt (oder die ganze `T`-Schritt-Schleife für einen Batch) in **einem
Launch** erledigt und die Aktivierungen über die `T` Schritte in Shared Memory/Registern
resident hält (adressiert die höhere Raum-Komplexität von PC).

## Build-Pfad (PyTorch C++/CUDA-Extension)

1. **Prototyp mit `load_inline`** (JIT) — schnelle Iteration, kein Build-Setup.
   `torch.utils.cpp_extension.load_inline(...)`.
2. **Produktiv mit `setuptools`** (AOT) — `setup.py`, sauberes Modul. Für PyTorch ≥ 2.10
   die **ABI-stabile LibTorch-API** nutzen (läuft ohne Recompile über PyTorch-Versionen).
3. C++-Wrapper (`.cpp`): Tensoren entgegennehmen, Properties prüfen (device, dtype,
   contiguity), an die `.cu`-Funktion forwarden, via `pybind11` nach Python exposen.
4. Kernel (`.cu`): `__global__` (vom Host gestartet) / `__device__` (vom GPU gerufen).

### Quellen (kanonisch → praktisch)
- **PyTorch offiziell**: "Custom C++ and CUDA Extensions" und das neuere "Custom C++ and
  CUDA Operators" (docs.pytorch.org/tutorials/advanced/) — `load_inline` vs. `setuptools`,
  ABI-stabile API ab 2.10. **Primärquelle.**
- **`pytorch/extension-cpp`** (GitHub) — das offizielle End-to-End-Beispiel (`mymuladd`),
  zwei Varianten: ATen/LibTorch und ABI-stable. Als Vorlage kopieren.
- **GPU MODE Lecture 1** (Kopf & Saroufim; Notes z.B. christianjmills.com) — `load_inline`,
  Triton, **Profiling mit Nsight Compute**, `torch.cuda.synchronize()` vor jeder Messung
  (CUDA ist asynchron!). Pflicht fürs Benchmarking.
- NVIDIA "CUDA C++ Programming Guide" — Referenz für Memory/Synchronisation/Occupancy.
- Hinweis (RFC #152032): 2025 entstehen pythonische GPU-DSLs (Triton, `cuTile`,
  `cuda-python`). **Triton** ist eine legitime, einfachere Alternative zum reinen
  CUDA-C++-Kernel — als Vergleichspunkt oder Fallback erwägen.

## Womit vergleichen (Baselines)

- **PRECO** (`bjornvz/PRECO`, PyTorch) — PC-Netze **und** -Graphen in PyTorch, mit
  Tutorial+Survey (van Zwol et al., doi:10.1145/3797870). Nächste Referenz, weil PyTorch.
- **JPC** (`thebuckleylab/jpc`, JAX, arXiv:2412.03676) — <1000 LOC, nutzt **ODE-Solver**;
  ein Second-Order-Solver ist deutlich schneller als Standard-Euler. Saubere Vergleichs-
  zahlen für "Solver-Wahl beeinflusst Settling-Zeit".
- **PCX** (`liukidar/pcx`, JAX, arXiv:2407.01163) — die Benchmark-Library; misst Zeit/Epoche
  auf A100 gegen Song's Library und gegen BP. Direkt vergleichbares Messprotokoll.
- **EO / ePC** (Goemaere et al., arXiv:2505.20137) — adressiert exakt Settling-Effizienz
  (digitale Hardware); konvergiert Größenordnungen schneller. Lesen, bevor man optimiert —
  ggf. die EO-Reparametrisierung im Kernel mitimplementieren.

## Benchmark-Design (fürs Paper)

- Metrik: Wall-Clock pro Settling-Schritt **und** pro Epoche; gegen PyTorch-Referenz und
  gegen ein gleich großes BP-MLP (fair, gleiche Hardware — eine A100 reicht).
- Sweep: Settling-Schritte `T`, Layer-Breite, Tiefe (zeigt, wo Fusion am meisten bringt).
- Korrektheit zuerst: identische Endgewichte/Accuracy wie die PyTorch-Referenz beweisen,
  **dann** Speed messen. Immer `torch.cuda.synchronize()` vor der Zeitmessung.
- Profiling-Artefakt: Nsight-Compute-Report (Kernel-Launches vorher/nachher) ins Paper.
