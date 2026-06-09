# pcn/kernels — OPTIONAL Phase-3 / M6 CUDA settling kernel

**Nicht nötig für Phase 1–2.** Der Default-Backend (`pcn/settling.py`, PyTorch) ist voll
funktionsfähig. Der CUDA-Pfad ist ein **bewusst zuzuschaltendes** Add-on (Hook A).

## Status (2026-06-09): gebaut (v1+v2+v3), correctness-verifiziert, gebenchmarkt, in §4g-Studie integriert
- **Die CUDA-Quelle steht in `settling_kernel.cu`** (echte `.cu`, kein auskommentierter Stub —
  von `settling_cuda.py` via `load(sources=[...])` kompiliert); `settling_cuda.py` ist nur noch
  Wrapper + Dispatch. Zwei fused Settling-Kernel (ganze T-Schleife in einem Launch): **v1** ein
  Block pro Sample (small-batch), **v2** ein Block pro TB=8-Sample-Kachel mit Gewicht-Reuse
  (large-batch). `settle()` dispatcht hybrid nach Batch-Größe.
- **Korrektheit:** beide matchen `_settle_pytorch` auf ~1e-6 (allclose; tanh/identity/sigmoid,
  act 0/1/2; `scripts/verify_kernel.py`; **mit stabiler Config** prüfen — instabiles Settling
  verstärkt Float-Diffs und täuscht Fehler vor).
- **Benchmark** (`scripts/bench_kernel.py`, RTX 3080 Ti, `[784,256,256,10]`): **3,2× @ B=64**
  (v1), **~2,0× @ B=256** (v3, P0-Cache), ~break-even @ B=1024, langsamer @ B≥2048
  (→ `backend="pytorch"`/cuBLAS). Batch-Tiling (v2) + P0-Caching (v3) schoben den Crossover von
  ~256 auf ~1024–1500; B≥2048 ist cuBLAS-Land (Compute-bound, naiver In-Kernel-GEMM). `docs/12` §4d.
- **Aktivierung bewusst:** `is_available()` → `True` nur mit Env-Flag `PCN_CUDA_KERNEL=1`
  **und** CUDA-Device. Sonst `False` → `settle(backend="cuda")` wirft `NotImplementedError`
  (kein stilles Fallback; Default-/Test-Pfad bleibt PyTorch).

## Bauen / Nutzen
1. CUDA-torch passend zum lokalen Toolkit (hier cu126 / nvcc 12.6; pyproject-Index gepinnt).
2. Build braucht **nvcc + MSVC + ninja** — auf Windows in einer **vcvars64-Shell** ausführen.
3. `PCN_CUDA_KERNEL=1 ... uv run python scripts/bench_kernel.py` (kompiliert `settling_kernel.cu`
   via `load(sources=[...])` beim ersten Aufruf, cached danach). Korrektheits-Gate: `scripts/verify_kernel.py`.
4. Im Code: `train_and_eval({"backend": "cuda", ...})` oder `run_alternating(..., backend="cuda")`
   mit gesetztem Env-Flag. §4g-Studie: `run_experiments.py alternating --backend cuda`.

## Scope & nächste Schritte (docs/09)
Spezialisiert auf 2-Hidden-Layer (`model.n==3`); fixes T (kein tol); keine Energie-
Aufzeichnung; float32. **v2+v3 erledigt:** Batch-Tiling (Gewicht-Reuse) + P0-Caching (φ(Input)-Tile in Shared) →
Crossover ~256→~1024–1500. **B≥2048 bleibt cuBLAS-Land** (Compute-bound; der fused-resident
Entwurf und ein register-geblocktes GEMM sind architektonisch gegenläufig — docs/12 §4d).
**Offen (v4, jenseits sinnvollen Aufwands):** voll register-geblocktes Tiled-GEMM. Wertvoller:
allgemeine Tiefe, EO-Vergleichsarm, Nsight-Compute-Report, AOT-Build (`CUDAExtension`).
