# 05 — Paper-Outline

## Format

- **8–12 Seiten**, NeurIPS/ICML-Stil, arXiv-fähig
- **Sprache: Englisch** (Norm für arXiv/NeurIPS; die Projekt-Doku bleibt Deutsch)
- Kein 80-Seiten-Thesis-Format. "So wenig Seiten wie möglich" ist hier goldrichtig:
  ein straffes Paper ist fürs Portfolio wirksamer als eine lange Arbeit.

## Ehrliche Novelty-Framing

Ein From-Scratch-MNIST-PCN, das bekannte Resultate reproduziert, ist **Verständnis +
saubere Reproduktion**. Das "Neue" muss aus **einem** fokussierten Winkel kommen — nicht
aus einem Durchbruch. Das ist die richtige Erwartung, kein Defizit. Reviewer akzeptieren
eine "überraschende, aber reale" Beobachtung als Beitrag; sie erwarten von einem solchen
Paper keinen Paradigmenwechsel.

## Novelty-Hooks (genau einen als Hauptbeitrag)

- **(A) CUDA-Settling-Kernel** — "Wie schnell wird PC-Inference auf GPU, und was ist der
  reale Overhead vs. Backprop?" Konkret, messbar, anschlussfähig ans LLM-Projekt.
  Adressiert eine reale offene Engineering-Frage (siehe `06`, Problem 2). **Empfohlen
  als Hauptbeitrag.**
- **(B) Empirische Ablation** — Effekt von `T` und Präzisions-Schedules auf MNIST, mit
  Energie-Dynamik. Wird ohnehin vom Such-Loop (`04`) produziert.
- **(C) Generativ / Robustheit** — PCN als generatives Modell: Occlusion + Anomalie +
  Rausch-Robustheit gegen das MLP. Als kurzer qualitativer Abschnitt mit Bildern.

**Empfehlung:** **(A)** als Hauptbeitrag + **(C)** als kurzer qualitativer Abschnitt für
die Figuren + **(B)** als Ablations-Tabelle. Das hält das Paper bei ~10 Seiten.

## Struktur

| § | Inhalt | ~Seiten |
|---|--------|---------|
| Abstract | Beitrag in 4 Sätzen | — |
| 1 Introduction | Motivation: Backprop-Implausibilität, FEP, warum PC | 1 |
| 2 Background | PC-Gleichungen, Bezug zu BP (Whittington & Bogacz, Song, Millidge) | 1.5 |
| 3 Method | From-Scratch-Implementierung + CUDA-Settling-Kernel (Hook A) | 2.5 |
| 4 Experiments | MNIST-Accuracy, Energie-Dynamik, Benchmark, Ablation (B), generativ (C) | 3 |
| 5 Limitations | **ehrlich**: Depth-Scaling, Skala, PC≈BP-Frage (siehe `06`) | 1 |
| 6 Conclusion | Was gezeigt, was offen | 0.5 |
| References | — | — |

## Limitations-Abschnitt (wichtig — Reviewer achten darauf)

Ehrlich benennen, gestützt auf `06_open_problems_and_approaches.md`:
- Depth-Scaling-Problem ("mehr Layer → schlechter") — bei MNIST-Skala umgangen, nicht
  gelöst
- kein Transformer-Scale-Beweis
- die offene Frage, *wann* PC überhaupt etwas anderes tut als BP (arXiv:2602.07697)

Ein ehrlicher Limitations-Abschnitt stärkt das Paper — er zeigt, dass die Position im
Feld verstanden ist.

## Workflow

LaTeX-Skeleton ab Phase 2 parallel anlegen; Figuren aus den geloggten
`train_and_eval`-Metriken generieren (reproduzierbar). Schreiben läuft parallel zu
Phase 2–4, nicht erst danach.

> **Hinweis (M7):** Die obige Outline ist der *Plan*. Der Draft unten reflektiert die
> *tatsächlichen* Befunde — und die weichen vom ursprünglichen Optimismus ab: Hook B wurde
> ein **Null- + Confound-Resultat** (kein PC-Vorteil), Hook C funktioniert erst mit einem
> *generativen* PC. Das macht das Paper ehrlicher und (post-Replikationskrise) wertvoller.

---

# Paper Draft (v1, M7) — English

**Title:** *A Sober Reckoning of Predictive Coding on MNIST: Where the Compute Goes, Why
PC-vs-Backprop Comparisons Mislead, and What a Generative PC Buys You.*

## Abstract

Predictive Coding (PC) is a biologically motivated, local-learning alternative to
backpropagation (BP), but the literature reports inconsistent advantages. We build a
from-scratch State-Optimization PC network on MNIST in pure PyTorch (no autograd in the
learning rule; the settling/weight gradients are verified against autograd) and use it for a
deliberately sober, reproducible study with three focused contributions. **(A) Systems:** among
open PC libraries — all JAX/JIT (PCX, JPC, ngc-learn) or PyTorch-autograd (PRECO, Torch2PC, pypc,
ePC) and *none* shipping a custom inference kernel — we contribute, to our knowledge, the first
hand-written fused CUDA *settling* kernel for PC. The persistent, states-resident, single-launch
technique is itself established prior art (cuDNN persistent RNN; LLM megakernels); our contribution
is its **transfer to PC** plus the measurement that PC inference at MNIST scale is
**launch-overhead-bound** (PyTorch ~0.5 ms/step regardless of batch, GPU ~25–49% utilized; a known
"framework tax", not our discovery). Fusing the loop into one launch gives **3.2× at batch 64**;
batch-tiling + input-activation caching push the crossover to ~1024; and a fused-resident kernel
and a cuBLAS-class GEMM are architecturally opposed, so at batch ≥2048 cuBLAS wins. **(B) Methodology:**
a fair PC-vs-BP comparison (matched architecture, initialization, data, and loss; per-method
LR tuned on a validation split; ≥3 seeds with bootstrap CIs) finds **PC ≈ BP** across bulk
accuracy, noise robustness, sample efficiency, and continual learning (domain-IL and class-IL,
including a Song-style Split-FashionMNIST setup). We identify **three confounds** —
loss function (cross-entropy vs MSE), the stability–plasticity trade-off in forgetting metrics,
and an untuned baseline learning rate — each of which manufactures a spurious "PC advantage" in
naive comparisons, explaining the field's inconsistency. **(C) Capability:** a generatively
trained PC (label→image) recovers PC's "one mechanism, many tasks" property — classification,
class-conditional generation, inpainting, and anomaly detection (AUC 1.0 vs uniform-noise OOD)
from a single settling rule, where a discriminatively trained PC produces noise. All code,
36 tests, and experiment artifacts are released.

## 1 Introduction

Backpropagation requires a non-local backward pass (the "weight transport problem"), motivating
local alternatives. PC minimizes a layer-wise precision-weighted prediction-error energy via a
settling (inference) phase followed by a local Hebbian weight update, and is claimed to (i)
approximate BP, (ii) need fewer samples, (iii) forget less in continual learning, and (iv)
double as a generative model. Yet results are inconsistent across the literature. We take the
honest position that a from-scratch MNIST PCN is primarily *understanding + clean reproduction*;
novelty comes from three focused angles, not a breakthrough. **Contributions:** (A) a fused
CUDA settling kernel and the first launch-overhead characterization of PC inference; (B) a
confound-controlled fair PC-vs-BP protocol and the finding that the advantages largely vanish
under it; (C) a working generative PC and the discriminative-vs-generative distinction.

## 2 Background

We use the State-Optimization (feedforward-indexing) formulation: layer *i* predicts layer
*i+1* via `pred_{i+1} = W_i φ(s_i) + b_i`; the energy is `E = ½ Σ_k ‖s_k − pred_k‖²` (isotropic
precision Π=I, the modern default — PCX fixes Σ=I, JPC uses plain SSE). Inference relaxes the
free states by `s_k ← s_k − λ(ε_k − φ'(s_k)⊙(Wᵏᵀ ε_{k+1}))` to (approximate) equilibrium, then
a local Hebbian update `ΔW_i ∝ ε_{i+1} φ(s_i)ᵀ` is applied. This is "prospective configuration"
(Song et al. 2024): the *equilibrium of standard energy-based PC*, not a separate algorithm.
PC approximates BP under the fixed-prediction assumption (Whittington & Bogacz 2017; Millidge,
Tschantz & Buckley 2022) and provably converges to BP only in restricted regimes (linear
residual nets, width≫depth, equilibrated activities; Innocenti, Achour & Bogacz 2026).

## 3 Methods

**Implementation.** Pure PyTorch tensor ops; every state/weight update under `@torch.no_grad`.
The hand-derived settling and Hebbian gradients are checked against `torch.autograd` to ~1e-5
(used only as an external oracle in tests). A convergence criterion (relative energy-delta `tol`)
makes "settling-steps-to-converge" a measurable metric.

**(A) Fused CUDA settling kernel.** The PyTorch backend launches many tiny ops per settling
step. The kernel runs the entire *T*-step loop in one launch with states resident across steps:
**v1** one block per sample (max block parallelism, small batch); **v2** one block per tile of
*TB*=8 samples, each weight read once and reused across the tile (large-batch weight reuse);
**v3** additionally caches the (clamped, hence constant) φ(input) tile in shared memory,
removing ~256× redundant global reads. `settle()` dispatches by batch size. Both kernels match
the PyTorch backend to ~1e-6 (verified in a *stable* settling regime — unstable settling
amplifies float-order differences and falsely looks like a bug).

**(B) Fair-comparison protocol.** The BP-MLP baseline *clones the PCN's init* and uses the
*identical* forward function, so only the learning rule differs. We add an MSE-to-one-hot BP
arm so the loss is matched (BP-MSE) alongside the practical BP-CE. LR is selected per method on
a held-out validation split and *reported* on test; bootstrap 68% CIs over ≥3 seeds; PC's state
and weight LRs are tuned **jointly** (a stability frontier couples them).

**(C) Generative PC.** Flipping the orientation to `[10, 256, 256, 784]` (label→image) and
training generatively (clamp label and image, settle hidden) yields a generative model;
generation is the forward pass label→image, inpainting clamps visible output pixels, anomaly
reads residual energy at the predicted label.

## 4 Experiments

**4.1 (A) Kernel.** PyTorch-SO is ~0.5 ms/step **independent of batch** (64→4096) — launch-bound;
GPU 25–49% utilized. *Mechanistic evidence:* a profiler launch count (cudaLaunchKernel) shows
PyTorch-SO issues **31 kernel launches per settling step** (620/1240/2480 for T=20/40/80), scaling
linearly with T, whereas the fused kernel issues **exactly one** regardless of T — a 620–2480×
launch reduction, confirming launch overhead (not compute) is what the kernel removes. Fused
speedups (RTX 3080 Ti, `[784,256,256,10]`): **3.2× @ B=64**,
**~2.0× @ B=256** (v3), ~1.0× @ B=1024, ~0.33× @ B≥2048. Batch-tiling + φ(input)-caching move the
crossover from ~256 to ~1024–1500. At B≥2048 *compute* dominates and our naive in-kernel GEMM
cannot match cuBLAS's register-blocked efficiency — and the fused-resident design (which kills
launch overhead) and a cuBLAS-class GEMM are architecturally opposed (residency consumes the
register/shared budget a fast GEMM needs). The kernel therefore wins exactly in the small/medium
batch (online) regime relevant to PC.

**4.2 (B) PC vs BP (matched loss, tuned, CI'd).**
- *Bulk accuracy (10k):* PC(MSE) 83.5% [82.8,84.3] vs **BP(MSE) 83.9% [83.6,84.3]** — gap +0.4 pp,
  CIs overlap. BP(CE) 89.9%; the **loss effect** BP(CE)−BP(MSE) = +5.9 pp — i.e. most of a naive
  "BP beats PC" gap is the *loss function*, not the learning rule.
- *Noise (σ=1.0):* BP(MSE) 60.4% vs PC(MSE) 52.6% (most of a naive ~28 pp gap was loss).
- *Sample efficiency, domain-IL (Permuted-MNIST):* no PC advantage.
- *Class-IL (Split-FashionMNIST 2×5, Split-MNIST 5×2; BP-LR tuned by learn-accuracy):* PC ≈ BP;
  the earlier "PC forgets less" was a double artifact — comparing against BP(CE) (which forgets
  more) and ignoring that PC *learns each task less well* (lower diagonal accuracy). A stuck,
  untuned BP-MSE even manufactured a false PC win, caught by the learn-accuracy gate.
- *Confounds:* (1) loss function, (2) plasticity (forgetting must be read against learn-accuracy
  and retained task-0 accuracy), (3) baseline LR (an untrained baseline fakes a PC win). PC's
  state/weight LRs must be tuned jointly (stability frontier).

**4.3 (B) Autonomous search (Phase 4).** A bounded random search over (T, η_x, η_w) as a thin
layer over `train_and_eval` independently rediscovers the η_x↔η_w interaction (best: low η_x,
high η_w, high T) — an internal consistency check.

**4.4 (C) Generative PC.** A discriminative PCN cannot generate (clamping a label and settling
the input yields noise; `feedforward_init` is a trivial zero-energy equilibrium). The
generative PC produces **recognizable per-class digit prototypes**, reconstructs occluded
halves, and separates uniform-noise OOD with **anomaly AUC 1.0** (in-dist energy 5.1 vs 94.5).
Stability note: the 784-dim output makes the hidden-settling gradient ~√784 larger, so the
state LR must scale as ~1/√(output-dim) or settling diverges.

## 5 Limitations (honest)

MNIST/FashionMNIST scale, two hidden layers, vanilla SO-PC. The class-IL study is *Song-style*,
not Song's exact alternating-training protocol, so our null does **not** globally refute Song —
it shows that *at this scale with a fairly tuned baseline* the advantage vanishes. The kernel
loses at batch ≥2048 (cuBLAS regime) and is specialized to depth-2; the "first PC CUDA kernel"
novelty claim is not yet verified against an exhaustive literature search. Generative output is
class *prototypes* (one-hot input → ≤10 distinct images), not diverse samples; uniform noise is
an easy OOD (near-OOD such as FashionMNIST would be a harder test). Precision schedules and the
iPC update variant are reserved, not implemented. Depth-scaling and the scale gap to
Transformers are untouched (frontier-lab work).

## 6 Conclusion

PC's headline advantages over BP are **fragile to experimental confounds**: matched-loss,
fairly tuned, CI'd comparisons on MNIST show parity, and we give a three-item confound checklist
that explains the literature's inconsistency. PC's inference cost is dominated by kernel-launch
overhead (fixable at small batch with a fused kernel; bounded by an architectural GEMM trade-off
at large batch). And a *generatively* trained PC genuinely delivers the "one mechanism, many
tasks" property. The contribution is a reproducible, honestly-scoped baseline plus two
transferable insights (the launch-overhead characterization and the confound checklist).

## References (corrected — see `docs/11`)

Rao & Ballard 1999 (Nat. Neurosci., DOI 10.1038/4580); Bogacz 2017 (J. Math. Psychol.,
10.1016/j.jmp.2015.11.003); Whittington & Bogacz 2017 (Neural Comput. 29(5),
10.1162/NECO_a_00949); Millidge, **Seth** & Buckley 2021 (arXiv:2107.12979); Millidge, Tschantz
& Buckley 2022 — PC approximates backprop along arbitrary graphs (arXiv:2006.04182); Song,
Millidge, Salvatori, Lukasiewicz, Xu & Bogacz **2024** (Nat. Neurosci. 27(2):348–358,
10.1038/s41593-023-01514-1); Zahid, Guo & Fountas **2023** (Neural Comput. 35(12):1881–1909,
peer-reviewed; arXiv:2304.02658); Goemaere, Oliviers, Bogacz & Demeester 2025/26 — **ePC**,
"Overcoming Exponential Signal Decay…" (arXiv:2505.20137); Salvatori et al. 2024 — iPC
(arXiv:2212.00720, ICLR); Qi, Forasassi, Lukasiewicz & Salvatori 2025 (arXiv:2506.23800);
Innocenti, Achour & Bogacz 2026 — Infinite Width/Depth limits (arXiv:2602.07697, *read in full
before citing*); Pinchetti/Salvatori et al. — PCX (arXiv:2407.01163); Millidge et al. — JPC
(arXiv:2412.03676); van Zwol et al. — PRECO/survey (10.1145/3797870, arXiv:2407.04117);
Lopez-Paz & Ranzato 2017 — GEM (arXiv:1706.08840); Lomonaco et al. 2021 — Avalanche
(arXiv:2104.00405); PyTorch Custom C++/CUDA Operators + `extension-cpp`.

**Technique lineage / related systems work (for Hook A's honest positioning — see `docs/15`):**
Diamos et al. 2016 — Persistent RNNs (Baidu; `baidu-research/persistent-rnn`) + cuDNN
`CUDNN_RNN_ALGO_PERSIST_*` — the original persistent, states-resident, single-launch kernel
motivated by per-step launch overhead at small batch; the technique we transfer to PC. LLM
megakernels (Mirage, FlashFormer, Hazy-Research "No Bubbles", 2025–26) — same pattern for
transformer decode. DEQ (Bai et al.; `locuslab/deq`) — closest fixed-point analog (PyTorch, no
custom kernel). ePC (Goemaere et al., arXiv:2505.20137) — closest PC-side related work, attacks
the *same* settling-cost goal *algorithmically* (error- vs state-reparametrization), orthogonal
to and combinable with our kernel-level approach. "Framework tax" of many small ops
(arXiv:2302.06117) — the launch-overhead observation is generic, not claimed here.

**Equilibrium Propagation (closest algorithmic relative — audited).** We additionally audited EP
across its canonical (Scellier & Bengio 2017; Laborieux et al. 2021; Ernoult et al. 2019/2020;
Laborieux & Zenke 2022) and recent (StochEP, arXiv:2511.11320; Scalable CRNN-EP, arXiv:2508.15989)
implementations: all run the free-/nudged-phase settling via PyTorch-autograd, JAX/XLA-JIT, or a
Python host loop. The only EP repo with hand-written GPU kernels (`autonull/bioplausible`) uses
per-step *elementwise Triton* kernels with the settling iteration on the host — not a hand-written
CUDA, fused, persistent-resident in-kernel settling loop, and EP rather than PC. GPU-side EP
settling acceleration otherwise lives in analog/neuromorphic hardware (memristor, coupled
oscillators), not CUDA. To our knowledge, no EP implementation ships a hand-written CUDA settling
kernel either — so the "first for PC" claim is bounded by neither the PC nor the EP literature.
