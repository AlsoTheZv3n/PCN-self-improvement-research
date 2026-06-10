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
>
> **Hinweis (M8):** Der Draft ist auf den aktuellen Stand gebracht: Songs *exaktes* alternierendes
> Protokoll ist jetzt repliziert (§4.3, n=10, kein ≥1σ-Gap; Framing von einer adversarialen
> 3-Linsen-Prüfung korrigiert), der Kernel ist auf **beliebige Tiefe** verallgemeinert und treibt
> die §4.3-Studie (1,45×), und iPC + EWC sind als optionale Tiefen implementiert (§4.6). Frühere
> Limitationen (depth-2, „Novelty unverifiziert", „iPC reserved", „Song-style nicht exakt") sind
> damit aufgelöst. Vier Figuren (`figures/`) sind eingebunden. Nächster Schritt: LaTeX-Satz.

---

# Paper Draft (v2, M8) — English

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
and a cuBLAS-class GEMM are architecturally opposed, so at batch ≥2048 cuBLAS wins. The kernel
generalizes to **arbitrary depth** (validated to ~1e-6 across 1–4 hidden layers) and, **integrated
into our own continual-learning study**, accelerates it 1.45× end-to-end with per-seed-identical
results — tying the systems contribution to the science one. **(B) Methodology:**
a fair PC-vs-BP comparison (matched architecture, initialization, data, and loss; per-method
LR tuned on a validation split; ≥3 seeds with bootstrap CIs) finds **PC ≈ BP** across bulk
accuracy, noise robustness, sample efficiency, and continual learning (domain-IL and class-IL).
We identify **three confounds** — loss function (cross-entropy vs MSE), the stability–plasticity
trade-off in forgetting metrics, and an untuned baseline learning rate — each of which manufactures
a spurious "PC advantage" in naive comparisons, explaining the field's inconsistency. We then build
an **architecture-faithful replication of Song et al.'s *exact* alternating Split-FashionMNIST
protocol** and, treating training budget as a controlled axis, find **no PC-vs-BP(MSE) difference
exceeding 1σ at any budget** (n=10) — a result whose framing was corrected by an **adversarial
three-lens review** that caught us misattributing a learning-rate-robustness claim to Song's
interference figure (a reusable safeguard against the field's confounds). **(C) Capability:** a generatively
trained PC (label→image) recovers PC's "one mechanism, many tasks" property — classification,
class-conditional generation, inpainting, and anomaly detection (AUC 1.0 vs uniform-noise OOD)
from a single settling rule, where a discriminatively trained PC produces noise. We additionally
add incremental-PC (faster per-step learning at a ~T× smaller weight LR) and an EWC baseline (which
does forget less, validating the harness) as optional extensions. All code, 40 tests, figures, and
experiment artifacts are released publicly.

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
removing ~256× redundant global reads. `settle()` dispatches by batch size. A **general-depth** variant (one block per sample; the
per-layer weights/states/biases passed as device-pointer arrays with shared-memory offsets
computed at launch) lifts the two-hidden-layer specialization to *arbitrary depth*; activations
cover tanh, identity and sigmoid. All kernels match the PyTorch backend to ~1e-6 (verified in a
*stable* settling regime — unstable settling amplifies float-order differences and falsely looks
like a bug; checked across 1–4 hidden layers).

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
launch reduction (Fig 4), confirming launch overhead (not compute) is what the kernel removes. Fused
speedups (Fig 1; RTX 3080 Ti, `[784,256,256,10]`): **3.2× @ B=64**,
**~2.0× @ B=256** (v3), ~1.0× @ B=1024, ~0.33× @ B≥2048. Batch-tiling + φ(input)-caching move the
crossover from ~256 to ~1024–1500. At B≥2048 *compute* dominates and our naive in-kernel GEMM
cannot match cuBLAS's register-blocked efficiency — and the fused-resident design (which kills
launch overhead) and a cuBLAS-class GEMM are architecturally opposed (residency consumes the
register/shared budget a fast GEMM needs). The kernel therefore wins exactly in the small/medium
batch (online) regime relevant to PC. *Generality and use.* The general-depth variant reproduces
the PyTorch equilibrium to ~1e-6 across 1–4 hidden layers (16 configurations, tanh+sigmoid,
B∈{32,256}), so the kernel is not tied to a fixed topology; and run as the backend of our own
Song-exact study (§4.3, batch 32, the small-batch regime) it delivers **1.45× end-to-end** at
**per-seed-identical** accuracy — the kernel is a working engine for the science, not a microbenchmark.

**4.2 (B) PC vs BP (matched loss, tuned, CI'd).** (Fig 2.)
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

**4.3 (B) Architecture-faithful replication of Song's *exact* protocol (Fig 3).** Our class-IL above
follows Song's *regime* but not his exact training; we therefore also reproduce his alternating
Split-FashionMNIST setup faithfully — a 4-layer 32–32 sigmoid Xavier net, two disjoint 5-class tasks
sharing one 5-output head, trained by **alternating at the minibatch level** (swap every 4 minibatches,
batch 32). Two findings reframe the comparison. *(i) Framing.* An **adversarial three-lens review**
(literature, statistics, alternative-explanations) caught that Song's Fig 4e is a *less-interference*
claim with the LR optimized **per method**, not a learning-rate-robustness claim (that is his Fig 3);
we therefore tune the LR per method and report the worse-task accuracy (*min-both*) as the
interference tell alongside Song's mean-both metric. *(ii) Budget as a controlled axis.* Song's exact
84-iteration budget leaves every method at chance in our hands (a faithfulness-vs-trainability
tension), so we sweep the budget from 84 to convergence. Result (n=10, run on the fused kernel,
per-seed-identical to PyTorch): **no PC-vs-BP(MSE) difference exceeds 1σ at any budget.** Directionally
PC learns *slower* early and pulls marginally ahead by convergence (Δmin-both +5.7 to +11.0 pp at the
800–2500-iteration budgets) but the gap stays inside the (large) seed spread; raising seeds 5→10 did
not make it significant. This is the most faithful replication we could build, and it agrees with the
§4.2 parity verdict rather than overturning it.

**4.4 (B) Autonomous search (Phase 4).** A bounded random search over (T, η_x, η_w) as a thin
layer over `train_and_eval` independently rediscovers the η_x↔η_w interaction (best: low η_x,
high η_w, high T) — an internal consistency check.

**4.5 (C) Generative PC.** A discriminative PCN cannot generate (clamping a label and settling
the input yields noise; `feedforward_init` is a trivial zero-energy equilibrium). The
generative PC produces **recognizable per-class digit prototypes**, reconstructs occluded
halves, and separates uniform-noise OOD with **anomaly AUC 1.0** (in-dist energy 5.1 vs 94.5).
Stability note: the 784-dim output makes the hidden-settling gradient ~√784 larger, so the
state LR must scale as ~1/√(output-dim) or settling diverges.

**4.6 Optional depth: incremental PC and an EWC baseline.** Two further variants, each
implemented and validated. *Incremental PC* (iPC; Salvatori et al. 2024) interleaves a weight
update with every settling step rather than updating once at equilibrium; a correctness anchor
shows its state step reproduces a standard settling step exactly (at zero weight LR). Because it
applies T weight updates per minibatch, its effective weight LR is ~T× larger and it diverges at
the standard rate; at a ~T×-smaller rate it is stable and in a fast-training regime *outperforms*
standard PC (74.3% vs 63.1% on a short MNIST run) — i.e. iPC trades LR sensitivity for faster
per-step learning. *EWC* (Kirkpatrick et al. 2017), added as a continual-learning baseline (a
diagonal-Fisher quadratic anchor on the BP arm), behaves as designed in a learning regime: it
forgets less than the matched BP (BWT −41.1% vs −45.7%, final 65.7% vs 62.0%) — a positive control
that the continual-learning harness measures forgetting correctly.

## 5 Limitations (honest)

MNIST/FashionMNIST scale, vanilla SO-PC. Our null does **not** globally refute Song: even the
*exact* alternating replication (§4.3) still differs from his in scale and seed budget, so the
honest claim is that *at MNIST/FashionMNIST scale with a fairly tuned baseline* no PC advantage
exceeds noise — not that none can exist anywhere. The kernel still loses at batch ≥2048 (the
compute/cuBLAS regime; the fused-resident design and a register-blocked GEMM are architecturally
opposed), and while the per-sample path is now general-depth, the large-batch tiled path remains
depth-2. The "first hand-written fused CUDA settling kernel for PC" claim is bounded by a
literature audit of 11 PC libraries *and* the Equilibrium-Propagation family (none ship one;
`docs/15`), but the usual "to our knowledge" caveat applies. Generative output is class
*prototypes* (one-hot input → ≤10 distinct images), not diverse samples; uniform noise is an easy
OOD (near-OOD such as FashionMNIST would be a harder test). iPC is LR-sensitive (needs ~LR/T) and
precision schedules remain reserved. Depth-scaling and the scale gap to Transformers are untouched
(frontier-lab work).

## 6 Conclusion

PC's headline advantages over BP are **fragile to experimental confounds**: matched-loss,
fairly tuned, CI'd comparisons on MNIST show parity, and we give a three-item confound checklist
that explains the literature's inconsistency. PC's inference cost is dominated by kernel-launch
overhead (fixable at small batch with a fused kernel — now general-depth and used as the engine of
our own study; bounded by an architectural GEMM trade-off at large batch). And a *generatively*
trained PC genuinely delivers the "one mechanism, many tasks" property. The contribution is a
reproducible, honestly-scoped baseline plus three transferable insights: the launch-overhead
characterization of PC inference, the PC-vs-BP confound checklist, and an adversarial-verification
protocol that caught us misframing a literature claim *before* it reached the writeup.

## Figures (in `figures/`, PNG + PDF; regenerable via `scripts/make_figures.py` from `results/*.json`)

- **Figure 1** — `fig1_kernel_speedup`: fused CUDA settling-kernel speedup over the PyTorch backend
  vs batch size (T=20, 40; RTX 3080 Ti, MLP [784,256,256,10]). Win at small batch (launch-overhead
  bound), break-even ~1024, cuBLAS regime at batch ≥2048. *(§4.1)*
- **Figure 2** — `fig2_pc_vs_bp`: matched-loss PC vs BP. *Left:* bulk test accuracy — PC ≈ BP(MSE),
  CIs overlap; BP(CE) leads by the +5.9 pp loss effect. *Right:* noise robustness vs σ. *(§4.2)*
- **Figure 3** — `fig3_alternating_budget_sweep`: Song-exact alternating Split-FashionMNIST,
  mean-both (Song's metric) and worse-task min-both (interference tell) accuracy vs training budget
  (n=10, ±1σ). No PC-vs-BP(MSE) gap exceeds 1σ at any budget. *(§4.3)*
- **Figure 4** — `fig4_launch_count`: CUDA kernel launches per settle vs settling steps T —
  PyTorch-SO grows 620→2480, the fused kernel issues exactly one (T-independent). *(§4.1)*

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
Lopez-Paz & Ranzato 2017 — GEM (arXiv:1706.08840); Kirkpatrick et al. 2017 — EWC, "Overcoming
catastrophic forgetting in neural networks" (PNAS 114(13):3521–3526, 10.1073/pnas.1611835114);
Lomonaco et al. 2021 — Avalanche (arXiv:2104.00405); PyTorch Custom C++/CUDA Operators +
`extension-cpp`.

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
