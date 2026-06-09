# 15 - Kernel-Neuheit: Verifikations-Verdikt (M7)

*Stand 2026-06-09. Ergebnis eines 4-Sucher-Workflows (PC-Library-Quell-Audit + Literatur + adjazente Energy/Equilibrium-Methoden + Skeptiker). Klaert, wie weit die Hook-A-Neuheitsbehauptung verteidigbar ist und welche Formulierung das Paper (docs/05) benutzen darf. Korrigiert Ueberzeichnungen in docs/07/09/12/README/CLAUDE.*

## Kernel-Neuheit — Verdikt (Stand 2026-06-09)

### 1. Verdikt in einem Satz
Die Behauptung ist **verteidigbar — aber nur in der eng gefassten, PC-spezifischen Form mit dem "to our knowledge"-Vorbehalt**: Sub-Claim (a) ist durch direkte Repo-Inspektion aller relevanten PC-Libraries stark belegt (kein einziges Repo enthält einen handgeschriebenen CUDA-Settling-Kernel), während Sub-Claim (b) gilt, sofern *nur die Anwendung auf PC* als neu beansprucht wird — die Technik (fused, single-launch, states-resident persistent kernel) selbst ist **nicht** neu und muss explizit auf bekannte Prior Art (cuDNN/Baidu persistent RNN, LLM-Megakernels) zurückgeführt werden.

### 2. Belege für (a) — PC-Libraries ohne eigenen CUDA-Settling-Kernel

| Library | Framework | Eigener CUDA-Settling-Kernel? | Beleg |
|---|---|---|---|
| PCX / pcax (liukidar) | JAX (jit/vmap/value_and_grad + optax) | nachweislich **nein** (87 Files, 0 .cu/.cuh/.cpp) | github.com/liukidar/pcx; arXiv:2407.01163 (Sec. 6: JIT "unable to parallelize the executions of the layers", vmap-Fix nur "in the unpractical case where all the layers have the same dimension") |
| JPC (thebuckleylab) | JAX / Equinox / Diffrax / Optax | nachweislich **nein** (0 .cu, Settling = Diffrax-ODE-Solver) | github.com/thebuckleylab/jpc; arXiv:2412.03676 |
| ngc-learn (NACLab, modern) | JAX (jnp advance_state) | nachweislich **nein** (466 Files, 0 .cu; nur `expKernel.py` = Python-Mathekernel, False Positive) | github.com/NACLab/ngc-learn |
| ngc-learn-legacy | TensorFlow 2 | nachweislich **nein** (einzige C-Files = INFIMNIST-Datengenerator, settling-fremd) | github.com/NACLab/ngc-learn-legacy |
| µPC | JAX (PCX-basiert) | nachweislich **nein** (Skalierung via Parametrisierung) | arXiv:2505.13124 |
| PRECO (bjornvz) | PyTorch (manuelle analytische Tensor-Ops, for-t-Loop) | nachweislich **nein** (12 Files, 0 .cu) | github.com/bjornvz/PRECO (PCN.py L116; structure.py L101-129) |
| Torch2PC (Rosenbaum) | PyTorch (autograd vjp + manuelle Updates) | nachweislich **nein** (3 Files, 0 .cu) | github.com/RobertRosenbaum/Torch2PC |
| pypc (infer-actively) | PyTorch (manuelle matmul-Ops) | nachweislich **nein** (0 .cu; load_inline/nvcc/__global__/triton = 0) | github.com/infer-actively/pypc (models.py L87; layers.py L79-89) |
| pybrid (alec-tschantz) | PyTorch (manuelle Ops) | nachweislich **nein** (0 .cu) | github.com/alec-tschantz/pybrid (hybrid.py L159) |
| ePC / error_based_PC (Goemaere) | PyTorch/Lightning (autograd E.backward()+SGD) | nachweislich **nein** (0 .cu) | github.com/cgoemaere/error_based_PC (pc_e.py L84-100); arXiv:2505.20137 |
| Song "zil" / Prospective-Configuration | PyTorch (manuelle matmul-Ops) | nachweislich **nein** (selbst das 3193-File-Repo: 0 .cu) | github.com/YuhangSong/zil; github.com/YuhangSong/Prospective-Configuration |

**Fazit (a):** Über alle benannten PC-Libraries hinweg null `.cu/.cuh/.cpp`-Files und null Treffer für `cpp_extension | load_inline | __global__ | nvcc | CUDAExtension | triton.jit` (Klon + grep sowie GitHub git/trees-API bestätigt). Settling läuft überall als Python-Loop über Framework-Ops, als JAX-JIT/XLA-Compile oder als Diffrax-ODE-Solve — **nie als ein fusionierter Custom-CUDA-Kernel**. Sub-Claim (a) ist damit **stark belegt (hoch)**.

**Unklar:** Equilibrium Propagation (EP) — Code ist über viele kleine Repos verstreut und wurde nicht erschöpfend enumeriert; gezielte Suche fand nur Theorie-/Analog-Hardware-Arbeiten, keinen GPU-Settling-Kernel (`has_custom_cuda_settling_kernel: unclear`). EP ist algorithmisch der nächste Verwandte von PC, aber kein gefundener Beleg schwächt die Behauptung.

### 3. Stärkste Gegen-Evidenz (begrenzt die Neuheit)
Die gefährlichste Prior Art ist **nicht** PC-spezifisch, sondern die *Technik selbst*:

- **cuDNN Persistent RNN (`CUDNN_RNN_ALGO_PERSIST_STATIC/DYNAMIC`) + Baidu persistent-rnn** — der kanonische handgeschriebene persistente CUDA-Kernel: **ein** Launch, dessen Thread-Blöcke über alle rekurrenten Timesteps resident bleiben, Gewichte/Aktivierungen im SM-Registerfile/Shared-Memory, Synchronisation per Global-Barrier — explizit motiviert durch Per-Step-Launch-Overhead bei kleiner Batch (~15x über cuBLAS-per-step bei Mini-Batch 4). Echtes Custom-CUDA (recurrent_ops.cu). Belege: svail.github.io/persistent_rnns/, github.com/baidu-research/persistent-rnn. **Das ist exakt dasselbe Engineering-Muster** und der stärkste Weakener-by-Analogy.
- **LLM-Megakernels (2025-2026)** — Mirage Persistent Kernel (arXiv:2512.22219), Hazy Research Llama-1B "No Bubbles", FlashFormer (arXiv:2505.22758): ganze Forward-/Decode-Pässe in einem persistenten Kernel, state resident, Small-Batch-Launch-Overhead-Win. Belege: arXiv:2512.22219, hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles, arXiv:2505.22758.
- **DEQ (locuslab/deq, torchdeq) + ePC (arXiv:2505.20137)** — die nächsten *konzeptuellen* Verwandten (Fixpunkt/Energie-Minimierung). DEQ löst rein in PyTorch (Anderson/Broyden, 0 .cu); ePC greift dasselbe Endziel (schnelles PC-Settling auf digitaler Hardware, "orders of magnitude faster") an, aber **algorithmisch** (Error- statt State-Reparametrisierung), nicht per Kernel. Beide **widerlegen nicht**, da kein Custom-CUDA-Settling-Kernel — aber **ePC muss zwingend zitiert-und-abgegrenzt werden**, sonst wirkt es wie eine übersehene Closest-Related-Work.

**Wie stark schränkt das ein?** Es entwertet die Neuheit der *Technik/des Designs* vollständig (das persistente, fusionierte Single-Launch-Muster ist seit 2016 etabliert), lässt aber den Kern "**erste Anwendung auf PC-Settling**" intakt. Auch die *Beobachtung* "Launch-Overhead dominiert bei Small-Batch / many-small-ops" ist bereits generisch publiziert (Framework Tax arXiv:2302.06117; kernel-looping arXiv:2410.23668; PCX nennt selbst "multiple smaller operations") und darf **nicht** als Beitrag beansprucht werden.

### 4. Die genau verteidigbare Formulierung

**Empfohlener Wortlaut:**
> "To our knowledge, this is the first hand-written, fused, single-launch CUDA kernel that runs the entire T-step Predictive Coding settling loop with layer states held resident on-chip across steps, eliminating per-step kernel-launch overhead at small/medium batch. Among open PC libraries — which are JAX/JIT (PCX, JPC, ngc-learn, µPC) or PyTorch-autograd/manual-tensor (PRECO, Torch2PC, pypc, pybrid, ePC, Song's code) — none ship such a kernel. The persistent/fused-single-launch, states-resident technique itself is established prior art, originating in cuDNN's persistent RNN and Baidu's persistent-rnn (2016) and re-applied in LLM megakernels (Mirage, FlashFormer); our contribution is its first transfer to — and measurement on — the PC settling loop (with the layer-parallel / T-sequential decomposition and precision-weighted, φ'-gated bidirectional update)."

**Pflicht-Beiwerk:** "to our knowledge"-Vorbehalt behalten (Negativ-Beweis über unindizierten/Industrie-Code unmöglich); ePC (arXiv:2505.20137) als nächste verwandte Arbeit zitieren-und-abgrenzen (algorithmisch vs. kernel-level, orthogonal/kombinierbar); cuDNN/Baidu persistent RNN explizit als Technik-Lineage nennen (aktuell fehlt das in docs/07 und README — die LLTM-Tutorial-Referenz ist schwächer und weniger on-point).

**Was man NICHT behaupten darf:**
- Kein unqualifiziertes "FIRST" ohne "for PC, kernel-level, to our knowledge".
- **Nicht**, das fusionierte/persistente Single-Launch-/States-resident-*Design selbst* sei neu (widerlegt durch persistent RNN / Megakernels). Jede Formulierung wie "the fused-single-launch, states-resident design is new" (so aktuell in docs/07/README/CLAUDE.md) muss umformuliert werden zu "first application to PC".
- **Nicht** die *Beobachtung* "Launch-Overhead dominiert bei Small-Batch" als Beitrag (generisch publiziert).
- Kein genereller Speedup-Claim: Gewinn nur bei Small/Medium-Batch (3-4x @ B=64, Break-even ~B=1024, cuBLAS gewinnt bei B≥2048 wegen naiver In-Kernel-GEMM); Artefakt ist auf 2 Hidden Layers (model.n==3), float32, festes T, tanh/identity/sigmoid (act 0/1/2) spezialisiert — also "first", nicht "general-purpose".
- Hinweis fürs Paper: der hand-geschriebene Kernel IST `pcn/kernels/settling_kernel.cu` (echte `.cu`, von `settling_cuda.py` via `load(sources=[...])` kompiliert; `settling_cuda.py` = Wrapper+Dispatch). Kein ungebauter Stub und kein Inline-String mehr — die CUDA-Quelle steht dort, wo nvcc sie kompiliert.

### 5. Konfidenz
**Hoch** für Sub-Claim (a) (vier unabhängige Reports, direkte Repo-/git-trees-Inspektion aller benannten PC-Libraries, 0 CUDA-Treffer, korroboriert durch PCX' eigene Limitations-Aussage). **Mittel-bis-hoch** für Sub-Claim (b) in der eng gefassten "first-for-PC"-Form; **niedrig**, falls man Design-/Technik-Neuheit beanspruchen wollte (dann widerlegt).

Konfidenz-steigernde Zusatzprüfungen:
- **EP-Lücke schließen:** gezielte Enumeration der verstreuten Equilibrium-Propagation-Repos (Scellier/Bengio-Nachfolge, Laborieux, neuromorphe Implementierungen) nach `.cu`/`__global__`/`load_inline` — derzeit der einzige als "unclear" markierte Nachbar.
- **2025/2026-Preprint- und Industrie-Sweep** ("predictive coding CUDA kernel", "PC settling fused kernel", "energy-based GPU megakernel") wiederholen, um den "to our knowledge"-Vorbehalt zu härten.
- **Nsight-Launch-Count-Artefakt** (vorher/nachher, docs/07 Benchmark-Plan) beilegen, um Reviewern der "Solver-Sicht" (JPC/ePC) zu belegen, dass Launch-Overhead tatsächlich der adressierte Bottleneck ist.

---

# Anhang: EP-Nachbar-Audit (M7, schliesst die letzte Luecke)

## EP-Lücke — Verdikt (Stand 2026-06-09)

### 1. Verdikt in einem Satz

Die EP-Lücke ist geschlossen: Kein Equilibrium-Propagation-Repository versendet einen handgeschriebenen, persistent-residenten CUDA-Settling-Kernel (.cu/.cuh, `cpp_extension`/`load_inline`, `__global__`) für die Free-/Nudged-Phase-Relaxation — alle settlen via PyTorch-Autograd, JAX/XLA-JIT oder einen Python-Host-Loop —, sodass der PC-Claim "first hand-written fused CUDA settling kernel for PC" durch den nächsten algorithmischen Nachbarn EP nicht widerlegt, sondern gehärtet wird; einzige relevante (aber nicht äquivalente) Prior Art sind die per-Step-elementweisen Triton-Kernels in `autonull/bioplausible`.

### 2. EP-Repos-Tabelle

| Repo | Paper | Framework | Custom-CUDA-Settling-Kernel? | Beleg |
|---|---|---|---|---|
| bscellier/Towards-a-Biologically-Plausible-Backprop | Scellier & Bengio 2017, arXiv:1602.05179 | Theano | nein | https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop |
| Laborieux-Axel/Equilibrium-Propagation | Laborieux et al. 2021, arXiv:2006.03824 / 2101.05536 | PyTorch-autograd | nein (Euler-Iteration von dPhi, reine Torch-Ops) | https://github.com/Laborieux-Axel/Equilibrium-Propagation |
| ernoult/updatesEPgradientsBPTT | Ernoult et al. 2019 (NeurIPS), arXiv:1905.13633 | PyTorch-autograd | nein (Python-`stepper`/`forward`-Loop) | https://github.com/ernoult/updatesEPgradientsBPTT |
| ernoult/continualEP | Ernoult et al. 2020 (Cosyne), arXiv:2005.04168 | PyTorch-autograd | nein | https://github.com/ernoult/continualEP |
| Laborieux-Axel/holomorphic_eqprop | Laborieux & Zenke 2022 (NeurIPS), arXiv:2209.00530 | JAX + Haiku | nein (eager `while`-Loop über `batched_fwd`, kein `lax.scan`/Pallas/Triton) | https://github.com/Laborieux-Axel/holomorphic_eqprop/blob/main/dynamics.py |
| bscellier/agnostic-equilibrium-propagation | Scellier et al. 2022, arXiv:2205.15021 | PyTorch | nein | https://github.com/bscellier/agnostic-equilibrium-propagation |
| smonsays/equilibrium-propagation | Referenz-PyTorch-EP (Scellier-Stil) | PyTorch-autograd | nein (`for i in range(n_relax): u_step` mit `batch_E.backward()`) | https://raw.githubusercontent.com/smonsays/equilibrium-propagation/master/lib/energy.py |
| jlaydevant/Binary-Equilibrium-Propagation | Laydevant et al. 2021 (CVPR Workshop) | PyTorch | nein | https://github.com/jlaydevant/Binary-Equilibrium-Propagation |
| zeligism/eqprop | Community-Reimpl. Scellier & Bengio | PyTorch (autodiff + manuell, beide Torch-Ops) | nein | https://github.com/zeligism/eqprop |
| jgammell/equilibrium_propagation | Community-Reimpl. Scellier & Bengio | PyTorch-autograd | nein (100% Python, `eqp.py`) | https://github.com/jgammell/equilibrium_propagation |
| NeuroCompLab-psu/EqProp-SeqLearning | Sequence Learning using EP (IJCAI 2023), arXiv:2209.09626 | PyTorch-autograd | nein (`--T`/`--K`-Timestep-Loop, Autograd von Phi) | https://github.com/NeuroCompLab-psu/EqProp-SeqLearning |
| QUVA-Lab/spiking-equilibrium-prop | O'Connor et al. 2019 (AISTATS) | PyTorch-autograd (spiking) | nein | https://github.com/QUVA-Lab/spiking-equilibrium-prop |
| Lin, Bal & Sengupta (Scaling SNN-EP to ConvNets) | ICONS 2024, arXiv:2405.02546 | PyTorch-autograd | nein (Standard-PyTorch auf GPU) | https://arxiv.org/abs/2405.02546 |
| StochEP | arXiv:2511.11320 (2025/26) | PyTorch-autograd | nein (Zitat: "implemented in Python using the PyTorch framework … NVIDIA RTX A5000 GPU") | https://arxiv.org/html/2511.11320v1 |
| Scalable EP via Intermediate Error Signals | arXiv:2508.15989 (TMLR 2026) | PyTorch-autograd | nein (Zitat: "conducted in Python using the PyTorch library, on an NVIDIA RTX A5000 GPU") | https://arxiv.org/html/2508.15989v1 |
| ml-jku/hopfield-layers (Nachbar: Modern Hopfield) | Hopfield Networks is All You Need, arXiv:2008.02217 | PyTorch-autograd | nein (Native-Attention-Ops, ~1 Update) | https://github.com/ml-jku/hopfield-layers |
| Lemon-cmd/energy-transformer-graph (Nachbar: Energy Transformer) | Energy Transformer, arXiv:2302.07253 | JAX/XLA | nein | https://github.com/Lemon-cmd/energy-transformer-graph |
| ruvnet/RuVector (Nachbar: thermodynamic learning) | Demo `equilibrium_propagation.rs` | Rust (CPU Langevin) | nein (kein GPU/.cu) | https://github.com/ruvnet/RuVector/tree/main/examples/exo-ai-2025/research/10-thermodynamic-learning |
| **autonull/bioplausible** | Multi-Algo-Framework (EqProp, FA, Hebbian, MEP) | Triton + CuPy (Settling-Loop auf Host) | **nein** (handgeschriebene Triton-Kernels, aber per-Step-elementweise; siehe §3) | https://raw.githubusercontent.com/autonull/bioplausible/HEAD/bioplausible/models/triton_kernel.py |

### 3. Nächste Prior Art (gefunden) und ihre Einschränkung des PC-Claims

Das einzige echte Treffer-Repo, das EP mit handgeschriebenen GPU-Kernels verbindet, ist **`autonull/bioplausible`**. Es liefert handgeschriebene Triton-Kernels für EP-Dynamik — `_eqprop_step_kernel`, `_eqprop_step_kernel_with_bias`, `_eqprop_step_linear_kernel`, `_neural_cube_update_kernel` (in `bioplausible/models/triton_kernel.py`, re-exportiert über `bioplausible/acceleration/triton_kernels.py`).

Warum es den PC-Claim **bindet, aber nicht widerlegt**:
- **Es ist nicht PC, sondern EP** — ein verwandter, aber anderer Algorithmus; "first … for PC" bleibt unberührt.
- **Es ist Triton, kein handgeschriebenes CUDA** (kein `.cu`/`.cuh`/`__global__`/`cpp_extension`/`load_inline`; das Repo hat null CUDA-Source-Dateien).
- **Es ist kein fused, persistent-residenter In-Kernel-Settling-Loop**: Die Kernels berechnen je einen elementweisen Relaxationsschritt `h <- (1-a)h + a*tanh(pre_act)`; das Matmul ist **nicht** mitgefused, und die **Settling-Iteration läuft auf dem Host** in Python (`bioplausible/models/causal_transformer_eqprop.py` Z. 168-182: `for _ in range(steps): … h = TritonEqPropOps.step(h, h_target, alpha=…)`).
- **Falschnamen-Warnung**: `bioplausible/mep/mep/cuda/kernels.py` definiert zwar `fused_settle_step`/`fused_settle_step_inplace` unter einem `cuda`-Pfad, doch die Rümpfe sind reine Torch-Ops (`torch.lerp`/`mul_`/`add_`); der Docstring räumt selbst ein: "PyTorch still launches separate kernels". `bioplausible/acceleration/kernel.py` ist explizit eine "Pure NumPy/CuPy implementation".

Fazit: `autonull/bioplausible` ist die nächstgelegene EP-seitige Prior Art und sollte vorsorglich zitiert werden, erfüllt aber keines der drei Claim-Merkmale (PC / handgeschriebenes CUDA / fused persistent-resident In-Kernel-Settling-Loop).

### 4. Auswirkung auf docs/15

Die Formulierung **"to our knowledge, the first hand-written fused CUDA settling kernel for PC" bleibt verteidigbar** — durch den PC-Audit (kein PC-Lib mit Custom-CUDA-Settling-Kernel) und nun den EP-Nachbar-Audit (kein EP-Repo mit handgeschriebenem CUDA-Settling-Kernel) doppelt abgesichert.

Empfohlener Satz fürs Paper (EP explizit als geprüften Nachbarn nennen):

> "We additionally audited Equilibrium Propagation — the closest algorithmic relative of Predictive Coding — across its canonical (Scellier & Bengio 2017; Laborieux et al. 2021; Ernoult et al. 2019/2020; Laborieux & Zenke 2022) and recent (StochEP, arXiv:2511.11320; Scalable CRNN-EP, arXiv:2508.15989) implementations: all perform the free-/nudged-phase settling via PyTorch-autograd, JAX/XLA-JIT, or a Python host loop. The only EP repository with hand-written GPU kernels (`autonull/bioplausible`) uses per-step *elementwise Triton* kernels with the settling iteration on the host — not a hand-written CUDA, fused, persistent-resident in-kernel settling loop, and EP rather than PC. To our knowledge, no EP implementation ships a hand-written CUDA settling kernel."

Optionaler Zusatzsatz (Scope-Caveat, ehrlichkeitshalber): EP-Settling-Beschleunigung lebt in **Analog-/neuromorpher Hardware** (z. B. IEEE Xplore 9181250 "Analog Circuits to Accelerate the Relaxation Process in EqProp", ~250×; Coupled-Oscillator/Kuramoto, arXiv:2402.08579) — physische Geräte, keine CUDA-Kernels, also keine Prior Art für einen GPU-Kernel-Claim, aber einer Erwähnung wert.

### 5. Konfidenz + Restunsicherheit

**Konfidenz: hoch (~0,9).** Drei unabhängige Audits decken alle kanonischen EP-Linien plus Community-Reimplementierungen, jüngste 2024–2026-Paper und benachbarte Energy-Based-/Modern-Hopfield-/Energy-Transformer-Libraries ab; Beleg über direkte File-Tree-Inspektion (`gh git/trees` rekursiv), Quelltext-Verifikation (`smonsays/energy.py`, `holomorphic_eqprop/dynamics.py`, `bioplausible`) und authentifizierte GitHub-Code-Suche — alle Kernel-Signaturen (`__extension`, `__global__`, `cpp_extension`, `load_inline`, `triton.jit`) für EP leer bzw. nur Falschpositive (EDA-Placement `eqProp`, Lattice-QCD `eqprop`).

Restunsicherheit:
- `jgammell/equilibrium_propagation` wurde in Report 1 als "unclear" markiert (File-Tree nicht einzeln geöffnet), in Reports 2/3 jedoch als 100% Python bestätigt — geringe Restunsicherheit.
- GitHub-Code-Suche ist nicht erschöpfend und war rate-limitiert (HTTP 403 nach ~10 Queries); ein sehr neues oder unverlinktes/privates EP-Repo könnte unentdeckt sein.
- Analog-/Hardware-EP (Memristor, Oszillator-Ising, EqSpike) ist bewusst ausgeklammert — physische Settling-Geräte, keine CUDA-Kernels, daher keine Prior Art für einen GPU-Kernel-Claim.
- Abdeckung gilt für Software-Simulatoren auf konventionellen GPUs; eine etwaige unveröffentlichte In-House-Implementierung kann ein Literatur-/Repo-Audit grundsätzlich nicht ausschließen — daher die Formulierung "to our knowledge".
