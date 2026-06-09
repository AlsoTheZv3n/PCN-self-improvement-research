# 11 — Quellen (Original-URLs)

Alle Web-Quellen, aus denen die Infos in `01`–`10` stammen, mit Original-Links.

**Legende:** ⭐ = in voller Länge gelesen/durchgearbeitet · 🔬 Primär (Paper/Doku/Repo) ·
📰 Sekundär (Blog/Aggregator, nur zur Orientierung). arXiv-Links als abs-Seite (PDF/HTML
unter derselben ID erreichbar).

---

## A. Predictive Coding — Skalierung, Tiefe, offene Probleme (→ `06`, `09`)

- 🔬⭐ **Goemaere, Oliviers, Bogacz, Demeester 2025** — "Error Optimization: Overcoming
  Exponential Signal Decay in Deep PC Networks" (ePC/EO). arXiv:2505.20137 —
  https://arxiv.org/abs/2505.20137
  - Code: https://github.com/cgoemaere/pc_error_optimization
  - OpenReview-Variante (PDF): https://openreview.net/pdf/cf0489ce82f5f6e7584c7c38f187f56d322ec1f0.pdf
- 🔬 **Innocenti, Achour, Buckley 2025** — "μPC: Scaling Predictive Coding to 100+ Layer
  Networks" (NeurIPS 2025). arXiv:2505.13124 — https://arxiv.org/abs/2505.13124
  - OpenReview: https://openreview.net/forum?id=lSLSzYuyfX
- 🔬 **"On the Infinite Width and Depth Limits of PCNs" 2026** — arXiv:2602.07697 —
  https://arxiv.org/abs/2602.07697
- 🔬 **van Zwol et al. — "Towards Scaling DNNs with Predictive Coding: Theory and Practice"**
  (Dissertation, enthält JPC-Kapitel). arXiv:2510.23323 — https://arxiv.org/abs/2510.23323
- 🔬 **"Faster Convergence in Deep-Predictive-Coding Networks"** (Sledge & Principe).
  arXiv:2101.06848 — https://arxiv.org/abs/2101.06848 ·
  PubMed: https://pubmed.ncbi.nlm.nih.gov/34714752/
- 🔬 **Millidge et al. 2021 — "Brain-Inspired Computational Intelligence via Predictive
  Coding"** (Review). arXiv:2308.07870 — https://arxiv.org/abs/2308.07870
- 🔬 **"Predictive Coding algorithms induce brain-like responses in ANNs"** (PLOS Complex
  Systems 2025) — https://journals.plos.org/complexsystems/article?id=10.1371/journal.pcsy.0000076
- 🔬 **"Predictive Coding Light"** (Nature Communications 2025) —
  https://www.nature.com/articles/s41467-025-64234-z
- 📰 **VERSES — "Benchmarking Predictive Coding Networks – Made Simple"** (Blog) —
  https://www.verses.ai/research-blog/benchmarking-predictive-coding-networks-made-simple
- 📰 **EmergentMind — "Predictive Coding Networks"** (Themenseite) —
  https://www.emergentmind.com/topics/predictive-coding-networks

## B. PC-Libraries / Implementierungen (→ `07`, `09`)

- 🔬 **JPC** (JAX, Innocenti et al.). Repo: https://github.com/thebuckleylab/jpc ·
  Paper arXiv:2412.03676 — https://arxiv.org/abs/2412.03676
- 🔬 **PCX** (JAX, Pinchetti/Salvatori-Umfeld). Repo: https://github.com/liukidar/pcx ·
  Paper "Benchmarking PCNs Made Simple" (HTML): https://arxiv.org/html/2407.01163v1 ·
  Oxford-PDF: https://www.mrcbndu.ox.ac.uk/sites/default/files/9795_Benchmarking_Predictive_C.pdf ·
  📰 Übersicht: https://www.emergentmind.com/topics/pcx-library
- 🔬 **PRECO** (PyTorch, PC-Netze + Graphen, van Zwol et al.). Repo:
  https://github.com/bjornvz/PRECO
- 🔬 **PyHGF** (JAX/Rust, PC/HGF). Repo: https://github.com/ComputationalPsychiatry/pyhgf

## C. CUDA / PyTorch-Extensions (→ `07`, `09`)

- 🔬 **PyTorch — "Custom C++ and CUDA Extensions"** (offiziell) —
  https://docs.pytorch.org/tutorials/advanced/cpp_extension
- 🔬 **PyTorch — "Custom C++ and CUDA Operators"** (offiziell, neuer; ABI-stable ≥2.10) —
  https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- 🔬 **`pytorch/extension-cpp`** (offizielles Referenz-Repo, `mymuladd`) —
  https://github.com/pytorch/extension-cpp
- 🔬 **RFC: "The State of Custom CUDA extensions in PyTorch"** (Issue #152032; Triton/cuTile) —
  https://github.com/pytorch/pytorch/issues/152032
- 📰 **GPU MODE Lecture 1 — Profiling (Christian Mills' Notes)** (`load_inline`, Nsight,
  `torch.cuda.synchronize()`) — https://christianjmills.com/posts/cuda-mode-notes/lecture-001/
- 📰 **Hongpei Li — "Use Customized CUDA kernel in your PyTorch Code"** —
  https://lhongpei.github.io/post/guide_pytorch_custom_cuda_kernel/
- 📰 **APXML — "Building Custom CUDA Extensions"** —
  https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-cuda-extensions
- 📰 **Vrushank Desai — "Integrating a Custom CUDA Kernel & CUDA Graphs in PyTorch"** —
  https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-viii---integrating-a-custom-cuda-kernel-cuda-graphs-in-pytorch
- 📰 **`chrischoy/pytorch-custom-cuda-tutorial`** —
  https://github.com/chrischoy/pytorch-custom-cuda-tutorial

## D. PC vs. BP / Prospective Configuration / Continual Learning (→ `08`, `10`)

- 🔬⭐ **Song, Millidge, Salvatori, Lukasiewicz, Xu, Bogacz 2024** — "Inferring neural
  activity before plasticity as a foundation for learning beyond backpropagation"
  (Prospective Configuration; Nature Neuroscience). doi:10.1038/s41593-023-01514-1
  - Verlag: https://www.nature.com/articles/s41593-023-01514-1
  - PMC (Volltext): https://pmc.ncbi.nlm.nih.gov/articles/PMC7615830/
  - Preprint (bioRxiv PDF): https://www.biorxiv.org/content/10.1101/2022.05.17.492325.full.pdf
- 🔬 **Zahid, Guo, Fountas 2023 — "Predictive Coding as a Neuromorphic Alternative to
  Backpropagation: A Critical Evaluation"** (Neural Computation, MIT Press). arXiv:2304.02658
  - arXiv: https://arxiv.org/abs/2304.02658
  - MIT Press: https://direct.mit.edu/neco/article/35/12/1881/117833/Predictive-Coding-as-a-Neuromorphic-Alternative-to
- 🔬 **Commentary zu Prospective Configuration** (Intelligent Computing) —
  https://spj.science.org/doi/10.34133/icomputing.0244
- 🔬 **"A Theoretical Framework for Inference and Learning in PCNs"** — arXiv:2207.12316 —
  https://arxiv.org/abs/2207.12316
- 🔬 **"Neuroscience-Inspired Memory Replay for Continual Learning: PC vs. BP"** (Dez 2025).
  arXiv:2512.00619 — https://arxiv.org/abs/2512.00619

## E. Autonome Forschungs-Agenten ("Hermes"-Kontext) (→ `04`)

- 🔬 **Sakana AI — "The AI Scientist" (Nature 2026)** — https://sakana.ai/ai-scientist-nature/ ·
  Repo v2: https://github.com/SakanaAI/AI-Scientist-v2
- 🔬 **AlphaEvolve** (DeepMind) — arXiv:2506.13131 — https://arxiv.org/abs/2506.13131
- 🔬 **EvoScientist** — arXiv:2603.08127 — https://arxiv.org/abs/2603.08127
- 🔬 **Deep Ideation** — arXiv:2511.02238 — https://arxiv.org/abs/2511.02238
- 🔬 **Idea2Story** — arXiv:2601.20833 — https://arxiv.org/abs/2601.20833
- 🔬 **"Towards a Medical AI Scientist"** — arXiv:2603.28589 — https://arxiv.org/abs/2603.28589
- 📰 **GEN — "Can AI Agents Automate Scientific Discovery?"** (Kosmos) —
  https://www.genengnews.com/topics/artificial-intelligence/can-ai-agents-automate-scientific-discovery/
- 📰 **TechXplore — "AI assistants can accelerate scientific discoveries"** (Co-Scientist, Robin) —
  https://techxplore.com/news/2026-05-ai-scientific-discoveries.html
- 📰 **C&EN — "AI companies introduce new agent-based tools for scientific discovery"** —
  https://cen.acs.org/articles/104/web/2026/05/ai-companies-introduce-agent-based-research-tools.html

## F. Transformer-Alternativen (Kontext zur "radikal andere Architektur"-Frage) (→ README)

- 📰 **Chojecki — "Going Beyond LLMs & Transformers"** (RWKV, RetNet, xLSTM) —
  https://pchojecki.medium.com/going-beyond-llms-transformers-39f3291ba9d8

---

## In den Quellen zitierte Schlüssel-Arbeiten (nicht separat abgerufen)

Fundamente, die in den obigen Papers (v.a. der EO-Referenzliste) zitiert werden — für die
Bibliografie deines Papers:

- **Rao & Ballard 1999** — Predictive coding in the visual cortex (Nature Neuroscience). Ursprung.
- **Friston & Kiebel 2009** — "Predictive coding under the free-energy principle" (Phil. Trans. R. Soc. B).
- **Bogacz 2017** — "A tutorial on the free-energy framework" (J. Math. Psychology).
- **Whittington & Bogacz 2017** — PC approximiert BP mit lokaler Hebb-Plastizität (Neural Computation 29:5).
- **Song et al. 2020** — "Can the Brain Do Backpropagation?" (NeurIPS 33). Exaktes BP via PC.
- **Millidge, Tschantz, Buckley 2022a** — "Predictive coding approximates backprop along
  arbitrary computation graphs" (Neural Computation 34:6; arXiv:2006.04182).
- **Millidge et al. 2022b** — "Predictive Coding: Towards a Future of DL beyond
  Backpropagation?" (IJCAI; arXiv:2202.09467).
- **Pinchetti et al. 2025** — "Benchmarking Predictive Coding Networks – Made Simple"
  (ICLR; OpenReview-ID sahQq2sH5x). Quelle des Depth-Scaling-Befunds.
- **Qi, Lukasiewicz, Salvatori 2025** — "Training Deep Predictive Coding Networks"
  (OpenReview-ID s3E08R4AMK). Gradienten-Fix.
- **Salvatori et al. 2024** — "Incremental Predictive Coding (iPC): A parallel and fully
  automatic learning algorithm" (ICLR). Die iPC-Variante.
- **Alonso, Krichmar, Neftci 2024** — "Understanding and Improving Optimization in PCNs"
  (AAAI). G-IL / IL-prox, Online-Robustheit.

---

## Provenienz-Hinweis

⭐-Quellen (EO-Paper, Prospective-Configuration-Paper) wurden vollständig gelesen und bilden
das Rückgrat von `09`/`10`. Die übrigen Einträge stammen aus Such-Ergebnissen (Titel +
Snippet); vor Übernahme in dein Paper jeweils im Original verifizieren — besonders die
📰-Sekundärquellen, die nur zur Orientierung dienen.

> **Update 2026-06-09:** Die folgende Sektion ergänzt diesen Hinweis durch eine
> systematische Web-Verifikation aller Primärquellen/Libraries. Wo sie Status/Autoren/
> Jahr/Titel korrigiert, hat sie Vorrang vor den obigen Roh-Einträgen.

---

## Verifizierter Quellen-Status (Stand 2026-06-09)

Diese Tabelle dokumentiert die Web-Verifikation aller in `01`–`10` zitierten Primärquellen
und Libraries. Sie ergänzt (und korrigiert teilweise) die obigen Roh-Links. Status-Legende:
✅ verifiziert · ⚠️ unsicher (Metadaten bestätigt, aber nicht im Volltext gegengeprüft) ·
❌ nicht gefunden · ✏️ korrigiert (existiert, aber Zitat/Autoren/Venue/Titel waren falsch).

| Zitat | Kennung (arXiv/DOI/Repo) | Status | Zentrale belegte Aussage |
|---|---|---|---|
| Rao & Ballard 1999 — Predictive coding in the visual cortex (Nature Neuroscience 2(1):79–87) | DOI 10.1038/4580 · PMID 10195184 | ✅ | Ursprung der Prädiktionsfehler-Definition `eps = Ist-Aktivität − Top-down-Prädiktion`; Feedback trägt Prädiktionen, Feedforward die Residual-Fehler. |
| Bogacz 2017 — A tutorial on the free-energy framework (J. Math. Psychol. 76:198–211) | DOI 10.1016/j.jmp.2015.11.003 · PMCID PMC5341759 · PMID 28298703 | ✅ (Metadaten) / ⚠️ (Gleichungs-Glyphen) | Präzisionsgewichtete Freie-Energie (Eq. 7); Inferenz-Dynamik mit `g'(φ)`-Gating der Bottom-up-Fehler (Eq. 8/9); Hebb-Gewichtsregel `eps·h(φ)` (Eq. 25/29). Volltext nur via PMC-HTML (PDF nicht extrahierbar). |
| Whittington & Bogacz 2017 — Approximation of Backprop in a PCN with local Hebbian plasticity (Neural Computation 29(5):1229–1262) | DOI 10.1162/NECO_a_00949 · PMID 28333583 | ✅ | Supervised-PC-Template: Value-/Error-Knoten, Transpose-Gewichts-Feedback gegated durch Aktivierungs-Ableitung, lokale Hebb-Regel — approximiert Backprop ohne nichtlokale Plastizität. |
| Millidge, **Seth**, Buckley 2021 — Predictive Coding: a Theoretical and Experimental Review | arXiv:2107.12979 | ✏️ | Mehrschicht-Freie-Energie als präzisionsgewichtete SSE (Eq. 11); Prädiktionsfehler `eps_l = μ_l − f_l(...)` (Eq. 12); Hebb-Gewichtsregel (Eq. 13). **Autoren-Korrektur: Seth, NICHT Tschantz.** |
| Song, Millidge, Salvatori, Lukasiewicz, Xu, Bogacz **2024** — Inferring neural activity before plasticity (Prospective Configuration; Nature Neuroscience **27(2):348–358**) | DOI 10.1038/s41593-023-01514-1 · PMID 38172438 · PMC7615830 | ✏️ | „Settle-to-Equilibrium, dann Plastizität". PC-Vorteil explizit behauptet für: Online-Lernen, Continual Learning, wechselnde Umgebungen, wenig Trainingsdaten, RL. **Jahr-Korrektur: 2024 (nicht 2023).** |
| Zahid, Guo, Fountas 2023 — PC as a Neuromorphic Alternative to Backprop: A Critical Evaluation (**Neural Computation 35(12):1881–1909**) | arXiv:2304.02658 · DOI 10.1162/neco_a_01620 · PMID 37844326 | ✏️ | Beweis: BP-approximierende PC-Varianten (FPA-PC, Z-IL) sind nachweisbar nicht schneller als BP und verlieren die Variational-Bayes-Begründung. **Venue-Korrektur: peer-reviewed, KEIN bloßer Preprint.** |
| Goemaere, Oliviers, Bogacz, Demeester 2025 — Error Optimization / **ePC: Overcoming Exponential Signal Decay in Deep PC Networks** (ICML 2026) | arXiv:2505.20137 · Repo github.com/cgoemaere/error_based_PC | ✏️ | Reparametrisierung über Prädiktionsfehler statt States (`s = ŝ + eps`) eliminiert exponentiellen Signal-Decay; konvergiert um Größenordnungen schneller, erreicht BP-Niveau auch in tiefen Netzen. **Titel/Repo-Korrektur (siehe Warnungen).** |
| Innocenti, Achour, Buckley 2025 — μPC: Scaling Predictive Coding to 100+ Layer Networks | arXiv:2505.13124 | ✅ | Depth-μP-Parametrisierung trainiert bis zu 128-Layer-Residual-PCNs; Zero-Shot-Transfer von Gewichts- UND Aktivitäts-Lernraten über Breite/Tiefe. |
| Salvatori et al. 2024 — A Stable, Fast, and Fully Automatic Learning Algorithm for PCNs (iPC; ICLR 2024) | arXiv:2212.00720 | ✅ | iPC = inkrementelles PC: Gewichts- und State-Updates simultan in jedem Settling-Schritt; Konvergenzgarantien, vollautomatisch. (v1-Titel war „Incremental Predictive Coding".) |
| Innocenti, Achour, **Bogacz** 2026 — On the Infinite Width and Depth Limits of PCNs | arXiv:2602.07697 | ✅ (existiert real) / ⚠️ (zukunftsdatiert, vor Zitat im Volltext lesen) | Für lineare Residual-Netze: PC-Energie mit äquilibrierten Aktivitäten konvergiert gegen den quadratischen BP-Loss im Regime **Breite ≫ Tiefe** und liefert dieselben Gradienten wie BP. KEINE unbedingte „PC = BP"-Aussage. |
| Qi, Forasassi, Lukasiewicz, Salvatori 2025 — Towards the Training of Deeper PCNs | arXiv:2506.23800 | ✅ | „Precision Schedule" (isotrop/spiking/decaying) als zitierbares Konzept; Skalar-pro-Layer-Präzision; Heuristik **T ≥ L** (nicht „L < T < 2L"). |
| Millidge et al. 2024 — JPC: Flexible Inference for PCNs in JAX | arXiv:2412.03676 · Repo github.com/thebuckleylab/jpc | ✅ | JAX-Library; Settling via ODE-Solver (Diffrax); Heun (2. Ordnung) schneller als Euler; Kern < 1000 LOC. Energie = reine SSE (Präzision implizit = Identität). |
| Pinchetti/Salvatori et al. 2024/2025 — Benchmarking PCNs — Made Simple (PCX; ICLR 2025) | arXiv:2407.01163 · Repo github.com/liukidar/pcx | ✅ | A100-Timing: PC ≈ 5.33 s/Epoche vs. BP ≈ 1.61 s (CIFAR-100, VGG-5). Inferenz ist inhärent sequenziell, nicht voll parallelisierbar (vmap nur bei gleichdimensionalen Layern). **Σ explizit = Identität fixiert.** |
| van Zwol, Jefferson, van den Broek 2026 — PCNs and Inference Learning: Tutorial and Survey (ACM Comput. Surv.) | DOI 10.1145/3797870 · arXiv:2407.04117 · Repo github.com/bjornvz/PRECO | ✅ (DOI/Preprint) / ⚠️ (Vol./Issue medium-confidence) | PRECO = PyTorch-Implementierung von PCNs UND Graph-PCNs (PCGs) mit Tutorial-Notebooks. ACM-Seite paywalled, via doi.org-Resolver + arXiv bestätigt. |
| Ororbia / NAC Lab — ngc-learn (NeuroAI in Python) | Repo github.com/NACLab/ngc-learn · DOI 10.1038/s41467-022-29632-7 | ✅ (Backend-Korrektur) | Reale PC/NeuroAI-Library; modernes ngc-learn ist **JAX-basiert** (`pip install ngclearn`); ngc-learn-legacy (TF-Ära) ist separat. |
| Rosenbaum — Torch2PC | Repo github.com/RobertRosenbaum/Torch2PC · PLOS ONE 2022 | ✅ | PyTorch; `PCInfer` mit Varianten `Strict`/`FixedPred`/`Exact` (Exact reproduziert BP-Gradienten) — als numerischer Gradienten-Crosscheck nutzbar. |
| Tschantz & Millidge — pypc | Repo github.com/infer-actively/pypc | ✅ | Reale PC-Library, **PyTorch** (torch+torchvision+numpy); `scripts.generative` / `scripts.supervised`. Forschungs-Skripte, kein pip-Paket. |
| Tschantz — pybrid (Hybrid PC) | Repo github.com/alec-tschantz/pybrid | ✅ (Repo) / ⚠️ (Backend inferiert) | Hybrides (amortisiert + iterativ) PC; `scripts.hybrid`, MIT. README nennt Framework nicht explizit (PyTorch inferiert). |
| Lopez-Paz & Ranzato 2017 — Gradient Episodic Memory (NeurIPS 2017) | arXiv:1706.08840 | ✅ | Kanonische CL-Metriken aus R-Matrix: ACC, BWT (Forgetting = −BWT), FWT — Formeln direkt aus dem NeurIPS-PDF verifiziert. |
| Lomonaco et al. 2021 — Avalanche (CVPR-W / ContinualAI) | arXiv:2104.00405 · PyPI `avalanche-lib` 0.6.0 | ✅ | `SplitMNIST(n_experiences=…)`, `PermutedMNIST(...)`; Metriken `accuracy_/forgetting_/bwt_/forward_transfer_metrics`; PCN via Subklasse von `SupervisedTemplate`. Installation: `uv add avalanche-lib`. |
| PyTorch — Custom C++ and CUDA Operators (Tutorial v2.12) | docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html | ✅ | Op-Registrierung via `TORCH_LIBRARY` / `torch.library` (NICHT pybind11), damit Autograd/`torch.compile`/FakeTensor komponieren; ABI-stabile Variante ≥ 2.10. |
| PyTorch — torch.utils.cpp_extension / torch.library / CUDA semantics (Docs 2.12) | docs.pytorch.org/docs/2.12/ | ✅ | `load_inline`-Signatur (Code-Strings), `load` (Datei-Pfade), `opcheck`; Benchmark-Disziplin: Warmup + `torch.cuda.synchronize()` bzw. CUDA-Events (GPU-Ops sind asynchron). |
| pytorch/extension-cpp (Referenz-Repo, `mymuladd`) | github.com/pytorch/extension-cpp (Branch `master`) | ✅ | Verifizierte verschachtelte Pfade `extension_cpp/extension_cpp/csrc/…`; Standard-CUDA-Launch (256 Threads/Block, `at::cuda::getCurrentCUDAStream()`); `setup.py` mit `CUDAExtension`/`BuildExtension`. |

---

### ⚠️ Korrekturen & Warnungen

Diese Punkte MÜSSEN vor Übernahme in das Paper berücksichtigt werden. Sie betreffen
falsche Autoren/Jahre/Titel, Backend-Annahmen und insbesondere die Theorie-Anker-Frage.

- **arXiv:2602.07697 — existiert real und ist KEINE Halluzination.** Trotz Zukunftsdatum
  (eingereicht 7. Feb 2026, v2 22. Mai 2026, relativ zum Januar-2026-Wissensstand) löst die
  ID auf ein echtes Paper auf: **Innocenti, Achour & Bogacz — „On the Infinite Width and
  Depth Limits of Predictive Coding Networks"** (verifiziert über arXiv-Abstract-Seite,
  export.arxiv.org-API und Web-Suche, alle konsistent). Die frühere Projekt-Annotation
  „NOT verified / future-dated" ist damit hinfällig. **ABER:** Es ist KEIN generischer
  „PC ≈ BP"-Anker. Die PC→BP-Konvergenz gilt nur (a) für **lineare Residual-Netze**,
  (b) im Regime **Breite ≫ Tiefe** und (c) bei **tatsächlich erreichtem Aktivitäts-
  Äquilibrium**. Im Paper exakt so eingeschränkt zitieren — nicht als unbedingte
  Äquivalenz. Da das Paper nach dem Wissensstand datiert, muss es vor dem Zitieren im
  Volltext gelesen werden, niemals aus dem Gedächtnis paraphrasiert.
- **Für einen FUNDAMENTALEN „PC approximiert BP"-Anker** ist 2602.07697 nicht das richtige
  Paper. Die kanonischen Anker sind stattdessen:
  - **Whittington & Bogacz 2017** (Neural Computation 29(5):1229–1262, DOI 10.1162/NECO_a_00949), und
  - **Millidge, Tschantz & Buckley — „Predictive Coding Approximates Backprop along
    Arbitrary Computation Graphs"** (Neural Computation 34(6); **arXiv:2006.04182**).
  - Vorsicht: **arXiv:2106.13082** ist „On the relationship between predictive coding and
    backpropagation" von **Robert Rosenbaum (Einzelautor)**, ein verwandter Review — NICHT
    Millidge zuschreiben.
- **arXiv:2505.20137 — Titel-/Repo-Korrektur:** Die **echten** Titel sind „Error
  Optimization: Overcoming Exponential Signal Decay in Deep Predictive Coding Networks" (v1)
  bzw. „ePC: Overcoming Exponential Signal Decay in Deep Predictive Coding Networks"
  (aktuell). Der Titel „ePC: Fast and Deep Predictive Coding in Digital Simulation" ist
  FALSCH (Fetch-Konfabulation) — NICHT verwenden. Die Bezeichnung „EO/energy optimization"
  ist ein Misnomer: es geht um **error-based PC (ePC)**, Optimierung über *Fehler*.
  **Repo-Korrektur:** korrektes Code-Repo ist **github.com/cgoemaere/error_based_PC**.
  Akzeptiert für **ICML 2026 (Main Track)**.
- **Millidge et al. 2021 (arXiv:2107.12979) — Autoren-Korrektur:** korrekt sind **Millidge,
  Seth, Buckley** (NICHT „Millidge, Tschantz, Buckley").
- **Song et al. — Jahr-Korrektur:** Article-of-Record ist **Nature Neuroscience
  27(2):348–358, 2024** (Epub 3. Jan 2024, PMID 38172438). Im Paper als **2024** zitieren.
- **Zahid, Guo & Fountas — Venue-Korrektur:** KEIN bloßer arXiv-Preprint. Peer-reviewed in
  **Neural Computation 35(12):1881–1909 (2023), DOI 10.1162/neco_a_01620, PMID 37844326**.
- **Framing-Falle (keine Halluzination, aber Überzeichnung vermeiden):** Song et al. und
  Zahid et al. widersprechen sich NICHT direkt zur selben Größe. Song et al. behaupten den
  PC-Vorteil (voll relaxiert) auf der Achse **Sample-/Episoden-Effizienz**; Zahid et al.
  beweisen, dass die **BP-äquivalenten** Varianten (FPA-PC, Z-IL) auf der Achse
  **Wall-Clock/Compute** langsamer sind. Eine faire Benchmark-Harness muss beide Achsen
  trennen.
- **iPC (arXiv:2212.00720) — Titel-Falle:** aktueller arXiv-Titel „A Stable, Fast, and Fully
  Automatic Learning Algorithm for PCNs"; „Incremental Predictive Coding" war der v1-Titel /
  Algorithmus-Name. Per arXiv-ID + ICLR 2024 zitieren.
- **ngc-learn — Backend-Korrektur:** modernes ngc-learn ist **JAX**-basiert; die TF-Ära ist
  separates `ngc-learn-legacy`.
- **Settling-Schritte T vs. Tiefe L — Faustregel-Korrektur:** Die Annahme **„L < T < 2L"
  ist durch die Primärquellen NICHT gedeckt.** Qi et al. (2506.23800) geben **T ≥ L** als
  Untergrenze; ePC (2505.20137) zeigt, dass naives sPC wegen Signal-Decay weit mehr als 2L
  Schritte braucht (≈ 100 Schritte für 20 Layer, ~5× Tiefe). Obergrenze decay-abhängig.
- **Präzision Π — Doku-Behauptung herabstufen:** Die kanonische Literatur (Millidge Eq. 11,
  Bogacz Eq. 7) behandelt Π/Σ als Kernbestandteil; die modernen Benchmark-Libraries fixieren
  Σ jedoch **explizit auf die Identität** (PCX: „Σ_l fixed to identity"; JPC: reine SSE).
  Das Projekt hartkodiert Π = I — korrekt als isotroper Spezialfall, aber jede Doku-Aussage
  „Π ist implementiert/Kern" ist falsch. Entweder skalar-pro-Layer-`precision_schedule`
  (spiking/decaying, Qi et al.) implementieren ODER Doku auf „isotrope Präzision (Π = I)"
  herabstufen. Nicht-Identitäts-Präzision lohnt v.a. bei tiefen Netzen (> 7 Layer).
- **Paywall-/Fetch-Hinweise (Provenienz):** Nature (Rao & Ballard, Song et al.), ACM
  (van Zwol et al.), MIT Press und bioRxiv waren für WebFetch paywalled/403; die Angaben
  wurden über PubMed/PMC, doi.org-Resolver, arXiv und NeurIPS-PDF gegenbestätigt. Exakte
  Gleichungs-Glyphen (Millidge Eq. 11–13, ePC, Bogacz-PDF) vor wörtlichem Zitat gegen das
  offizielle PDF prüfen — die **strukturelle Form** jeder Gleichung ist verifiziert.

---

### ⭐ Vollständig verifizierte Primärquellen

Diese Quellen sind in Identität, Autorenschaft und zentraler Aussage abgesichert und können
ohne Vorbehalt als Rückgrat des Papers genutzt werden (Korrekturen oben jeweils beachten):

- **Rao & Ballard 1999** — DOI 10.1038/4580 / PMID 10195184 (PubMed-Primärrecord).
- **Whittington & Bogacz 2017** — DOI 10.1162/NECO_a_00949 / PMID 28333583 (Supervised-PC-Template, lokale Hebb-Regel).
- **Song et al. 2024** — DOI 10.1038/s41593-023-01514-1 / PMID 38172438 / PMC7615830 (Prospective Configuration; Volltext via PMC gelesen). **Jahr = 2024.**
- **Zahid, Guo & Fountas 2023** — arXiv:2304.02658 / DOI 10.1162/neco_a_01620 / PMID 37844326 (peer-reviewed Kritik). **Venue = Neural Computation.**
- **Lopez-Paz & Ranzato 2017** — arXiv:1706.08840 (NeurIPS-PDF; ACC/BWT/FWT-Formeln glyphengenau verifiziert).
- **μPC** — arXiv:2505.13124 (Innocenti, Achour, Buckley; 128-Layer-PCNs, Depth-μP).
- **iPC** — arXiv:2212.00720 (Salvatori et al., ICLR 2024; simultane State+Weight-Updates).
- **JPC** — arXiv:2412.03676 + github.com/thebuckleylab/jpc (Diffrax-ODE-Settling, < 1000 LOC).
- **PCX** — arXiv:2407.01163 + github.com/liukidar/pcx (A100-Timing PC vs. BP; Σ = I fixiert; vmap-Limit).
- **Torch2PC** — github.com/RobertRosenbaum/Torch2PC (PyTorch; `Exact`-Variante = BP-Gradienten, Gradienten-Crosscheck).
- **Avalanche** — arXiv:2104.00405 + PyPI `avalanche-lib` 0.6.0 (SplitMNIST/PermutedMNIST + ACC/BWT/FWT-Metriken; via `uv add`).
- **PyTorch Custom-Ops-Doku + `extension-cpp`** — docs.pytorch.org (2.12) + github.com/pytorch/extension-cpp (`TORCH_LIBRARY`-Registrierung, `load_inline`, `opcheck`, Sync-Benchmark-Disziplin).
- **Innocenti, Achour & Bogacz 2026** — arXiv:2602.07697 (ID real und aufgelöst; Aussage nur im Regime *linear-residual, Breite ≫ Tiefe, äquilibriert* zitieren — vor Gebrauch im Volltext lesen).
