# Master's Thesis — LaTeX source

From-scratch Predictive Coding on MNIST: local learning vs. backpropagation, with a fused CUDA
settling kernel. English thesis; the research content lives in `../docs/` (00–15) and `../docs/05`
(the condensed paper draft), which this thesis expands to full length.

## Build

No `biber` needed (uses `natbib` + BibTeX):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or simply:
latexmk -pdf main.tex
```

Produces `main.pdf`. The figures are pulled from `../figures/` (regenerate with
`uv run python ../scripts/make_figures.py`).

## Structure

```
thesis/
  main.tex                 # document class, preamble, frontmatter, \include chapters, bibliography
  frontmatter/
    titlepage.tex          # title page (FILL the \thesis* macros in main.tex)
    declaration.tex        # declaration of authorship (replace with your faculty's exact wording)
    abstract.tex           # English abstract + German Zusammenfassung (written)
  chapters/
    01_introduction.tex    # 1 Introduction
    02_background.tex       # 2 Background (PC theory, SO formulation, relation to BP)
    03_related_work.tex     # 3 Related Work (PC libs, EP, systems, continual learning)
    04_method.tex           # 4 Method & Implementation (PCN, kernel, protocols, iPC, EWC)
    05_experiments.tex      # 5 Experiments & Results (kernel, PC-vs-BP, Song-exact, generative)
    06_discussion.tex       # 6 Discussion
    07_limitations_future.tex # 7 Limitations & Future Work
    08_conclusion.tex       # 8 Conclusion
    A_appendix.tex          # Appendix (derivations, kernel details, full tables, reproducibility)
  references.bib            # bibliography (populated from docs/11)
```

## Status (skeleton stage)

- **Done:** compilable skeleton, title page + declaration + abstract/Zusammenfassung, full
  chapter/section structure with per-section "planned content" notes, populated `references.bib`,
  four figures wired in.
- **To fill:** the `\thesis*` metadata macros in `main.tex` (name, university, supervisors, dates,
  matriculation number) and your faculty's exact declaration wording; then the chapter prose,
  written section by section from the cited `docs/`.

## Provenance note

Replace the declaration with your examination office's mandated wording; some programmes also
require disclosing tool/AI assistance — check your regulations.
