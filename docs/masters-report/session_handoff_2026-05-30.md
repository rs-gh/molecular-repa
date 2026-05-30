# Session handoff — Proteina chapter prose + assets (2026-05-30)

Resume doc for the Proteina chapter (Ch~\ref{ch:proteina-study}) of the master's
report. Read top-to-bottom; everything needed to pick up cleanly is here. This
supersedes the figures/tables-focused handoff in
[session_handoff_2026-05-28.md](session_handoff_2026-05-28.md) (still useful for
the conventions + data-source map).

## TL;DR — where we are

The Proteina chapter has been **fully rewritten into a new structure with prose
drafted end-to-end**, all result tables/figures are **self-contained `\input`
files with bold message-first captions**, and the draft **compiles to 38 pages,
0 undefined refs**. Remaining work is polish + a few data TODOs, not structure.

## ⚠️ Build gotcha (READ FIRST)

- `report-draft.tex` needs `upquote.sty`, which is **missing on this machine's
  TeX install** (it is in the user's real build env). To compile here, drop a
  stub then remove it:
  ```bash
  cd docs/masters-report
  printf '\\NeedsTeXFormat{LaTeX2e}\n\\ProvidesPackage{upquote}[stub]\n\\endinput\n' > upquote.sty
  pdflatex -interaction=nonstopmode report-draft.tex   # run twice for refs
  rm -f upquote.sty
  ```
  On the real env just `make pdf` (or `pdflatex report-draft.tex` twice). Do NOT
  commit the stub.
- **Caption-only change → recompile only.** Plot change → rerun the figure
  script (`python figures/scripts/figNc_*.py`) THEN recompile.
- Pre-commit hooks run (trailing-whitespace, eof-fixer, ruff-format). They will
  reformat `.py`/`.tex` and abort the commit; re-`git add` and re-commit. Never
  stage `.aux/.log/.out/.toc` (hooks choke on them).

## ☠️ Hard-won lesson: NEVER bulk-regex the .tex

A multiline non-greedy regex over `report-draft.tex` **deleted the whole middle
of the document** once this session (recovered from git HEAD + reconstruction).
For multi-edit passes use a Python script that does **exact `str.replace` with a
`count==1` assertion per change** and writes only if all matched. Always
recompile + check page count after.

## Chapter structure (current, in order)

All sections live in `report-draft.tex`. Tables/figures are `\input{}` one-liners.

| Section | Label | Key assets |
|---|---|---|
| Intro (unbolded para 1; bold headline para 2) | — | `figures/fig01_fid_des_convergence` |
| §Experimental setup | `sec:proteina-setup` | — |
| ··· Datasets | — | `tables/table_setup_datasets` |
| ··· Models | — | `tables/table_setup_model` |
| ··· Hyperparameters | — | `tables/table_setup_hparams` †|
| ··· Metrics | — | `tables/table_setup_metrics` †, `tables/table_setup_cath` |
| §REPA installs structural representations… | `sec:proteina-rep` | `figures/fig02_representation`, `tables/table_rep_quality` |
| §REPA pulls the trunk's geometry… | `sec:proteina-alignment` | `figures/fig03_alignment` |
| §REPA accelerates generation… | `sec:proteina-convergence` | `tables/table_speedup`, `figures/fig04_fid_des_gen_vs_rep` |
| §REPA improves whole-model coverage but trades off… | `sec:proteina-anatomy` | `tables/table_proteina_700k`, `tables/table_concentration`, `tables/table_proteina_1m_reduced` |
| §REPA's gains hold across sampler noise levels | `sec:proteina-sampler` | `tables/table_sampler`, `tables/table_ode` |
| §What we do not claim | `sec:proteina-scope` | — |
| §Robustness across scale | `sec:proteina-scale` | — |
| §Summary | `sec:proteina-summary` | — |

† `table_setup_hparams.tex` + `table_setup_metrics.tex` were created by **another
session** and are now wired into the report. **Do not edit those two**, nor
`table_setup_model.tex` (its caption references `tab:setup-hparams`) — that
session owns them. Everything else is this session's.

## Locked conventions (apply to any new asset)

- **Captions = opinion only.** Pattern: **bold message-first takeaway** →
  concise interpretation → terse `(metadata)` parenthetical at the very end.
  NO "caption as metadata". Metadata (n, seeds, bands, scale, colour key, Δ
  convention) goes in the trailing parenthetical or onto the figure itself.
- **Figures carry their own metadata:** panel titles `(a)–(d)` say what each
  panel is (e.g. "(a) PDB-trained: FID ↓"); axis labels include "(log scale)"
  where applicable; bands are min/max over seeds (stated on-figure or in the
  parenthetical). No matplotlib `suptitle` (looked bad — removed).
- **Tables:** baseline row first, then random control where available, then
  Δ-from-baseline everywhere, coloured green (`\gd`) / red (`\rd`); `\gb`
  (darker green) marks best-per-column in the rep-quality table. Macros defined
  in the preamble of `report-draft.tex`.
- **Results-section prose:** every paragraph opens with a `\textbf{}`
  topic-sentence; keep sentences short; prose is opinion-overlay, the
  tables/figures are the evidence. (Setup/intro para 1 are NOT bolded — only
  claim-bearing paragraphs are.)
- Best variant per dataset: **PDB = L9-MPNN, AFDB = L4-GearNet**.
- Use `\ref{}` for all chapter/section/figure/table refs, never hard numbers.

## Narrative threads (woven, not yet pulled together — that's for Ch7)

- **T1** mechanistic-not-regularisation (random control gains least / near-chance)
- **T2** multi-factorial reps → encoder-routed "family of interventions"
  (seeded one-liner in intro para 2; **paid off as the bold backbone sentence of
  the Models subsection**)
- **T3** regime asymmetry (PDB transient / AFDB durable; tracks the baseline)
- **T4** the T-D designable-diversity trade-off (load-bearing; §anatomy)
- **T5** sampler composability
- **T6** "REPA works until it doesn't" (HASTE, `\cite{HASTE}`, arXiv:2505.16792) —
  cited at the trade-off + acceleration; the capacity-mismatch / straitjacket
  framing. Early-termination is flagged as motivated future work, NOT tested.

## Git / uncommitted state (as of this handoff)

⚠️ **Another session has been committing to `main` in parallel.** Recent log:
```
34f2a1e Plots, tables, and skeleton for proteina chapter
85b93cc docs: tablify Ch6 setup — hyperparameters and metrics   (other session)
e93c942 docs: match figure fonts to report body (Computer Modern) (other session)
abf5c65 docs: wire up Ch6 intro citations and seed encoder-family thread
7de2cf5 docs: rewrite Proteina chapter (Ch6) into new structure
ddd1a53 docs: colour result-table cells green/red, drop bold-best
```
Most of this session's caption/figure/table work is therefore **already
committed** (folded into `34f2a1e` and others). The two † setup tables
(`table_setup_hparams`, `table_setup_metrics`) are committed and **wired into the
report** (`\input` at lines ~1321, ~1327) — already integrated, not pending.

**Only three files are currently dirty in the working tree** (this session's
latest prose/caption tweaks not yet committed):
- `report-draft.tex` — intro/setup/datasets/models prose edits (Tabasco
  shortening, family-of-interventions one-liner + Models backbone sentence,
  designability-bias paragraph, `chapter[conclusion]`→`\ref{ch:conclusions}` fix)
- `tables/table_setup_datasets.tex` — caption rewrite to new style
- `report-draft.pdf` — rebuilt output

To commit: `git add report-draft.tex tables/table_setup_datasets.tex
report-draft.pdf` (+ this handoff) and commit. Never stage build artifacts
(`.aux/.log/.out/.toc`) or `tables/preview/` (gitignored). Because the other
session is live, **`git pull`/rebase-aware: check `git log` before committing**
to avoid clobbering their in-flight work; coordinate on `report-draft.tex` (both
sessions edit it).

## Outstanding TODOs

**Data / analysis (flagged in captions as TODO):**
- `table_concentration`: **single-seed**; multi-seed backfill pending
  (rerun `evaluation/proteina/generation/scripts/paper/exp_beta_stratified_diversity.py`
  over the 3 seeds that exist for most cells). Also: **random-control row** is a
  `[TODO]` placeholder there — random was trained long enough but not yet
  β-stratified-eval'd.
- `table_speedup` / `table_ode`: designability-acceleration is clean on PDB; AFDB
  saturates too early (proxy bias) → not a meaningful number; n128-AFDB needs more
  baseline ckpts. ODE table is single-seed.
- `table_proteina_1m`: PDB-L4-random not run past 700K; AFDB-L9-GN stops at 900K.

**Prose / polish:**
- `fig05_td_tradeoff` is currently the **envelope plot**; the chapter's trade-off
  story is carried by `table_concentration` (β-stratified, the stronger asset).
  Decide: keep fig05 in-text, demote to appendix, or cut. (Alternative candidates
  A/B/C + a colored-table mock live in `scratch/fig5_candidates/`.)
- Word count on the cover page is **stale** (says 8895; pre-rewrite). Re-run
  `make wordcount` on the real env (page range auto-derives from `.aux`; note the
  buggy multi-page `gs txtwrite` on THIS machine under-counts — use per-page sum
  or `pdftotext -f 7 -l 36 report-draft.pdf - | egrep '[A-Za-z]{3}' | wc -w`).
- Ch3 (Evaluation) is still scaffold-only; the `\ref{ch:evaluation}` callbacks
  (FID/designability "defined in", supercolumn suite, CATH probe principle) will
  sharpen to section-refs once that chapter is written + labelled.
- "What we do not claim" section is drafted — re-read against final results to
  ensure every retraction still holds (novelty, β-preservation generality,
  learned-encoder-required-at-all-scales, γ=1 artifact, scRMSD bimodality,
  no-formal-mediation, AFDB designability proxy).

## Next-step menu

A. **Commit the uncommitted batch** (low-risk; see staging note above).
B. **Read-through pass** of the full chapter for flow/tone now it's one structure
   (render Ch6 page range to PNG and review).
C. **Resolve fig05** (keep/appendix/cut) and finalise the trade-off visual.
D. **Backfill data TODOs** (multi-seed β-stratified + random row for the
   concentration table is the highest-value one).
E. **Move on to Ch7 (Conclusions)** where threads T1–T6 converge, or back-fill
   Ch3 so the callbacks resolve to sections.

## Key paths

- Chapter source: `docs/masters-report/report-draft.tex` (Ch6 ≈ lines 1085–1470)
- Tables: `docs/masters-report/tables/*.tex`
- Figure snippets + PNGs + scripts: `docs/masters-report/figures/` (+ `scripts/`)
- Canonical plan / claims: `docs/masters-report/proteina-chapter-flow.md`
- Evidence docs: `docs/research/proteina_narratives.md`,
  `proteina_td_crossover.md`, `proteina_sampler_regime_audit_2026-05-28.md`,
  `cath_dataset_characterisation.md`, `datasets.md`
- Data sources for every metric: see §Data sources in
  [session_handoff_2026-05-28.md](session_handoff_2026-05-28.md)
