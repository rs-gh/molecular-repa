# Session handoff — Proteina chapter §Experimental setup (2026-05-31)

Resume doc for the Proteina chapter (Ch~\ref{ch:proteina-study}) of the master's
report. This session overhauled the **Experimental setup** section end-to-end.
Supersedes [session_handoff_2026-05-30.md](session_handoff_2026-05-30.md) for the
setup section; that doc still holds the broader chapter structure + conventions.

## TL;DR — where we are

The **Experimental setup section is in solid shape end-to-end**. This session
restructured it: moved justification tables to the appendix, merged the model +
hyperparameter tables into one, fully rewrote the Metrics prose and table,
added an Evaluation-protocol subsection, and rewrote the transition into the
results. All committed and **pushed to `origin/main` (tip `5dbd391`)**. Draft
compiles to **42 pages, 0 undefined refs/citations**. Next natural move: the
first results section (§rep) or picking off tracker items.

## ⚠️ Build gotcha (READ FIRST)

`report-draft.tex` needs `upquote.sty`, **missing on this machine**. To compile
here, drop a stub then remove it (do NOT commit the stub):
```bash
cd docs/masters-report
printf '\\NeedsTeXFormat{LaTeX2e}\n\\ProvidesPackage{upquote}[stub]\n\\endinput\n' > upquote.sty
pdflatex -interaction=nonstopmode report-draft.tex   # run twice for refs
rm -f upquote.sty
```
On the real env: `make pdf`. Pre-commit hooks (trailing-whitespace, eof-fixer,
ruff-format) will reformat and abort the commit — re-`git add` and re-commit.
Never stage `.aux/.log/.out/.toc`.

## ☠️ Hard-won lessons from THIS session

1. **Commit promptly — an IDE revert silently wiped uncommitted edits.** I added
   the ProteinWorkshop/Ingraham citations, compiled clean (41pp), then the working
   copy got reverted to ~HEAD (almost certainly an editor "discard changes" on the
   open file) and the citations vanished. Had to re-apply. **Since the user edits
   `report-draft.tex` live, commit after each substantive change.**
2. **A "recommended regime" in the research notes ≠ what the figure actually
   plots.** I wrote that CATH-architecture uses the blinded (xclean) eval per the
   audit doc's *recommendation*; the user pushed back, and
   `figures/scripts/fig2_representation.py:93-96` showed IF/dihedral read from
   `XCLEAN` but **CATH-A reads from `CLEANTRAIN`**. Fixed in `0a54d02`. **Verify
   cleanliness/regime claims against the figure code, not the audit's framing.**
3. Parallel session is (was) committing to `main` — always `git fetch` before
   push; all our pushes fast-forwarded cleanly.

## What this session changed (commits, newest first)

- `5dbd391` setup close: generation-sampling leads with "follows the Proteina
  protocol at a reduced sample budget"; **n=128 protocol parked** (tracked);
  replaced the family-of-interventions closer with a **question-framed transition**
  ("…what it changes — representations, alignment, or outputs… We now turn to the
  results.").
- `0a54d02` **fix**: per-residue probes → blinded (xclean); per-chain CATH →
  probe-clean (cleantrain). Corrected protocol prose, `app:leakage`, and the fig02
  caption.
- `b4d5333` probing protocol: 5,000 train chains; blinded ~325 proteins (~43k res)
  for residue probes; probe-clean ~3,190 chains for CATH. New appendix section
  **`app:leakage`** (model vs probe leakage) — summarised from
  `docs/research/pdb_split_leakage_audit.md`.
- `4fadacb` ssJSD-2D appendix section **`app:ssjsd`** (we introduce it; Proteina
  has no SS metric) + prose tweaks.
- `9e91785` / `e980f74` Metrics prose rewrite: hypothesis-grounded intro, three
  bold family heads, enumerate of the three probes, independence/holistic point,
  secondary/tertiary + CATH primers, fidelity-vs-diversity framing (corrected:
  REPA improves fidelity + whole-set diversity together, trades off
  *designable-subset* diversity).
- `74fe657` cite **ProteinWorkshop** (Jamasb 2024) + **Ingraham2019** for the
  probe suite (inline + bibitems).
- `8b94c24` restructure: datasets table → appendix (`app:datasets`); CATH-level
  table → appendix (`app:cath`); merged model+hparams into one grouped
  Property/Value table; expanded metrics table (per-metric defs, Definition/Content
  split, bold/italic two-level hierarchy, whole-set/designable terms); dropped
  "supercolumn".

## Current §Experimental setup structure (`report-draft.tex`, Ch6 ≈ 1299–1349)

- Intro (1301): integrate REPA into Proteina.
- **Datasets** (1303): essentials inline (origin contrast, ≤256 crop, train/val/
  test sizes, one-line leakage caveat); full table + justification in appendix.
- **Models** (1311): base architecture, encoder/depth ablation (GearNet =
  fold-classifier checkpoint from Proteina; ProteinMPNN), random-GearNet control,
  hyperparameters run-in para → **merged config table** `tab:setup-model`.
- **Metrics** (1322): three families (rep quality / rep alignment / generation
  quality), motivated prose + `tab:setup-metrics`.
- **Evaluation protocol** (1343): Representation probing + Generation sampling.
- Transition closer → results.

Appendix (`ch:appendix`, ≈1857+): `app:datasets`, `app:cath`, `app:leakage`,
`app:ssjsd`, then a `\section{Placeholder section}` stub (remove eventually).

## Locked conventions (still apply)

- Setup-table captions are **descriptive, not opinion** (unlike result-table
  captions). Bold header rows across all setup tables.
- Baseline-first; per-residue/per-chain vs whole-set/designable terminology.
- Best variants: **PDB = L9-MPNN, AFDB = L4-GearNet**.
- `\ref{}` everything; never hard-code numbers.
- Don't bulk-regex the .tex (deleted the doc once in an earlier session); use
  exact `str.replace`-style edits and recompile + page-count check after.

## Outstanding work — see the tracker

**[appendix_and_cleanup_todo.md](appendix_and_cleanup_todo.md)** is the running
list (created this session). Highlights:
- **Citations:** add `\href` arXiv/DOI to `ProteinWorkshop` + `Ingraham2019`
  (author/title/venue only right now); add a P-SEA/Biotite cite in `app:ssjsd`;
  **confirm CKNNA is what REPA + BoltzREPA actually report** (unverified).
- **Appendix content still to write:** encoder/depth sweep results, optimisation
  ablations (λ, batch size, lr), projector-depth ablation. Remove the Placeholder
  section once real.
- **Parked:** n=128 generation/representation protocol → document in
  `sec:proteina-scale`.
- **Build nits:** appendix datasets + CATH tables have ~17pt/~16pt overfull hboxes;
  orphaned `tables/table_setup_hparams.tex` (merged into `table_setup_model.tex`,
  safe to `git rm` — kept pending OK); cover-page word count stale (says 8895;
  main chapters now ~8232).

## Key paths

- Chapter source: `docs/masters-report/report-draft.tex`
- Setup tables: `docs/masters-report/tables/table_setup_{model,metrics,datasets,cath}.tex`
  (hparams merged into model; `table_setup_hparams.tex` now orphaned)
- Figure + script: `docs/masters-report/figures/fig02_representation.{tex,png}`,
  `figures/scripts/fig2_representation.py`
- Tracker: `docs/masters-report/appendix_and_cleanup_todo.md`
- Evidence: `docs/research/pdb_split_leakage_audit.md` (leakage scheme),
  `docs/research/proteina_narratives.md`, `proteina_claims_compilation.md`
- Probe code (for protocol facts): `evaluation/proteina/representation/` (sweep_config.yaml,
  lib/probes/), `evaluation/proteina/generation/` (sweep_config.yaml, scripts/evaluate.py)
