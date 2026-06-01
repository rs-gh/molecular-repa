# Session handoff — Ch6 §Results, generation subsection (2026-06-01)

Resume doc for the **Proteina results §6.2.3** (the generation/envelope subsection)
of the master's report. This session rewrote §6.2.3 end-to-end — prose, two
tables, and the figure. Separate concern from
[session_handoff_2026-06-01.md](session_handoff_2026-06-01.md) (multi-seed bands
for Fig 6.2, which the user is handling in other sessions). Broader chapter
state: [session_handoff_2026-05-31.md](session_handoff_2026-05-31.md).

## TL;DR — where we are

**§6.2.3 is done end-to-end and all pushed to `origin/main` (tip `5bc164a`).**
Draft compiles to **44 pp, 0 undefined**. The user is happy through §6.2.2;
this session did §6.2.3. **Next natural move: §6.2.4** (the anatomy / coverage-
vs-designable-diversity trade-off, `sec:proteina-anatomy`) — it still has the
old dense, em-dash-heavy draft prose and wants the same treatment.

## ⚠️ Build gotchas (READ FIRST)

- `report-draft.tex` needs `upquote.sty`, missing here. Compile with a stub then
  remove it (do NOT commit the stub):
  ```bash
  cd docs/masters-report
  printf '\\NeedsTeXFormat{LaTeX2e}\n\\ProvidesPackage{upquote}[stub]\n\\endinput\n' > upquote.sty
  pdflatex -interaction=nonstopmode report-draft.tex   # run twice for refs
  rm -f upquote.sty
  ```
- **Pre-commit hooks**: `ruff` auto-fixes the figure `.py` (it stripped an unused
  `numpy` import this session) and **aborts the commit** — re-`git add` and
  re-commit. Never stage `.aux/.log/.out/.toc`.
- **LaTeX headings won't break at an en-dash compound** (`representation--generation`):
  `\allowbreak` is ignored. Fix long titles with the optional-arg + `\\` pattern
  (see the §6.2.3 title) — `\subsection[toc-version]{line one\\ line two}`.

## What this session changed (commits, newest first, all pushed)

- `5bc164a` bold Table 6.3 (rep-quality) headers + add a `Variant` col header
  (the one table missing bold heads).
- `459baad` fix the §6.2.3 subsection-title overflow with a two-line heading.
- `ee524e3` Fig 6.4 caption — lead with the equal-compute claim, note plotted ckpts.
- `2e37d1e` Fig 6.4 rewrite: CATH-A x-axis, drop random, thinned per-run trajectories.
- `9c7668f` **the big one** — rewrite Ch6 generation section around the rep-gen
  envelope (prose merge + Table 6.4 + Table 6.5 + their generator scripts).
- (`b629562`, parallel) regenerate seed-aware n256 rep sweeps.

## §6.2.3 structure now (merged subsection)

We **merged** the old speedup subsection and the envelope subsection into one:
title *"REPA accelerates generation quality by climbing the representation--generation
envelope faster"* (labels `sec:proteina-convergence` **and** `sec:proteina-genrep`).
Bold-topic-sentence spine:

1. *intro* — scope to the **Quality** metrics (designability + FID, `tab:setup-metrics`);
   defer tertiary/secondary to §anatomy.
2. **early head start** — at matched compute (400K, Fig 6.4 anchor) most variants lead;
   up to +28% FID, +149% designability → Table 6.4.
3. **consistent with the representation-bottleneck hypothesis** — Yu et al. coupling;
   correlation confirms it → Table 6.5.
4. **REPA climbs the envelope faster** — Fig 6.4; traverses the *existing* curve faster,
   not a separate shortcut.
5. **the link is encoder-matched** (strongest causal sign) — GearNet→fold-distribution,
   MPNN→designability; rank-orders coincide. Rests on §rep + §anatomy (NOT the table —
   the table no longer shows routing after fJSD was dropped).
6. **headroom** — *"REPA's lasting advantage tracks the headroom the baseline leaves"*:
   AFDB-FID permanent, PDB-des durable-but-narrowing, PDB-FID baseline catches up; AFDB-des
   is a saturated proxy (random scores highest); cross-ref the small-molecule study (Ch5).

## Artifacts (all generated from scripts unless noted)

- **Table 6.4** `tables/table_speedup.tex` ← `tables/scripts/make_speedup_table.py`.
  Two questions split into two coloured %-delta columns: **Acceleration** (vs baseline
  @400K) and **Long run** (own best within a **2.0M-step window** vs baseline best).
  4 cells (AFDB/PDB × FID/des); green/red `\gd`/`\rd`; training extent inlined in the
  variant name; generation = mean over up to 3 sampling seeds.
- **Table 6.5** `tables/table_genrep_corr.tex` ← `tables/scripts/make_genrep_corr.py`.
  rep-quality × generation partial(+raw) Pearson, **FID + Designability only** (fJSD-A
  dropped), no bold. Strong for designability, moderate for FID. *Caption is hand-edited
  by the user* — re-running the script overwrites it, so re-apply the user's caption if
  you regenerate.
- **Fig 6.4** `figures/fig04_fid_des_gen_vs_rep.{png,tex}` ← `figures/scripts/fig4c_gen_vs_rep_combined.py`.
  x = **CATH-A** (cleantrain, best layer); baseline + REPA-MPNN-L9; **thinned per-run
  trajectories** (~400K spacing, `thin()`); 400K matched-compute arrow; (a) FID / (b)
  designability; shared legend, no suptitle. Render needs the plotting env
  (`source .venv/bin/activate`); caption is in the `.tex`, hand-edited.
- **Table 6.3** `tables/table_rep_quality.tex` — **hand-maintained and STALE** (numbers are
  a 2026-05-30 snapshot). Now has bold headers; when finally scripted, keep them bold.

## Key decisions / conventions locked this session

- **Acceleration ≠ long-run advantage** — report them separately (Table 6.4's two columns).
- **400K = matched-compute anchor** everywhere (Table 6.4 acceleration col + Fig 6.4 arrow).
- **2.0M window** for long-run (caps the baseline's longer training without dropping any
  headline number). The step-matching bias runs *against* REPA on the durable cells, so
  those claims are conservative — see the tracker's big re-eval to-do.
- **CATH-A is the rep axis** for the envelope (vs dihedral/IF): tightest band, cleanest
  baseline-vs-REPA separation, smoothest. Per-run fits/curves were rejected (the rep-leads-
  gen lag distorts them; pooled is fine, but trajectories matched the REPA-paper look best).
- **"PDB-trained"/"AFDB-trained"**, never bare "PDB"/"AFDB", when qualifying `n<=256`
  (new memory `feedback_pdb_trained_phrasing`).
- Generation values are **seed-means (up to 3 seeds)**; representation is **single-seed**
  (multi-seed pending — that's the *other* handoff's blocked job; it will also denoise
  Fig 6.4's CATH-A axis).

## Outstanding — see [appendix_and_cleanup_todo.md](appendix_and_cleanup_todo.md)

New/relevant tracker items from this session:
- **Step-matching / training-extent bias** — re-evaluate every model on a common step grid
  (the proper de-bias; current tables note it but use cumulative/within-window bests).
- **Cross-study metric-saturation/headroom theme** — pay off in Conclusions + seed in Ch5;
  §6.2.3 already carries the backward-ref to Ch5 (which Ch5 must then deliver).
- **Encoder-matched rank-order panel** (optional, would strengthen the causal claim).
- **Regenerate `table_rep_quality` from a script** (stale; widen window to 1.4M once the
  L4-random 1.3M/1.4M evals land; keep bold headers; add CATH-C column).

## Key paths

- Chapter: `docs/masters-report/report-draft.tex` (§6.2.3 ≈ lines ~1397–1420; §anatomy follows)
- Table generators: `docs/masters-report/tables/scripts/make_{speedup,genrep_corr}_table*.py`
- Figure: `docs/masters-report/figures/scripts/fig4c_gen_vs_rep_combined.py`
- Gen data: `evaluation/proteina/generation/results/paper/n256_convergence_{pdb,afdb}/sweep_results.clean.jsonl`
- Rep data: `evaluation/proteina/representation/results/paper/n256_{xclean_afdb_pdb,convergence_cleantrain_pdb}/pretrained_sweep_results.csv`
