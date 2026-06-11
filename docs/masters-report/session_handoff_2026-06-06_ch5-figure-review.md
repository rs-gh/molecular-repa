# Session handoff — Ch5 no-acceleration finding + Fig 5.1, Table 6.4 redesign, external-review triage (2026-06-06)

## TL;DR — where we are

Two commits landed this session, both pushed to `origin/main`:
- `dab1911` — Ch5 no-acceleration finding, Table 6.4 redesign, small polish, Ch3/Ch7 review-thread notes.
- `ad924fa` — Fig 5.1 (Tabasco validation curves), wired into Ch5; cover-sheet word count refreshed 10378 → **12345**.

PDF builds clean: **56 pages, 0 overfull, 0 undefined refs**. The draft is in good shape on Ch4–Ch6; the still-unwritten chapters are **Abstract, Ch3, Ch7** (all scaffold-only).

Everything this session was committed by **explicit path** (only `docs/masters-report/`). The repo has a **pre-existing half-merged state** in `evaluation/proteina/` (15 unmerged files) that is NOT mine and is still pending your resolution — see Gotchas.

---

## What was done this session (all committed + pushed)

### 1. External-review triage (the trigger for everything below)
Shreyas ran the draft through two external AIs and asked what's worth acting on. Full verdict saved to memory `project_external_review_2026-06`. Key outcome:
- **Most concrete line-edits were FALSE POSITIVES** (PDF render artifacts, not source bugs): the "`1 ˚A` → `1~\AA`" fix (source already uses `\,\AA{}`), the "space before colon in Table 4.1" (no space in source), and "recompute Table 6.4 FID math" (the math was correct — different denominators). DO NOT re-act on these if a future review repeats them.
- **Real takeaways** acted on: "degenerate dimensions" → "inactive dimensions"; "a-priori/post-hoc" → "a priori/post hoc"; the Table 6.4 legibility redesign; Ch3/Ch7 framing threads (as comments).

### 2. Ch5 — "REPA does not accelerate generation quality" finding
- Promoted the absence of speed-up to its own bolded finding in §5.3 (`\textbf{There is no acceleration in generation quality.}`). Anchored to **REPA's own image-domain speed-up claim (Yu et al., §2.2)**, NOT a forward-reference to Ch6 — Shreyas's call (self-contained, avoids over-flattening Ch6's nuanced transient/durable story).
- Scoped honestly: no-early-separation covers only axes tracked over training; **FCD is endpoint-only** (not logged over training).
- Added a §5.4 sentence scoping the diagnosis: encoder + loss-form varied, but **alignment layer and projector held at image-domain defaults** → frames "neither headroom nor usable teacher" as the most parsimonious account, not a proof.
- Chapter-opening headline updated to answer both "accelerate OR improve".

### 3. Fig 5.1 — Tabasco validation curves (the "no separation" visual)
- New: `figures/scripts/fig5_1_tabasco_curves.py`, `figures/fig5_1_tabasco_curves.png`, `figures/fig5_1_tabasco_curves.tex`. Wired into Ch5 (`\input` + restored `\ref{fig:tabasco-curves}`).
- 1×3 panels: **validity, connectivity, atom-type distribution** (the quality block of Table 5.2). Three series: baseline + REPA-CheMeleon + REPA-MACE.
- **Data source: the on-disk `validation_curves.csv`** (no WandB needed). Script only re-plots.
- **Cropped to common epoch range 0–14** (MACE stops at 14) → strictly apples-to-apples, no "baseline ran longer" caveat needed.
- Whole-number x-ticks (`MaxNLocator(integer=True)`).
- Caption makes the saturation point: all series "track together and reach near-ceiling on every axis within the first epoch or two, leaving no headroom for alignment to fill."
- **IMPORTANT — these are GENERATION metrics, not representation.** The `val/` prefix and `validation/` folder are misleading: the `molecule_metrics.py` callback SAMPLES 100 molecules/epoch and scores their quality. Confirmed by reading the callback. Caption says "generation quality" explicitly.

### 4. Table 6.4 (table_speedup.tex) — two-denominator legibility fix
- The old design showed `+7%` (accel, vs baseline@400K) next to `−50%` (long-run, vs baseline-BEST) on one row — different denominators, sign flip, read as an arithmetic error (it fooled a reviewer).
- Fix: long-run column is now **absolute-only** (best value + step), coloured green/red against a **plain baseline-best anchor row** (Shreyas chose plain over bold). Column headers name each baseline ("@400K (vs base)" / "(abs. best)"). Caption tightened.
- Generator: `tables/scripts/make_speedup_table.py` (durable — re-runs produce the fixed design).

### 5. Small polish + scaffold notes
- "inactive dimensions" in `table_profiling_diagnostics.tex`; "a priori/post hoc" normalised.
- Ch3 scaffold: `% RT-Ch3-*` comments (keep §3.2 imaging-vs-molecular prominent; pre-empt "why these diagnostics" in §3.6).
- Ch7 scaffold: `% RT-Ch7-*` comments (framework as first-class contribution; reframe question to "when can external reps help"; SCOPE the "predictive framework" claim carefully — §4.4 did NOT pre-register the routing; trim rhetorical density in final pass).

---

## THE NEXT JOBS (priority order)

1. **Write Ch3** (Evaluating molecular generation) — scaffold-only at present, but **load-bearing**: Ch6 leans hard on the T-W/T-D supercolumn taxonomy that Ch3 is meant to introduce. The `% RT-Ch3-*` notes are in the scaffold. Shreyas intends to write Ch3 + Ch7 soon.
2. **Write Ch7** (Conclusions) — scaffold + `% RT-Ch7-*` notes ready. This is where the profiling framework earns first-class billing and the spine pays off.
3. **Write the Abstract** — still placeholder ("Write a summary of the whole thing").
4. **(Optional) Fig 5.1 polish** — currently shows 3 series (baseline + 2 additive variants). If you'd rather the figure label REPA-CheMeleon as the *same*-projector run instead of *fused*, swap `chemeleon_additive_fused` → `chemeleon_additive_same` in SERIES (curves nearly identical; match whatever Table 5.2's numbers came from).

---

## Gotchas (carried + new)

- **PRE-EXISTING MERGE STATE — not mine, still pending.** `git status` shows 15 unmerged `evaluation/proteina/...` files (stages 1/2/3: `plot_convergence_*.py`, several `sweep_results.jsonl`, two `.png`) PLUS 3 already-staged-at-stage-0 edits (`plot_helix_sheet_ratio.py`, `plot_pareto_des_div_nov.py`, `plot_ssjsd_vs_des.py`). This blocks `git reset` ("cannot reset in the middle of a merge") and makes a bare `git commit` dangerous (would sweep in the 3 stage-0 files + try to finalize the conflicts). **Resolve this when you get a chance.** Until then, commit report work by explicit path only.
- **Commit by explicit path.** `git commit -- docs/masters-report/<files>` works correctly despite the dirty index (verified: 0 evaluation files in both commits). `git diff --cached` will *list* evaluation files (unmerged entries always show) — that's not the same as them being committed; check stage numbers with `git ls-files --stage` if unsure.
- **Pre-commit hook reformats + aborts.** The `ruff-format` hook rewraps Python and then fails the commit so you re-stage. Expect a first-attempt failure on any new/edited `.py`; re-stage the formatted file and re-commit. The hook also stash/restores the full working tree, which **defeats per-hunk atomic staging** — atomic splitting has to be by-file, not by-hunk.
- **Validation ≠ evaluation (Tabasco).** `validation_curves.csv` = 100 mols/epoch in-training callback (generation quality). `evaluation_summary.csv` / Table 5.2 = 1000 mols, final checkpoint. Absolute numbers differ; don't try to reconcile Fig 5.1's curves with Table 5.2 row-for-row.
- **Cover-sheet word count is hand-maintained.** It's hardcoded in two places (headline + verbatim block, lines ~152 and ~166). `make wordcount` gives the number; update both after substantive prose changes. Currently 12345.
- **IDE auto-builds the PDF on save of `report-draft.tex`** but NOT on edits to `tables/`/`figures/` includes — rebuild manually with `make pdf` after regenerating a table/figure.

---

## Key paths

- Draft: `docs/masters-report/report-draft.tex` (build: `make pdf` from `docs/masters-report/`)
- Word count: `make wordcount` (texcount, scoped by `%TC:ignore`)
- Fig 5.1: `figures/scripts/fig5_1_tabasco_curves.py` → reads `evaluation/tabasco/generation/results/geom/validation/validation_curves.csv`
- Fig 5.1 CSV regen (needs WandB): `evaluation/tabasco/generation/scripts/geom/compile_wandb_curves.py`
- Table 6.4: `tables/scripts/make_speedup_table.py` → `tables/table_speedup.tex`
- Report figure style: `figures/scripts/style.py`
- Chapter flow doc (Ch6): `docs/masters-report/proteina-chapter-flow.md`
- Memory: `project_external_review_2026-06` (review verdict + false positives)

---

## Open decisions

- Fig 5.1 CheMeleon variant: fused vs same (see Next Jobs #4). Currently fused.
- Whether Ch7 states the "framework was predictive" claim — and if so, scoped to ONLY what §4.4 pre-registered (no small-mol gain; GearNet/MPNN likely help). The routing detail (GearNet→fold, MPNN→local) is post-hoc inference, NOT a prediction — do not retro-fit it. See `% RT-Ch7-3`.
