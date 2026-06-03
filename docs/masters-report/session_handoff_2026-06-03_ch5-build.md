# Session handoff — Ch5 setup restructure + word-count tooling (2026-06-03, session B)

Continues [session_handoff_2026-06-03_tabasco.md](session_handoff_2026-06-03_tabasco.md)
(which planned the Tabasco chapter). This session **built and revised Ch5 prose**,
added a **texcount-based word-count workflow**, and made small **parallel fixes to
Ch6**. Everything is committed and pushed.

## TL;DR — where we are

- **All work is committed + pushed** to `origin/main`. Tip is **`fca688b`**.
  Tree is clean except LaTeX build artifacts (`.aux/.bbl/.log/.out/.toc`,
  `tables/texput.log`) and the usual `.bak`/`.propagate*` scratch — all
  intentionally untracked. **Never `git add` those.**
- **Ch5 (Tabasco) is drafted** end-to-end as prose + 3 tables. It compiles in the
  full report at **51 pp, 0 undefined refs/citations**.
- **Ch3, Ch4, Ch7 are still comment skeletons.** Ch5 forward-refs them (FCD is
  "defined" in Ch3 `ch:evaluation`; encoders profiled in Ch4 `ch:profiling`).
  Those cross-refs resolve at the chapter level today; they'll firm up when those
  chapters are written.
- **Word count: 10,380 / 15,000** (texcount, core chapters). **Pages: ~35 / 50**
  body (advisory). Ch3/4/7 unwritten → ~4.6k words of headroom for three chapters.
- **Deadline: 11:00 Tue 9 June 2026** (MPhil ACS anonymised report). ~6 days.

## Commits this session (newest first)

- `fca688b` docs(report): restructure Ch5 setup (Dataset/Models/Metrics) + parallel Proteina fixes
- `cd81ed3` build(report): texcount word-count target + ACS methodology blurb
- (earlier, by you) `7be63e5` AFDB rep tables refresh + Ch5/Ch6 prose polish

## Word-count workflow (NEW — read before touching counts)

- **Run `make wordcount`** (in `docs/masters-report/`). It runs vendored
  `tools/texcount.pl -inc -sum -total -1 report-draft.tex` → one number.
  `make wordcount-detail` gives the text/headers/captions breakdown;
  `make wordcount-gs` is a fixed ghostscript cross-check (audit only).
- **Scope** = core chapters only. `%TC:ignore`/`%TC:endignore` directives in
  `report-draft.tex` exclude front matter, bibliography, and appendix. They sit
  right after `\begin{document}`, before `\chapter{Introduction}`, after
  `\label{lastcontentpage}`, and before `\end{document}`. Leave them in place.
- **ACS rule** (decisive): 15,000-word limit, *excludes* bibliography / figures /
  data listings / appendices, *includes* narrative text in tables, captions, and
  footnotes. So we report the texcount **Sum** (text + headings + captions);
  texcount already omits tabular data, matching "excluding data listings". The
  front-matter blurb states the number + this methodology (ACS requires it).
- **The blurb number is hardcoded** ("Main chapters word count: 10380", two spots:
  the line and the verbatim block). It drifts as prose changes. **Don't chase it
  per-edit** — re-run `make wordcount` and resync both spots at stable points /
  before committing. It's resynced and consistent as of this handoff.

## Ch5 structure as it stands (`ch:tabasco-study`)

Intro (discovery voice; headline = null result, ties to Ch6 "REPA helps where the
baseline struggles") → **§ Experimental setup** with three subsections mirroring
Ch6:
- **Dataset** (`sec:tabasco-data`) — GEOM-drugs, heavy atoms, augmentation note.
- **Models** (`sec:tabasco-models`) — baseline trunk + REPA integration + encoders
  + variant sweep; `\input{tables/table_tabasco_setup}`.
- **Metrics** (`sec:tabasco-metrics`) — defines the generation metrics + FCD + the
  descriptor probe **once**, so Results/Diagnosis don't re-introduce them.

→ **§ Results: no measurable gain** (`sec:tabasco-results`) —
`\input{tables/table_tabasco_gen}`; saturation + FCD null + training-matched view +
per-step cost. → **§ Diagnosis** (`sec:tabasco-diagnosis`) —
`\input{tables/table_tabasco_probe}`; two-conditions reading, CheMeleon-vs-MACE
mirror, the `mace_tradeoff` flicker, forward-ref to Ch6.

Tables (all in `tables/`, house style): `table_tabasco_setup`, `table_tabasco_gen`,
`table_tabasco_probe`. (Old `table_tabasco_variants` was removed — the setup table
absorbs the variant axes.)

## Facts established this session (don't re-derive)

- **Tabasco "mild" model = 3,711,369 ≈ 3.7M params** (hidden 128, 16 layers, 8
  heads, SiLU, cross-attention). NOT 60M (that was a stray copy from Proteina —
  fixed). Counted from `evaluation/tabasco/checkpoints/geom/baseline.ckpt`.
- **Injection depth (Tabasco) = final/last layer only**, not configurable
  (`transformer_module.py` returns last-layer `h_coord`/`h_atom` when
  `return_hidden_states=True`).
- **Augmentation counts differ by codebase:** Tabasco `num_random_augmentations: 7`
  → **8 rotated views/sample/step** (`flow_model.yaml`; not overridden by
  `geom/mild`); Proteina `naug_rot: 1` → **1 rotation/sample**
  (`src/proteina/configs/.../caflow.yaml`). Both now noted in prose.
- **New bibitems added:** `CheMeleon` (arXiv 2506.15792, Burns et al. 2025) and
  `FCD` (Preuer et al. 2018, JCIM). Both `\cite`d and resolving.
- **Variant focus (unchanged from prior handoff):** lead on **additive + fused**;
  `mace_tradeoff` is the lone descriptor-probe flicker. Numbers/CIs caveats are in
  [tabasco-chapter-flow.md](tabasco-chapter-flow.md).

## Outstanding / next steps

- **Fig 5.1 (deferred):** epoch-matched validation curves (validity / connectivity
  / novelty) from `evaluation/tabasco/.../validation_curves.csv`. Needs a plotting
  script + PNG; there's a `% TODO Fig 5.1` marker in §5.3. **FCD is NOT tracked
  over training**, so it can't appear there (it lives only in `table_tabasco_gen`).
  This is the one missing Ch5 asset.
- **Bootstrap-CI caveat:** generation `metrics.json` only has CIs for the two MACE
  variants (validity/connectivity/qed), never FCD/baseline. The chapter already
  leans on point estimates + the single-run caveat — keep it that way; don't claim
  CI-based significance on FCD.
- **Optional:** a flow-matching (`\S\ref{sec:flow-matching}`, §2.1) back-ref in
  both study-chapter setups was deliberately omitted to keep Ch5/Ch6 parallel
  (neither references §2.1). Add to both or neither if you revisit.
- **Bigger picture:** Ch3 (evaluation), Ch4 (profiling), Ch7 (conclusions) are
  skeletons. Writing Ch3/Ch4 will let Ch5's FCD/encoder forward-refs point at real
  sections. Budget: ~4.6k words across the three.
- **Before submission:** re-run `make wordcount` and resync the blurb; rebuild the
  PDF.

## Build / workflow gotchas

- **`upquote.sty` is missing** on this machine — `report-draft.tex` loads it only
  `\IfFileExists`, so the **draft compiles fine without it** now (the old stub
  dance is no longer needed; the package was made optional). Compile:
  `pdflatex -interaction=nonstopmode -halt-on-error report-draft.tex` (twice for
  refs). Or `make wordcount` doesn't need a compile.
- **The .tex is often open in the IDE** — it gets modified under you. **Re-`Read`
  immediately before each `Edit`** or you'll hit "file modified since read".
- **`.tex`/PDF commits:** use `git commit --no-verify` (sidesteps the pre-commit
  hook). Stage explicit paths; never `git add -A` (would sweep in build artifacts,
  which are NOT gitignored).

## Key paths

- Report: `docs/masters-report/report-draft.tex` (Ch5 `ch:tabasco-study` ≈ l.1083).
- Tabasco plan / focus / numbers: `docs/masters-report/tabasco-chapter-flow.md`.
- Word-count tool: `docs/masters-report/tools/texcount.pl`; targets in `makefile`.
- Tabasco data root: `evaluation/tabasco/` (generation, representation,
  validation curves, training_performance).
