# Session handoff — Tabasco chapter (Ch5) sketch + Ch6 Discussion/§6.2.6 rework (2026-06-03)

Continues the report work from
[session_handoff_2026-06-02_ch6-tables.md](session_handoff_2026-06-02_ch6-tables.md).
This session (a) reworked the **Ch6 ending** — §6.2.6 "What we do not claim" and
a new top-level **Discussion** section — and (b) produced the **plan/sketch for
the Tabasco chapter (Ch5)**, which is what the next session should pick up.

## TL;DR — where we are

- **The Tabasco plan lives in a new flow doc:**
  [tabasco-chapter-flow.md](tabasco-chapter-flow.md) (untracked). That is the
  primary thing to pick up. It is the *sketch*, not prose.
- **Ch6 ending is done** (§6.2.6 list + new Discussion section). Some of it landed
  in the parallel session's commit `dd676bf`; the latest refinements (~27 lines)
  are **uncommitted in the working tree**. Nothing was committed by us
  deliberately — user asked us not to commit while a concurrent session runs.
- **A concurrent session is live on `main`** (it owns "everything up till §6.2.6"
  + appendix + eval). Tip is `4f4eedf`. Pull before pushing; `report-draft.tex`
  was restructured several times mid-session.
- **Nothing for Tabasco is written into `report-draft.tex` yet** — Ch5 is still
  the comment skeleton. The flow doc is the bridge to writing it.

## ⚠️ Build + workflow gotchas (READ FIRST)

- `report-draft.tex` needs `upquote.sty` (missing). Compile with a stub, then
  remove it (do **not** commit the stub):
  ```bash
  cd docs/masters-report
  printf '\\NeedsTeXFormat{LaTeX2e}\n\\ProvidesPackage{upquote}[stub]\n\\endinput\n' > upquote.sty
  pdflatex -interaction=nonstopmode report-draft.tex   # twice for refs
  pdflatex -interaction=nonstopmode report-draft.tex
  rm -f upquote.sty
  ```
  Draft currently compiles to **47 pp, 0 undefined refs, 0 errors**.
- **Concurrent session collision.** `report-draft.tex` is being edited *and
  committed* by another session. Edits to `report-draft.tex` work but: re-`Read`
  immediately before each `Edit` (the file moves under you — got a "file modified
  since read" mid-session), and our staged changes got swept into the other
  session's commit once. **Ch5 (Tabasco) is upstream of §6.2.6 = squarely the
  other session's territory** → strongly prefer writing Ch5 prose to a scratch
  file, or coordinate before editing it directly. Figures/tables are new files →
  collision-free, safe to build anytime.
- **Speedup plots are BANNED** (memory `feedback_speedup_plots_removed`). The
  Tabasco per-step cost story stays in prose / a small table — never a figure.
- Never stage `.aux/.log/.out/.toc`. For `.tex`/PDF-only commits, `git commit
  --no-verify` sidesteps the pre-commit hook tangle.

## Tabasco chapter — the plan (full detail in the flow doc)

**Thesis:** null result + diagnosis. REPA does not measurably improve Tabasco,
because *neither* REPA-enabling condition holds — (1) no metric headroom
(saturated baseline), (2) weak/ill-matched encoders (CheMeleon 2D-saturated,
MACE descriptor-bottlenecked). This is the **control** that licenses the report's
"when does REPA work?" question and the contrast Proteina satisfies. Frame as a
genuine investigation; earn the diagnosis in §5.4.

**Beats:** 5.1 regime + question → 5.2 setup (**Table 5.1** variant grid) → 5.3
results/null (**Fig 5.1** epoch-matched validation curves, **Table 5.2**
generation metrics + bootstrap CIs) → 5.4 diagnosis (**Table 5.3** descriptor
probe, ties to Ch4; two-conditions; forward-ref Proteina).

**It is a writing task, not an experiment** — all data is on disk:
- Generation: `evaluation/tabasco/generation/results/geom/evaluation/`
  (`evaluation_summary.csv` = baseline + **6** variants — CheMeleon full 2×2 +
  MACE additive/tradeoff; **MACE has no same/fused split**, so the grid is not a
  clean cube; per-variant `metrics.json` has `bootstrap_ci`).
- Over training: `.../validation/validation_curves.csv` (+ `validation_epoch_matched.png`).
- Rep probes: `evaluation/tabasco/representation/{FINDINGS.md,results/results.md}`.
- Encoder profiling: `encoder_profiling/tabasco/{chemeleon,mace}/FINDINGS.md`.
- Cost: `.../training_performance/training_performance.csv`.

**Four wrinkles that constrain the claims:**
1. **No multi-seed generation** — one run/variant. Null rests on **bootstrap CIs
   over the 1000-mol set**, NOT seed replication. Do not write "seed variance."
2. **Baseline trained ~2× longer** (≈152k/33ep vs ≈73k/15ep) → use the
   **epoch/step-matched** comparison.
3. **Speedup plots banned** (above).
4. **Diagnosis sharpening:** even FCD (the one non-saturated metric) doesn't
   improve → not *only* saturation; the encoders can't exploit the headroom.
   `mace_tradeoff` is the lone flicker (3D teacher, tradeoff mode).

**Open decisions (need user steer):**
- Prose placement: scratch file vs direct into Ch5.
- Scope: 1 fig + 3 tables, or fold Table 5.1 into prose (→ 2 tables).

**Resolved 2026-06-03:** semantics verified in code + variant focus set by user.
additive = regularizer (`diffusion + λ·repa`, the headline); tradeoff = convex
squeeze (`(1−λ)·diffusion + λ·repa`, the "ring it out" attempt). fused = concat
coord+atom → one projector → one alignment (the variant to lead on); same =
separate plumbing. Spotlight pair = `chemeleon_additive_fused` + `mace_additive`;
`mace_tradeoff` is the lone flicker. See flow doc "Resolved decisions".

**Build order (collision-free first):** Fig 5.1 script (`validation_curves.csv` →
PNG) → Tables 5.1/5.2/5.3 `.tex` in `tables/` → prose §5.1–5.4 (scratch or Ch5).

## Ch6 ending — what changed this session (in `report-draft.tex`)

- **§6.2.6 "What we do not claim"** — converted run-on paragraph → 6-item list:
  no asymptotic *generation-quality* win on PDB (transient), unclear designability
  on synthetic (ODE-floor caveat), no reliable novelty, no general β-preservation,
  co-movement-not-mediation, and **unconditional-only** scope (length specified,
  fold/motif not — refs `sec:flow-matching`). Lead is **unbolded** (other session
  unbolded it; §6.2.6 is the only unbolded subsection lead — decide if intentional).
- **New `\section{Discussion}`** (`sec:proteina-discussion`, promoted from a
  subsection; the flat "Summary" was dropped). Four bold-led paragraphs + a
  one-line Ch7 bridge:
  1. *establishes REPA accelerates Proteina's convergence on generation quality,
     and shows how* (headline + encoder-routed mechanism).
  2. *designability/diversity trade-off — REPA reaches a better spot than the
     baseline* (ODE floor = better learned distribution; encoder-specific cost).
  3. *helps most where the baseline struggles* (regime → data-efficiency).
  4. *points past Proteina* (multi-factorial routing spine — **gestured**, Ch7
     delivers).
- §6.2.7 "Robustness across scale" deliberately **left untouched** (user deferred
  it). Its appendix promise (training-dynamics ablations) is tracked as `[P5]` in
  the appendix TODO block — data not collected; write the appendix table or delete
  the clause before submission.

## Outstanding / next steps

- **Start the Tabasco chapter** from [tabasco-chapter-flow.md](tabasco-chapter-flow.md):
  build the collision-free assets first (Fig 5.1 + Tables 5.1–5.3), then prose.
- Resolve the three open Tabasco decisions above.
- Decide whether to commit the uncommitted Ch6 refinements (~27 lines) as their
  own commit, or let them keep riding with the other session.
- §6.2.6 bold-vs-unbold consistency call.
- §6.2.7 + the Conclusions chapter (Ch7) are still future work (Ch7 is a commented
  beat skeleton 7.1–7.6).

## Key paths

- Report: `docs/masters-report/report-draft.tex`
  (§6.2.6 ≈ l.1473; Discussion `\section` ≈ l.1491; Ch5 Tabasco skeleton ≈ l.1076).
- Tabasco plan: `docs/masters-report/tabasco-chapter-flow.md`
- Proteina plan (style reference): `docs/masters-report/proteina-chapter-flow.md`
- Tabasco data root: `evaluation/tabasco/`
