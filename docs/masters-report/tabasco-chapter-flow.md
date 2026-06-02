# Tabasco chapter (Ch5, `ch:tabasco-study`) — flow / plan sketch

Companion to [proteina-chapter-flow.md](proteina-chapter-flow.md), same style.
This is the **plan**, not prose. Deliberately a smaller chapter than Ch6.

---

## TL;DR — the thesis

REPA does **not** measurably improve Tabasco small-molecule generation. The
chapter is a **null result + diagnosis**: the two conditions REPA needs are both
absent —

1. **No evaluation headroom** — the baseline is near-saturated on the metric
   surface (validity, connectivity, uniqueness all ≈ceiling).
2. **Weak / ill-matched encoders** — CheMeleon (2D, saturated projector) and
   MACE (3D but descriptor-bottlenecked), per Ch4 profiling.

This is **not a failure to report** — it's the *control* that licenses the
report's "*when* does REPA work?" question, and the contrast Proteina (Ch6)
satisfies. Frame as a genuine investigation; the reader earns the diagnosis in
§5.4, not before.

## Role in the report (why this chapter has to exist)

- Answers Intro **Q(i)** ("Does REPA work in the molecular domain?") for the
  small-molecule regime: *no, under our setup*.
- **Validates the Ch4 profiling verdicts** (CheMeleon/MACE flagged weak) — this
  is the *small-molecule payoff* of the profiling framework. Without Ch5 that
  half of Ch4 is orphaned.
- Sets up the **Ch7 synthesis**: domain is the wrong axis; the conditions
  (headroom + transferable encoder info) are what predict gain. Tabasco = neither
  condition → Proteina = both → REPA helps.

---

## Beats (5.1 → 5.4)

### §5.1 The small-molecule regime + the question
- GEOM-drugs scale, ~900 Da cutoff; why this is a fundamentally different data
  regime from proteins (smaller, 2D-graph-dominated, strong saturated baseline).
- Pose the chapter's open question: *does REPA's mechanism transfer to
  small-molecule 3D generation?* Do **not** pre-commit to the saturation answer.

### §5.2 Experimental setup → **Table 5.1**
- Tabasco baseline (vanilla flow-matching transformer on 3D coords + atom types).
- REPA integration: projector on trunk hidden states (`hidden_states_coord` /
  `hidden_states_atom`).
- The variant grid → **Table 5.1**: encoder {CheMeleon, MACE} × combination mode
  {additive, tradeoff} × plumbing {same, fused}. **NOT a clean cube: CheMeleon
  runs the full 2×2 (4 variants); MACE has only additive/tradeoff, no same/fused
  split (2 variants). → 6 REPA variants + baseline, 7 rows total** (verified on
  disk against `evaluation_summary.csv`).
- Encoders sourced from Ch4 profiling — name the two and why they were the
  candidates.

### §5.3 Results — the null → **Fig 5.1**, **Table 5.2**
- **Fig 5.1**: validation curves vs epoch (validity / connectivity / FCD),
  baseline + variants, **epoch-matched** — everything saturates early, variants
  track baseline. The "no separation" visual.
- **Table 5.2**: generation metrics with **bootstrap CIs**, baseline vs variants.
  Differences sit inside the CIs. Baseline already at validity 0.98 / connectivity
  0.998 / uniqueness 1.0.
- Honest reading: no variant beats baseline on FCD (the one non-saturated metric);
  most are slightly worse.

### §5.4 Diagnosis → **Table 5.3**
- **Table 5.3**: descriptor-regression probe (baseline vs variants vs frozen
  encoders) — ties straight to Ch4.
- Two-conditions reading:
  - **Surface saturated** → no headroom (preview from Ch3).
  - **Encoders weak/ill-matched** → even where headroom exists (FCD), the teacher
    can't supply usable signal. CheMeleon 2D + saturated (adds noise); MACE 3D but
    descriptor-orthogonal (signal doesn't transfer).
- `mace_tradeoff` is the lone flicker of rep-signal (3D teacher, *tradeoff* mode
  forces adoption) — name it, don't oversell it.
- Forward-reference Ch6 (Proteina) as the contrasting regime where both
  conditions are met.

---

## Figures & tables (proposed)

| Asset | Content | Source |
|---|---|---|
| **Table 5.1** | Variant grid: encoder × mode × plumbing | (design table, no data) |
| **Fig 5.1**   | Validation curves vs epoch, baseline + variants (epoch-matched) | `validation_curves.csv` |
| **Table 5.2** | Generation metrics + bootstrap CIs, baseline vs variants | `evaluation_summary.csv` + per-variant `metrics.json` (`bootstrap_ci`) |
| **Table 5.3** | Descriptor-probe R², baseline vs variants vs frozen encoders | `representation/results/results.md` |

All booktabs / no vertical rules, house style. Table 5.1 could fold into prose
(→ 2 tables) if we want it leaner. Optional Fig 5.2 sample gallery — low priority.

---

## Data sources (all on disk — this is a *writing* task, not an experiment)

- Generation: `evaluation/tabasco/generation/results/geom/evaluation/`
  - `evaluation_summary.csv` (baseline + 6 variants: CheMeleon 2×2 + MACE additive/tradeoff)
  - per-variant `metrics.json` carries `bootstrap_ci`
- Over training: `.../validation/validation_curves.csv` (+ `validation_epoch_matched.png`)
- Representation probes: `evaluation/tabasco/representation/{FINDINGS.md,results/results.md}`
- Encoder profiling: `encoder_profiling/tabasco/{chemeleon,mace}/FINDINGS.md`
- Training cost: `.../training_performance/training_performance.csv`

---

## Wrinkles / honesty (MUST handle — these shape what we can claim)

1. **No multi-seed generation runs.** One run per variant. The null rests on
   **bootstrap CIs over the 1000-molecule set**, *not* seed replication. Do **not**
   write "seed variance swallows the signal." Flag single-seed-per-variant as a
   limitation.
2. **Baseline trained ~2× longer** (≈152k steps / 33 ep vs ≈73k / 15 ep for REPA
   variants). Raw head-to-head is not apples-to-apples → use the **epoch/step-matched**
   comparison (`validation_epoch_matched.png` exists). At matched epochs REPA
   neither leads nor trails meaningfully.
3. **Speedup plots are BANNED** (project memory `feedback_speedup_plots_removed`).
   The per-step cost story (MACE ~2×, CheMeleon ~2× baseline) stays in prose / a
   small table — never a speedup figure.
4. **Diagnosis sharpening:** even FCD (the one non-saturated metric) doesn't
   improve → it's not *only* "saturated surface"; where headroom exists, the
   *encoders* can't exploit it. This is the cleaner two-conditions payoff.

---

## Key numbers (for grounding the prose / tables)

- **Generation (gen budget = 1000 mols):** baseline FCD **5.61**, validity 0.98,
  connectivity 0.998, uniqueness 1.0, diversity 0.886, novelty 0.966. REPA-variant
  FCD all **worse** (5.83–7.43); other metrics within noise.
- **Rep probe — atom-type (P3):** baseline 0.999 / macro-F1 0.993; variants
  0.998–1.0 → **saturated, no signal** (model sees atom types as input).
- **Rep probe — descriptor R² (P4):** baseline already high (MolWt 0.92, LogP 0.71,
  NumRings 0.74, RotBonds 0.64). `mace_tradeoff` the **only** variant to beat
  baseline on 3/4 (MolWt 0.955, LogP 0.730, RotBonds 0.666). CheMeleon variants
  flat/worse. Frozen encoders: CheMeleon MolWt R² **0.993** (near-perfect, 2D),
  MACE **0.022** (descriptor-bottlenecked), dummy 0.822.
- **Cost (s/step):** baseline 0.38; CheMeleon ~0.74; MACE GPU ~2.6; MACE cached ~0.78.

---

## Resolved decisions (2026-06-03)

- **Variant focus (user steer):** lead the narrative on **additive + fused**.
  - **additive** = `total = diffusion + λ·repa` — REPA as a pure *regularizer*,
    generation objective untouched (verified `losses.py:151`). The principled,
    honest test. **tradeoff** = `(1−λ)·diffusion + λ·repa` — a convex combination
    that *down-weights generation* to force the rep signal in; it was the attempt
    to "ring something out," not the default. Report additive as the headline;
    `mace_tradeoff` appears only as the lone flicker (forcing adoption surfaces
    MACE's weak 3D signal, exactly at the cost the convex form implies).
  - **fused** = concat coord+atom hidden states → **one** projector → **one**
    alignment (verified `losses.py:273`). This is the "former" — two reps brought
    into a single alignment, which is what we want to talk about. **same** = the
    separate-plumbing contrast. Lead on fused; `chemeleon_additive_fused` is also
    the strongest CheMeleon variant on the descriptor probe (MolWt R² 0.943).
  - So the spotlight pair is `chemeleon_additive_fused` + `mace_additive`; the
    other four are the surrounding grid that shows "no separation anywhere."

## Open decisions (need user steer)

- **Prose placement:** scratch file vs straight into Ch5 of `report-draft.tex`
  (Ch5 is upstream of §6.2.6 = parallel session's territory → collision risk).
- **Scope:** 1 fig + 3 tables, or fold Table 5.1 into prose (1 fig + 2 tables).

---

## Build order (collision-free first)

1. Fig 5.1 generation script (`validation_curves.csv` → PNG) — new file, safe.
2. Tables 5.1 / 5.2 / 5.3 `.tex` in `tables/` — new files, safe.
3. Prose §5.1–5.4 — into scratch file or Ch5 per the placement decision.
