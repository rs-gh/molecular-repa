# Plan — per-caption seed/spread annotations (Chapter 6)

**No edits made yet.** This is for your review. Below is every Ch6 float (plus
the appendix tables carrying Ch6 data), what its data *actually* is on the
seed/spread axis, and the exact caption text I propose to add.

---

## What I found (the key facts)

- The standard generation eval is **n = 256 generated backbones per seed**
  (`n256` runs); the n=128 anchor rows are 128. Designability is computed on a
  per-length subset (lengths 50/100/150/200/250). There is **no single sample-n**
  that applies cleanly to every metric, so I propose annotating **seeds**, not
  sample-n, as the uniform fact (sample-n offered as an option below).
- Seeds used across the chapter are **42 / 1042 / 2042**. Coverage is **not
  uniform** — it ranges from 1 to 3 depending on run and step:
  - Early training-step points are often **single-seed (42)**; later points
    have 2–3.
  - The **GearNet-PDB** runs (L4-GearNet, L9-GearNet) and **AFDB L4-MPNN** are
    **single-seed everywhere**.
  - CKNNA (Fig 6.3) is **single-seed with a bootstrap band**, not a seed band.
  - The sampler table's non-default γ columns are **single-seed**; only γ=0.45
    has 3 (PDB) / 2 (AFDB).

So the chapter-level guide's blanket "mean over three sampling seeds" is
**slightly optimistic** and should be softened to "up to three" — see item 0.

---

## Proposed convention (the format I'd add)

A short trailing parenthetical, consistent across floats. Two facts:

> **Seeds:** `<count or range>` (42/1042/2042); **spread:** `<band meaning>`.

- For **trajectory figures**: `Seeds: 1–3 per point; bands show min/max across seeds.`
- For **snapshot tables**: `Seeds: 3 (42/1042/2042)` — with explicit exceptions
  called out where coverage drops to 1.
- Where a caption *already* states the seed status (6.2, 6.7, 6.8, A.8), I only
  standardise the wording.

**Decision for you — interpretation of "min/max count":**
- **(A, my default)** the *seed-count range* across points (e.g. "1–3 seeds"),
  matching the existing "bands show min/max" language.
- **(B)** also state the *generated-sample size* (n=256/seed). Adds a number but
  it's metric-dependent (designability uses a per-length subset), so it risks
  implying more precision than is true.

I've drafted everything under (A); say the word and I'll fold (B) in.

---

## Item 0 — fix the chapter-level guide (line 1568)

**Current:**
> …we display the mean over three sampling seeds, with shaded bands showing the
> min/max.

**Proposed:**
> …we display the mean over **up to three** sampling seeds (42/1042/2042), with
> shaded bands showing the min/max across them; **single-seed points show no
> band**. Per-float captions state their exact seed coverage.

---

## Figures

### Fig 6.1 — `fig01_fid_des_convergence` (proteina-fid) — FPSD/Des convergence, **both regimes**
- **Facts:** mean over seeds, band = min/max over seeds; coverage 1–3 per point.
- **Current caption:** no seed statement.
- **Add:** `(Mean over up to 3 seeds [42/1042/2042]; bands show min/max across seeds, absent where a point is single-seed.)`

### Fig 6.2 — `fig02_representation` (proteina-rep) — probe-quality trajectories
- **Facts:** seed 42 throughout, 1042/2042 on the tails; band = min/max over
  seeds, degenerate (no band) where single-seed. Probe `n_train=1000`.
- **Current caption:** no seed statement (confirmed).
- **Add:** `(Best-layer probe, n_train=1000; mean over up to 3 seeds [42/1042/2042], bands = min/max across seeds.)`

### Fig 6.3 — `fig03_alignment` (proteina-cknna) — CKNNA alignment
- **Facts:** **single seed (42)**; bands are **5–95% bootstrap** intervals (NOT
  seed spread). Already noted at chapter level.
- **Current caption:** mentions bootstrap median; **no seed count**.
- **Add:** `(Single seed [42]; lines are bootstrap medians, bands the 5–95% bootstrap interval.)`

### Fig 6.4 — `fig04_fid_des_gen_vs_rep` (proteina-genrep) — gen vs rep scatter
- **Facts:** gen axis = **seed-mean** (1–3 seeds), rep axis = best-layer probe
  at **seed 42**. Each point a checkpoint.
- **Current caption:** no seed statement (confirmed).
- **Add:** `(Generation: seed-mean over up to 3 seeds; representation: best-layer probe, seed 42.)`

---

## Tables

### Table 6.1 — `table_setup_model` (setup-model) — config
- **No data / no seeds.** **No change.**

### Table 6.2 — `table_rep_quality` (proteina-rep) — rep rank-order Δ
- **Facts:** single seed (42), averaged over the 700K–1.2M window. `n_train=1000`.
- **Current caption:** already says *"(Single-seed; averaged over the 700K–1.2M
  training-step window.)"* → standardise to: `(Single seed [42], n_train=1000; averaged over the 700K–1.2M window.)`

### Table 6.3 — `table_speedup` (proteina-speedup) — acceleration % + long-run best **[auto-generated]**
- **Facts:** each value is a **seed-mean** over all available seeds (up to 3 for
  n256; some n128 anchor rows up to 5). Mixed coverage.
- **Current caption:** no seed statement.
- **Add:** `(Seed-mean over up to 3 seeds [42/1042/2042] where available; n=256.)`
- **Note:** auto-generated — I'd add this to `make_speedup_table.py`'s caption
  string, not the .tex, so a regen won't wipe it.

### Table 6.4 — `table_sampler` (proteina-sampler) — γ robustness @400K
- **Facts:** **γ=0.45 column** = 3 seeds (PDB) / 2 (AFDB); **all other γ columns
  = single seed (42)**. n=256.
- **Current caption:** no seed statement.
- **Add:** `(n=256; the γ=0.45 column is a 3-seed mean [2 for AFDB], the other γ columns single-seed.)`

### Table 6.5 — `table_genrep_corr` (proteina-genrep-corr) — partial correlation **[auto-generated]**
- **Facts:** gen = seed-mean; rep = **seed 42** best-layer; correlation pooled
  over **n=59 checkpoints**.
- **Current caption:** already says *"(n=59 checkpoints pooled…)"*.
- **Add:** extend to `(n=59 checkpoints; generation seed-mean, representation seed 42.)` — in `make_genrep_corr.py`.

### Table 6.6 — `table_proteina_13m` (proteina-13m) — 1.3M centerpiece
- **Facts (verified):** 9 of 12 rows are **3-seed means** (42/1042/2042);
  **single-seed (42):** PDB L4-GearNet, **PDB L9-GearNet**, AFDB L4-MPNN. The
  step-tagged rows are earlier checkpoints (some 3-seed, some 1).
- **Current caption:** no seed statement.
- **Add:** `(Means over 3 seeds [42/1042/2042], except the two GearNet-PDB rows and AFDB L4-MPNN, which are single-seed; n=256.)`
- **Optional:** dagger-mark the 3 single-seed cells. Flagging because
  **L9-GearNet-PDB is not step-tagged**, so a reader would assume it's a full
  3-seed cell. Recommend at least the caption note; dagger is your call.

### Table 6.7 — `table_ode_floor` (proteina-ode-floor) — ODE floor
- **Facts:** **single seed** (ODE was not multi-seeded). n=256.
- **Current caption:** already ends "Single-seed." → standardise to `(Single seed [42]; n=256.)`

### Table 6.8 — `table_concentration` (proteina-concentration) — β-stratified pwTM
- **Facts:** **single seed**. n=256, binned by β-fraction.
- **Current caption:** already ends "Single-seed." → standardise to `(Single seed [42]; n=256.)`

---

## Appendix tables with Ch6 data (in scope? — your call)

| # | float | seed facts | proposed |
|---|-------|-----------|----------|
| A.3 | `proteina-rep-full` | seed 42, full probe table | `(Single seed [42], n_train=1000.)` |
| A.4 | `proteina-rep-afdb` | seed 42, AFDB | `(Single seed [42], n_train=1000.)` |
| A.5 | `cknna-matrix` | single seed, bootstrap | `(Single seed [42]; bootstrap medians.)` |
| A.6 | `proteina-genrep-corr-afdb` | gen seed-mean, rep seed 42 | mirror 6.5 |
| A.7 | `ss-composition` | **3 seeds**, reports per-seed ranges | already discusses seed spread; standardise to `(3 seeds [42/1042/2042]; ranges are across seeds.)` |
| A.8 | `proteina-ode` | single seed, 700K | already says single-seed → `(Single seed [42]; n=256.)` |

---

## Open questions before I edit

1. **Interpretation A vs B** (seed-range only, or also sample-n=256). Default: A.
2. **Scope:** Ch6 main floats only, or include the appendix Ch6 tables above?
   (I'd include them for consistency.)
3. **Table 6.6 single-seed cells:** caption note only, or also dagger-mark the
   3 cells?
4. **Auto-generated tables (6.3, 6.5):** OK to edit the caption string inside
   `make_speedup_table.py` / `make_genrep_corr.py` so regen preserves it?
5. **Exact wording:** the snippets above are drafts — happy to match a tighter
   house style (e.g. a fixed `Seeds: … · Spread: …` tag).
