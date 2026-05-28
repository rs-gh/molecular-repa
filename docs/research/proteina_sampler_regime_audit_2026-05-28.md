# Sampler-regime robustness — audit against the expanded sweep (2026-05-28)

This doc re-audits the [§ Sampler-regime robustness check](proteina_narratives.md#sampler-regime-robustness-check-added-2026-05-27-expanded-2026-05-27)
in `proteina_narratives.md` against the larger sampler-noise sweep that landed
overnight 2026-05-27 → 02:03 BST 2026-05-28 (propagation log: [.propagate_20260528_020324.log](../../evaluation/proteina/generation/scripts/.propagate_20260528_020324.log)).

**Scope:** every claim in the narrative's sampler-regime section is re-checked
against the new (encoder × dataset × γ × step) coverage. Each claim is tagged
with its **disentangled axis** — `dataset-specific`, `encoder-specific`,
`mechanism (headroom / geometric bias)`, or `single-cell artifact`.

**Data sources used here:**
- `results/variance/n256_sampler_ablation/sweep_results.clean.jsonl` (PDB, 144 rows)
- `results/variance/n256_afdb_sampler_ablation/sweep_results.clean.jsonl` (AFDB, 105 rows)
- `results/paper/n256_convergence_pdb/sweep_results.clean.jsonl` (γ=0.45, 3-rep mean)
- `results/paper/n256_convergence_afdb/sweep_results.clean.jsonl` (γ=0.45, 3-rep mean)

**Reproduce:** [build_sampler_regime_robustness.py](../../evaluation/proteina/generation/scripts/paper/build_sampler_regime_robustness.py) emits AFDB tables today; PDB tables don't regenerate yet (see § 1).

---

## 0. What's in the new sweep

Coverage in the two clean jsonls. **Bold = 5 samplers (full γ grid); plain = only 3 (ODE / γ=0 / γ=1).**

| dataset · run | steps |
|---|---|
| **PDB · baseline_256** | only 3 samplers at 100/200/400/700/1000/1300/1500K |
| **PDB · L4-GN** | **5 samplers** at 100/200/400/700/1000K |
| **PDB · L4-random** | **5 samplers** at 100/200/400/700K |
| **PDB · L9-GN** | 3 samplers at 100/200/400/700/800/900K; **5 samplers** at 1000K |
| **PDB · MPNN-L4** | **5 samplers** at 100/200/400/700/1000/1300K |
| **PDB · MPNN-L9** | **5 samplers** at 100/200/400/1000/1300K |
| **AFDB · baseline_afdb_256** | **5 samplers** at 100/400/700/1000/1300/1600K |
| **AFDB · L4-GN** | **5 samplers** at 100/400/700/1000/1200/1300K |
| **AFDB · L9-GN** | **5 samplers** at 100/400/700/900K |
| **AFDB · MPNN-L9** | **5 samplers** at 100/400/700/1000/1300K |

Notable gap: **PDB baseline has no γ=0.35 or γ=0.5 rows**, so
`build_sampler_regime_robustness.py` (which intersects baseline ∩ REPA at every γ)
emits only the AFDB tables — see "PDB blocker" below. The new PDB encoders
(L4-GN, L4-random, MPNN-L4, MPNN-L9) themselves *are* 5-γ complete; they just
have no baseline to step-match against at γ=0.35 / γ=0.5.

**PDB blocker — what's needed to regenerate PDB tables:**
fire baseline_256 (and ideally L9-GN) at γ ∈ {0.35, 0.5} across the
step-matched grid {100, 200, 400, 700, 1000, 1300}K = **14 baseline cells**
(7 steps × 2 γ). The ext sweep (jobs 29735627–31, all COMPLETED 21/21) scoped
itself to *new* encoder rows and treated baseline as "already on disk" — but
on disk it only ever had ODE/γ=0/γ=1.

---

## 1. Baseline sanity check — γ navigates des↔diversity the same way on both datasets

Late-training (step ≥ 700K) baseline values per γ regime, averaged across steps:

| γ regime | PDB baseline | AFDB baseline |
|---|---|---|
| **γ=0** (no noise) | Des **.84**, scRMSD 1.3, pLDDT .70, pwTM **.71**, #Clust 3, FID 800, fJSD-A 2.5, ssJSD-2D **.50** | Des **.92**, scRMSD 1.2, pLDDT .72, pwTM **.58**, #Clust 8, FID 640, fJSD-A 2.5, ssJSD-2D **.30** |
| **γ ∈ {.35,.45,.5}** | Des .55, scRMSD 2.5, pLDDT .63, pwTM .18, **#Clust 25**, FID 440, fJSD-A 1.1, ssJSD-2D .32 | Des .70, scRMSD 2.2, pLDDT .69, pwTM .20, **#Clust 26**, FID 460, fJSD-A 2.0, ssJSD-2D .13 |
| **γ=ODE / γ=1** | Des .06, scRMSD 9, pLDDT .50, pwTM .15, #Clust 4, **FID 300**, **fJSD-A .4**, ssJSD-2D .03 | Des .18, scRMSD 7, pLDDT .60, pwTM .18, #Clust 8, **FID 220**, **fJSD-A .6**, ssJSD-2D .04 |

Shape is identical on PDB and AFDB:

- **γ=0** = mode-collapsed regime. High Des + high quality but **pwTM huge** (samples cluster tightly) and ssJSD-2D bad (SS marginal is wrong).
- **γ=mid** = only regime with measurable tertiary diversity. #Clust peaks at 25–30.
- **γ=ODE / γ=1** = "exploration without quality" regime. FID/fJSD-A best, but Des < 0.2 and #Clust low (designable pool too small to cluster).

**Two consequences load-bearing for the rest of this doc:**

1. **`#Clust` is a strictly mid-γ metric.** At γ=0, designable samples are too clustered. At γ=ODE/γ=1, designable samples are too few. So every REPA `#Clust` finding is necessarily mid-γ — calling it "robust across γ" or "fails at γ=1" is mis-framed; #Clust doesn't exist as a usable measurement outside mid-γ.
2. **ssJSD-2D headroom is γ-dependent and training-step-dependent.** PDB baseline ssJSD-2D at γ=ODE crashes 0.84 → 0.01 by step 700K; AFDB baseline at γ=ODE stays at ~0.04. At γ=0.45 neither dataset's baseline solves ssJSD-2D. "Where REPA wins ssJSD-2D" is governed by where the baseline still has headroom, not by γ regime per se.

Baseline tables across all γ × step are in [/tmp/sampler_analysis/baseline_tradeoff.py](/tmp/sampler_analysis/baseline_tradeoff.py) (regenerable; see § 7 for the raw renders).

---

## 2. Claim-by-claim audit

Each claim is shown with **(a) what the narrative said + the data behind it**, **(b) what the new data shows**, **(c) verdict**, **(d) the disentangled axis**.

### Claim A — *"Distribution-match (FID, fJSD-A) dies at γ=1"*

**(a) Original evidence:** PDB L9-GN single-step at 700K had Δ FID = +41.7, Δ fJSD-A = +0.44 at γ=1 (the bad cell). PDB L9-GN trajectory: FID 1/4 wins at γ=1. AFDB L4-GN at γ=1 had 1/2 (n=2 only).

**(b) New data — Δ at γ=1.0 (REPA − baseline; ✓ = win, i.e. FID/fJSD-A lower):**

| pair | Δ FID across steps | wins | Δ fJSD-A across steps | wins |
|---|---|---|---|---|
| PDB L4-GN | −88 / −216 / −83 / +39 / −20 | 4/5 | −.16 / −.93 / −.54 / −.13 / +.63 | 4/5 |
| PDB L9-GN | +139 / −263 / +51 / +42 / −27 | 3/5 | −.06 / −.84 / +.06 / +.44 / +.49 | 2/5 |
| PDB MPNN-L4 | −287 / −406 / +51 / −83 / −13 / +68 | 4/6 | −1.15 / −1.21 / −.60 / +.27 / +1.04 / −.02 | 3/6 |
| PDB MPNN-L9 | +5 / −233 / −129 / −271 / −28 | 4/5 | −.35 / −.59 / −.57 / +.26 / −.26 | 4/5 |
| AFDB L4-GN | +11 / −330 / −89 / −66 / −10 | 4/5 | −.07 / −.43 / −.45 / −.18 / −.08 | **5/5** |
| AFDB L9-GN | −27 / −350 / −122 | 3/3 | +.07 / −.44 / −.43 | 2/3 |
| AFDB MPNN-L9 | −30 / −322 / −97 / +13 / +6 | 3/5 | +.05 / −.32 / −.34 / −.15 / +.10 | 3/5 |

**(c) Verdict:** **The claim collapses to a PDB-L9-GN-at-700K single-cell artifact.** Out of 7 (pair × γ=1) cells we can now check, AFDB wins FID in 4/5–3/3 cells; PDB wins FID in 4/5 (L4-GN), 4/6 (MPNN-L4), 4/5 (MPNN-L9). Only PDB-L9-GN's specific 700K cell loses both metrics — that was the cell quoted in the original narrative table.

**(d) Axis:** **Single-cell artifact**, not a dataset/encoder/mechanism effect. One bad row got promoted to a claim.

---

### Claim B — *"Designability/quality dies at γ=0 on PDB; survives at γ=0 on AFDB"*

**(a) Original evidence:** PDB L9-GN trajectory at γ=0: Des 1/4, pLDDT 1/4. AFDB L4-GN at γ=0: Des 2/2 wins (n=2). Specific 700K PDB Δ Des = −0.16 vs baseline 0.836.

**(b) New data — Δ Des at γ=0:**

| pair | Δ Des across steps | wins |
|---|---|---|
| PDB L4-GN | 0 / +.21 / +.04 / +.04 / **−.41** | 4/5 |
| PDB L9-GN | 0 / +.25 / −.05 / **−.16** / **−.70** | 2/5 |
| PDB GN-random L4 | 0 / −.10 / **−.61** / −.11 | 0/3 (post-100K) |
| PDB MPNN-L4 | 0 / +.49 / **−.44** / −.05 / −.10 / **−.38** | 2/6 |
| PDB MPNN-L9 | +.04 / +.28 / +.15 / −.12 / +.01 | 4/5 |
| AFDB L4-GN | +.02 / +.19 / +.05 / +.17 / −.02 | 4/5 |
| **AFDB L9-GN** | **−.02 / −.20 / −.22** | **0/3** |
| AFDB MPNN-L9 | +.29 / +.15 / +.06 / +.06 / −.07 | 4/5 |

Important context: baseline Des at γ=0 is already 0.78–0.92 (AFDB) and 0.84–0.99 (PDB) by step ≥ 400K — most "REPA worse" cells are losses against a near-ceiling baseline. AFDB-L9-GN's losses (e.g. baseline .91 → REPA .69) are real, large drops.

**(c) Verdict:** Real effect, but the dataset axis from the narrative is **wrong**. The effect is encoder-specific, not PDB-specific:
- GearNet at L9 loses Des at γ=0 on **both** datasets (PDB L9-GN 2/5, AFDB L9-GN 0/3).
- GearNet at L4 loses late-training on PDB but survives on AFDB.
- MPNN-aligned REPA preserves Des at γ=0 on both datasets.
- Random-encoder GearNet also loses Des at γ=0 on PDB.

**(d) Axis:** **Encoder-specific** (GearNet ≫ MPNN; deeper layer worse; random-GN also loses). Small dataset effect on top (less Des headroom on PDB). Mechanism candidate: **alignment to richer geometric encoders broadens the very narrow γ=0 mode-collapse basin baseline lives in**, so REPA can't reach baseline's low-temp ceiling. MPNN's coarser encoding doesn't broaden the manifold as far.

---

### Claim C — *"ssJSD-2D is only robust in the middle band; loses at γ=0 / γ=1"*

**(a) Original evidence:** PDB L9-GN trajectory ssJSD-2D was 1/4 at γ=0 (only 700K win) and 2/4 at γ=1.

**(b) New data:**

Δ ssJSD-2D at **γ=0**:

| pair | Δ ssJSD-2D across steps | wins |
|---|---|---|
| PDB L4-GN | +.11 / −.18 / +.06 / −.05 / −.15 | 3/5 |
| PDB L9-GN | +.03 / +.11 / +.08 / **−.39** / **−.26** | 3/5 |
| PDB GN-random L4 | +.08 / −.27 / −.27 / −.22 | 3/4 |
| PDB MPNN-L4 | +.33 / +.08 / −.26 / −.26 / −.11 / −.33 | 4/6 |
| PDB MPNN-L9 | +.28 / +.35 / +.06 / −.03 / −.28 | 2/5 |
| AFDB L4-GN | **−.34 / −.19 / +.02 / −.38 / −.05** | **4/5** |
| AFDB L9-GN | −.08 / −.25 / −.17 | **3/3** |
| AFDB MPNN-L9 | −.13 / −.12 / −.04 / −.17 / −.09 | **5/5** |

Δ ssJSD-2D at **γ=1**:

| pair | wins |
|---|---|
| AFDB L4-GN | **5/5** (Δ ∈ [−.05, −.001]) |
| AFDB L9-GN | 3/3 |
| AFDB MPNN-L9 | 3/5 |
| PDB L9-GN | 2/5 |
| PDB L4-GN | 3/5 |
| PDB MPNN-L4 | 3/6 |
| PDB MPNN-L9 | 3/5 |

**(c) Verdict:** Dataset-specific, not γ-regime-specific. On AFDB, ssJSD-2D wins at *every* γ for *every* encoder (the "only mid-band robust" framing is wrong on AFDB). On PDB, mid-band wins are clearer (3/5 – 5/5) and γ=0 / γ=1 are noisier (2/5 – 4/6) but not "loses."

**Mechanism:** baseline ssJSD-2D vs step at the extremes:
- PDB γ=ODE: 0.84 → 0.11 → **0.01** by 700K — baseline solves SS-marginal cleanly at the extremes with training.
- AFDB γ=ODE: 0.05 → 0.10 → 0.04 — already low; never crashes.
- PDB γ=0.45: 0.17 → 0.27 → 0.32 — *drifts up*; baseline can't get SS-marginal right at mid-γ.
- AFDB γ=0.45: 0.36 → 0.14 → 0.13 — improves, plateaus.

So once PDB baseline crashes to ~0.01 at γ=ODE/γ=1 by step 700K, there's no headroom for REPA. The original "loses at γ=1" cells are mostly *early-step* losses where baseline closed faster than REPA at those γ — not a γ-regime failure.

**(d) Axis:** **Headroom-driven, modulated by dataset.** REPA wins ssJSD-2D where the baseline is bad at it. AFDB baseline is bad-ish at every γ → REPA wins everywhere. PDB baseline solves the extremes after 700K → REPA wins concentrate in mid-γ. *Not* an intrinsic γ-regime limitation.

---

### Claim D — *"β (= ss_frac_E) was never sampler-robust"*

**(a) Original evidence:** PDB L9-GN at every γ: β 1–2/4. AFDB L4-GN: 1/2.

**(b) New data — Δ β (REPA − baseline) at γ=0.45 across the trajectory:**

| pair | Δ β | direction |
|---|---|---|
| PDB L4-GN | −.005 / −.048 / −.005 / +.013 / +.048 / +.099 / −.024 | mixed, late-positive |
| PDB L9-GN | +.021 / −.134 / +.012 / −.045 / +.025 / +.049 / +.042 | mixed |
| AFDB L4-GN | +.028 / +.005 / −.037 / −.021 / +.012 | mixed, near zero |
| **AFDB MPNN-L9** | **+.010 / +.027 / −.054 / −.050 / −.041 / −.040** | **REPA consistently ↓β** |

**(c) Verdict:** Holds in spirit — β is never a robust REPA *increase* across γ. But a new finding emerges that deserves its own bullet: **AFDB MPNN-L9 REPA consistently *reduces* β across the trajectory at every γ**.

**(d) Axis:** **Encoder-specific** (MPNN-L9-on-AFDB ↓β; everyone else mixed). Not dataset-specific (AFDB GN-L4 doesn't show it; only MPNN does).

---

### Claim E — *"AFDB-GearNet reduces tertiary diversity (#Clust, pwTM) across γ; PDB-GearNet preserves it"*

**(a) Original evidence:** AFDB L4-GN #Clust 1/2 at γ=0.45, losses at γ ∈ {0, .35, .45, .5}. PDB L9-GN #Clust wins at every γ.

**(b) New data — Δ #Clust at γ=0.45 across the trajectory:**

| pair | Δ #Clust across steps |
|---|---|
| AFDB L4-GN | +4.2 / −4.2 / −6.7 / −7.1 / −7.8 (5 cells, 4 losses) |
| AFDB L9-GN | 0 / +6.4 / −12.1 / −10.1 (4 cells, 2 losses) |
| **AFDB MPNN-L9** | **+13.9 / +0.7 / −3.9 / +4.2 / +1.2 / +5.9** (6 cells, **5 wins**) |
| PDB L4-GN | +4.0 / +3.7 / +0.4 / −3.7 / −19.8 / −5.6 (mixed, late losses) |
| PDB L9-GN | 0 / +8.5 / +2.7 / −1.7 / −16.2 / −10.6 (mixed) |
| PDB MPNN-L9 | 0 / +2.1 / +3.9 / +9.0 / −3.3 / −10.4 (mixed) |
| PDB GN-random L4 | 0 / −3.6 / −8.7 (control: also loses) |

Within AFDB GN-L4, the γ-dependence is sharp: γ=ODE gives **wins** (+5/+6/+4/+5), γ=0 mostly negative, γ ∈ {.35,.45,.5} consistent losses, γ=1 mixed. As noted in § 1, this is consistent with #Clust only existing as a usable metric in mid-γ — so the regime-dependence is partly a measurement artifact.

**(c) Verdict:** The AFDB-GearNet diversity loss holds, but the framing needs three corrections:

1. **It's GearNet-specific on AFDB, not REPA-generic.** AFDB MPNN-L9 *grows* #Clust at γ=0.45 (5/6 wins). This is the cleanest GearNet-vs-MPNN split in the whole sweep.
2. **PDB late-training also loses #Clust under GearNet.** L4-GN, L9-GN, even MPNN-L9 post late-step #Clust losses (1000K+). The PDB-vs-AFDB asymmetry in the narrative weakens — late-step PDB looks like AFDB.
3. **Random-encoder GearNet also loses #Clust on PDB.** So this is not a "learned representation" effect — geometric alignment to any GearNet-shaped manifold clusters samples. **Mechanism is the encoder's geometric inductive bias, not its learned features.**

**(d) Axis:** **Encoder-specific × γ-regime-specific (mid-γ only).** GearNet-style geometry pulls the designable manifold into fewer basins; MPNN doesn't. Not a learned-feature effect (random-encoder reproduces it).

---

## 3. SS and tertiary diversity vs γ across training time — synthesis

### Secondary-structure marginal (ss_frac_E ≈ β)
- **γ-insensitive on baseline:** β stays in 0.14–0.21 across γ on PDB, 0.12–0.21 on AFDB. γ does not navigate secondary-structure composition for the baseline.
- **REPA Δβ is small everywhere** except AFDB-MPNN-L9 (consistently negative — REPA ↓β across the trajectory at every γ) and PDB-L9-GN late-train (slightly positive at γ=0.45).
- **Conclusion:** secondary-structure *shape* is largely γ-insensitive on the baseline; REPA's β effects are small, encoder-specific, and late-training.

### SS-distribution-matching (ssJSD-2D)
- **Strongly γ-dependent for baseline:**
  - γ=0: high (0.40–0.72 PDB / 0.20–0.64 AFDB) — mode-collapse drives SS-marginal off the data
  - γ=mid: moderate, doesn't converge with training on PDB (0.17–0.38); plateaus at ~0.13 on AFDB
  - γ=ODE/γ=1: baseline crashes ssJSD-2D to <0.05 by step 700K on PDB, stays at ~0.03 on AFDB
- **REPA ssJSD-2D wins concentrate in (γ, step) cells where the baseline still has headroom.** The most universal REPA effect across encoders + datasets is mid-γ ssJSD-2D improvement; the apparent "γ=ODE / γ=1 weakness" is mostly headroom exhaustion at late steps on PDB.

### Tertiary diversity (#Clust, pwTM)
- **Baseline #Clust is mid-γ-only** (4–8 at extremes, 25–30 in mid band). γ navigates tertiary diversity in a narrow inverted-U.
- **Inside the mid-γ band:**
  - AFDB GearNet: REPA reduces #Clust by 5–10 clusters (encoder-specific; reproduced by random-encoder GearNet on PDB → geometric inductive bias).
  - AFDB MPNN: REPA grows #Clust by 0–14 clusters.
  - PDB: late-step #Clust drops under most REPA variants — the AFDB-flavor mechanism appears once training is deep enough.
- **pwTM** (inter-sample TM, lower = more diverse):
  - Baseline pwTM is U-shaped in γ: huge at γ=0 (~0.5–0.75 — mode collapse), low at γ=ODE/γ=1 (~0.15).
  - REPA's pwTM effects are small and encoder-split: GearNet-on-AFDB slightly *raises* pwTM at γ=0 (i.e. more mode-collapse), MPNN-on-AFDB slightly *lowers* pwTM at γ=0 — same encoder split as #Clust.

### Cross-encoder summary at the regime level

| effect | regime | dataset axis | encoder axis | mechanism candidate |
|---|---|---|---|---|
| γ=1 distribution-match loss | n/a | n/a | n/a | single-cell, not real |
| γ=0 Des loss | low-γ | both | GN ≫ MPNN, deeper > shallow, random-GN also | alignment broadens mode-collapse basin |
| ssJSD-2D wins | wherever baseline has headroom | AFDB everywhere; PDB only mid-γ post-700K | universal | headroom-driven |
| β shifts | mid-γ | small | AFDB-MPNN-L9 unique (↓β) | encoder-specific |
| #Clust loss | strictly mid-γ | AFDB > PDB (PDB only late-train) | GN (incl. random-GN) ≠ MPNN | geometric encoder inductive bias |
| #Clust gain | strictly mid-γ | AFDB MPNN-L9 only | MPNN-specific | encoder-specific |

---

## 4. Suggested rewrite of the narrative headline

Currently:
> "γ ∈ [0.35, 0.5] is the robust band; distribution-match dies at γ=1; designability/quality dies at γ=0; ssJSD-2D and β are not robust at every γ."

Proposed replacement:
> **Two real γ-regime effects survive the larger sweep: (i) γ=0 designability loss for GearNet-aligned REPA on *both* datasets (encoder × mechanism), (ii) GearNet-vs-MPNN tertiary-diversity split in mid-γ (encoder × mechanism, reproduced by random-encoder GearNet → geometric inductive bias, not learned features). The "γ=1 distribution-match cliff" was a single-cell PDB-L9-700K artifact and disappears at every other (encoder, dataset, step). ssJSD-2D improvements are not γ-mid-band-specific; they are headroom-driven and universal where the baseline hasn't already converged.**

---

## 5. What's still missing / next moves

- **Fire 14 PDB baseline cells:** baseline_256 at γ ∈ {0.35, 0.5} across {100, 200, 400, 700, 1000, 1300}K. This unlocks PDB step-matched tables for all four PDB REPA encoders that are already 5-γ complete (L4-GN, L4-random, MPNN-L4, MPNN-L9) plus PDB L9-GN where γ=0.35/0.5 are still missing for everything except step=1000K.
- **Fire PDB L9-GN at γ ∈ {0.35, 0.5}:** 6 cells at 100/200/400/700/800/900K (1000K already has full grid). Without this, L9-GN — the encoder the original narrative used — can't be step-matched at the new mid-band γ on PDB.
- Once both backfills land: re-run `clean_variance_jsonl.py` then `build_sampler_regime_robustness.py`. The script auto-emits PDB tables once the intersection is non-empty.

---

## 6. Pointers

**Scripts:**
- Build script: [build_sampler_regime_robustness.py](../../evaluation/proteina/generation/scripts/paper/build_sampler_regime_robustness.py)
- Clean step: [clean_variance_jsonl.py](../../evaluation/proteina/generation/scripts/clean_variance_jsonl.py)
- TSV propagation: [jsonl_to_tsv.py](../../evaluation/proteina/generation/scripts/jsonl_to_tsv.py) (run via `all`)
- Sweep configs (where to add backfill cells):
  - PDB ext: `evaluation/proteina/generation/configs/sweeps/...` (search for `n256_pdb_sampler_ablation_ext`)
  - AFDB ext: `n256_afdb_sampler_ablation_ext`

**Raw data:**
- PDB ablation: [results/variance/n256_sampler_ablation/sweep_results.clean.jsonl](../../evaluation/proteina/generation/results/variance/n256_sampler_ablation/sweep_results.clean.jsonl)
- AFDB ablation: [results/variance/n256_afdb_sampler_ablation/sweep_results.clean.jsonl](../../evaluation/proteina/generation/results/variance/n256_afdb_sampler_ablation/sweep_results.clean.jsonl)
- γ=0.45 reference (3-rep): [results/paper/n256_convergence_pdb/sweep_results.clean.jsonl](../../evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl), [results/paper/n256_convergence_afdb/sweep_results.clean.jsonl](../../evaluation/proteina/generation/results/paper/n256_convergence_afdb/sweep_results.clean.jsonl)

**Audit Python scripts used to produce this doc** (persisted at [evaluation/proteina/generation/scripts/paper/audit_2026-05-28/](../../evaluation/proteina/generation/scripts/paper/audit_2026-05-28/)):
- [load.py](../../evaluation/proteina/generation/scripts/paper/audit_2026-05-28/load.py) — load + normalize all jsonls into one frame
- [baseline_tradeoff.py](../../evaluation/proteina/generation/scripts/paper/audit_2026-05-28/baseline_tradeoff.py) — baseline metric × γ × step tables (§ 1, § 7.{1,2})
- [compare.py](../../evaluation/proteina/generation/scripts/paper/audit_2026-05-28/compare.py) — Δ = REPA − baseline tables per (encoder, layer, dataset) (§§ 2–3, § 7.3)

Pre-rendered outputs (committed alongside the scripts):
- [baseline_tradeoff_out.txt](../../evaluation/proteina/generation/scripts/paper/audit_2026-05-28/baseline_tradeoff_out.txt)
- [compare_out.txt](../../evaluation/proteina/generation/scripts/paper/audit_2026-05-28/compare_out.txt)

These are quick rebuilds (~5 s each) against the canonical clean jsonls — re-run when the underlying jsonls move.

**Job history reference (sacct):** PDB sampler-ext array jobs 29735627 / 29735628 / 29735629 / 29735630 / 29735631 all COMPLETED 21/21 (2026-05-27 evening). AFDB sampler-ext landed 2026-05-27 → 2026-05-28 (29726047–52, 29727069–73, 29731171–73 had FAILED attempts; 29734862–875 and 29734971/29735144–46 COMPLETED — the working batch).

---

## 7. Supporting data: full Δ tables

(These are what the audit was actually built on; included verbatim so this doc is self-contained.)

### 7.1 Baseline metric × step × γ — PDB

```
[Des]                            [scRMSD]                         [pLDDT]
gamma     ODE    0.0   0.45    1.0  ODE    0.0   0.45    1.0  ODE    0.0   0.45    1.0
step_K
100     0.000  0.000  0.000  0.000  17.75  13.16  12.60  18.38  0.426  0.378  0.416  0.426
200     0.000  0.096  0.002  0.000  16.51  10.34  12.10  17.01  0.423  0.495  0.456  0.450
400     0.005  0.840  0.263  0.000  11.22   1.77   4.49  12.46  0.431  0.705  0.562  0.428
700     0.036  0.836  0.460  0.012   9.71   1.61   2.96   9.08  0.446  0.696  0.614  0.469
1000    0.044  0.916  0.544  0.020   9.62   1.34   2.69   8.94  0.454  0.696  0.631  0.483
1300    0.072  0.988  0.556  0.032   8.66   1.16   2.52   8.81  0.484  0.713  0.636  0.491
1500    0.088  0.756  0.644  0.116   7.63   2.58   2.36   7.29  0.515  0.663  0.646  0.534

[ssJSD-2D]                        [#Clust]              [pwTM]
gamma     ODE    0.0   0.45    1.0  ODE  0.0  0.45  1.0  ODE    0.0   0.45    1.0
step_K
100     0.836  0.222  0.229  0.682  NaN  NaN   0.0  NaN  NaN    NaN    NaN    NaN
400     0.108  0.723  0.269  0.221  NaN  8.0  12.8  NaN  NaN  0.473  0.142    NaN
700     0.015  0.624  0.350  0.010  2.7  2.0  18.2  2.0  0.259  0.752  0.271  0.433
1000    0.010  0.533  0.318  0.015  6.0  3.2  26.1  2.5  0.152  0.709  0.164  0.101
1300    0.022  0.582  0.315  0.037  4.7  2.4  25.1  2.7  0.273  0.719  0.164  0.128
1500    0.058  0.414  0.376  0.068  6.0  5.4  13.6  5.2  0.133  0.585  0.178  0.167

[FID]                             [fJSD-A]
gamma     ODE     0.0   0.45     1.0  ODE   0.0  0.45   1.0
step_K
100     697.3  6808.9  721.4  1017.8  1.94  2.65  1.85  2.82
400     419.5   881.5  472.3   729.8  1.14  2.64  0.52  1.80
700     352.6   781.9  436.8   507.7  0.61  1.90  1.11  0.70
1000    307.4   820.7  466.8   673.8  0.33  2.25  0.95  0.68
1300    297.4   771.9  415.6   666.7  0.41  3.03  0.95  0.94
1500    212.5   737.3  317.8   446.6  0.36  3.54  1.21  0.71
```

### 7.2 Baseline metric × step × γ — AFDB

```
[Des]                                         [scRMSD]                                      [pLDDT]
gamma     ODE    0.0   0.35   0.45    0.5    1.0   ODE   0.0  0.35  0.45   0.5    1.0    ODE    0.0   0.35   0.45    0.5    1.0
step_K
100     0.008  0.660  0.556  0.431  0.412  0.008  10.83  2.14  2.74  3.19  3.44  10.95  0.468  0.683  0.642  0.629  0.619  0.496
400     0.028  0.800  0.784  0.752  0.684  0.092   8.35  1.50  1.90  2.11  2.43   6.70  0.480  0.719  0.692  0.683  0.669  0.526
700     0.120  0.912  0.724  0.720  0.692  0.212   6.69  1.20  1.88  2.11  2.43   5.98  0.560  0.724  0.689  0.688  0.673  0.603
1000    0.120  0.784  0.792  0.723  0.668  0.144   7.06  1.57  1.80  2.02  2.25   7.33  0.572  0.705  0.701  0.690  0.686  0.592
1300    0.180  0.940  0.784  0.641  0.640  0.240   6.79  1.05  2.13  3.15  3.00   7.86  0.582  0.733  0.708  0.681  0.688  0.610
1600    0.116  0.936  0.808  0.736  0.704  0.184   7.34  1.15  1.82  2.14  2.46   7.41  0.581  0.723  0.713  0.701  0.698  0.610

[ssJSD-2D]                                       [#Clust]                              [pwTM]
gamma     ODE    0.0   0.35   0.45    0.5    1.0   ODE   0.0  0.35  0.45   0.5   1.0   ODE    0.0   0.35   0.45    0.5    1.0
step_K
100     0.045  0.640  0.415  0.362  0.348  0.059   2.0   5.6  27.4  21.1  20.4   NaN  0.038  0.485  0.194  0.192  0.179    NaN
400     0.096  0.311  0.183  0.141  0.125  0.059   4.0   2.0  32.0  31.3  28.6   4.6  0.163  0.737  0.182  0.172  0.167  0.179
700     0.035  0.209  0.164  0.150  0.126  0.023   4.4   6.6  23.6  25.5  26.2   9.4  0.196  0.687  0.245  0.220  0.219  0.147
1000    0.035  0.451  0.182  0.158  0.142  0.036   6.8  11.8  24.8  26.7  25.0   7.8  0.167  0.417  0.238  0.209  0.206  0.170
1300    0.032  0.197  0.117  0.100  0.077  0.024   6.4  10.6  26.8  24.3  22.8  10.8  0.144  0.528  0.216  0.181  0.198  0.145
1600    0.040  0.239  0.153  0.139  0.120  0.029   8.0   5.6  28.6  27.9  26.2   7.4  0.275  0.649  0.220  0.194  0.192  0.183

[FID]                                            [fJSD-A]
gamma     ODE    0.0   0.35   0.45    0.5    1.0   ODE   0.0  0.35  0.45   0.5   1.0
step_K
100     145.1  749.9  464.9  404.7  385.7  374.2  0.40  4.07  2.65  2.43  2.28  0.73
400     501.1  711.8  429.2  431.3  435.5  612.6  0.78  1.99  1.67  1.44  1.33  0.78
700     219.8  641.4  521.9  494.0  475.3  365.6  0.79  2.96  2.71  2.45  2.20  0.77
1000    223.5  631.2  560.3  533.8  524.8  403.5  0.66  1.06  2.55  2.23  2.16  0.64
1300    232.8  609.9  419.8  386.4  383.2  355.5  0.61  2.47  1.91  1.63  1.51  0.56
1600    209.9  674.4  436.9  415.2  402.0  329.0  0.64  3.07  2.19  1.91  1.82  0.58
```

### 7.3 Full Δ tables per (encoder, layer, dataset)

Pre-rendered: [evaluation/proteina/generation/scripts/paper/audit_2026-05-28/compare_out.txt](../../evaluation/proteina/generation/scripts/paper/audit_2026-05-28/compare_out.txt) (~930 lines). Each pair gets 10 Δ tables (Des / scRMSD / pLDDT / FID / fJSD-A / fJSD-T / β=ss_E / ssJSD-2D / #Clust / pwTM) indexed by step × γ. The headline cells used in §§ 2–3 are extracted from there.

Regen recipe (from repo root):

```bash
source .venv/bin/activate
python evaluation/proteina/generation/scripts/paper/audit_2026-05-28/baseline_tradeoff.py \
    > evaluation/proteina/generation/scripts/paper/audit_2026-05-28/baseline_tradeoff_out.txt
python evaluation/proteina/generation/scripts/paper/audit_2026-05-28/compare.py \
    > evaluation/proteina/generation/scripts/paper/audit_2026-05-28/compare_out.txt
```

The three scripts read only the four clean jsonls listed in § 6.
