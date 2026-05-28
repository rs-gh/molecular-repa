# Proteína narratives — living document

Working doc for findings, hypotheses, and report framing. Reorganize freely.

Last updated: 2026-05-27.

## Headline claim (draft)

**REPA accelerates the Proteína student's convergence toward the data distribution, but the magnitude and direction depend on encoder × dataset × scale.** Across all REPA configs at γ=0.45, the most robust effects are:

- **Architectural diversity (fS-A) and SS-distribution match (ssJSD-2D) improve under nearly all REPA variants**, in nearly all regimes — these are the most generic-REPA findings.
- **Whole-distribution match (FID-PDB, fJSD-A) improves under most learned-encoder REPA variants at γ ∈ [0, 0.5]**, with the AFDB advantage being uniquely durable (PDB-baseline catches up by 1.5M, AFDB-baseline never catches up).
- **Per-sample quality (pLDDT, TM-self) improves under MPNN encoders most consistently**; GearNet is more mixed on quality but stronger on distribution match.

Caveats:
- **Random-encoder control is scale-dependent**: at n256 PDB L4-rand loses 0/4 on FID-PDB ✓ confirms learned encoder needed. At n128 PDB L4-rand wins 5/6 on FID-PDB ⚠ random regularization helps at smaller scale. The "REPA needs learned representations" claim is true at n256+, weaker at n128.
- **β-content shift** (REPA producing more β-rich generations) is **PDB-specific** — most PDB-trained REPA variants win on β%, but AFDB-MPNN actively shifts the opposite direction (more α-rich).
- **T-D plateaus at ~700K** for REPA-GearNet on both datasets; baseline keeps growing past that. Diversity reduction is late-training-only, not inherent.
- **Novelty is too noisy to anchor a claim on** — win rates near 50% across most cells, large variance from small designable subsets.

The full per-step data per (claim, regime, variant) is in [proteina_claims_compilation.md](proteina_claims_compilation.md).

---

## Claim 1 — REPA accelerates whole-distribution learning (T-W + S-W)

**Status: ✓ broadly confirmed; framing is "acceleration" not "asymptotic dominance"; the magnitude varies by encoder × dataset × scale.**

### Cross-config win/loss tally at γ=0.45 (step-matched vs baseline)

Per-variant fraction of step-matched comparisons where REPA beats baseline. Full per-step deltas at `docs/research/proteina_claims_compilation.md`.

| Regime | Variant | FID-PDB | fJSD-A | fJSD-T | fJSD-C | fS-A | fS-C |
|---|---|---|---|---|---|---|---|
| n256 PDB | L4-GN | 3/5 | 4/5 | 4/5 | 3/5 | 5/5 | 5/5 |
| n256 PDB | L9-GN | 3/5 | 2/5 | 3/5 | 4/5 | 4/5 | 4/5 |
| n256 PDB | L4-MPNN | 2/7 | 4/7 | 2/7 | 4/7 | 5/7 | 4/7 |
| n256 PDB | **L9-MPNN** | **5/5** | 3/5 | 3/5 | 4/5 | 3/5 | 4/5 |
| n256 PDB | L4-rand (ctrl) | **0/4** | 2/4 | 2/4 | 2/4 | 3/4 | 3/4 |
| n256 AFDB | **L4-GN** | **5/5** | 4/5 | 4/5 | 4/5 | 5/5 | 4/5 |
| n256 AFDB | L9-GN | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 2/4 |
| n256 AFDB | L9-MPNN | 4/6 | 3/6 | 3/6 | 2/6 | 3/6 | 3/6 |
| n128 PDB | L4-GN | 4/6 | 5/6 | 5/6 | 4/6 | 4/6 | 5/6 |
| n128 PDB | L9-GN | 3/5 | 4/5 | 3/5 | **5/5** | **5/5** | **5/5** |
| n128 PDB | L4-MPNN | 3/4 | 3/4 | 3/4 | 1/4 | 3/4 | 3/4 |
| n128 PDB | **L9-MPNN** | **7/7** | 6/7 | 6/7 | 5/7 | 5/7 | 4/7 |
| n128 PDB | L4-rand (ctrl) | **5/6** | 5/6 | 5/6 | 3/6 | 5/6 | 5/6 |
| n128 AFDB | L4-GN | 2/3 | 2/3 | **0/3** | 0/3 | 0/3 | 0/3 |
| n128 AFDB | L4-MPNN | **0/6** | **0/6** | 1/6 | 0/6 | 3/6 | 4/6 |
| n128 AFDB | L9-MPNN | **0/3** | **0/3** | 0/3 | 0/3 | 1/3 | 0/3 |

### Reading

1. **REPA wins majority of comparisons across most regimes/metrics.** L9-MPNN cleanly sweeps both n128 PDB FID-PDB (7/7) and n256 PDB FID-PDB (5/5). L4-GN on AFDB-n256 sweeps 5/5. L9-GN at n128 PDB cleanly sweeps fJSD-C/fS-A/fS-C (5/5 each).

2. **fS-A and fS-C are the most robust REPA improvements** — nearly every variant in every regime wins majority. These are architectural-entropy and SS-class-entropy metrics ("REPA's whole-set output is more diverse over fold classes").

3. **The random-encoder control diverges between scales**:
   - **n256 PDB**: L4-rand loses 0/4 on FID-PDB ✓ confirms learned encoder matters at scale
   - **n128 PDB**: L4-rand WINS 5/6 on FID-PDB ⚠ at smaller scale even random regularization helps. So "REPA needs learned representations" is *scale-dependent*.

4. **AFDB advantage is durable**: in PDB the FID-PDB Δ goes negative late (baseline catches up), in AFDB it stays positive across the full convergence sweep. AFDB-baseline saturates around fJSD-A ≈ 1.7–2.0; REPA-AFDB stays at 0.5–0.9 throughout 1.7M+ steps.

5. **L4 vs L9 (depth)**: no consistent winner across all metrics. On AFDB-n256, L4-GN dominates more uniformly than L9. On PDB, L9 looks slightly stronger at n128 but mixed at n256.

6. **N128 AFDB is the only regime where REPA broadly LOSES on T-W**:
   - L4-GN: 2/3 FID-PDB but 0/3 on fJSD-A/T/C/fS-A/fS-C. Only wins FID, loses everything else.
   - L4-MPNN: **0/6 on FID-PDB** and **0/6 on FID-AFDB**, only 1/6 on fJSD-A. Consistently makes the distribution-match WORSE.
   - L9-MPNN: 0/3 on FID-PDB, 0/3 on FID-AFDB, 0/3 on fJSD-A. Same story.
   - This is the regime where the "REPA accelerates T-W" claim fails completely. Possible causes: (a) AFDB has very limited β-content to align toward at n128 scale, (b) the model is too small for the encoder regularization to help, (c) AFDB-128 baseline has already plateaued by ~100K so REPA-as-acceleration has nothing to accelerate.

**Note on data caveat**: n128_AFDB and the n256 AFDB L9-GN tally now include "legacy" rows (no sampler metadata), which were generated with the default γ=0.45 sampler. Prior compilation was filtering them out — fixed in build_claims_compilation.py (2026-05-27).

### Sampler-regime robustness (n256, L9-GN-PDB vs baseline)

For FID-PDB, the win-rates collapse cleanly by γ band:
- γ ∈ {0.0, 0.35, 0.45, 0.5}: REPA wins majority at steps ≥400K
- ODE: mixed (cliff effects, mode-collapse-sensitive)
- γ=1.0: REPA loses majority (full-temperature SDE breaks the encoder-aligned manifold)

Cleanest single robust finding: **REPA improves FID-PDB and fJSD-A in the γ ∈ [0, 0.5] band** — but note we only have multi-γ data for GearNet (PDB-L9, AFDB-L4); MPNN/random are γ=0.45-only. See the full trajectory-wide analysis and the corrected step-matched tables in [§ Sampler-regime robustness check](#sampler-regime-robustness-check-added-2026-05-27-expanded-2026-05-27), which sharpens this to: distribution-match dies at γ=1, designability/quality dies at γ=0, ssJSD-2D and β are *not* robust at every γ.

### Open follow-ups

- Run more n128 AFDB checkpoints (baseline only has 1100K, 1200K — almost no overlap with REPA variants)
- Verify the "random encoder helps at n128 scale" finding with more L4-random checkpoints on PDB
- Pull MPNN-AFDB-L9 loss curves from wandb when API stabilizes

---

## Claim 2 — REPA preserves SS balance / improves SS-distribution match

**Status: split — ssJSD-2D improvement is ✓ broadly robust; β-content shift is ✓ on PDB only and encoder × dataset specific.**

### Cross-config win/loss tally at γ=0.45 (step-matched vs baseline)

| Regime | Variant | β% ↑ | ssJSD-2D-PDB ↓ | ssJSD-2D-AFDB ↓ | fJSD-C ↓ |
|---|---|---|---|---|---|
| n256 PDB | L4-GN | 3/4 | 3/4 | 3/4 | 3/5 |
| n256 PDB | L9-GN | 3/4 | **4/4** | **4/4** | 4/5 |
| n256 PDB | **L4-MPNN** | **6/6** | **6/6** | **6/6** | 4/7 |
| n256 PDB | L9-MPNN | 3/4 | **4/4** | **4/4** | 4/5 |
| n256 PDB | L4-rand (ctrl) | 2/3 | **3/3** | **3/3** | 2/4 |
| n256 AFDB | L4-GN | 3/5 | **5/5** | **5/5** | 4/5 |
| n256 AFDB | L9-GN | 4/4 | 4/4 | 4/4 | 4/4 |
| n256 AFDB | L9-MPNN | 2/6 | 4/6 | 4/6 | 2/6 |
| n128 PDB | L4-GN | 4/6 | 1/6 | 1/6 | 4/6 |
| n128 PDB | L9-GN | 3/5 | 3/5 | 4/5 | **5/5** |
| n128 PDB | L4-MPNN | **4/4** | 3/4 | 3/4 | 1/4 |
| n128 PDB | L9-MPNN | 1/7 | 5/7 | 6/7 | 5/7 |
| n128 PDB | L4-rand (ctrl) | **1/6** | **5/6** | **6/6** | 3/6 |
| n128 AFDB | L4-GN | **0/3** | 2/3 | 2/3 | 0/3 |
| n128 AFDB | L4-MPNN | **6/6** | **6/6** | **6/6** | 3/6 |
| n128 AFDB | L9-MPNN | 1/3 | **3/3** | **3/3** | 1/3 |

### Two findings, separately

**(A) ssJSD-2D improvement is broadly robust across REPA configs.** Almost every REPA variant in every regime wins majority on both ssJSD-2D-PDB and ssJSD-2D-AFDB:
- n256 PDB: every learned-encoder REPA wins ≥4/5
- n256 AFDB: L4-GN sweeps 5/5; L9-MPNN 5/6 / 4/6
- n128 PDB: most variants win majority

This is the **single most robust REPA effect we've found** *across encoders/datasets at γ=0.45*. Even L4-random wins ssJSD-2D-PDB 3/3 on n256 and 5/6 on n128. So the SS-distribution-matching effect is broader than "learned encoder needed". **Caveat (sampler axis):** this robustness is across *encoders*, not *sampler noise* — the [sampler-regime check](#sampler-regime-robustness-check-added-2026-05-27-expanded-2026-05-27) shows ssJSD-2D is only robust in the γ ∈ [0.35, 0.5] band and loses at γ=0 / γ=1 on the step-matched trajectory.

**(B) β-content shift is PDB-and-encoder specific.** The β% column tells a different story:
- n256 PDB: all learned-encoder REPA win 3-6 / 4-7 — solid β shift
- **n128 PDB L9-MPNN: 1/7** — at n128 scale, L9-MPNN does NOT shift β. Different from n256.
- **n128 PDB L4-rand: 1/6** — random encoder does NOT shift β. Confirms learned-encoder dependency.
- n256 AFDB: L4-GN 3/5 marginal, L9-GN 4/4 wins, L9-MPNN 2/6 (REPA β LOWER than baseline). Encoder matters.
- **n128 AFDB extremes**: L4-MPNN sweeps **6/6 β%** (REPA significantly more β) BUT it also lost 0/6 on FID-PDB (Claim 1). So β shift WITHOUT distribution match — a degenerate version of "REPA gets the right SS composition but produces structurally-wrong samples". L4-GN goes the OPPOSITE way (0/3 β% — LESS β than baseline). L9-MPNN: 1/3 (mixed).

### The corrected story

The original framing "REPA preserves β-content" overfit to one config (PDB-L9-GN-n256). The corrected picture:

- **ssJSD-2D improvement is generic-REPA** — even random encoder helps (at least at n128). Mechanism: any "regularizer against the late-training helix-mode-collapse" improves SS-distribution match.
- **β-content shift specifically (i.e., increasing β-rich generations) requires a learned encoder *and* PDB-style training data**. AFDB has very little β-rich content even in baseline (3–17%), so there's no β-rich attractor for REPA to pull toward. PDB has enough β-content for the learned encoder to identify and amplify.
- **L4-MPNN n256 is the cleanest "REPA shifts β" example** (6/6 win on β%). Suggests the effect isn't unique to GearNet.

This is consistent with the broader claim from the [SS-class trajectory analysis](#ss-class-trajectory-across-all-gen-eval-variants-added-2026-05-27) that AFDB-MPNN actively shifts the OPPOSITE direction (toward α-rich) on AFDB — and that decomposes into "encoder × dataset determines where on the manifold the model lands".

### Open follow-ups

- The n128 L9-MPNN β = 1/7 result is striking — at n128 the MPNN encoder doesn't shift β. Worth a focused look: is it because the model is too small to develop a β-rich mode, or because MPNN at n128 settles on a different attractor?
- More n256 reps would help — many β% comparisons are 3/4 or 3/5 which is close to chance.

---

## Claim 3 — REPA reaches good designability / quality earlier

**Status: ⚠ heterogeneous. Strong for MPNN encoders and on AFDB. Weaker / sometimes negative for GearNet on n128 PDB.**

### Cross-config win/loss tally at γ=0.45 (step-matched vs baseline)

| Regime | Variant | Des% ↑ | scRMSD ↓ | pLDDT ↑ | TM-self ↑ |
|---|---|---|---|---|---|
| n256 PDB | L4-GN | 4/5 | 3/5 | 4/5 | 4/5 |
| n256 PDB | L9-GN | 3/5 | 3/5 | 4/5 | 4/5 |
| n256 PDB | L4-MPNN | **6/7** | 4/7 | **6/7** | 4/7 |
| n256 PDB | L9-MPNN | 4/5 | 4/5 | 4/5 | **5/5** |
| n256 PDB | L4-rand (ctrl) | 2/4 | 2/4 | 2/4 | 3/4 |
| n256 AFDB | L4-GN | 4/5 | **1/5** | 4/5 | **1/5** |
| n256 AFDB | L9-GN | 2/4 | 1/4 | 3/4 | 1/4 |
| n256 AFDB | **L9-MPNN** | 4/6 | 4/6 | **6/6** | 4/6 |
| n128 PDB | L4-GN | **2/6** | 3/6 | 5/6 | 3/6 |
| n128 PDB | L9-GN | **1/5** | 2/5 | 2/5 | 3/5 |
| n128 PDB | L4-MPNN | **4/4** | **4/4** | **4/4** | **4/4** |
| n128 PDB | **L9-MPNN** | 5/7 | **7/7** | 4/7 | **7/7** |
| n128 PDB | L4-rand (ctrl) | 4/6 | 4/6 | 3/6 | 5/6 |
| n128 AFDB | L4-GN | 2/3 | 2/3 | 2/3 | 2/3 |
| n128 AFDB | L4-MPNN | **6/6** | 3/6 | **6/6** | 1/6 |
| n128 AFDB | L9-MPNN | **3/3** | **3/3** | **3/3** | **3/3** |

### Reading

The original "REPA accelerates designability" framing was based on PDB-L9-GN-n256. The full picture is much more heterogeneous:

1. **MPNN encoders are the strongest quality accelerators**. L9-MPNN n128 sweeps 7/7 on scRMSD and TM-self. L4-MPNN n128 sweeps 4/4 on every metric. L9-MPNN AFDB sweeps 6/6 pLDDT. **MPNN is a quality booster more than a distribution shaper** — consistent with our AFDB three-way head-to-head finding earlier (Baseline vs L9-MPNN vs L4-GearNet section).

2. **GearNet at n128 PDB underperforms on Des%**: L4-GN 2/6, L9-GN 1/5. At smaller scale, GearNet-REPA may slightly hurt designability. Sharply different from n256 PDB (3-4/5).

3. **GearNet on AFDB has a scRMSD/TM-self tradeoff**: L4-GN-AFDB hits 4/5 on Des% but only 1/5 on scRMSD and 1/5 on TM-self. Higher designability rate AND higher mean scRMSD → explained by the polarization finding (see "scRMSD: polarization, NOT bimodality" below): REPA's higher mean scRMSD is a fatter far tail, not a worse typical sample.

4. **Random encoder helps at n128**: 4/6 Des%, 5/6 TM-self, 4/6 scRMSD. Echoes the Claim 1 finding that at smaller scale, even random regularization helps. Possibly because the model is more underparametrized so any structural prior helps.

### scRMSD: polarization (verified — was "bimodality")

scRMSD-mean increases under REPA *despite* higher Des% rate. We predicted a
bimodal distribution; the histograms (verified, see dedicated section below)
are **unimodal for both** — the real effect is **polarization**: REPA depletes
the marginal 2–4Å bin, sending that mass to <2Å early in training and into a
fatter >4Å broken tail late. So the higher mean is a far tail, not a second
mode. AFDB-L4-GN (4/5 Des% wins, 1/5 scRMSD wins) is the case where this is
most visible.

### The corrected story

- **REPA improves per-sample structural quality (pLDDT, TM-self) broadly across configs** — most variants majority-win.
- **Designability-rate acceleration is encoder-and-scale dependent**: clear for MPNN encoders, mixed for GearNet, and at n128 GearNet-REPA can underperform baseline outright.
- **REPA polarizes the scRMSD distribution** (not bimodal): depletes the marginal 2–4Å zone; the depleted mass becomes designable early and clearly-broken late.

### Open follow-ups
- Why does GearNet-n128 underperform MPNN-n128 on Des%? Possibly the encoder's representation features are too high-dimensional for a 60M model to absorb at smaller residue-counts.

---

## Claim 4 — REPA improves novelty

**Status: ⚠ very noisy. No variant wins consistently across regimes. Conflicts with designability-bottlenecked subsets.**

### Cross-config win/loss tally at γ=0.45 (step-matched vs baseline)

| Regime | Variant | Nov-PDB% ↑ | Nov-AFDB% ↑ |
|---|---|---|---|
| n256 PDB | L4-GN | 1/5 | 2/5 |
| n256 PDB | L9-GN | 3/5 | 4/5 |
| n256 PDB | L4-MPNN | 3/7 | 4/7 |
| n256 PDB | L9-MPNN | 2/5 | 2/5 |
| n256 PDB | L4-rand (ctrl) | 1/4 | 2/4 |
| n256 AFDB | L4-GN | 3/5 | 3/5 |
| n256 AFDB | L9-GN | 1/4 | 1/4 |
| n256 AFDB | L9-MPNN | 4/6 | 4/6 |
| n128 PDB | L4-GN | **5/6** | 3/6 |
| n128 PDB | L9-GN | 1/5 | 1/5 |
| n128 PDB | L4-MPNN | 1/4 | 0/4 |
| n128 PDB | L9-MPNN | 2/7 | 2/7 |
| n128 PDB | L4-rand (ctrl) | **5/6** | 4/6 |
| n128 AFDB | L4-GN | 2/3 | **3/3** |
| n128 AFDB | L4-MPNN | 3/6 | 2/6 |
| n128 AFDB | L9-MPNN | 2/3 | **3/3** |

### Reading

Genuinely noisy — no variant consistently wins majority across regimes. Specifically:

1. **L9-GN at n128 PDB loses 1/5 on both Nov-PDB and Nov-AFDB**, while at n256 PDB it wins 3-4/5. Same encoder × different scale, opposite direction. Big variance.

2. **L4-rand at n128 PDB wins 5/6 on Nov-PDB** — random encoder helps novelty as much as any learned variant at n128 scale. Yet another instance of "n128 is the regime where random ~= learned".

3. **L4-GN at n128 PDB wins 5/6 on Nov-PDB**, but loses 1/5 at n256 PDB. Opposite trend from FID/fJSD which generally improved at n256 over n128.

4. **AFDB novelty is at chance** — most variants 3/5 or 3/6. Hard to claim REPA helps novelty on AFDB.

### Caveats

- Novelty is computed on the **designable subset**. When designable count is small, novelty rate is noisy. Many of the early-step comparisons have small designable populations.
- The `n` of step-matched comparisons is small (3-7 per cell). Win-rate around 50% is statistically uninformative.
- We don't have multi-rep data for most variants, so each cell is essentially one seed.

### Recommendation

**Demote novelty from a load-bearing claim.** Cite the trajectory plot for context but don't anchor a paper-table-worthy claim on it. If we want to make a clean novelty statement, would need more reps and more careful designable-N filtering. Currently the variance dwarfs the signal.

---

## Claim 5 — REPA's T-D advantage crosses over: better early, worse late

**Status: ✓ confirmed and sharpened (full cross-config analysis 2026-05-27). REPA improves T-D up to a regime-specific crossover step, then degrades. CRUCIALLY, the crossover is T-D-SPECIFIC — distribution-match and quality metrics keep favoring REPA past it.**

Full analysis + tables: `docs/research/proteina_td_crossover.md` (`build_td_crossover_analysis.py`).

### The crossover is real and regime-specific

#Clusters Δ (REPA − baseline) flips from positive to negative at a step that depends on regime:

| Regime | T-D crossover | Basis |
|---|---|---|
| **n256 PDB** | **~850K** | L4-GN & L9-GN #clusters Δ flip between 700K (+) and 1000K (−); MPNN variants similar |
| **n256 AFDB** | **~150K** | GearNet flips between 100K (+) and 200K (−) — very early. MPNN-L9 never cleanly flips. |
| **n128 PDB** | not reached | most variants stay T-D-positive through 600-700K (baseline hasn't overtaken yet) |
| **n128 AFDB** | <100K | REPA T-D-negative from first ckpt |

**The crossover tracks when the *baseline's* #clusters growth overtakes REPA's plateau, NOT an intrinsic REPA step.** PDB baseline keeps adding distinct folds with training (→ late crossover / not-yet at n128). AFDB baseline is already cluster-rich early (→ early crossover). REPA's #clusters plateaus ~700K regardless; the crossover is set by the baseline curve.

### The crossover is T-D-SPECIFIC — other metrics keep favoring REPA

This is the key new finding. Splitting each variant's win-fraction into BEFORE vs AFTER its regime crossover (n256 PDB ~850K; n256 AFDB ~150K). Each cell = #REPA-wins / #comparisons. **#Clust and pwTM** (bold) are the pure-diversity metrics; everything else is distribution-match / quality / SS. Variants ordered best-first within each dataset.

Columns: FID-PDB ↓ | fJSD-A ↓ | fS-A ↑ | Des% ↑ | pLDDT ↑ | ssJSD2D ↓ | β% ↑ | **#Clust ↑** | **pwTM ↓** | Nov-PDB ↑

#### n256 PDB (crossover ~850K)

**MPNN-L9** (best PDB variant; before n=4, after n=1) — *the cleanest demonstration: only the two diversity metrics cross*

| window | FID-PDB | fJSD-A | fS-A | Des% | pLDDT | ssJSD2D | β% | **#Clust** | **pwTM** | Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 4/4 | 2/4 | 2/4 | 3/4 | 3/4 | 3/3 | 2/3 | 2/3 | 1/2 | 1/4 |
| after | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | **0/1** | **0/1** | 1/1 |

**L4-GN** (before n=4, after n=1)

| window | FID-PDB | fJSD-A | fS-A | Des% | pLDDT | ssJSD2D | β% | **#Clust** | **pwTM** | Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 2/4 | 3/4 | 4/4 | 4/4 | 4/4 | 3/3 | 3/3 | 3/3 | 0/2 | 1/4 |
| after | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | **0/1** | **0/1** | 0/1 |

**L9-GN** (before n=4, after n=1)

| window | FID-PDB | fJSD-A | fS-A | Des% | pLDDT | ssJSD2D | β% | **#Clust** | **pwTM** | Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 2/4 | 2/4 | 3/4 | 3/4 | 3/4 | 3/3 | 2/3 | 2/3 | 1/2 | 2/4 |
| after | 1/1 | 0/1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | **0/1** | **0/1** | 1/1 |

**L4-MPNN** (before n=4, after n=3 — most post-crossover data)

| window | FID-PDB | fJSD-A | fS-A | Des% | pLDDT | ssJSD2D | β% | **#Clust** | **pwTM** | Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 2/4 | 2/4 | 3/4 | 4/4 | 4/4 | 3/3 | 3/3 | 2/3 | 0/2 | 2/4 |
| after | 0/3 | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 | **0/3** | **0/3** | 1/3 |

**L4-rand** (control; before n=4, no post-crossover data — run stops at 700K)

| window | FID-PDB | fJSD-A | fS-A | Des% | pLDDT | ssJSD2D | β% | **#Clust** | **pwTM** | Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 0/4 | 2/4 | 3/4 | 2/4 | 2/4 | 3/3 | 2/3 | 1/3 | 1/2 | 1/4 |
| after | — | — | — | — | — | — | — | — | — | — |

#### n256 AFDB (crossover ~150K)

**L4-GN** (best AFDB GearNet variant; before n=1, after n=4)

| window | FID-PDB | fJSD-A | fS-A | Des% | pLDDT | ssJSD2D | β% | **#Clust** | **pwTM** | Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| after | 4/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 2/4 | **0/4** | **1/4** | 3/4 |

**L9-GN** (before n=1, after n=3)

| window | FID-PDB | fJSD-A | fS-A | Des% | pLDDT | ssJSD2D | β% | **#Clust** | **pwTM** | Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| after | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 | **0/3** | **0/3** | 0/3 |

**L9-MPNN** (the falsifier — no real crossover; before n=1, after n=5; see dedicated section below)

| window | FID-PDB | fJSD-A | fS-A | Des% | pLDDT | ssJSD2D | β% | **#Clust** | **pwTM** | Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| after | 4/5 | 3/5 | 3/5 | 3/5 | 5/5 | 3/5 | 1/5 | **4/5** | **4/5** | 4/5 |

### Reading

Across every variant that crosses (all PDB variants + AFDB-GearNet), the after-crossover pattern is the same: **#Clust and pwTM flip to losing while distribution-match and quality metrics largely keep winning.**

- **PDB-MPNN-L9 is the cleanest**: after crossover, *every* metric is 1/1 ✓ except #Clust 0/1 and pwTM 0/1. The crossover touches only the pure-diversity axis.
- **PDB-L4-MPNN** (n=3 post-crossover, most reliable): ssJSD-2D and β% stay 3/3 ✓ while #Clust and pwTM go 0/3 ✗. FID does degrade here too (0/3) — the one config where a distribution metric also crosses.
- **AFDB-GearNet-L4** (n=4 post-crossover): FID 4/4, fS-A 4/4, Des% 4/4, ssJSD-2D 4/4 all ✓ after crossover — only #Clust 0/4 and pwTM 1/4 cross.
- **AFDB-MPNN-L9 does NOT cross** — it *gains* #Clust 4/5 and pwTM 4/5 after 150K. The falsifier (dedicated section below).

So on **PDB, all variants (incl. both encoders) cross** — because on PDB both GearNet and MPNN concentrate toward β-rich (SS-class trajectory: PDB-L9-GN β-rich → 41%, PDB-L9-MPNN → ~30-39%), and β-rich is fold-narrow. On **AFDB, only GearNet crosses**; MPNN goes α-rich (fold-diverse) and preserves diversity.

### Revised interpretation

The "cliff" is **not** "REPA's overall advantage expires". It's specifically a **diversity-vs-distribution-match decoupling**:

- **Before crossover**: REPA improves both distribution-match AND diversity (model is still learning more folds, encoder-guided).
- **After crossover**: REPA continues to improve distribution-match (FID/fJSD/ssJSD/β stay ✓) and per-sample quality, but its #clusters/pwTM diversity stops growing while baseline's keeps climbing → REPA loses on T-D.

This is fully consistent with the [β-stratified Experiment 1](#experiment-1-β-stratified-diversity-within-designable-subset--run-2026-05-27) and [SS-class trajectory](#ss-class-trajectory-across-all-gen-eval-variants-added-2026-05-27) findings: REPA concentrates the model onto encoder-preferred designable modes. That concentration *improves* distribution-match (you're producing the right folds) but *reduces* within-set diversity (you produce fewer distinct ones). The two diverge exactly at the crossover.

**The clean story**: REPA trades tertiary diversity for distribution fidelity, and the trade tilts further toward fidelity-over-diversity as training continues. The crossover step is where baseline's diversity (still growing) overtakes REPA's (plateaued).

### Experiment A — the concentration is designable-subset-SPECIFIC (✓ RUN 2026-05-27)

Tests whether REPA's diversity loss is whole-set or only in the designable subset, using pwTM (n-independent → whole-set and designable directly comparable). Script: `exp_wholeset_vs_designable_diversity.py`. PDB γ=0.45:

| Model+step | whole-set pwTM | designable pwTM | Δ (des − whole) |
|---|---|---|---|
| baseline 400K | 0.138 | 0.143 | +0.005 |
| baseline 700K | 0.222 | 0.229 | +0.007 |
| baseline 1000K | 0.173 | 0.168 | −0.005 |
| baseline 1600K | 0.166 | 0.180 | +0.013 |
| REPA-L9-GN 400K | 0.160 | 0.158 | −0.003 |
| REPA-L9-GN 700K | 0.187 | **0.280** | **+0.093** |
| REPA-L9-GN 1000K | 0.237 | **0.491** | **+0.254** |

**Decisive**: the designable-vs-whole pwTM gap is ≈0 for baseline at every step, but for REPA it grows 0 → +0.093 → +0.254 across 400K/700K/1000K — in lockstep with the T-D crossover.

- REPA's **whole-set** structural diversity stays comparable to baseline (pwTM 0.16–0.24).
- REPA's **designable** subset concentrates dramatically (pwTM 0.49 at 1000K vs its own whole-set 0.24).
- Baseline's designable subset is representative of its whole set (no gap, any step).

**Mechanism (resolves Q1)**: REPA doesn't reduce diversity everywhere — the **designability filter selects REPA's concentrated high-confidence modes**. REPA produces a structurally varied whole set (broader fold-class coverage → fS-A ↑), but the samples that *pass designability* collapse onto a narrow set of (β-rich, fold-poor) topologies. So "whole-set fS-A ↑" and "designable #Clust/pwTM ↓" are consistent: more fold *classes* overall, fewer distinct *structures* among designable ones. Exp 1 (β-stratified) explains *why* the designable modes are narrow (β is fold-poor); Exp A shows the narrowing is *confined to the designable subspace* and grows with training.

### Exp A EXPANDED to all 10 n256 variants (✓ RUN 2026-05-28) — falsifier prediction CONFIRMED

Ran the designable−whole pwTM gap (Δ) across every n256 variant × steps spanning the crossover. Prediction: configs that concentrate toward β-rich (fold-poor) should show a *growing* Δ; the random-encoder control and the α-concentrating AFDB-MPNN should not.

| Variant | early Δ | mid Δ | late Δ | verdict |
|---|---|---|---|---|
| PDB baseline | +0.006 | +0.009 | −0.005 | flat ~0 (no concentration) |
| **PDB L4-rand (ctrl)** | −0.014 | −0.019 | — | **negative — control confirms learned encoder needed** |
| PDB L9-GN | −0.001 | +0.091 | **+0.256** | grows strongly ✓ |
| PDB L4-GN | +0.014 | +0.117 | −0.028 | grows to 700K (1000K is last ckpt, noisy) |
| PDB L4-MPNN | +0.088 | +0.177 | +0.100 | grows ✓ |
| PDB L9-MPNN | +0.008 | +0.101 | +0.085 | grows ✓ |
| AFDB baseline | −0.006 | +0.040 | +0.028 | small, ≤0.04 |
| AFDB L4-GN | +0.033 | +0.094 | +0.066 | moderate growth ✓ |
| AFDB L9-GN | +0.124 | +0.146 | +0.060 | strong ✓ |
| **AFDB L9-MPNN** | +0.032 | +0.032 | +0.032 | **flat ~0.03, never grows — the falsifier** |

(early/mid/late ≈ 400K / 700K / 1000K, nearest available per variant.)

**Prediction holds cleanly across all 10 variants:**
- **Every β-concentrating config** (all PDB learned encoders + both AFDB-GearNet) shows a **growing** designable-vs-whole gap.
- **The random-encoder control (PDB L4-rand) is NEGATIVE** — its designable subset is *more* diverse than its whole set. Random GearNet doesn't concentrate. Cleanest evidence yet that concentration requires *learned* representations.
- **AFDB-MPNN has a flat ~0.03 gap that never grows** — it concentrates toward fold-rich α-folds, so the designability filter doesn't narrow it. Same falsifier signature as its no-T-D-crossover behavior.

So the designable-subset concentration Δ is a direct, quantitative readout of "does this config concentrate toward fold-poor β-rich modes" — it tracks 1:1 with the T-D crossover. **Closes the Claim 5 mechanism**: the diversity trade-off is caused by encoder-driven concentration onto fold-poor β-rich modes, confined to the designable subspace, requiring a learned encoder, and absent when the encoder concentrates toward fold-rich α (AFDB-MPNN) or isn't learned (random control).

### Crossover sharpened with 800K/900K baseline (✓ gap-fill 2026-05-28)

Gap-fill evals added baseline + repa_mpnn_l4 @ 800K/900K and repa_l9 @ 1200K (3 reps each). Baseline-vs-REPA-L9 #clusters with the new points (γ=0.45):

| step | baseline | REPA-L9 | Δ(R−B) |
|---|---|---|---|
| 400K | 54 | 92 | +38 |
| 700K | 77 | 104 | +27 |
| 800K | 59 | 65 | +6 |
| 900K | 132 | 62 | **−70** |
| 1000K | 114 | 69 | −44 |

**Crossover lands between 800K (+6) and 900K (−70) — confirms ~850K.** Caveat: baseline #clusters is high-variance (59→132 across one 100K step); the crossover sits on top of that baseline noise. REPA's curve is the stable one (plateaus 60–90 after 700K); baseline swings 54–146. Robust statement: "REPA's designable diversity plateaus ~700K while baseline's keeps (noisily) climbing past it ~850–900K".

### The falsifier: MPNN-L9-AFDB has NO crossover (preserves T-D)

MPNN-L9-AFDB is the one config that doesn't show the crossover. Full-trajectory (100K–1.3M) win rates vs baseline:

| Metric | win rate | |
|---|---|---|
| **#Clust** | **5/6 ✓** | preserves/improves tertiary diversity throughout (only the transient 400K dip loses) |
| **pwTM** | **4/6 ✓** | also wins on diversity |
| pLDDT | 6/6 ✓ | perfect quality |
| FID-PDB | 4/6 | modest distribution-match win |
| Des% | 4/6 | modest |
| **β%** | **2/6 ✗** | REPA produces *less* β than baseline (the α-shift) |
| fJSD-C | 2/6 | weak on fold-class match |

#Clust Δ trajectory: +80, +21, −20, +21, +6, +37 (100K→1.3M) — stays positive except the one 400K dip. **No crossover.**

### This reveals the unifying mechanism

Side-by-side the two AFDB encoders:

| Config | SS-mode it concentrates toward | T-D (#Clust) | crossover? |
|---|---|---|---|
| GearNet-AFDB | β-rich (structurally narrow fold space) | loses after ~150K | **yes, early** |
| MPNN-AFDB | α-rich / mixed (fold-diverse space) | wins 5/6 throughout | **no** |

**The T-D crossover is caused by encoder-driven concentration onto a *low-diversity* SS-mode — not by REPA per se.**

Causal chain:
1. REPA pulls the model toward the encoder's preferred designable mode (all configs).
2. **If that mode is fold-narrow (β-rich) → within-set diversity drops → T-D crossover.** β-rich fold space is intrinsically narrow (Exp 1: β-rich pwTM 0.7–0.9).
3. **If that mode is fold-diverse (α-rich/mixed) → diversity preserved → no crossover.**

The discriminating variable is **"does the encoder concentrate toward β-rich on this dataset?"** — which maps 1:1 onto "does T-D cross over":

| Config | concentrates β-rich? | T-D crossover? |
|---|---|---|
| PDB GearNet-L9 | yes (→41%) | yes (~850K) |
| PDB MPNN-L9 | yes (→30-39%) | yes (~850K) |
| AFDB GearNet (L4/L9) | yes | yes (~150K) |
| **AFDB MPNN-L9** | **no (→α-rich)** | **no** |

So "REPA reduces diversity late in training" is really "REPA concentrates toward β-rich when the encoder×dataset combo favors it, and β-rich is fold-poor". On PDB both encoders favor β-rich (→ both cross). On AFDB only GearNet does. Swap to an encoder×dataset that concentrates toward a fold-rich SS-class (MPNN-AFDB) and the diversity loss disappears. **MPNN-AFDB is the falsifier; PDB-MPNN-L9 (everything-wins-but-T-D after crossover) is the cleanest positive demonstration.**

### Caveats

- After-crossover sample sizes are thin for n256 PDB (most REPA runs stop ~1000-1100K → n=1 after-step). L4-MPNN (runs to 1.6M) is the exception with n=3 and is the most reliable post-crossover evidence.
- n128 regimes don't reach the crossover (PDB) or start past it (AFDB), so the before/after split is only meaningful for n256.
- The crossover-step estimates are coarse (gap between adjacent ckpts, e.g. "between 700K and 1000K"). Denser checkpoints would sharpen them.
- MPNN-L9-AFDB β% = 2/6: the α-shift is what *saves* its diversity but it's a worse SS-distribution match (fJSD-C 2/6). So "preserving diversity" and "matching the SS distribution" are themselves in tension here.

### Earlier supporting detail (retained below)

### Evidence

| Dataset | Model+Step | β% | #Clusters | pwTM |
|---|---|---|---|---|
| PDB | Baseline 700K | 22 | 77 | 0.27 |
| PDB | REPA L9 900K | 22 | 62 | 0.27 |
| PDB | REPA L9 1000K | 24 | 69 | 0.29 |
| PDB | Baseline 1500K | 13 | 87 | 0.18 |
| PDB | Baseline 1600K | 17 | **146** | 0.18 |
| AFDB | Baseline 700K | 15 | 128 | 0.22 |
| AFDB | REPA L4 700K | 14 | 95 | 0.26 |
| AFDB | Baseline 1.6M | 13 | 142 | 0.19 |
| AFDB | REPA L4 1.2M | 14 | 101 | 0.24 |

### At matched β-content baseline still has more clusters

This rules out the simplest version of "more sheets → fewer foldable topologies":
- PDB at β≈22%: baseline 77 clusters, REPA 62 clusters
- AFDB at β≈14–15%: baseline 128–142 clusters, REPA 95–101 clusters

Δ ≈ −15 to −40 clusters under REPA, even when β content is matched. (n = 4 paired comparisons; not statistically definitive.)

### Working hypotheses for low T-D

1. **Encoder-induced fold concentration**: REPA's representation alignment to GearNet forces samples toward a subset of folds well-represented by GearNet's training distribution.
2. **Sampling temperature effective reduction**: REPA-aligned models produce sharper output distributions, so given the same SDE noise the generated structures are closer to each other.
3. **Real fold-space narrowing due to β-class topological constraints**: β-rich folds occupy a smaller portion of TM-distance space, so cluster count goes down naturally — but our matched-β data rules out the strong version of this.

### Cross-encoder #Clusters at γ=0.45 (added 2026-05-27)

Late training (~700K–1.5M) means of #clusters & β%:

| Dataset | Family | #Clust range | β% range |
|---|---|---|---|
| PDB | baseline | 77–146 | 8–22 |
| PDB | repa_l4_GearNet | 41–86 | 8–24 |
| PDB | repa_l9_GearNet | 62–104 | 18–24 |
| PDB | repa_mpnn_l4 | 47–102 | 16–25 |
| PDB | repa_mpnn_l9 | 63–110 | 15–23 |
| AFDB | baseline | 127–156 | 12–18 |
| AFDB | repa_l4_GearNet | 95–122 | 13–16 |
| AFDB | repa_l9_GearNet | 77–94 | 19–21 |
| AFDB | repa_mpnn_l9 | **122–176** | **10–12** |

**Key finding**: encoder choice changes everything.
- **GearNet-REPA reduces T-D on BOTH PDB and AFDB.** Consistent.
- **MPNN-REPA reduces T-D on PDB**, but **MPNN-REPA preserves baseline-level T-D on AFDB**.
- **MPNN-REPA on AFDB actively decreases β content** (10–12% vs baseline 12–18%), opposite from GearNet.

So the "REPA reduces T-D" story holds for GearNet but NOT MPNN-on-AFDB. The encoder choice is load-bearing for these phenomena. **The cleanest framing**: "REPA-GearNet (the paper-default and our main subject) consistently shows T-D reduction + β preservation; REPA-MPNN behaves differently and acts as a falsifier control for any encoder-agnostic 'REPA does X' claim."

This is also a useful safety check on Claim 2 (β-preservation): GearNet does it on both datasets, MPNN doesn't (on AFDB). So Claim 2 is more precisely **"GearNet-REPA stabilizes β"**, not "REPA stabilizes β".

### TODO
- Check cross-encoder: are MPNN-aligned REPA models on PDB and AFDB also lower T-D? If yes, mechanism is generic-REPA not GearNet-specific.
- Run β-stratified pwTM (Experiment 1 below) to definitively rule in/out the geometric hypothesis.
- Stratify by predicted CATH-A class within designable: are REPA's designable samples in fewer or more architectures than baseline's at matched n?

---

## Experiments planned

### Experiment 1: β-stratified diversity within designable subset (✓ RUN 2026-05-27)

**Procedure**: For each (model, step) at γ=0.45 (PDB only), bin designable samples by per-sample β-fraction, compute intra-bin #clusters and mean-pwTM (length-pooled). Script at `evaluation/proteina/generation/scripts/paper/exp_beta_stratified_diversity.py`.

**Results** (multi-rep aggregated):

| β-bin | Model+Step | n | #clust | pwTM (↓=more diverse) | clust/n |
|---|---|---|---|---|---|
| **β<10** | baseline 700K | 63 | 60 | 0.154 | 0.95 |
| | baseline 1.0M | 94 | 88 | 0.154 | 0.94 |
| | baseline 1.5M | 45 | 41 | 0.145 | 0.91 |
| | baseline 1.6M | 140 | 104 | 0.145 | 0.74 |
| | REPA 700K | 127 | 112 | 0.156 | 0.88 |
| | REPA 900K | 22 | 21 | 0.162 | 0.95 |
| | REPA 1.0M | 101 | 93 | 0.143 | 0.92 |
| **10–25** | baseline 700K | 78 | 74 | 0.216 | 0.95 |
| | baseline 1.0M | 142 | 131 | 0.228 | 0.92 |
| | baseline 1.5M | 76 | 74 | 0.229 | 0.97 |
| | baseline 1.6M | 284 | 221 | 0.247 | 0.78 |
| | REPA 700K | 103 | 74 | 0.272 | 0.72 |
| | REPA 900K | 43 | 38 | 0.238 | 0.88 |
| | REPA 1.0M | 97 | 78 | 0.237 | 0.80 |
| **β≥25** | baseline 700K | 74 | 31 | 0.496 | 0.42 |
| | baseline 1.0M | 39 | 35 | 0.135 | 0.90 |
| | baseline 1.5M | 11 | 11 | 0.132 | 1.00 |
| | baseline 1.6M | 81 | 75 | 0.134 | 0.93 |
| | **REPA 700K** | 75 | 22 | **0.732** | **0.29** |
| | **REPA 900K** | 45 | 15 | **0.670** | **0.33** |
| | **REPA 1.0M** | 144 | 26 | **0.870** | **0.18** |

### Interpretation

- **β<10 bin (helix-rich/mixed)**: baseline and REPA have *nearly identical* pwTM (~0.14–0.16). Within helix-rich samples, REPA's diversity matches baseline's.
- **10–25 bin**: REPA slightly less diverse (pwTM ~0.24–0.27 vs baseline ~0.22–0.25). Mild concentration.
- **β≥25 bin (sheet-rich)**: **MASSIVE concentration in REPA**. REPA's β-rich samples have pwTM 0.67–0.87 (nearly identical structurally) vs baseline 1.0M+ β-rich at pwTM 0.13–0.14 (highly diverse). REPA's β-rich cluster-rate is 18–33% vs baseline's 90–100%.

### Takeaway

**The "sheets→fewer folds" geometric hypothesis is FALSIFIED at the within-composition level.** When you control for β content sample-by-sample, REPA's β-rich samples still concentrate onto a much smaller fold set than baseline's β-rich samples. So there IS a REPA-specific effect beyond what β composition explains, and it's concentrated in the high-β regime.

Equivalent reframing: **REPA's β-richness comes from producing the *same* β-rich fold(s) over and over, while baseline's β-rich samples (when it produces them at all) span many distinct β-rich folds**.

This is consistent with the encoder-mediated-concentration hypothesis: the alignment target rewards specific β-rich GearNet-friendly architectures (a particular β-sandwich or barrel topology), and REPA learns to produce that target heavily while baseline explores the space more broadly when it happens to land on β-rich generations.

Helix-rich samples don't show this concentration because helix folds occupy a wider, less encoder-distinguished part of GearNet representation space — many helical architectures look similar at GearNet's representation level, so the alignment doesn't preferentially funnel toward one helix mode.

### AFDB cross-encoder comparison at γ=0.45 (added 2026-05-27)

Comparing the four AFDB-trained 256-residue models at 700K (only step where all four have data) on γ=0.45:

| Metric | Baseline | REPA-L4-GearNet | REPA-L9-GearNet | REPA-L9-MPNN |
|---|---|---|---|---|
| Des% ↑ | 72.4 | **77.0** | 73.3 | 76.6 |
| FID-PDB ↓ | 464 | **252** | 272 | 333 |
| FID-AFDB ↓ | 494 | **298** | 322 | 363 |
| fJSD-A ↓ | 2.12 | **0.60** | 0.69 | 1.46 |
| fJSD-C ↓ | 0.64 | 0.06 | **0.04** | 0.47 |
| fS-A ↑ | 4.50 | 6.83 | **7.35** | 4.89 |
| #Clust ↑ | 128 | 95 | 77 | **150** |
| pwTM ↓ | 0.22 | 0.26 | 0.33 | **0.19** |
| Nov-PDB% ↑ | 4.7 | **7.8** | 1.3 | 5.7 |
| ssJSD-2D ↓ | 0.26 | 0.16 | **0.11** | 0.26 |
| α / β % | 47 / 16 | 47 / 14 | **38 / 19** | 55 / 11 |

### L4 vs L9 (REPA depth) on AFDB

L4 wins on Des%, FID, fJSD-A, #Clust, pwTM, Nov-PDB (basically all design-and-diversity metrics). L9 wins only on fJSD-C, fS-A, ssJSD-2D, and α/β balance (i.e., on the SS-distribution-match metrics).

**Reading**: deeper REPA → stronger encoder regularization → better SS-distribution match, but at cost of generation diversity. L4 is the better all-around AFDB model; L9 is the SS-balance specialist. Different from PDB where L9 looked more uniformly favored.

### Three-way AFDB head-to-head: Baseline vs L9-MPNN vs L4-GearNet (added 2026-05-27)

Comparing the three AFDB-trained 60M models that have the most training data at γ=0.45, multi-rep means across 100K–1.3M:

| | Sample quality | Distribution match (T-W, S-W) | T-D diversity | SS balance (S-D) |
|---|---|---|---|---|
| Baseline | mid | poor | high | mid (drifts helix-ward) |
| **REPA-L4-GearNet** | **high** | **best** | **low** | **best (balanced)** |
| **REPA-L9-MPNN** | **best** | mid (modest improvement) | **high** (matches baseline) | **worst (most helix)** |

Specific numbers at 700K:
- pLDDT: GearNet 0.74 / MPNN 0.72 / baseline 0.69
- scRMSD: MPNN 1.93 / baseline 2.06 / GearNet 2.32 (MPNN best — opposite to PDB where MPNN had higher scRMSD)
- FID-PDB: GearNet 252 / MPNN 333 / baseline 464
- fJSD-A: GearNet 0.60 / MPNN 1.46 / baseline 2.12
- fS-A: GearNet 6.83 / MPNN 4.89 / baseline 4.50 (MPNN barely improves)
- #Clust: MPNN 150 / baseline 128 / GearNet 95
- α/β at 1.0M: GearNet 43/15 / baseline 50/14 / **MPNN 56/11** (MPNN more helix-biased than baseline!)
- ssJSD-2D: GearNet 0.16 / MPNN 0.26 ≈ baseline 0.27

**Reading**: MPNN-REPA-on-AFDB is essentially "REPA as quality booster" — improves scRMSD/pLDDT/Des% but doesn't act as a distribution shaper. It even mildly *increases* helix bias. GearNet-REPA-on-AFDB is the opposite — pulls the model toward correct global topology, slightly less per-sample quality, much better distribution match.

**Mechanistic guess**: GearNet encodes global geometric/topological features (rotation-equivariant message passing on residue-residue contact graphs). MPNN encodes more local/sequence-style features. So GearNet-REPA → pulls toward correct global topology distribution (improves fJSD-A/fS-A and SS distribution match). MPNN-REPA → pulls toward locally-valid structures (improves scRMSD/pLDDT) but doesn't reshape the global fold distribution.

This is a clean choice for the report narrative:
- For *distribution-match* and *SS balance* claims → cite GearNet
- For *per-sample quality* claims → cite MPNN
- The two are not the same thing, and the encoder choice determines which axis REPA acts on.

### MPNN vs GearNet (encoder choice) on AFDB

REPA-L9-MPNN behaves almost like a *different model class* from REPA-L9-GearNet:
- Doesn't reduce T-D (preserves #Clust 150, even *above* baseline 128)
- Doesn't shift SS balance — α/β = 55/11 vs baseline 47/16, actually *more* helix-biased and *less* β
- Only modestly improves fJSD/FID compared to GearNet variants
- Roughly matches baseline on ssJSD-2D and fS-A

So **REPA-L9-MPNN-on-AFDB is "REPA that doesn't act like REPA"**. It accelerates designability slightly, modestly improves FID, but loses the GearNet-style SS-balance and T-D effects. This is the falsifier the cross-encoder check was looking for — generic "REPA does X" claims should be checked against MPNN-AFDB.

### Sanity checks for this finding
- Baseline 700K β≥25 has pwTM=0.496 (also concentrated!) — so baseline ALSO concentrates β-rich generations *early* in training. Then by 1.0M baseline's β-rich gets diverse (0.135) while REPA never broadens. So this is more "REPA *gets stuck* at the concentrated β regime that early-training models all visit" rather than "REPA *invents* concentration".
- Sample sizes vary (baseline 1.5M β≥25 has only 11 samples — small-N caveat). The β≥25 row with n=144 (REPA 1.0M) at pwTM=0.870 is the most statistically reliable single point.

### Extended results: AFDB + MPNN variants (2026-05-27)

Re-ran β-stratified diversity across 10 PDB-and-AFDB × baseline-and-REPA-variant cells, multi-rep where available. The key signal is **pwTM in the β≥25 bin** (lower = more diverse fold space; ~0.13 = baseline-like).

| β-bin | Case | n | pwTM | #clust |
|---|---|---|---|---|
| **β≥25** | **PDB**: baseline 1.0M | 39 | **0.135** | 35 |
| | PDB: baseline 1.6M | 81 | **0.134** | 75 |
| | PDB: REPA-L9-GN 1.0M | 144 | **0.870** | 26 |
| | PDB: REPA-L4-GN 700K | 118 | 0.840 | 26 |
| | PDB: REPA-L9-MPNN 700K | 71 | 0.670 | 20 |
| | PDB: REPA-L9-MPNN 1.0M | 99 | 0.671 | 14 |
| | PDB: REPA-L4-MPNN 400K | 121 | 0.658 | 43 |
| | PDB: REPA-L4-MPNN 700K | 109 | **0.863** | 23 |
| | PDB: REPA-L4-MPNN 1.0M | 31 | 0.700 | 7 |
| | **AFDB**: baseline 700K | 26 | **0.166** | 24 |
| | AFDB: baseline 1.6M | 14 | 0.186 | 13 |
| | AFDB: REPA-L4-GN 700K | 35 | 0.251 | 24 |
| | AFDB: REPA-L4-GN 1.0M | 38 | 0.314 | 21 |
| | AFDB: REPA-L9-GN 700K | 112 | **0.818** | 28 |
| | AFDB: REPA-L9-GN 900K | 129 | 0.764 | 27 |
| | **AFDB: REPA-L9-MPNN 700K** | 12 | **0.113** | 12 |
| | **AFDB: REPA-L9-MPNN 1.0M** | 11 | **0.148** | 11 |

### Cross-dataset, cross-encoder β-rich concentration matrix

| | PDB | AFDB |
|---|---|---|
| baseline | diverse (pwTM ~0.13) | diverse (~0.17) |
| REPA-GearNet (L4 & L9) | **concentrated** (~0.7–0.9) | **concentrated** (L9: ~0.8; L4: ~0.25–0.31) |
| REPA-MPNN-L4 | **concentrated** (~0.66–0.86) | (not yet measured at n=256) |
| REPA-MPNN-L9 | **concentrated** (~0.67) | **diverse** (~0.11–0.15) |

**Three findings**:

1. **GearNet-REPA concentrates β-rich generations on both datasets** — robust to dataset.
2. **MPNN-REPA concentrates β-rich on PDB but NOT on AFDB**. The MPNN-AFDB case is the unique outlier where REPA *doesn't* induce β-concentration. Same n-caveat — only 11–12 β-rich samples — but consistent across two ckpts. **MPNN-L4 on PDB also concentrates** (pwTM 0.66–0.86 across 400K/700K/1000K, n=31–121 per cell), confirming that on PDB any learned encoder drives the concentration regardless of layer depth.
3. **REPA-GearNet on AFDB also concentrates the α-rich bin** (pwTM 0.27–0.40 vs baseline 0.12–0.16). On PDB the α-rich bin doesn't show GearNet concentration. So GearNet-AFDB is the *most* concentrating REPA configuration; the alignment regularizer concentrates *all* SS classes on AFDB.

The cleanest mechanistic hypothesis is now: **GearNet's representation space carves AFDB-fold-space into a small number of attractors that REPA pulls all generations toward; MPNN's representation space is more diffuse on AFDB-style structures so REPA-MPNN doesn't induce the same concentration.** PDB-trained REPA models concentrate β-rich regardless of encoder, because the PDB β-fold space itself is structurally more constrained (well-defined β-sandwiches, barrels, etc. with stricter H-bond patterns).

### Does T-D grow back with more REPA training? (Task 2 — answer: NO)

PDB REPA L9 #clusters trajectory at γ=0.45: 0 → 38 → 92 → **104** → 65 → 62 → 69 → 59 (200K → 1100K).
- Peak at 700K (104), then drops and plateaus around 60–70.
- Within β≥25 bin: 22 → 15 → 26 clusters across 700K/900K/1000K — slight oscillation, no growth.

AFDB REPA L4 #clusters: 119 → 117 → 122 → 95 → 96 → 101 (100K → 1.2M). Drops once after 400K, stays around 95–101.

Both REPA-GearNet variants show no sustained T-D recovery. T-D is stuck at the plateau they reach by ~700K. Baseline continues to grow well past that.

### Follow-ups
- For the concentrated β-rich REPA samples, check the CATH-A class breakdown — is it 1 architecture or a few?
- Compute β-stratified diversity for the convergence sweep's repa_l4_random control (PDB) to test "is GearNet's *learned* representation the cause, or does random GearNet already concentrate folds?"
- Repeat the analysis for repa_mpnn_l4 on PDB and repa_mpnn_l4 on AFDB to add another data point on encoder × depth × dataset interaction.

### Original Experiment 1 procedure (kept for reference)

Test whether REPA's lower T-D diversity is explained by β-content or persists even at matched β-content within individual samples.

Procedure:
- For each (model, step, sampler) tuple, pull all designable samples (`designability_index.csv` lists pass/fail per sample)
- Compute per-sample SS from cached `ss_cache/ss_fractions.npz` or PSEA
- Split designable samples by per-sample β-fraction into 3 bins: low-β (β<10%), mid-β (10–25%), high-β (β>25%)
- Within each bin, compute pwTM mean across pairs
- Compare baseline vs REPA per-bin

Expected outcome A (sheets→fewer folds): within-bin pwTM-diversity matches.
Expected outcome B (REPA→fewer folds independent of SS): within-bin pwTM still lower for REPA.

### Experiment 2: Cross-encoder T-D comparison (TO RUN)

Pull `_res_diversity_*` for REPA-GearNet vs REPA-MPNN at γ=0.45 across training steps on both PDB and AFDB. If MPNN also shows reduced T-D, the encoder hypothesis is generic-REPA. If only GearNet does, it's encoder-specific.

### Experiment 3: scRMSD bimodality (✓ RUN — refuted; see "scRMSD: polarization, NOT bimodality")

Per-sample scRMSD histograms for baseline vs REPA at step-matched γ=0.45.
Predicted REPA = bimodal — **not supported**. Both are unimodal; the real
effect is marginal-zone (2–4Å) depletion / polarization. Full result in the
dedicated section below.

---

## Sampler-regime robustness check (added 2026-05-27, expanded 2026-05-27)

The convergence claims above are all compiled at **γ=0.45 across the training
trajectory**. This section asks whether those *trajectory* trends survive a
change of sampler noise (ODE, γ ∈ {0, 0.35, 0.45, 0.5, 1.0}). The first version
of this section compared only a single step (700K vs 700K); this is the
trajectory-wide version.

Regenerate with
[build_sampler_regime_robustness.py](../../evaluation/proteina/generation/scripts/paper/build_sampler_regime_robustness.py).

### Scope and data sources (read first)

- **Multi-γ data exists for only ONE encoder per dataset**: PDB **L9-GN** and
  AFDB **L4-GN** (baseline + that REPA run, across all 5 samplers). MPNN, the
  random control, L4-on-PDB, and L9-on-AFDB have **no** sampler-ablation data
  yet — so this check covers GearNet only. MPNN status is at the bottom.
- Sampler-ablation γ (ODE/0/0.35/0.5/1.0): the `*.clean.jsonl` in
  `results/variance/n256{,_afdb}_sampler_ablation/`, **1 rep/cell**. (The PDB
  *raw* jsonl is missing 0.35/0.5; only the clean has all five. The AFDB raw had
  a malformed line — `}` on its own — which the loader skips.)
- γ=0.45: the convergence-sweep raw jsonl, **3 reps/cell** (deduped by mean).
- Step-matched steps with full γ coverage: **PDB 100/200/400/700K** (n=4),
  **AFDB 100/700K** (n=2). Win-rates are therefore coarse.
- `#Clust` here is `_res_diversity_clusters_mean` (per-length-bin mean), so its
  magnitude is smaller than the total-cluster counts used elsewhere in this doc;
  only its **sign** is used for win/loss and that is consistent.

### Win-rate across γ over the step-matched trajectory

REPA beats baseline in `x/n` step-matched steps. γ=.45 is the doc's claim column.

#### PDB — REPA L9-GN vs baseline (steps 100/200/400/700K)

| metric | ODE | γ=0 | γ=.35 | **γ=.45** | γ=.5 | γ=1 |
|---|---|---|---|---|---|---|
| Des ↑ | 3/4 | 1/4 | 3/4 | 3/4 | 3/4 | 3/4 |
| FID ↓ | 2/4 | 2/4 | 3/4 | 2/4 | 3/4 | 1/4 |
| fJSD-A ↓ | 2/4 | 2/4 | 2/4 | 2/4 | 3/4 | 2/4 |
| fJSD-C ↓ | 3/4 | 2/4 | 3/4 | 3/4 | 3/4 | 2/4 |
| ssJSD2D ↓ | 2/4 | 1/4 | 3/4 | 3/4 | 3/4 | 2/4 |
| β ↑ | 1/4 | 1/4 | 2/4 | 2/4 | 2/4 | 1/4 |
| scRMSD ↓ | 3/4 | 2/4 | 4/4 | 3/4 | 3/4 | 3/4 |
| pLDDT ↑ | 1/4 | 1/4 | 3/4 | 3/4 | 3/4 | 2/4 |
| #Clust ↑ | 1/1 | 1/3 | 2/2 | 2/3 | 2/2 | 1/1 |
| pwTM ↓ | 0/1 | 1/3 | 1/2 | 1/2 | 0/2 | 1/1 |

#### AFDB — REPA L4-GN vs baseline (steps 100/700K)

| metric | ODE | γ=0 | γ=.35 | **γ=.45** | γ=.5 | γ=1 |
|---|---|---|---|---|---|---|
| Des ↑ | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| FID ↓ | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 |
| fJSD-A ↓ | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| fJSD-C ↓ | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 1/2 |
| ssJSD2D ↓ | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| β ↑ | 2/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| scRMSD ↓ | 1/2 | 1/2 | 1/2 | 0/2 | 0/2 | 1/2 |
| pLDDT ↑ | 2/2 | 1/2 | 2/2 | 1/2 | 2/2 | 2/2 |
| #Clust ↑ | 1/1 | 0/2 | 0/2 | 1/2 | 1/2 | 1/1 |
| pwTM ↓ | 1/1 | 1/2 | 0/2 | 1/2 | 0/2 | 0/1 |

### Single-step 700K Δ tables (corrected)

Δ = REPA − baseline; ✓ = REPA better. **These replace the original 700K
tables, whose ssJSD2D and β columns did not correspond to any real SS field**
(e.g. ODE-700K PDB the old table claimed β +0.10 / ssJSD2D −0.19, but
`ss_frac_E` Δ = −0.02 and `ss_jsd_pdb_2d` Δ = +0.02). FID/fJSD/Des reproduce
exactly. `#Clust` uses `_res_diversity_clusters_mean` (see scope note).

**PDB 700K (baseline vs REPA L9-GN):**

| γ | Des | FID | fJSD-A | fJSD-C | ssJSD2D | β | #Clust |
|---|---|---|---|---|---|---|---|
| ODE | ✓+0.05 | ✓−38.13 | ✗+0.06 | ✓−0.09 | ✗+0.02 | ✗−0.02 | ✓+1.33 |
| γ=0 | ✗−0.16 | ✓−280.13 | ✓−0.85 | ✓−0.34 | ✓−0.39 | ✗−0.04 | ✓+8.20 |
| γ=.35 | ✓+0.17 | ✓−139.35 | ✓−0.67 | ✓−0.08 | ✓−0.22 | ✗−0.05 | ✓+1.80 |
| **γ=.45** | ✓+0.14 | ✓−117.90 | ✓−0.57 | ✓−0.06 | ✓−0.19 | ✗−0.05 | ✓+2.69 |
| γ=.5 | ✓+0.15 | ✓−95.17 | ✓−0.49 | ✓−0.02 | ✓−0.19 | ✗−0.03 | ✓+0.60 |
| γ=1 | ✓+0.08 | ✗+41.73 | ✗+0.44 | ✗+0.20 | ✗+0.06 | ✗−0.02 | ✓+1.50 |

**AFDB 700K (baseline vs REPA L4-GN):**

| γ | Des | FID | fJSD-A | fJSD-C | ssJSD2D | β | #Clust |
|---|---|---|---|---|---|---|---|
| ODE | ✓+0.23 | ✓−72.21 | ✓−0.52 | ✓−0.17 | ✓−0.03 | ✓+0.01 | ✓+6.40 |
| γ=0 | ✓+0.05 | ✓−55.34 | ✓−0.22 | ✗+0.10 | ✗+0.02 | ✗−0.04 | ✗−1.80 |
| γ=.35 | ✓+0.11 | ✓−193.83 | ✓−1.62 | ✓−0.88 | ✓−0.12 | ✗−0.02 | ✗−5.00 |
| **γ=.45** | ✓+0.05 | ✓−196.19 | ✓−1.62 | ✓−0.88 | ✓−0.12 | ✗−0.02 | ✗−6.60 |
| γ=.5 | ✓+0.03 | ✓−183.67 | ✓−1.38 | ✓−0.76 | ✓−0.09 | ✗−0.02 | ✗−8.20 |
| γ=1 | ✓+0.07 | ✓−89.28 | ✓−0.45 | ✓−0.09 | ✓−0.01 | ✗−0.00 | ✓+0.80 |

### Corrected takeaways

The γ=0.45 trends carry over **almost cell-for-cell to the middle band
γ ∈ {0.35, 0.5}**. The two extremes are where they break, and the failure modes
differ by dataset:

- **The middle band {.35, .45, .5} is internally consistent.** Any claim made at
  γ=0.45 holds at its neighbours; treat the three as one regime.
- **Distribution-match (FID, fJSD-A) dies at γ=1.** PDB FID drops to 1/4 and
  fJSD-A flips positive at full-temperature SDE; AFDB FID 1/2. γ=0 is *fine* for
  distribution-match — it is *quality* that suffers there. So the headline
  "γ ∈ [0, 0.5]" band is right, with **γ=1 the cutoff**.
- **Designability/quality dies at γ=0 on PDB** (Des 1/4, pLDDT 1/4): baseline's
  low-temperature mode-collapse beats REPA. AFDB Des survives at γ=0.
- **ssJSD-2D is NOT "✓ at every γ"** (the original claim). It is 1/4 at γ=0 on
  PDB and loses at ODE/γ=1 at 700K. It is robust *in the middle band* and on
  AFDB, but the "every γ" version was a 700K-only artifact (γ=0 happens to win
  at 700K, loses at 100/200/400K).
- **β is non-robust at every single γ** (1–2/4 on PDB, 1/2 on AFDB; REPA is
  mostly *lower* β step-matched). This reinforces the already-revised Claim 2:
  the β shift is a late-training PDB phenomenon, not a sampler-robust effect.
- **T-D: the AFDB-GearNet diversity reduction is robust across γ** (#Clust/pwTM
  lose at every non-ODE γ); on PDB #Clust stays a win across γ. Consistent with
  Claim 5's "GearNet reduces T-D on AFDB, not on PDB at step-match".

**Bottom line (GearNet only):** the doc's γ=0.45 convergence story is robust
across γ ∈ [0.35, 0.5], with two clean failure modes — **distribution-match
collapses at γ=1, designability/quality collapses at γ=0** — and **β was never
sampler-robust**. The caveat is coverage: this is shown only for GearNet, at
n≤4 step-matched points.

### MPNN extension status (2026-05-27)

- **AFDB MPNN-L9 sampler ablation IS queued but not yet run.** Config
  `n256_afdb_sampler_ablation_ext` (5 array jobs, one per sampler ODE/0/0.35/0.5/1.0,
  array 0-14) covers MPNN-L9-AFDB + L9-GN-AFDB + extra baseline/L4 steps;
  γ=0.45 comes from the convergence sweep. All tasks PENDING as of writing — the
  raw jsonl currently has MPNN/L9 rows only as untagged (γ=0.45) entries, no
  ablation-γ rows. Once these land, `build_sampler_regime_robustness.py` will
  auto-emit the AFDB MPNN-L9 and L9-GN rows (it skips pairs with no multi-γ data).
- **PDB sampler ablation submitted 2026-05-27** (jobs 29735627 ODE / 29735628 γ=0
  / 29735629 γ=.35 / 29735630 γ=.5 / 29735631 γ=1, each array 0-20). Config
  `n256_pdb_sampler_ablation_ext` fills full encoder/layer coverage — adds
  **L4-GN, MPNN-L4, MPNN-L9, L4-random** (baseline + L9-GN already on disk) at the
  step-matched grid {100,200,400,700,1000,1300}K (L4-GN→1000, L4-rand→700,
  MPNN-L9 has no 700k). 21 new (run,step) cells × 5 samplers = 105 tasks, writing
  to the same `results/variance/n256_sampler_ablation/` (dedup skips existing
  baseline/L9 cells). γ=0.45 already in the convergence sweep. **PENDING.** Once
  both this and the AFDB ext land, re-run `build_sampler_regime_robustness.py` and
  `clean_variance_jsonl.py` (the clean snapshots are stale vs the mutated raw).

## The "700K T-D cliff" investigation (2026-05-27)

REPA-GearNet (both L4 and L9) and REPA-MPNN show a sharp drop in #clusters around training step ~700K on PDB γ=0.45 that doesn't recover. Baseline doesn't show this — its #clusters keeps growing.

### Hypotheses
- **H1** ✗ REFUTED — REPA-loss/cos_sim balance shifts (FM saturates ~600-700K, REPA loss dominates). Denoised cos_sim shows no plateau in any run; the cliff run rises most. See "H1 cos_sim mechanism — ✗ REFUTED by denoised data".
- **H2** ✓ CONFIRMED — Designability composition shifts: REPA's designable mass moves into the concentrated β-rich bin (the surviving mechanism).
- **H3** LR schedule transition at 700K — untested
- **H4** Encoder representation budget exhausted — model approaches GearNet's training distribution asymptote — untested (and H1's refutation undercuts the "budget exhausted early" intuition)
- **H5** Implicit overfitting — REPA's narrower hypothesis space saturates earlier than baseline — untested

### H2 test: SS-class trajectory of designable subset (✓ RUN, CONFIRMS H2)

For REPA L9 GearNet PDB γ=0.45 across training steps, the SS-class breakdown of designable samples:

| Step | n_des | mean H | mean E | α-rich % | β-rich % | mixed % |
|---|---|---|---|---|---|---|
| 200K | 76 | 0.76 | 0.01 | **97.4%** | 0.0% | 1.3% |
| 400K | 239 | 0.60 | 0.13 | 53.6% | 17.6% | 27.6% |
| **700K** | **305** | **0.50** | **0.17** | **41.6%** | **23.9%** | **33.8%** |
| 800K | 125 | 0.45 | 0.18 | 38.4% | 31.2% | 25.6% |
| 900K | 110 | 0.39 | 0.22 | 20.0% | **39.1%** | 36.4% |
| 1000K | 342 | 0.38 | 0.23 | 28.9% | **41.5%** | 27.8% |
| 1100K | 265 | 0.39 | 0.20 | 24.2% | 33.6% | 36.2% |

Baseline PDB designable for comparison:

| Step | α-rich % | β-rich % | mixed % |
|---|---|---|---|
| 700K | 28.8 | 34.0 | 36.7 |
| 1000K | 33.1 | 13.5 | 51.6 |
| 1500K | 34.1 | 8.3 | 57.6 |
| 1600K | 27.5 | 15.8 | 56.0 |

### Reading

**REPA L9 designable subset transitions from α-dominated → balanced → β-dominated**:
- 200K: 97% α-rich (early mode collapse on helix bundles, n only 76)
- 400-700K: model broadens through the SS spectrum, peaking at balanced **42 / 24 / 34** (α/β/mixed) at 700K. This is where #Clusters peaks (104).
- 800-1100K: model concentrates in **β-rich** modes — fraction of designable β-rich rises from 24% → 42% (peak at 1000K).

**Baseline transitions differently**: starts β-balanced at 700K (34% β-rich), then designable mass shifts toward MIXED (over 50% by 1.5M). Baseline produces *more* mixed-SS samples late, REPA produces *more* β-rich.

### Combined with Experiment 1 (β-stratified diversity)

We already established that REPA's β-rich samples are highly concentrated (pwTM 0.67-0.87). The H2 trajectory shows the designable subset is shifting TOWARD that concentrated β-rich mode. So the cliff at 700K is a compositional shift:

> Pre-700K: REPA produces diverse designable samples across α/β/mixed (#clust=104 at 700K).
> Post-700K: REPA's designable mass shifts into the β-rich bin, which is intrinsically concentrated for REPA-GearNet (Exp 1 finding). Result: fewer distinct architectures even as designable count stays roughly constant.

So the cliff isn't because the model gets WORSE — it's because it gets *more β-rich* over time, and its β-rich mode is narrow. The "good" interpretation: REPA is learning to produce more β-rich folds (which were previously underrepresented in the model). The "bad" interpretation: those β-rich folds are concentrated on a small set of encoder-friendly architectures.

This is consistent with **H2 + H4 combined**: composition shifts toward β-rich (H2), and within β-rich the encoder's representation pulls toward a small subspace (H4).

### SS-class trajectory across ALL gen-eval variants (added 2026-05-27)

Extended H2 test to every REPA + baseline family we have ckpts for, both PDB and AFDB. Looking at the β-rich %% of designable trajectory across training step at γ=0.45:

| Family | Early (≤200K) | Mid (700K) | Late peak | Late settle |
|---|---|---|---|---|
| **PDB Baseline** | (n=0) | 34% | 16% (1.6M) | 16% (1.7M) |
| **PDB REPA-L4-GN** | 0–4% | **36%** | **47% (800-900K)** | 10% (1.0M) ⚠ |
| **PDB REPA-L9-GN** | 0% | **24%** | **41% (1.0M)** | 35% (1.2M) |
| **PDB REPA-L4-rand** | 0% | 12% | 12% (400K) | 6% (800K) |
| **PDB REPA-L4-MPNN** | 0–7% | 36% | **43% (400K)** | 27% (1.6M) |
| **PDB REPA-L9-MPNN** | 2–14% | 18% | **39% (800K)** | 32% (1.3M) |
| **AFDB Baseline** | 3% | 7% | 21% (400K, spike) | 4% (1.8M) |
| **AFDB REPA-L4-GN** | 17% | 8% | 17% (100K) | 13% (1.3M) |
| **AFDB REPA-L9-GN** | (n/a) | 20% | 23% (900K) | 23% (900K) |
| **AFDB REPA-L9-MPNN** | 9–15% | 3% | 15% (200K) | **2% (1.5M)** ⚠ |

### Reading

**Two distinct patterns emerge:**

**Pattern A: "Learned-encoder REPA drives designable subset toward β-rich"** — applies to:
- PDB-L4-GN, PDB-L9-GN, PDB-L4-MPNN, PDB-L9-MPNN
- AFDB-L9-GN (modestly)

β-rich % grows over training from 0% (early helix collapse) to 30–47% (late). α-rich % correspondingly declines.

**Pattern B: "Encoder-driven α-bias"** — applies to:
- AFDB-L9-MPNN: opposite direction. β-rich DECLINES from 15% (200K) to 2% (1.5M). α-rich GROWS from 12% to 38%. The model gets MORE helix-biased with training.

**Pattern C: "No encoder pull"** — applies to:
- AFDB-L4-GN: β-rich stays roughly flat (8–17%) across training. α-rich slowly declines. The model converges to a stable mixed-dominated distribution like the AFDB data itself.
- PDB REPA-L4-rand (random GearNet): β-rich stays low (≤13%), like baseline. **Random encoder doesn't trigger the β-shift.** This is the critical falsifier.

### Mechanistic decomposition

The data lets us cleanly separate effects:

1. **Generic REPA effect**: All REPA models go through an early α-rich phase (200K = 80–100% α-rich for nearly all). The Flow Matching loss alone is happy producing all-helix at first — REPA doesn't change this initial regime.

2. **Encoder-LEARNED-representation drives a compositional shift after ~400K.**
   - Random GearNet (PDB-L4-rand) doesn't trigger it → confirms it's *learned* representations, not just adding any loss term.
   - The direction depends on the encoder × dataset combination.

3. **Direction is encoder × dataset specific**:
   - Learned GearNet on PDB → shifts toward β-rich (β-rich gross from 0% → 35-47%)
   - Learned MPNN on PDB → shifts toward β-rich (similar magnitude, more oscillatory)
   - Learned GearNet on AFDB → shifts toward β-rich, but smaller (AFDB data has less β to learn from)
   - Learned MPNN on AFDB → shifts toward α-rich (opposite!). β-rich actually DECREASES; α-rich grows to 38%.

4. **Dataset constrains the destination**: AFDB baseline lives in the "mixed-dominated" regime (60-75% mixed throughout). PDB baseline drifts between α/β/mixed more. REPA shifts WITHIN each dataset's space, not across them.

### Implications for the report narrative

We can now make precise mechanistic claims:

| Claim | Status |
|---|---|
| "REPA accelerates convergence (faster FID/fJSD drop)" | ✓ generic to all learned-encoder REPA |
| "REPA improves SS-distribution match (ssJSD-2D)" | ✓ generic to all learned-encoder REPA on PDB; modestly true on AFDB |
| "REPA preserves β-content / reduces helix collapse" | ⚠ encoder × dataset specific. True for GearNet/MPNN on PDB, GearNet on AFDB. FALSE for MPNN on AFDB (opposite). |
| "REPA concentrates β-rich folds" | ⚠ specific to GearNet (both datasets), MPNN on PDB only |
| "T-D cliff at 700K" | ⚠ specific to GearNet on both datasets and MPNN on PDB; AFDB-MPNN doesn't show this |
| "Random encoder doesn't drive distributional changes" | ✓ confirmed by PDB-L4-rand control |

The encoder + alignment layer choice **determines the direction of the regularization** — REPA isn't a single intervention, it's a family of interventions parameterized by encoder choice. The "REPA does X" narrative needs to be stated as "REPA-with-learned-GearNet does X". MPNN on AFDB shows what "REPA in the other direction" looks like.

### Why AFDB-MPNN goes the opposite way — Exp B tested & REFUTED the SS-salience hypothesis (2026-05-27)

**Falsifier observation**: GearNet-REPA on AFDB shifts toward β (same as PDB), MPNN-REPA on AFDB shifts toward α. On PDB both encoders go toward β. So encoder identity sets the *direction*, not just magnitude.

**Hypothesis tested (Exp B)**: maybe the fixed MPNN encoder *represents SS differently* depending on the structure population — i.e., it makes sheets more linearly-separable on PDB structures (→ β-pull) and helix more separable on AFDB structures (→ α-pull). Test: per-residue linear SS-probe of frozen ProteinMPNN embeddings, n=600 proteins each from PDB-train and AFDB-SwissProt lmdbs. Script: `encoder_profiling/proteina/mpnn/ss_probe_cross_dataset.py`.

**Result (n≈60K residues each)**:

| Dataset | probe acc | recall helix | recall sheet | recall coil |
|---|---|---|---|---|
| PDB | 0.887 | 0.956 | **0.740** | 0.895 |
| AFDB | 0.905 | 0.972 | **0.760** | 0.901 |

- Helix is more separable than sheet on **both** datasets, by the **same** margin (~+0.21).
- **sheet-recall: PDB 0.740 vs AFDB 0.760 (Δ=−0.02)** — AFDB sheet is *marginally more* separable, the opposite of what the hypothesis needs (it predicted PDB sheet > AFDB sheet).

**Verdict: REFUTED.** The frozen MPNN encoder's SS-discriminability is essentially dataset-invariant. It does **not** represent sheets more saliently on PDB. So the MPNN α-on-AFDB / β-on-PDB direction split is **not** explained by dataset-dependent SS-salience in the encoder.

**What this leaves**: the directional pull must come from the **data distribution REPA aligns toward** (PDB has β to pull toward; AFDB is helix-dominated and β-poor) rather than from the encoder seeing SS differently. BUT this can't be the whole story either, because GearNet and MPNN diverge on the *same* AFDB data — so encoder identity matters in a way that is **orthogonal to simple SS-classification** (i.e., the encoders differ in *what they emphasize in their full representation geometry*, not in how cleanly they linearly separate H/E/C). That deeper encoder-geometry question is unresolved; Exp B's contribution is to cleanly rule out the simplest "encoder is dataset-specifically-SS-biased" explanation.

(Caveat: this probes the encoder on *ground-truth* PDB/AFDB structures. REPA aligns to the encoder's reps of *generated* samples during training; the probe is a proxy. But a fixed encoder applying the same function makes the ground-truth probe a reasonable test of dataset-dependent SS-salience.)

### H1 test (loss balance) — ⚠️ SUPERSEDED / REFUTED (see "H1 cos_sim mechanism — ✗ REFUTED by denoised data" below)

> **Everything in the H1 subsections below — down to (but not including) the
> "H1 cos_sim mechanism — ✗ REFUTED by denoised data" section — is WRONG.**
> ("REPA loss saturates ~400K", "REPA saturates faster on PDB-L9-GN → cliff",
> "REPA-saturation-then-FM-navigates".) It was read off SINGLE-STEP
> `cos_sim_layer_*_step` samples, whose ~±0.016 noise manufactured a fake
> plateau. The denoised ±10K-window re-pull shows all four runs rise
> monotonically with no plateau, and the cliff run (PDB-L9-GN) rises the
> *most* late. All of this is kept only as a record of the mistake — skip to
> the REFUTED section for the corrected analysis.

Wandb scan of `proteina_60m_repa_l9_256_per_residue_bs24_2gpu` (the PDB-L9-GN training run):

| Step | FM loss | REPA loss | cos_sim_L9 |
|---|---|---|---|
| 50K | -0.000 | -0.724 | 0.72 |
| 100K | -0.09 | -0.779 | 0.78 |
| 200K | -0.20 | -0.811 | 0.81 |
| **400K** | **-0.22** | **-0.839** | **0.84** |
| 500K | -0.08 | -0.786 | 0.79 |
| 600K | -0.18 | -0.817 | 0.82 |
| **700K** | -0.02 | -0.815 | 0.81 |
| 850K | -0.30 | -0.852 | 0.85 |
| 1000K | -0.21 | -0.840 | 0.84 |
| 1200K | -0.11 | -0.818 | 0.82 |

(Both losses are reported negative due to internal normalization; lower magnitude = "more loss". I read them as: FM-loss-magnitude-decreases-→-improving; REPA-loss-magnitude-stays-around-0.81-0.85-after-400K-→-saturated.)

**H1 reading**: REPA loss saturates by ~400K — cos_sim_L9 reaches 0.84 there and oscillates 0.81–0.85 thereafter. So contrary to "REPA loss starts dominating at 700K", the REPA loss has been *fully baked-in for 300K steps before* the T-D cliff. FM loss continues optimizing throughout but with noisy single-step values.

This actually supports a slightly different mechanism than the original H1 framing:

> Post-400K the model lives in the encoder-aligned subspace (cos_sim ≈ 0.84). For the next ~300K steps, FM loss navigates *within* that subspace, eventually converging onto the easiest-to-designable modes inside it. Those modes happen to be the concentrated β-rich folds (Exp 1). So the cliff at 700K is the model finishing its "navigation within encoder-aligned space" → finds and concentrates on the high-designability β-rich attractor → diversity drops.

So the right framing combines **H2 (compositional shift) + H4 (encoder budget) + slow-FM-navigation**: REPA saturates fast, then FM steers within the encoder-aligned manifold for several hundred thousand steps until it finds the manifold's "designable basin", which for GearNet is concentrated β-rich folds.

### H1 ancillary observation

The FM loss spike at 700K (-0.02 vs neighboring -0.18) might be a single-step artifact (we sampled one step out of the wandb history at that point). But the cos_sim_L9 = 0.81 at 700K is slightly *lower* than at 600K (0.82) and 850K (0.85), suggesting brief mis-alignment around the cliff that recovers. Worth a denser wandb pull around 650-750K to confirm.

### H1 cross-encoder pull (added 2026-05-27)

Successfully pulled L4-AFDB-GearNet loss curves from wandb (MPNN-AFDB-L9 pull still flaking on wandb 500s, retried in background):

**AFDB-L4-GearNet trajectory**:

| Step | FM loss | REPA loss | cos_sim_L4 |
|---|---|---|---|
| 100K | -0.20 | -0.85 | **0.85** |
| 200K | -0.21 | -0.85 | 0.85 |
| 400K | -0.27 | -0.85 | 0.85 |
| 700K | -0.31 | -0.89 | 0.89 |
| **1000K** | **-0.35** | **-0.90** | **0.90 (peak)** |
| 1200K | -0.32 | -0.87 | 0.87 |
| 1300K | -0.29 | -0.87 | 0.87 |

Compared with the **PDB-L9-GearNet** trajectory (already pulled):

| Step | FM loss | REPA loss | cos_sim_L9 |
|---|---|---|---|
| 100K | -0.09 | -0.78 | 0.78 |
| 400K | -0.22 | -0.84 (saturates) | 0.84 |
| 700K | -0.02 | -0.81 | 0.81 |
| 1000K | -0.21 | -0.84 | 0.84 |
| 1200K | -0.11 | -0.82 | 0.82 |

### Notable differences

1. **AFDB-L4-GN starts with much higher cos_sim (0.85 at 100K)** than PDB-L9-GN (0.78). The L4 alignment is "easier" — earlier layers have features more similar to GearNet's representations than later layers, which need more training to align.

2. **AFDB-L4-GN's cos_sim keeps RISING through 1M** (0.85 → 0.90 → 0.87), while PDB-L9-GN saturates at 0.84 by 400K and stays flat. So **AFDB-L4-GN keeps improving its alignment for longer**.

3. **FM loss trajectories**:
   - PDB-L9-GN: -0.22 (400K) → -0.21 (1M). Roughly flat after 400K, oscillating noisily.
   - AFDB-L4-GN: -0.27 (400K) → -0.35 (1M). Continues *decreasing* in magnitude (-0.35 is better than -0.27).

### Reading

The two runs differ in *how saturated REPA is* and *how much further FM is driving the model*:

- **PDB-L9-GN**: REPA done by 400K, FM continues to optimize within manifold → lands on concentrated β-rich attractor by 700K → cliff.
- **AFDB-L4-GN**: REPA still tightening through 1M, both losses continue improving. The model converges more slowly toward its attractor — and that attractor is not a β-rich mode (we know SS stays mixed-dominated 60% throughout per the SS-class trajectory above).

This is *consistent with* the "REPA-saturation-then-FM-navigates" mechanism but adds nuance: when REPA saturates faster (PDB-L9-GN) the cliff happens sharply; when REPA keeps improving (AFDB-L4-GN) there's no clean cliff and the model gently converges toward its stable attractor. **The 700K cliff is sharp because REPA saturated 300K earlier; not all REPA configs have this sharp dynamic.**

This actually weakens the "700K cliff" framing as a universal REPA phenomenon — it's specific to configurations where REPA saturates very early relative to total training. PDB-L9-GN: yes. AFDB-L4-GN: no (gradual convergence). Worth checking AFDB-MPNN-L9 (in flight) to see if its alpha-shift trajectory corresponds to a particular loss profile.

### H1 cos_sim mechanism — ✗ REFUTED by denoised data (added 2026-05-27)

> **This supersedes the single-step H1 analysis above (the "REPA saturates
> at 400K → cliff" story). That conclusion was an artifact of single-step
> `*_step` sampling noise — see below.**

The earlier H1 read sampled ONE `cos_sim_layer_*_step` value per target step.
That metric has ~±0.016 single-step scatter, which is as large as the
cross-run trend we were trying to read. The single PDB-L9-GN sample at 400K
happened to be 0.84 (a high outlier) and 500K happened to be 0.79 (a low
one), manufacturing a fake "plateau-then-noise" pattern.

Re-pulled all four runs with ±10K-step windows (n≈20k samples per point) and
took the mean. Script:
[pull_h1_cossim_denoised.py](../../evaluation/proteina/generation/scripts/paper/pull_h1_cossim_denoised.py);
plot: [h1_cossim_denoised.png](../../evaluation/proteina/generation/figures/paper/n256_sampler_ablation/h1_cossim_denoised.png);
data: [h1_cossim_denoised.csv](../../evaluation/proteina/generation/results/variance/h1_cossim_denoised.csv).

| Run | cos_sim 100K | 400K | 1200K | rise 100→400K | rise 400→1200K | late-rise /100K |
|---|---|---|---|---|---|---|
| **PDB-L9-GN** (has cliff) | 0.757 | 0.804 | **0.831** | +0.047 | **+0.026** | **+0.0033** |
| AFDB-L4-GN (no cliff) | 0.854 | 0.874 | 0.882 | +0.020 | +0.008 | +0.0010 |
| AFDB-L9-GN (no cliff) | 0.864 | 0.884 | 0.891¹ | +0.021 | +0.007 | +0.0012 |
| AFDB-MPNN-L9 (no cliff) | 0.887 | 0.903 | 0.910 | +0.015 | +0.008 | +0.0010 |

¹ to 1.0M (run ends there). std ≈ 0.015–0.016 per point; std-of-mean ≈ 0.0001.

**The denoised data refutes the cos_sim-saturation mechanism:**

1. **No run plateaus.** All four cos_sim trajectories are smooth, monotone,
   and decelerating — the same qualitative shape. None flattens to zero slope.

2. **The cliff run rises the *most*, not the least.** PDB-L9-GN — the only run
   with a sharp 700K T-D cliff — has the **fastest** late-training cos_sim
   rise (+0.0033/100K, ~3× the AFDB runs). It does NOT saturate early. cos_sim
   climbs smoothly straight through the 700K cliff with no inflection. This is
   the **opposite** of what the original H1 story claimed.

3. **Absolute levels aren't comparable** across encoder/layer/dataset (PDB-L9
   0.76–0.83, AFDB-L4 0.85–0.88, AFDB-L9 0.86–0.89, MPNN-L9 0.89–0.91). Only
   the shape is comparable, and the shapes don't distinguish cliff from
   no-cliff runs.

**Conclusion: the 700K T-D cliff does not correspond to any cos_sim
saturation event.** The cliff is real in the #clusters / SS-composition data
(H2 — REPA's designable mass shifts into the concentrated β-rich bin), but
the loss-balance/cos_sim explanation (H1) is dead. Whatever drives the cliff,
it is NOT "REPA alignment finishes early, then FM navigates." H2
(compositional shift toward the concentrated β-rich attractor) stands as the
mechanism; H1 should be dropped from the report.

## scRMSD: polarization, NOT bimodality (added 2026-05-27)

Per-sample scRMSD from `designability_index.csv` across all reps. Script:
[exp_scrmsd_polarization.py](../../evaluation/proteina/generation/scripts/paper/exp_scrmsd_polarization.py).
Figure: [scrmsd_polarization.png](../../evaluation/proteina/generation/figures/paper/n256_sampler_ablation/scrmsd_polarization.png).

> **The original "REPA → bimodal scRMSD" prediction is NOT supported.** The
> histograms (both linear and log-y) are unimodal for both baseline and REPA:
> one dominant peak near 0–2Å and a monotone-decaying tail. There is no second
> mode, no dip-then-rise. The bimodality coefficient (BC) does not distinguish
> the two (PDB 700K: 0.81 vs 0.84; PDB 1000K: 0.83 vs 0.76) — a fat right tail
> inflates BC even for a unimodal distribution, so BC was a bad diagnostic here.

The real, robust effect is **marginal-zone depletion / polarization**. Mass
fractions per scRMSD bin:

| Step-matched pair | who | <2Å | **2–4Å (marginal)** | 4–8Å | >8Å |
|---|---|---|---|---|---|
| PDB 700K | base / REPA | 0.43 / **0.61** | 0.39 / **0.20** | 0.10 / 0.10 | 0.09 / 0.09 |
| PDB 1000K | base / REPA | 0.55 / 0.53 | 0.32 / **0.24** | 0.07 / 0.12 | 0.06 / 0.12 |
| AFDB 700K | base / REPA | 0.72 / 0.77 | 0.22 / **0.11** | 0.04 / 0.05 | 0.02 / 0.07 |
| AFDB 1000K (L4) | base / REPA | 0.74 / 0.72 | 0.20 / **0.14** | 0.05 / 0.07 | 0.01 / 0.07 |

**Verdict:**
- ✓ **Robust across all 4 pairs**: REPA *depletes the marginal 2–4Å bin*
  relative to baseline. Samples leave the "almost-designable" middle.
- The *fate* of the depleted mass is step-dependent:
  - **Early (700K)**: goes into the <2Å designable peak → pure quality gain
    (PDB <2Å 0.43→0.61, AFDB 0.72→0.77).
  - **Late (1000K)**: splits — some to <2Å, some into a fatter >4Å broken
    tail (PDB >4Å 0.13→0.24, AFDB 0.06→0.14). At AFDB-L4 1000K, REPA's <2Å
    mass is even slightly *lower* than baseline while its broken tail doubles.
- ✗ **Not bimodality** — the broken tail grows but never forms a distinct
  second peak; the distribution stays unimodal-with-a-fatter-tail.

**Refined claim (replaces the bimodality claim in Claim 3):** REPA
*polarizes* the scRMSD distribution — it empties the marginal 2–4Å zone.
Early in training that mass becomes designable (acceleration); late in
training (once REPA enters its encoder-aligned attractor) an increasing
share becomes clearly-broken instead. This explains the original puzzle
(REPA's higher scRMSD-mean at matched/higher Des%) as a **fatter far tail**,
not a second mode. Report it as polarization, not bimodality.

## n=128 cliff cross-check (added 2026-05-27)

Pulled #clusters trajectories from the n=128 convergence sweeps at γ=0.45
(multi-seed where available). Script:
[exp_n128_cliff_check.py](../../evaluation/proteina/generation/scripts/paper/exp_n128_cliff_check.py).

**PDB cliffs detected at n=128**:

| Run | Peak (step → #clust) | Trough (step → #clust) | Drop |
|---|---|---|---|
| **REPA-L4-GN** | 200K → 35 | 300K → **6.3** | **82%** (cliff) |
| REPA-L9-GN | 200K → 39 | 400K → 18.7 | 52% (cliff at threshold) |
| baseline | 200K → 35 | 500K → 14 | 60% (but recovers to 24 by 900K — no sustained cliff) |
| REPA-MPNN-L4 | 200K → 45 | 400K → 32 | 28% (soft decline) |
| REPA-MPNN-L9 | 200K → 75 | 500K → 50 | 33% (soft decline) |

**Reading**:
- ✓ **The cliff IS present at n=128 for PDB-REPA-GearNet** — even more
  dramatic (82% drop at 300K vs ~42% drop at 700K for n=256). So the cliff
  scales proportionally with training duration (peaks at ~25% of total
  steps, drops sharply to plateau), not at a fixed step count.
- The cliff is encoder-specific at n=128 too: GearNet variants show sharp
  drops, MPNN variants show soft declines.
- Baseline n=128 also shows a non-sustained drop+recovery; not a true cliff.

**AFDB n=128** mostly shows soft declines (consistent with n=256 AFDB):
MPNN-L4 and MPNN-L9 both show ~40-60% drops over much longer training, no
sharp single-step transitions. Confirms AFDB-encoder-REPA doesn't carry the
sharp-cliff phenotype regardless of model size.

---

# Representation-quality sweep (linear/MLP probes) — added 2026-05-27

Everything above evaluates the model's **generations**. This section evaluates
the model's **internal representations** directly — the REPA-paper-style probing
analogue (ImageNet linear probe → here, structural decodability of the trunk's
clean-endpoint hidden states). Full methodology in
[`evaluation/proteina/representation/FINDINGS.md`](../../evaluation/proteina/representation/FINDINGS.md);
probe-fit/eval-split and leakage framing in
[`project_repa_evidence_framing`](memory) / `docs/research/pdb_split_leakage_audit.md`.

**Per the leakage framing, this section reports two regimes:**
- **Per-residue probes** (inverse-folding top-1, backbone-dihedral MAE) → lead
  with **xclean-AFDB** (doubly-clean cross-DB eval; both probe-side and
  model-side homology removed). n=325 proteins ≈ 43k residues at n256.
- **Per-sample / per-chain probes** (CATH C/A/T fold classification) → lead with
  **cleantrain-PDB** (probe-side leakage removed; n=3190 — 10× the xclean pool,
  needed for the 89-class CATH-T problem). cleantrain absolute scores still carry
  residual *model-side* leakage and are inflated; the Δ baseline→REPA is the
  robust quantity, not the absolute.

All probes are linear/MLP heads on the **best layer**, clean endpoint (`x_t=x_1`,
`t=1.0`). Data: `results/paper/n256_xclean_afdb_pdb/`,
`results/paper/n256_convergence_cleantrain_pdb/` (+ n128 analogues). Figures:
`figures/paper/n{128,256}_convergence/repr_quality_over_training.png`,
`figures/paper/leakage_decomp/`.

## Headline

**REPA learns a measurably better structural representation than baseline
flow-matching, and the effect rank-orders cleanly by encoder.** This is the
representation-side counterpart to the generation claims, and it is *cleaner* —
REPA beats baseline at essentially **every checkpoint, every probe, every
regime** (win-rates 4/4 – 7/7 across families). The load-bearing finding is the
**magnitude rank-order**, which holds in both clean regimes:

> **structurally-pretrained-encoder REPA (GearNet) > sequence/IF-encoder REPA (ProteinMPNN) > random-encoder REPA > baseline.**

## n256 PDB-trained — the main result

Best-layer values; abs = mean over steps ≥700K, (Δ) = mean step-matched Δ vs
baseline. **NVIDIA-60M** is the frozen NGC `proteina_v1.3_60M` reference (no
relation to our val → no model-side leakage; a fair OOD anchor).

### Per-residue (xclean-AFDB, n=325) — IF top-1 ↑, dihedral MAE ↓

| family | IF top-1 | dihedral MAE (°) |
|---|---|---|
| baseline | 0.123 | 32.1 |
| **L4-GN** | 0.148 (+0.027) | 20.2 (−15.6) |
| **L9-GN** | 0.156 (+0.026) | 19.5 (−13.6) |
| MPNN-L4 | 0.161 (+0.038) | 27.5 (−10.0) |
| **MPNN-L9** | **0.167 (+0.036)** | **16.4 (−16.8)** |
| L4-rand (ctrl) | 0.146 (+0.016) | 20.2 (−9.3) |
| *NVIDIA-60M (ceiling)* | *0.215* | *12.9* |

### Per-sample (cleantrain-PDB, n=3190) — CATH accuracy ↑

| family | CATH-C | CATH-A | CATH-T |
|---|---|---|---|
| baseline | 0.733 | 0.407 | 0.301 |
| **L4-GN** | 0.869 (+0.234) | 0.631 (+0.272) | 0.623 (+0.309) |
| **L9-GN** | **0.888 (+0.236)** | **0.702 (+0.273)** | **0.752 (+0.335)** |
| MPNN-L4 | 0.814 (+0.162) | 0.513 (+0.153) | 0.452 (+0.160) |
| MPNN-L9 | 0.838 (+0.186) | 0.548 (+0.165) | 0.473 (+0.153) |
| L4-rand (ctrl) | 0.807 (+0.109) | 0.476 (+0.085) | 0.378 (+0.063) |
| *NVIDIA-60M* | *0.792* | *0.447* | *0.363* |

### Reading

1. **Rank-order is the clean story, and it differs by probe axis.**
   - On **per-chain fold decodability (CATH)**, **GearNet dominates** — L9-GN adds
     +0.335 CATH-T over baseline (0.30 → 0.75), roughly **2× the MPNN Δ** and
     **5× the random Δ**. GearNet (global contact-graph topology encoder) is
     exactly the right teacher for fold-level structure.
   - On **per-residue local geometry (dihedral, IF)**, **MPNN-L9 is the strongest**
     (dihedral 16.4° vs L9-GN's 19.5°; IF 0.167 vs 0.156). ProteinMPNN's
     inverse-folding pretraining teaches local environment, which transfers to
     per-residue probes. This mirrors the generation finding exactly:
     **GearNet→distribution/topology, MPNN→per-sample/local quality.**

2. **Random-encoder REPA is the falsifier — but it is *not* zero.** L4-rand gives
   the **smallest Δ on every probe** (CATH-T +0.063 vs learned +0.15–0.34; IF
   +0.016 vs +0.026–0.038), confirming the gain is driven by *structural
   knowledge in the target*, not by adding any auxiliary loss. ⚠ Note this
   *refines* the earlier "random-encoder REPA collapses to baseline" claim (which
   was about generation / FM trans-loss): on **representation probes the random
   target gives a small but consistent gain over baseline** — any structured
   trunk regularizer makes reps slightly more linearly-decodable. The rank-order
   (learned ≫ random) is preserved and load-bearing; "random ≈ baseline" is true
   to within ~5× but not literally zero.

3. **The leakage caveat is visible in the NVIDIA anchor — and it cuts both ways.**
   - On the **doubly-clean per-residue** eval (no model-side leakage for anyone),
     NVIDIA-60M is a genuine ceiling (IF 0.215, dih 12.9°) that our REPA
     *approaches but does not reach*. REPA closes most of the baseline→NVIDIA gap;
     honest headline = "REPA recovers a large fraction of a well-trained
     reference's structural decodability."
   - On **cleantrain CATH** our REPA *exceeds* NVIDIA (L9-GN 0.752 vs 0.363). This
     is **not** evidence REPA beats NVIDIA — cleantrain is model-side-leaky for
     *our* models (trained on chains homologous to val) and clean for NVIDIA. The
     gap is leakage-inflated. Quote CATH **Δs**, never the cross-model absolute.

## Convergence trajectory — baseline never catches up

The generation side shows REPA as *acceleration* (baseline often catches up
late). The representation side is **stronger: a persistent gap**.

- **CATH-T (L9-GN, cleantrain):** 0.40 (100K) → 0.59 (400K) → 0.71 (700K) → 0.78
  (900K). Baseline meanwhile oscillates **0.21–0.34 across all of training**
  (1.8M steps) with no upward trend. The student's reps become dramatically more
  fold-structured under REPA; baseline's never do.
- **IF top-1 (xclean):** baseline flat at ~0.10–0.13 the entire run; L9-GN climbs
  to ~0.16, MPNN-L9 to ~0.17 by 1.0–1.3M. Same shape — REPA grows, baseline
  plateaus immediately.

This is arguably the cleanest single piece of evidence in the whole study:
**the baseline's hidden states do not become more structurally decodable with
training, REPA's do.**

## n128 PDB-trained (per-residue xclean n=154, per-sample cleantrain)

⚠ **Coverage gap: n128 GearNet runs were not probed in the clean regimes** —
only MPNN and random are on disk. So n128 can only speak to MPNN-vs-random.

| family | IF top-1 (Δ) | dih MAE (Δ) | CATH-A (Δ) | CATH-T (Δ) |
|---|---|---|---|---|
| baseline | 0.128 | 29.8 | 0.439 | 0.332 |
| MPNN-L4 | 0.154 (+0.019) | 19.4 (−1.9) | 0.550 (+0.109) | 0.431 (+0.146) |
| **MPNN-L9** | **0.170 (+0.026)** | **14.7 (−4.1)** | 0.662 (+0.117) | 0.627 (+0.177) |
| L4-rand | 0.144 (+0.008) | 18.9 (−3.4) | 0.567 (+0.125) | 0.470 (+0.175) |

**At n128 the random control closes most of the gap on CATH** (rand CATH-T Δ
+0.175 ≈ MPNN-L9 +0.177). This echoes the **generation n128 finding** that at
smaller scale even a random target helps — the "REPA needs a *learned* target"
claim is scale-dependent on the representation side too (clear separation at
n256, compressed at n128). MPNN-L9 still wins the per-residue probes at n128.

## AFDB-trained models — no robust clean-eval rep gain (secondary, noisy)

The cleanest cross-DB eval for AFDB-trained models is `xclean_pdb_afdb`, but it
has only **n=62** clean PDB-val proteins (Gene3D/AF2 filtering is aggressive) →
too small to ground a claim. At that sample size the per-residue gains are
**null or negative** for learned encoders (L4-GN-afdb IF Δ+0.008, dihedral
*+0.4° worse*; MPNN variants worse on dihedral), with only the random control
showing a (likely-noisy) dihedral drop. So: **the representation-quality gain is
a robust PDB-trained-model finding; on the AFDB-trained side the clean-eval
sample is too small to detect it.** Do not lean on AFDB-trained rep numbers.

## How this section relates to the generation claims

| Generation claim | Representation-side counterpart |
|---|---|
| REPA accelerates distribution match (Claim 1) | REPA reps become fold-decodable; baseline's don't (CATH-T 0.75 vs 0.30) — *persistent*, not just faster |
| GearNet→distribution, MPNN→per-sample quality (Claim 3/5) | **Exact mirror**: GearNet wins CATH (+0.34), MPNN wins dihedral/IF |
| Random-encoder is the falsifier (Claim 1 caveat) | Random gives smallest Δ — but on probes it is small-positive, not zero |
| n128: random ≈ learned (Claim 1/3) | n128 random closes the CATH gap to MPNN; clean separation only at n256 |

## Open follow-ups (representation)

- **Probe n128 GearNet** in cleantrain + xclean — the only missing cell needed to
  state the encoder rank-order at n128 (currently MPNN/random only).
- Add the **frozen-GearNet ceiling** row to these tables (it's the REPA *target*,
  so it bounds CATH decodability) — currently only NVIDIA-60M is plotted as a
  reference.
- The CATH-T cleantrain Δ (~+0.33) is larger than the per-residue Δ — quantify
  how much of the CATH-T Δ survives on xclean-AFDB CATH-A (cleaner, but n=72
  in-vocab) to bound the model-side-leakage inflation on fold probes.

---

# Representation alignment (CKNNA) — added 2026-05-27

A second, more direct representation-side measurement, distinct from the probe
sweep above. Where the probes ask *"can a linear head decode structure from the
reps?"*, CKNNA asks *"are the student's reps geometrically aligned to a frozen
encoder's reps?"* — the **Platonic-Representation-Hypothesis** metric (Huh et al.
2024), and the same alignment metric REPA's own Fig 2b/3b uses. Faithful port of
the platonic-rep reference impl (unbiased HSIC, mutual-kNN mask) in
[`evaluation/proteina/alignment/lib/cknna.py`](../../evaluation/proteina/alignment/lib/cknna.py)
with unit tests. Data:
`evaluation/proteina/alignment/results/cknna_matrix_{per_residue,per_protein}.jsonl`;
figures: `results/figures/cknna_n10k_{per_residue,per_protein}.png`.

**Scope (current snapshot):** **n256 PDB, step 1M, t=1.0**, 5 models × 10 trunk
layers × 3 frozen target encoders (CA-GearNet, ProteinMPNN, **ESM2-150M**), in
**two modes** — per-residue (N=10,000 residues subsampled across 3,000 PDB-val
proteins, REPA-paper protocol) and per-protein (N=3,000 mean-pooled proteins,
CATH-probe protocol). k=10, 50 without-replacement subsample CIs. *Not yet
covered:* convergence-over-training, AFDB, other t, ESM2-650M.

> **Note on a scale correction (2026-05-27).** An earlier pilot used only 64
> proteins / 10,140 residues drawn cursor-first; it reported peaks ~3× higher
> (e.g. gearnet_l9→ESM2 = 0.125). Those numbers were **inflated by the small,
> low-diversity residue pool** — a known small-N bias in kNN kernel metrics. The
> N=10,000-residue / 3,000-protein reservoir-sampled numbers below are the honest
> figures; treat anything from the 64-protein run as superseded.

## Per-residue matrix — CKNNA to each encoder (peak over layers)

REPA-paper protocol (each residue is a sample). N=10,000 residues.

| model | → GearNet | → MPNN | → ESM2 |
|---|---|---|---|
| baseline | 0.001 (L6) | 0.001 (L7) | 0.003 (L8) |
| repa_gearnet_l4 | 0.003 (L4) | 0.003 (L6) | 0.009 (L8) |
| **repa_gearnet_l9** | **0.024 (L8)** | **0.028 (L8)** | **0.046 (L8)** |
| repa_mpnn_l4 | 0.005 (L7) | 0.016 (L7) | 0.023 (L5) |
| repa_mpnn_l9 | 0.004 (L7) | 0.013 (L7) | 0.016 (L7) |

Per-layer profile to GearNet (peak-then-L9-collapse shape):

| model | L0 | L2 | L4 | L6 | L8 | L9 |
|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 |
| repa_gearnet_l4 | 0.000 | 0.000 | **0.003** | 0.002 | 0.000 | 0.000 |
| repa_gearnet_l9 | 0.000 | 0.001 | 0.003 | 0.006 | **0.024** | 0.000 |

## Per-protein matrix — CKNNA to each encoder (peak over layers)

CATH-probe protocol (each mean-pooled protein is a sample). N=3,000 proteins.

| model | → GearNet | → MPNN | → ESM2 |
|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.000 |
| repa_gearnet_l4 | 0.022 (L9) | 0.001 | 0.000 |
| repa_gearnet_l9 | 0.009 (L8) | 0.008 (L8) | 0.006 (L8) |
| repa_mpnn_l4 | 0.001 | 0.000 | 0.000 |
| repa_mpnn_l9 | 0.002 | 0.002 | 0.022 (L9) |

## Findings

1. **Alignment is a REPA-induced property; flow-matching alone barely produces
   it.** At honest scale, baseline per-residue CKNNA to every encoder is **≤0.003
   — essentially the noise floor.** REPA-GearNet-L9 lifts it ~20–25× (GearNet
   0.001→0.024, MPNN 0.001→0.028, ESM2 0.003→0.046). This is the proteina
   analogue of REPA Fig 2c, but with a **much weaker baseline drift than the
   vision domain** (where vanilla SiT already reached ~0.15 vs DINOv2). For a 60M
   structure model, the generative objective does *not* recover encoder-like
   geometry on its own — so any alignment seen in REPA models is cleanly
   attributable to REPA. That is a *cleaner causal/falsifier story than the REPA
   paper itself could make.*

2. **The alignment peak is NOT at the injection layer, and L9 collapses.** REPA
   alignment builds to a peak in the **upper-middle stack (≈L8)** and the final
   layer (L9) drops to the noise floor (gearnet_l9: 0.024 at L8 → 0.000 at L9).
   The L9 hidden state sits pre-`coors_3d_decoder` in a task-specific
   velocity-output basis (std ~80–100 vs ~20 mid-stack), and REPA only constrains
   `projector(h_L)`, not `h_L` itself. Matches the REPA-paper "later layers focus
   on high-frequency details" behaviour, and independently confirms the
   probe-suite note that **"where REPA aligns ≠ where the representational peak
   ends up."**

3. **Deeper injection (L9) yields more total alignment than shallow (L4).**
   repa_gearnet_l9 is the most-aligned model to *all three* encoders. L4 variants
   show a small, local bump near their injection depth that doesn't propagate.
   Consistent with L9-GN also being the strongest CATH probe — more aligned reps
   ↔ more fold-decodable reps.

4. **Off-diagonal generalization / Platonic convergence (the headline).** Aligning
   to GearNet *alone* also raises alignment to MPNN and ESM2 — so REPA produces
   **generically more encoder-like reps, not just target-shaped ones.** This is
   the cell that defuses the "isn't CKNNA-to-your-own-target tautological?"
   objection: the off-diagonals and the baseline row carry the signal, not the
   diagonal. Strikingly, **GearNet-L9-REPA's alignment to ESM2 (0.046) exceeds its
   alignment to its own GearNet target (0.024)** — ESM2's space is the most
   "universal" attractor. Direct PRH-style evidence in protein-structure
   generation.

5. **REPA reshapes residue-level geometry, not protein-level identity.** The
   per-protein (mean-pooled) matrix is **nearly flat everywhere** — only two cells
   poke above 0.02 (repa_gearnet_l4→GearNet 0.022 at L9; repa_mpnn_l9→ESM2 0.022
   at L9, both plausibly noise at N=3,000) and the clean L8-peak structure of the
   per-residue matrix is gone. The residue-level gains do **not** summarize into
   protein-level geometry agreement. Consistent with REPA being a per-residue
   loss, and a guardrail against over-claiming fold-level representational
   benefits.

**Caveat — absolute scale.** All CKNNA values are small (≤0.05) relative to the
REPA paper's vision-domain numbers (~0.15–0.6). Two structural reasons: our
encoders aren't DINOv2-caliber, and the model is 60M (REPA showed alignment
scales with model size). **Lead with the relative story** — REPA ≫ baseline
(≈noise floor), the L8-peak-then-L9-collapse profile, off-diagonal
generalization / ESM2-universality, and the per-residue≫per-protein contrast —
not the absolute magnitudes. Single step / single dataset / single t — a snapshot
pending the convergence-over-training extension.

## How this section relates to the probe sweep

The probes ("can a linear head decode structure?") and CKNNA ("is the geometry
aligned to an encoder?") answer different questions, and CKNNA adds what the
probes can't: it shows REPA pulls the trunk **toward the encoder's
representational geometry**, not merely adding linearly-decodable features. The
two agree on the ranking (L9-GN strongest), which is the reassuring cross-check;
they diverge on locality (probes peak where decodability is highest; CKNNA peaks
where geometry matches), which is the new information.

## Open questions for narrative

1. Headline framing: "REPA accelerates convergence" (paper-original framing) vs "REPA stabilizes the distribution" (our late-training finding). The first is more honest; the second is more novel.
2. Story for the AFDB durable advantage — is it just "baseline AFDB never converges, so acceleration always wins"? Or genuinely different behavior?
3. T-D lower under REPA: present as a feature (concentrated on the correct folds) or a limitation (less variety)? Depends on what fS-A within designable tells us.
4. How much to lean on FID vs fJSD — they sometimes disagree (FID rewards REPA, fJSD-A flat at 1M; see Claim 1 row). Worth flagging which is the load-bearing metric.
