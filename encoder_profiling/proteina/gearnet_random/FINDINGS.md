# CA-GearNet (random-init) — Architecture-Only Floor

**Date**: 2026-05-04 (re-run with RankMe; major narrative revision — see § "What changed under RankMe").
**Encoder**: `GearNetPerResidueEncoder(random_init=True, random_seed=s)` — same 8-layer, 512-dim CA-GearNet architecture as [../gearnet/](../gearnet/), but with **freshly initialised weights and no pretraining**. Output is whatever the architecture produces from CA coordinates alone, before any training signal.
**Data**: 200 PDB train proteins (42 034 residues), `--random-seed 0`.
**Seeds**: 3 (init seeds 0, 1, 2). All numbers below are **mean ± std across seeds** unless noted.
**SLURM**: 28852949 (full sweep). Aggregated results: [results/20260504_182201/mean_std.json](results/20260504_182201/mean_std.json). Per-seed run dirs: [seed0](results/20260504_182201/seed0/), [seed1](results/20260504_182201/seed1/), [seed2](results/20260504_182201/seed2/).
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md), [../gearnet/FINDINGS.md](../gearnet/FINDINGS.md). Q1 / Q2 / Q3 framing defined in the cross-encoder file.

## What changed under RankMe

The original 2026-04-29 numbers reported "effective rank 3.3 / 512" using entropy on σ² of the centered embedding. That was unusually small because σ²-weighting is extremely sensitive to spectral concentration; **under RankMe (entropy on raw σ of the *uncentered* matrix; Garrido et al. 2023) the random encoder has RankMe ≈ 100 / 512** — not rank-collapsed. The "rank decays monotonically through depth" claim from the previous version is also gone: under RankMe, per-layer rank rises from 109 at L0, peaks at 121 mid-network, then falls to 93 at L7 (see § layer-wise).

What is collapsed is the **distribution within that span** — `mean_direction_cos = 0.952`, `participation_ratio = 1.5`, `dims_for_95pct_var = 39 ± 8`. The random encoder spreads its outputs across ~100 raw-σ directions, but variance is overwhelmingly concentrated in 1–2 of them, so every residue lives near the same direction on the sphere. **PR (Gao et al. 2017), not RankMe, is the metric that captures this** — pretraining lifts PR ~25× (1.5 → 38) while it lifts RankMe only ~2.6× (100 → 256).

## Why this matters

Random init is the **architecture-only floor**. The trained CA-GearNet's metrics are only meaningful relative to what the architecture produces with no training: a Q1 / Q2 / Q3 metric that doesn't move noticeably off this floor is one where pretraining is contributing nothing and REPA has nothing useful to align against.

We run **three seeds** because random init has high seed variance, and a single point estimate would mislead. The floor is a distribution, not a constant.

## Headline numbers (random vs trained CA-GearNet)

| Metric                              | Random (3 seeds)         | Trained CA-GearNet | Trained gain |
|-------------------------------------|--------------------------|-------------------:|-------------:|
| Q3.2 RankMe                         | **99.8 ± 4.5** / 512     |              256.4 |       2.6×   |
| Q3.2 Participation ratio            | **1.48 ± 0.07**          |               37.9 |    **25×**   |
| Q3.2 Dims for 95% variance          | **38.7 ± 7.7**           |                119 |        3.1×  |
| Q1.4 Within-protein vs between Δ    | **0.036 ± 0.001**        |              0.222 |        **6×** |
| Q1.1 Linear AA-probe accuracy       | **0.127 ± 0.001**        |              0.137 |    +1pp      |
| Q2 Projector mean-dir baseline      | **0.952 ± 0.001**        |              0.425 |   −0.527 (lower is better headroom) |
| Q2 Best projector (`onehot+pos`)    | **0.954 ± 0.001**        |              0.434 |   −0.520    |
| Q2 Projector saturation **gap**     | **+0.003 ± 0.000**       |             +0.009 |    +0.006   |
| Q1.2 Pert@1 Å cos                   | **0.996 ± 0.000**        |              0.269 |   −0.727 (lower = more 3D signal) |

Read it as: pretraining lifts **PR** ~25× (the spectrum becomes much more "equally weighted") and **within-protein discrimination** (Q1.4) ~6×; lifts **RankMe** only ~2.6× (the count of nontrivial directions was already ~100 at random init); **does not** add residue-identity signal (CA-only input — neither random nor trained encoder can encode AA identity, Q1.1); and adds only a thin slice of projector headroom (Q2) on top of an already much-easier baseline.

## Q1. What information does the encoder encode? (random)

### 1.1 Residue identity

- Probe accuracy **0.127 ± 0.001** vs trained 0.137 — ~chance for a 20-class problem with mass concentrated on ALA/LEU/VAL.
- Centroid cos 0.998: per-AA-type centroids are nearly indistinguishable, just like the trained encoder.

CA-GearNet has no residue-type input, so neither random nor trained encodes AA identity. This row is essentially a sanity check, not a signal.

### 1.2 3D geometric sensitivity

| Perturbation | Random cos | Trained cos |
|--------------|-----------:|------------:|
| 0.1 Å | 0.9996 | 0.933 |
| 0.5 Å | 0.998  | 0.367 |
| 1.0 Å | 0.996  | 0.269 |
| 2.0 Å | 0.986  | 0.187 |
| 5.0 Å | 0.900  | −0.002 |
| Random rotation | 1.000 | 0.99968 |

Random init is **insensitive to coordinates** — embeddings barely move when you perturb the structure by 5 Å. This is *not* because the architecture is rotation-only invariant (it is, by construction) — it's because the random GearNet message-passing collapses inputs to a near-constant subspace before the perturbation can register. Pretraining is what gives the encoder its sub-Å sensitivity.

### 1.3 Structural context

Not separately reported under the random-init pipeline; the SS Δ is not a meaningful signal at random init because the embedding is near-constant across SS classes.

### 1.4 Protein-level identity (within-protein vs between-protein)

- Within-protein cos 0.937 ± 0.001
- Between-protein cos 0.901 ± 0.002
- **Δ 0.036 ± 0.001**

Random init still produces *some* protein-specificity (Δ 0.036 > 0), because two residues in the same protein share local graph structure and message passes propagate through it even with random weights. Trained CA-GearNet's Δ 0.222 is **6× this floor** — the metric where pretraining most clearly contributes.

## Q2. How much is reachable from cheap inputs? (random)

3-layer MLP, 80/20 train/test, 300 epochs.

| Input condition          | Test cos          |
|--------------------------|-------------------|
| Mean direction (no MLP)  | **0.952 ± 0.001** |
| Random 128-d             | 0.953 ± 0.001     |
| AA one-hot (21-d)        | 0.954 ± 0.001     |
| AA one-hot + position    | **0.954 ± 0.001** |

**Saturation gap = +0.003.** The mean-direction baseline is enormous (0.95) because the random encoder produces near-constant embeddings — every residue sits very close to the same direction on the sphere (PR 1.5 confirms ~1 dominant variance axis), so cosine to the dataset mean is already near-perfect. There is essentially no degree of freedom for the projector to fit.

This is the comparison that makes the trained gap (+0.009) interpretable: trained adds ~3× the projector-extractable signal that random does, so roughly 1/3 of the +0.009 trained gap is shared with random and ~2/3 is genuinely novel structural information.

## Q3. Is the encoder a tractable optimisation target? (random)

### 3.1 Sparsity & value distribution

- Mean 0.07 ± 0.08, std 3.29 ± 0.18.
- Negative fraction 49% — symmetric around zero, as expected for randomly initialised LeakyReLU.
- Exact zeros 0%, dead dims 0/512.

Random init is dense and well-distributed at the element level — the failure isn't in scalar statistics, it's in directionality (Q3.2 below).

### 3.2 Effective dimensionality

- **RankMe 99.8 ± 4.5** / 512.
- **Participation ratio 1.48 ± 0.07** — this is where the "near-1-D ray" character shows up.
- Dims for 90 / 95 / 99% var: 7 ± 2 / 39 ± 8 / 132 ± 5.

The random encoder maps every CA coordinate set into a roughly 100-direction span (RankMe), but variance is concentrated in 1–2 directions (PR 1.5) — the "near-1-D ray with structured noise" character is captured by PR, not RankMe. The 99%-var bound (~132 dims) is the long *tail* of small singular values; RankMe weights raw σ so it counts that tail almost as heavily as the top components, which is why it lands at ~100 rather than ~3. **PR is the conditioning metric where pretraining most visibly lifts the architecture (1.5 → 38, a 25× gain).** RankMe lift is a more modest 2.6×.

### 3.3 Norms & dead dimensions

Dead dims 0/512 (same as trained); norms not separately reported in the aggregated random-init result. The random encoder's norms grow with depth but stay at machine-friendly magnitudes — there's no MC-GearNet-style explosion.

## Layer-wise: random vs trained

Per-layer RankMe, averaged across seeds:

| Layer | Random RankMe | Trained RankMe | Trained / Random |
|------:|--------------:|---------------:|-----------------:|
|     0 |         109.5 |          145.4 |             1.33× |
|     1 |         120.9 |          193.1 |             1.60× |
|     2 |         121.4 |          225.6 |             1.86× |
|     3 |         119.5 |          247.4 |             2.07× |
|     4 |         116.0 |          259.6 |             2.24× |
|     5 |         108.3 |          264.9 |             2.45× |
|     6 |         100.5 |          262.1 |             2.61× |
|     7 |          93.2 |          239.2 |             2.57× |

The shapes are similar in form but very different in amplitude:

- **Random**: RankMe peaks at L1–L2 (~121) and decays gently towards the readout (93). Not a collapse — about 18% of full rank survives at L7. Variance is concentrated in 1–2 directions throughout (PR ≈ 1.5 at the readout).
- **Trained**: RankMe peaks at L5 (265) and dips slightly at the readout (239). The pretraining objective uses the architecture's bandwidth to spread variance across many directions while preserving overall rank, lifting trained / random from 1.3× at L0 to 2.6× by L7.

The trained-vs-random multiplier rises monotonically with depth, peaking at L6–L7. **This is the cleanest single signature of what pretraining does to CA-GearNet's spectrum** — the deeper layers gain the most from training. (Note: this contradicts the previous version of these notes, which used `entropy on σ²` and saw random RankMe collapse to 3.4 at L7. Under the RankMe convention used by Garrido et al. 2023 and the rest of the encoder-profiling pipeline, the collapse is partial, not absolute.)

## Implications for REPA

1. **The trained projector gap (Q2 +0.009) is already small; subtract the random gap (+0.003) to get the genuinely-novel part.** Net "structural signal beyond random" is ~+0.006 — tight. This is consistent with the modest empirical REPA gains observed at 128/256-residue scales.
2. **Pretraining's big contribution (Q3.2 PR, Q1.4 Δ) is not what REPA aligns against.** The 25× PR lift and 6× within-protein-Δ lift are the ways the trained encoder is dramatically better, but cosine similarity loss is mostly insensitive to where in the spectrum variance sits — it cares about direction. So REPA can't easily extract these gains.
3. **Sanity-check future encoders against this floor.** Any new candidate (PW variants, 3D-aware GNNs, etc.) should report its metrics relative to this random-init line. An encoder whose trained-vs-random gap on PR, within-protein-Δ, and projector saturation is comparable to CA-GearNet's is a credible REPA target. Without that gap, pretraining isn't doing useful work and the candidate should be rejected before training time is spent on it.

## Notes & caveats

- **Three seeds is a minimum, not a luxury.** Seed-to-seed std on most metrics is small (~0.001 on projector cos), but RankMe had visible seed variance (~4.5 std across the 3 seeds, with per-layer std up to 6.5). With one seed we'd have no error bar; with three we can see the floor is tight.
- **The same 200 proteins are used across seeds**, so within-seed noise is purely from random weight initialisation — not from data sampling.
- **Same architecture, same forward pass, same probe.** The only thing that differs from [../gearnet/](../gearnet/) is the weight initialisation. This isolates the effect of pretraining.
