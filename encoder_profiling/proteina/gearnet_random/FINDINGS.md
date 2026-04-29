# CA-GearNet (random-init) — Architecture-Only Floor

**Date**: 2026-04-29
**Encoder**: `GearNetPerResidueEncoder(random_init=True, random_seed=s)` — same 8-layer, 512-dim CA-GearNet architecture as [../gearnet/](../gearnet/), but with **freshly initialised weights and no pretraining**. Output is whatever the architecture produces from CA coordinates alone, before any training signal.
**Data**: 200 PDB train proteins (42 034 residues), `--random-seed 0`.
**Seeds**: 3 (init seeds 0, 1, 2). All numbers below are **mean ± std across seeds** unless noted.
**SLURM**: 28596016 (full sweep). Aggregated results: [results/20260429_135443/mean_std.json](results/20260429_135443/mean_std.json). Per-seed run dirs: [seed0](results/20260429_135443/seed0/), [seed1](results/20260429_135443/seed1/), [seed2](results/20260429_135443/seed2/).
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md), [../gearnet/FINDINGS.md](../gearnet/FINDINGS.md).

## Why this matters

Random init is the **architecture-only floor**. The trained CA-GearNet's metrics are only meaningful relative to what the architecture produces with no training: a metric that doesn't move noticeably off this floor is one where pretraining is contributing nothing and REPA has nothing useful to align against.

We run **three seeds** because random init has high seed variance, and a single point estimate would mislead. The floor is a distribution, not a constant.

## 1. Headline numbers (random vs trained CA-GearNet)

| Metric                              | Random (3 seeds)         | Trained CA-GearNet | Trained gain |
|-------------------------------------|--------------------------|-------------------:|-------------:|
| Effective rank                      | **3.3 ± 0.3** / 512      |               77.5 |       **23×** |
| Within-protein vs between Δ         | **0.036 ± 0.001**        |              0.222 |        **6×** |
| Linear AA-probe accuracy            | **0.128 ± 0.001**        |              0.137 |    +0.9pp    |
| Projector mean-dir baseline         | **0.952 ± 0.001**        |              0.425 |   −0.527 (lower is better headroom) |
| Best projector (`onehot+pos`) test  | **0.954 ± 0.001**        |              0.432 |   −0.522    |
| Projector saturation **gap**        | **+0.003 ± 0.000**       |             +0.006 |    +0.003   |
| Pert@1 Å cos                        | **0.996 ± 0.000**        |              0.269 |   −0.727 (lower = more 3D signal) |

Read it as: pretraining **massively** lifts effective rank and within-protein discrimination; **does not** add residue-identity signal (CA-only input — neither random nor trained encoder can encode AA identity); and adds only a thin slice of projector headroom on top of an already much-easier baseline.

## 2. What the floor looks like in detail

### Distribution & sparsity
- Mean 0.07 ± 0.08, std 3.29 ± 0.18.
- Negative fraction 49% — symmetric around zero, as expected for randomly initialised LeakyReLU.
- Exact zeros 0%, dead dims 0/512.

### Dimensionality
- Effective rank **3.3 ± 0.3** / 512.
- Participation ratio 1.5 ± 0.06.
- Dims for 90 / 95 / 99% var: 7 ± 2 / 39 ± 7 / 132 ± 5.

The random encoder maps every CA coordinate set to a near-1-D ray with some structured noise around it. The 99%-var bound (~132 dims) shows there *is* spread, but the top few directions dominate.

### 3D sensitivity
| Perturbation | Random cos | Trained cos |
|--------------|-----------:|------------:|
| 0.1 Å | 0.9996 | 0.933 |
| 0.5 Å | 0.998  | 0.367 |
| 1.0 Å | 0.996  | 0.269 |
| 2.0 Å | 0.986  | 0.187 |
| 5.0 Å | 0.900  | −0.002 |
| Random rotation | 1.000 | 0.99968 |

Random init is **insensitive to coordinates** — embeddings barely move when you perturb the structure by 5 Å. This is *not* because the architecture is rotation-only invariant (it is, by construction) — it's because the random GearNet message-passing collapses inputs to a near-constant subspace before the perturbation can register. Pretraining is what gives the encoder its sub-Å sensitivity.

### Residue-type discrimination
- Probe accuracy **0.128 ± 0.001** vs trained 0.137 — ~chance for a 20-class problem with mass concentrated on ALA/LEU/VAL.
- Centroid cos 0.998: per-AA-type centroids are nearly indistinguishable, just like the trained encoder.

CA-GearNet has no residue-type input, so neither random nor trained encodes AA identity. This row is essentially a sanity check, not a signal.

### Projector saturation
3-layer MLP, 80/20 train/test, 300 epochs.

| Input condition          | Test cos          |
|--------------------------|-------------------|
| Mean direction (no MLP)  | **0.952 ± 0.001** |
| Random 128-d             | 0.953 ± 0.001     |
| AA one-hot (21-d)        | 0.954 ± 0.001     |
| AA one-hot + position    | **0.954 ± 0.001** |

**Saturation gap = +0.003.** The mean-direction baseline is enormous (0.95) because the random encoder produces near-constant embeddings — every residue sits on the same ray, so cosine to the dataset mean is already near-perfect. There is essentially no degree of freedom for the projector to fit.

This is the comparison that makes the trained gap (+0.006) interpretable: "trained adds about as much projector-extractable signal as random does," which means most of the +0.006 trained gap is shared with random and only ~half is genuinely novel structural information.

### Within-protein vs between-protein
- Within-protein cos 0.937 ± 0.001
- Between-protein cos 0.901 ± 0.002
- **Δ 0.036 ± 0.001**

Random init still produces *some* protein-specificity (Δ 0.036 > 0), because two residues in the same protein share local graph structure and message passes propagate through it even with random weights. Trained CA-GearNet's Δ 0.222 is **6× this floor** — the metric where pretraining most clearly contributes.

## 3. Layer-wise: random vs trained

Per-layer effective rank, averaged across seeds:

| Layer | Random eff rank | Trained eff rank | Direction |
|------:|----------------:|-----------------:|-----------|
|     0 |            56.8 |             62.0 | comparable |
|     1 |            39.8 |             61.2 | trained higher |
|     2 |            24.8 |             62.0 | trained higher |
|     3 |            15.1 |             69.8 | trained higher |
|     4 |             9.8 |             78.8 | trained higher |
|     5 |             6.3 |             86.4 | trained higher |
|     6 |             4.4 |             87.3 | trained higher |
|     7 |             3.4 |             68.2 | trained higher |

The shapes are completely opposite:

- **Random**: rank decays *monotonically* with depth (56.8 → 3.4) — every additional message-passing step compresses the representation onto fewer directions, with no gradient pressure to maintain spread. This explains the eff rank 3.3 readout: deep random GNNs collapse.
- **Trained**: rank *rises* through depth (62 → 87 by L6) and then dips at the readout (L7 = 68). The pretraining objective explicitly resists collapse and uses the depth productively.

Layer 0 is the only layer where random and trained are comparable in rank. By the readout, the trained encoder has 20× more usable directions. **This is the cleanest single signature of what pretraining does to CA-GearNet.**

## 4. Implications for REPA

1. **The trained projector gap (+0.006) is already small; subtract the random gap (+0.003) to get the genuinely-novel part.** Net "structural signal beyond random" is ~+0.003 — very tight. This is consistent with the modest empirical REPA gains observed at 128/256-residue scales.
2. **Pretraining's big contribution is not what REPA aligns against.** The 23× rank lift and 6× within-protein-Δ lift are the ways the trained encoder is dramatically better, but cosine similarity loss is mostly insensitive to rank — it cares about direction. So REPA can't easily extract these gains.
3. **Sanity-check future encoders against this floor.** Any new candidate (PW variants, 3D-aware GNNs, etc.) should report its metrics relative to this random-init line. An encoder whose trained-vs-random gap on rank, within-protein-Δ, and projector saturation is comparable to CA-GearNet's is a credible REPA target. Without that gap, pretraining isn't doing useful work and the candidate should be rejected before training time is spent on it.

## Notes & caveats

- **Three seeds is a minimum, not a luxury.** Seed-to-seed std on most metrics is small (~0.001 on projector cos, ~0.3 on eff rank), but eff rank had visible seed variance (3.0 / 3.5 / 3.4 across seeds 0–2). With one seed we'd have no error bar; with three we can see the floor is tight.
- **The same 200 proteins are used across seeds**, so within-seed noise is purely from random weight initialisation — not from data sampling.
- **Same architecture, same forward pass, same probe.** The only thing that differs from [../gearnet/](../gearnet/) is the weight initialisation. This isolates the effect of pretraining.
