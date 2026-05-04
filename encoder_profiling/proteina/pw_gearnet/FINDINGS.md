# ProteinWorkshop GearNet-Edge characterization

**Date**: 2026-05-04 (torsional re-run with RankMe metric). Structure-denoising last profiled 2026-04-24 (job 28308561) and is **not** included in the latest sweep — see "stale" tags throughout.
**Encoder**: PWGearNet-Edge (`PWGearNetEdgePerResidueEncoder`, 6 layers, 3072-dim concat output)
**Checkpoints (Zenodo 8287754)**:
- Torsional: `pw_gearnet_torsional_denoising_ca_angles.ckpt`
- Structure: `pw_gearnet_structure_denoising_ca_angles.ckpt`
**Feature config**: `ca_angles` (43-d node input: AA[23] + seq_pe[16] + alpha[2] + kappa[2])
**Data**: 200 PDB train proteins (42 034 residues)
**SLURM**: 28852949 (full sweep, torsional only). Latest results: [results/torsional_20260504_183511/results.json](results/torsional_20260504_183511/results.json), [layerwise.csv](results/torsional_20260504_183511/layerwise.csv).
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md) — for the three-question framing (Q1 information, Q2 saturation, Q3 conditioning) used throughout.

## TL;DR

- **Torsional variant — borderline.** Q3.2 RankMe 207.7 / 3072 (PR 5.5; 95% variance in 36 dims). Q2 projector saturation gap = +0.013 (mean-dir 0.710 → best 0.723). Q1.2 3D-sensitive (cos = 0.92 at 0.5 Å). Q1.1 strong AA identity signal (probe 0.928). Usable if no better option, but ESM2 (gap +0.053) and CA-GearNet (3D headroom in geometric terms) are both stronger choices.
- **Structure variant — borderline-poor (stale numbers, old metric).** Last run 2026-04-24: Q3.2 effective rank 19.9 (under the old σ²-weighted definition; not comparable to the new RankMe numbers below), Q2 mean-dir 0.781, onehot 0.794 → gap ~0.013. Numbers below were collected before the audit (`projector_num_layers` 2→3 fix and per-residue averaging change) and the metric switch, so they aren't directly comparable to the current pipeline.
- Both PW variants sit between ESM2 and MC-GearNet-Edge in saturation. Same root-cause family as MC-GN-Edge (`concat_hidden=True`, no final LayerNorm), but mid-layer rank and the edge-to-node scatter rescue most of the signal — PW's RankMe (208) is ~17× MC-GearNet's (12).

## Port & forward pass (critical implementation detail)

PW differs from MC-GearNet-Edge — keep this section even when results drift; it's a frequent source of bugs when porting GearNet variants.

The Zenodo checkpoints are Lightning `.ckpt` files with all encoder weights under `encoder.*`. Architecture is 6-layer GearNet-Edge with `concat_hidden=True`:

```
node_dims  = [43, 512, 512, 512, 512, 512, 512]
edge_dims  = [89, 43,  512, 512, 512, 512, 512]   # line-graph chain
output_dim = 3072  (concat of 6 × 512 hidden layers)
```

Verified against `proteinworkshop/models/graph_encoders/gear_net.py`. The forward loop:

```python
for i in range(len(layers)):
    hidden = layers[i](batch, layer_input)           # uses RAW edge_feature (89-d)
    if hidden.shape == layer_input.shape:
        hidden = hidden + layer_input                # short-cut
    edge_hidden = edge_layers[i](line_graph, edge_input)   # line-graph chain
    node_out = edge_index[1] * num_relation + edge_index[2]
    update = scatter_sum(edge_hidden, node_out).view(N, num_relation * edge_hidden.dim)
    update = relu(layers[i].linear(update))          # REUSES node layer.linear!
    hidden = hidden + update
    hidden = batch_norms[i](hidden)                  # top-level BN
    edge_input = edge_hidden                         # chain edge state forward
```

Two subtleties that took iteration:

1. `layers[i].edge_linear` in the checkpoint has `in_features = 89` at **every** layer → node layers consume the **original** edge feature (`f_ji`), not the updated `edge_hidden`. Passing `edge_hidden` instead caused a `mat1 (2672×43) × mat2 (89×43)` dim mismatch (job 28307906).
2. `layers[i].linear` is used **twice** per layer: once for the standard `(num_relation × input_dim) → output_dim` aggregation, and once more for the edge-to-node scatter update. Works because edge_hidden's dim matches `layer.linear`'s input dim at every layer (43 at L0, 512 at L1+).

**Forgetting the edge-to-node scatter step** (job 28308226) produced a drastically degraded representation — under the old σ²-weighted metric, effective rank dropped to 3.3 (vs the complete forward's ~12), projector saturation collapsed to ~0.94, and 3D sensitivity disappeared (cos = 0.995 at 0.5 Å). The complete forward (current numbers) reaches RankMe ~208 / 3072 and 3D sensitivity (cos ≈ 0.92 at 0.5 Å).

## Q1. What information does the encoder encode?

### 1.1 Residue identity

**Torsional (2026-05-04)**

| Metric | Value |
|--------|-------|
| Linear probe accuracy | **0.928** |
| Mean cos between AA centroids | 0.977 |
| Within-type cos | 0.517 ± 0.227 |
| Between-type cos | 0.507 ± 0.233 |
| Δ (within − between) | **+0.010** |
| Residue-shuffle cos (coords fixed, labels permuted) | 0.995 ± 0.001 |

Probe is high (0.927) because residue type is a node-feature input, but the centroid-cos 0.977 and Δ ~0.01 say the per-AA cluster centers are nearly indistinguishable in direction — identity is encoded by axis-aligned subspace differences too small to dominate cosine similarity. Residue-shuffle cos 0.995 means the embedding is dominated by *structure*, not residue identity, even though residue type is in the node feature.

**Structure (2026-04-24, stale)**

| Metric | Value |
|--------|------:|
| Linear probe accuracy | 0.96 |
| Residue-shuffle cos | 0.995 |

Higher probe accuracy than torsional but the same residue-shuffle cos — same qualitative picture: identity is in the embedding but doesn't drive its direction.

### 1.2 3D geometric sensitivity

**Torsional (2026-04-29)**

| Perturbation | Cosine similarity |
|--------------|------------------:|
| 0.1 Å | 0.989 ± 0.003 |
| 0.5 Å | **0.922 ± 0.017** |
| 1.0 Å | 0.846 ± 0.030 |
| 2.0 Å | 0.739 ± 0.041 |
| 5.0 Å | 0.489 ± 0.058 |
| Random rotation | 0.99999 ± 5×10⁻⁶ |

3D-sensitive (0.5 Å → cos 0.92, vs MC's 0.81 and CA-GearNet's 0.37). Less responsive to sub-Å noise than CA-GearNet (which drops to 0.37) but clearly responsive overall. Fully rotation-invariant.

**Structure (2026-04-24, stale)**

| Perturbation | Cosine similarity |
|--------------|------------------:|
| 0.5 Å | 0.913 |

Comparable to torsional at the 0.5 Å mark; full sweep not available under the new pipeline.

### 1.3 Structural context (helix / sheet / loop)

**Torsional (2026-04-29)**

| AA  | within-SS cos | between-SS cos | Δ      |
|-----|--------------:|---------------:|-------:|
| ALA |         0.568 |          0.489 | +0.079 |
| GLY |         0.556 |          0.519 | +0.036 |
| LEU |         0.565 |          0.463 | +0.102 |
| VAL |         0.625 |          0.526 | +0.099 |

All four AAs show strong positive Δ — the torsional pretraining task explicitly depends on SS-specific local geometry, and this comes through clearly. Compare CA-GearNet's deltas (~0.03) and MC-GearNet's (mostly negative).

**Structure (2026-04-24, stale)**

Strongest SS-Δ AA: ALA +0.074. The structure variant compresses harder, which weakens the SS signal relative to torsional.

### 1.4 Protein-level identity (within-protein vs between-protein)

**Torsional (2026-04-29)**

| Metric                | Value          |
|-----------------------|----------------|
| Within-protein cos    | 0.635 ± 0.248  |
| Between-protein cos   | 0.533 ± 0.229  |
| **Δ**                 | **0.102**      |

Modest within-protein clustering — between CA-GearNet's 0.222 and ESM2's 0.098.

**Structure (2026-04-24, stale)** — not measured under the standardised pipeline.

## Q2. How much is reachable from cheap inputs?

### Torsional (2026-05-04)

3-layer MLP, 80/20 train/test, 300 epochs.

| Input condition          | Train cos | Test cos |
|--------------------------|----------:|---------:|
| Mean direction (no MLP)  |        —  | **0.710** |
| Random 128-d             |    0.781  |    0.646  |
| AA one-hot (21-d)        |    0.726  | **0.723** |

**Saturation gap = +0.013.** Above the random-init floor (+0.003) but well below ESM2 (+0.053). Identity alone takes the projector to 0.72.

The random-input row trains to 0.781 but only generalises to 0.646 (gap 0.14) — the MLP partially memorises noise, similar to CA-GearNet. The encoder's true cosine signal is genuinely small once the (onehot + pos) baseline is in place.

### Structure (2026-04-24, stale)

| Input condition          | Test cos |
|--------------------------|---------:|
| Mean direction (no MLP)  | 0.781    |
| AA one-hot              | 0.794    |
| **Estimated headroom**   | **~+0.013** |

Headroom slightly higher than torsional in absolute terms, but the higher mean-dir baseline says more of that signal is structural collapse rather than informative spread. Reprofile under the standardised pipeline before drawing conclusions.

## Q3. Is the encoder a tractable optimisation target?

### 3.1 Sparsity & value distribution

**Torsional (2026-04-29)**

| Metric | Value |
|--------|-------|
| Mean / Std | 0.092 / 0.702 |
| Min / Max | −5.83 / 26.99 |
| Exact zeros | 0.00% |
| Negative values | 49.7% |

Fully dense, near-symmetric. Healthy distribution.

**Structure (2026-04-24, stale)** — distribution stats not preserved in the older run.

### 3.2 Effective dimensionality & singular values

**Torsional (2026-05-04)**

| Metric | Value |
|--------|-------|
| Output dim | 3072 |
| **RankMe** | **207.7** |
| Participation ratio | 5.5 |
| Dims for 90 / 95 / 99% var | 16 / 36 / 125 |
| Top singular value (centered) | 2 864 |
| S[0] / S[−1] (centered) | 2.3 × 10⁷ |

RankMe 207.7 / 3072 = ~7% of capacity. Severely under-utilised in absolute terms, but ~17× MC-GearNet (RankMe 12) — the edge-to-node scatter rescues the spectrum from total collapse. PR 5.5 says variance still concentrates in roughly 5 effective directions.

**Structure (2026-04-24, stale, old metric)**

| Metric | Value |
|--------|-------|
| Effective rank (σ²-weighted, old) | 19.9 |

Higher than the torsional value under the old metric (12.2) but in the same regime; not directly comparable to the new RankMe number above. Both PW variants are starved compared to ESM2 (RankMe 1030) or CA-GearNet (256).

### 3.3 Norms & dead dimensions

**Torsional (2026-04-29)**

| Metric | Value |
|--------|-------|
| Mean L2 norm | 37.2 |
| Std L2 norm | 12.7 |
| Min / Max | 19.87 / 508.95 |
| Dead dimensions | 0 / 3072 |
| Dim std range | [4.9 × 10⁻⁶, 2.95] |

Healthy norms compared to MC-GearNet (mean 1.5 M). No dead dims, narrow std range, well-conditioned.

**Structure (2026-04-24, stale)**

| Metric | Value |
|--------|-------|
| Mean L2 norm | 22.7 |
| Dead dims | 22 / 3072 |

A handful of dead dims and slightly tighter norms — similar regime, no exploding pathology.

## Layer-wise representation

From [layerwise.csv](results/torsional_20260504_183511/layerwise.csv) (torsional only):

| Layer | Dim | RankMe | Mean norm |
|------:|----:|-------:|----------:|
|     0 | 512 |   50.4 |     19.3  |
|     1 | 512 |   47.7 |     19.1  |
|     2 | 512 |   63.8 |     16.5  |
|     3 | 512 | **82.6** | 11.2  |
|     4 | 512 |   57.2 |      9.0  |
|     5 | 512 |   34.5 |     11.2  |

RankMe peaks at **layer 3** (82.6 / 512) — mid-network, before further compression. Norms are all O(10), without the geometric blow-up that ruins MC-GearNet-Edge. **For REPA, aligning to layer 3 in isolation is plausibly better than the concat output** — much higher per-slab RankMe (82 vs 35 at L5), 6× smaller dim, no slab-imbalance problem. Worth a small experiment.

## Reproducing the structure variant under the new pipeline

```
python encoder_profiling/proteina/pw_gearnet/explore_pw_gearnet.py \
    --ckpt $DATA_PATH/metric_factory/model_weights/pw_gearnet_structure_denoising_ca_angles.ckpt \
    --variant structure --n-proteins 200 --random-seed 0
```

This will produce a `results/structure_*/` dir alongside the torsional one and bring the structure variant into apples-to-apples comparison.

## Comparison summary (across PW variants and other encoders)

For the cross-encoder ranking and headline projector-saturation gaps see [../FINDINGS.md](../FINDINGS.md). At a glance for the PW family:

| Property | PW Torsional (2026-05-04, RankMe) | PW Structure (2026-04-24, stale, old metric) |
|----------|----------------------------------:|---------------------------------------------:|
| Rank metric / 3072 | RankMe 207.7 | eff rank 19.9 (old σ²-weighted; not directly comparable) |
| Mean-dir baseline | 0.710 | 0.781 |
| Best projector test | 0.723 | 0.794 |
| Saturation gap | +0.013 | ~+0.013 |
| 3D cos @ 0.5 Å | 0.922 | 0.913 |
| AA probe | 0.928 | 0.96 |
| Strongest SS-Δ AA | LEU (+0.102) | ALA (+0.074) |

## Recommendations

1. **Prefer CA-GearNet** as the REPA target for 3D content. PW has higher absolute cosine similarity but most of it is mean-direction baseline (Q2); CA-GearNet's lower baseline leaves more room for the *geometric* signal (Q1.2) we actually want REPA to teach.
2. If we explicitly want a 3D + sequence-aware encoder, **PW torsional** is the usable variant. Its saturation gap (+0.013) is comparable to CA-GearNet's (+0.009) and the torsional pretraining task gives it the cleanest SS signal of any GearNet variant we've checked (Q1.3).
3. **Try aligning REPA to PW-torsional layer 3** rather than the concat output. Per-slab RankMe is 2.4× the L5 slab, dimensionality is 6× smaller, and the slab-imbalance problem disappears.
4. **Reprofile structure_denoising** under the new pipeline before drawing conclusions about it. The 2026-04-24 numbers used a different projector depth, averaging mode, and the old σ²-weighted rank metric.

## Files referenced

- Port: `src/proteina/proteinfoundation/metrics/gearnet_utils.py` — classes `_PWLayer`, `PWGearNetEdge`, `NoTrainPWGearNetEdge`.
- Wrapper: `src/proteina/proteinfoundation/repa/gearnet_encoder.py` — `PWGearNetEdgePerResidueEncoder`.
- Fetch: `hpc-scripts/proteina/data_prep/fetch_pw_gearnet.sh`.
- Training configs (`encoder.type: pw_gearnet`): `src/proteina/configs/experiment_config/training/{256,256/afdb,128/afdb}/pw_gearnet/per_residue/`.
