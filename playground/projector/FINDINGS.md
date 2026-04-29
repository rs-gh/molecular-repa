# Projector Analysis: CheMeleon, MACE, GearNet

**First written**: 2026-04-02. **Last refreshed**: 2026-04-29.
**Canonical script**: `playground/projector/encoder_analysis.py`
(legacy `projector_analysis.py` retained for reference; superseded.)
**Figures**: `playground/projector/figures/` — numbers below from `results.json`.

## Question

When REPA training reports high cosine similarity between projected transformer hidden states and encoder targets, how much of that alignment is genuine structural learning vs. the projector simply learning atom/residue type prototypes?

## Method

Three tests per encoder, 80/20 stratified train/test splits:

1. **Mean-direction floor**: cosine sim between the mean embedding and all targets — the floor any constant-output model achieves.
2. **Identity input**: one-hot atom/residue type → 2-layer MLP → encoder targets.
3. **Random input**: random 128-d vectors → same MLP → targets. Sanity check.

MLP `hidden_dim` matches the projector used in training: 128 for CheMeleon/MACE (tabasco `model.net.hidden_dim`), 512 for GearNet (proteina `model.nn.token_dim`).

## Results

| | Floor | Rand (train) | Rand (test) | ID (train) | ID (test) | REPA val |
|---|---|---|---|---|---|---|
| **CheMeleon** | 0.388 | 0.578 | 0.277 | 0.459 | **0.455** | 0.66–0.68 |
| **MACE**      | 0.755 | 0.826 | 0.711 | 0.863 | **0.861** | 0.56 |
| **GearNet**   | 0.419 | 0.433 | 0.413 | 0.426 | **0.426** | 0.80 |

REPA val sources: CheMeleon/MACE from final GEOM production runs (`docs/research/tabasco_training_runs.md`); GearNet from `proteina_60m_repa_layer4_v2`.

### Identity-vs-REPA gap

| | Identity (test) | REPA val | Gap |
|---|---|---|---|
| **GearNet**   | 0.426 | **0.80**      | **+0.374** |
| **CheMeleon** | 0.455 | **0.66–0.68** | **+0.21**  |
| **MACE**      | 0.861 | **0.56**      | **−0.30**  |

## Effective rank

The unified script computes all four definitions per encoder. Earlier per-encoder scripts (`encoder_profiling/tabasco/chemeleon/investigate.py`, `encoder_profiling/tabasco/mace/generate_figures.py`, `encoder_profiling/proteina/gearnet/explore_gearnet.py`) used different sample sets and preprocessing, so their absolute values do not match the table below — treat only this table as canonical.

| Definition              | CheMeleon | MACE | GearNet |
|---|---|---|---|
| Threshold (SV > 1% max) | 1265      | 110  | 512     |
| Entropy (norm SVs)      | 872.4     | 58.0 | 272.7   |
| Entropy (norm variance) | 212.0     | 10.5 | 82.6    |
| PCA 90% variance        | 341       | 12   | 82      |
| Total dims              | 2048      | 192  | 512     |

## Takeaways

- **GearNet has the largest gap (+0.374).** Identity barely predicts GearNet embeddings (0.426 ≈ floor 0.419), yet REPA reaches 0.80 — the transformer learns substantial structural information.
- **CheMeleon is moderate (+0.21).** Identity already accounts for ~⅔ of the alignment; REPA adds the rest.
- **MACE is negative (−0.30) but confounded — see caveat below.**
- **Random input overfits on small datasets.** Train→test for CheMeleon (0.578→0.277) and MACE (0.826→0.711) is large; GearNet (0.433→0.413) is flat thanks to its much larger sample count (45k vs 1.8k–5k).

## Caveat: the MACE negative gap is confounded

The two numbers being compared are not equivalent measurements:
- **Identity baseline**: standalone MLP on one-hot atom types → MACE embeddings of **clean** molecules, evaluated on held-out clean molecules.
- **REPA val**: cosine sim between projected **noisy** denoiser hidden states and clean MACE embeddings, averaged across **all timesteps**.

At t≈0 the denoiser sees near-pure noise, so its hidden states carry almost no atom identity. Averaging across all t dilutes the signal; the identity baseline gets clean one-hots every time. The negative gap partly reflects task difficulty, not necessarily a failure of the approach — MACE REPA still hits perfect validity (1.000 vs 0.980 baseline) and slightly better bond metrics in generation.

To resolve: timestep-stratified eval (see `timestep_stratified_eval.py`, not yet run) would reveal cos_sim at t≈1 specifically.

## Open questions

1. ~~Is using the same MLP architecture across encoders fair given different output dims?~~ **Fixed** — `encoder_analysis.py` uses per-encoder hidden_dim.
2. CheMeleon's gap (0.21) is meaningful but modest. Would a simpler atom-type classification auxiliary loss achieve similar generation-quality benefit without a frozen encoder?
3. GearNet's +0.374 gap: what structural features drive this — local geometry, contact patterns, secondary structure?
4. MACE REPA val (0.56) falls below identity (0.86). The projector hits 0.90 on training data, so it's a generalization issue, not capacity. Resolve with timestep-stratified eval (see caveat).
5. ~~Are 200-sample floor estimates reliable?~~ **Verified** — bootstrap shows std < 0.005 even at 500 samples.
