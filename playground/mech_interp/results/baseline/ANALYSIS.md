# Mechanistic Interpretability Analysis: GEOM Baseline

**Model**: GEOM-mild baseline (`evaluation_checkpoints/baseline.ckpt`)
**Architecture**: 16-layer reimplemented Transformer, 8 heads, hidden_dim=128, cross-attention output heads
**Analysis**: 50 molecules, 100 Euler steps, captured every 5 steps (20 capture points)
**Validity**: 48/50 generated molecules valid (96%)

---

## Summary of Findings

The baseline flow-matching model for molecular generation shows **depth-stratified** internal representations where the layer position determines behaviour far more than the generation timestep. This contrasts with proteina's findings where temporal dynamics (early vs late generation) are the dominant axis.

### Five metrics computed across [timestep x layer]:

| Metric | Early (t~0) | Mid (t~0.5) | Late (t~1) | Key pattern |
|--------|-------------|-------------|------------|-------------|
| Structure lens RMSD | 1.18 | 0.79 | 1.36 | Non-monotonic (see discussion) |
| Atom accuracy | 0.74 | 0.81 | 0.98 (L15) | Progressive refinement, last layer leads |
| Attention entropy | 0.71 | 0.64 | 0.60 | Depth-stratified bands, mild temporal decrease |
| Distance correlation | 0.46 | 0.61 | 0.64 | Monotonic increase; geometry emerges over time |
| Bond precision@N | 0.14 | 0.23 | 0.27 | Mid-layers (4-11) highest; NOT final layers |

---

## Detailed Analysis

### 1. Attention Entropy — Depth Determines Selectivity

The entropy heatmap reveals striking **horizontal banding**: mid-layers 7-9 have the lowest entropy (~0.35-0.45) throughout the entire trajectory, while early layers (0-3) and the last layer (15) remain diffuse (~0.65-0.75).

**Interpretation**: The mid-layers act as selective "bottleneck" processors regardless of input quality. Early layers spread attention broadly (mixing information), mid-layers focus on specific atom pairs, and the final layers re-expand attention (possibly for global coordination before the output heads).

**Temporal variation is weak** — entropy decreases only ~15% from t=0 to t=1. The model's attention patterns are largely pre-determined by architecture, not by the evolving input.

**Contrast with proteina**: In proteina, geometric pair bias B creates strong temporal dynamics (B dominates at late t). Tabasco's standard self-attention lacks this explicit geometric signal, so the attention pattern is more static and depth-dependent.

### 2. Bond Precision — Chemical Structure in Middle Layers

Bond precision (fraction of top-N attention pairs that are true bonds) peaks in **layers 4-11** (~0.25-0.33 at late timesteps). Layers 0-3 and 13-15 have markedly lower bond precision.

**This is the inverse of proteina's finding** that "structure specificity concentrates in the final 1-2 layers." In tabasco:
- **Early layers** (0-3): broad mixing, low structural specificity
- **Middle layers** (4-11): chemical bonding knowledge concentrates here
- **Final layers** (12-15): lower bond precision; these layers appear to do something other than track bonds — likely global molecular coordination for the output heads

**Temporal trend**: Bond precision increases over time (0.14 → 0.27), meaning attention becomes more bond-aware as the molecule crystallises. But the depth profile barely changes — the same layers are always most/least bond-aware.

### 3. Distance Correlation — Geometry Emerges Monotonically

Attention-distance correlation (Pearson r between attention weights and inverse pairwise distance) shows the clearest temporal story:
- **Early (t~0)**: r = 0.46 — already positive even from noise (the model has spatial priors)
- **Late (t~1)**: r = 0.64 — attention strongly tracks physical proximity

The last layer consistently has the highest correlation (~0.52 → 0.67), meaning the final layer is most spatially aware. This is the one metric where the final layer dominates.

### 4. Atom Type Accuracy — Last Layer Leads

Atom type accuracy from decoded intermediates:
- Starts at ~0.74 (well above random 1/9 ≈ 0.11 — strong atom embedding prior)
- Reaches 0.98 in the last layer at late timesteps
- Clear depth gradient: deeper layers → better accuracy
- Clear temporal gradient: later timesteps → better accuracy

This is the most "expected" result: the model progressively refines atom type predictions through both depth and time.

### 5. Structure Lens — Non-Monotonic (Caveat)

The RMSD between decoded intermediate coordinates and the final Euler-integrated output shows a non-monotonic pattern: lowest at mid-generation, higher at both early and late timesteps.

**Important context**: The model predicts endpoint coordinates x_1 (via `CenteredMetricInterpolant`), not velocities. The structure lens decodes intermediate hidden states as x_1 predictions. The final output `x_final` is the Euler-integrated result of 100 such predictions. The non-monotonic RMSD likely reflects:
- **Early t**: uncertain predictions that are close to the origin (averaged out)
- **Mid t**: predictions that happen to align well with the final integrated result
- **Late t**: highly specific predictions that diverge from the accumulated path

This metric may need refinement for flow-matching models — comparing to the model's own last-step prediction rather than the integrated output would be cleaner.

---

## Implications for REPA Alignment

### Where to align (depth)

The concentration of chemical structure knowledge in **layers 4-11** suggests that REPA alignment should target these layers, not just the final layer (current default). The ChemProp encoder captures 2D chemical graph structure (bonds, aromaticity) — this aligns naturally with the layers that already attend to bonds.

**Concrete suggestion**: Try multi-layer REPA with projectors at layers 4, 8, and 12 (early-structure, peak-structure, and post-structure). Compare to single-layer REPA at the last layer.

### When to align (time-weighting)

The weak temporal variation in attention structure suggests that **time-weighting of the REPA loss may be less important than layer selection**. The model's internal representation is surprisingly stable across generation time — the depth axis dominates.

However, the monotonic increase in distance correlation suggests that **late-timestep alignment** (t > 0.5) is where the geometric representations are most developed and most amenable to alignment.

### What the ablation experiments will test

The ablation sweep (next section) will test the **causal** role of each input component:
- **Coordinate ablation**: Does the model actually use 3D geometry, or does it reconstruct structure from atom types + positional encoding?
- **Atom type ablation**: Can the model generate valid structures from geometry alone?
- **Positional encoding ablation**: How important is the sinusoidal ordering?
- **Time encoding ablation**: Does the model need to know "where in the trajectory" it is?
- **Early vs late** ablation: Does the model need coordinates early (to set up the structure) or late (to refine it)?

---

## Files

- `trajectory_metrics.npz` — raw metric arrays [T=20, L=16] and [T=20, L=16, H=8]
- `figures/structure_lens_rmsd.png` — [layers x timesteps] heatmap
- `figures/attention_entropy.png` — [layers x timesteps] heatmap
- `figures/bond_precision.png` — [layers x timesteps] heatmap
- `figures/metric_lines.png` — 2x2 line plots by layer

## Reproduction

```bash
source .venv/bin/activate && export PROJECT_ROOT=$(pwd)/src/tabasco
python playground/mech_interp/run_analysis.py \
    --checkpoint evaluation_checkpoints/baseline.ckpt \
    --num-molecules 50 --num-steps 100 \
    --output-dir playground/mech_interp/results/baseline
```
