# ProteinWorkshop GearNet-Edge characterization (2026-04-24)

Characterization of the two ProteinWorkshop (Jamasb et al., ICLR 2024) GearNet-Edge
checkpoints from Zenodo 8287754, mirroring the analyses done for CA-GearNet, ESM2,
and MC-GearNet-Edge in [../gearnet/](../gearnet/), [../esm/](../esm/),
[../mc_gearnet/](../mc_gearnet/).

- Variants: `torsional_denoising`, `structure_denoising`
- Feature config: `ca_angles` (43-dim node input: aa[23] + seq_pe[16] + alpha[2] + kappa[2])
- 200 PDB train proteins, 45,503 residues total
- Job 28308561 (A100 80GB), Apr 24 00:51 BST

Full script: [explore_pw_gearnet.py](explore_pw_gearnet.py) •
sbatch: [run_pw_gearnet.sh](run_pw_gearnet.sh)

Raw log: `.local_ckpts/pw-gn-char-28308561.out`

## TL;DR

- **PW torsional** (`pw_gearnet_torsional_denoising_ca_angles.ckpt`) — **usable** REPA target. Eff rank 11.2/3072, projector saturation mean-dir=0.698, onehot=0.709 → headroom ~0.07–0.15, 3D-sensitive (cos=0.91 at 0.5 Å).
- **PW structure** (`pw_gearnet_structure_denoising_ca_angles.ckpt`) — **borderline**. Eff rank 19.9/3072 (better), but projector saturation is worse (mean-dir=0.781, onehot=0.794 → headroom ~0.03–0.08).
- Both sit **between** ESM2 L33 (mean=0.664) and MC-GearNet-Edge (mean=0.864) in saturation. Still noticeably worse than CA-GearNet (mean=0.419), which keeps the "gold standard" slot.

## Port & forward pass (critical implementation detail)

The Zenodo checkpoints are Lightning `.ckpt` files with all encoder weights under
`encoder.*`. Architecture is 6-layer GearNet-Edge with `concat_hidden=True`:

```
node_dims  = [43, 512, 512, 512, 512, 512, 512]
edge_dims  = [89, 43,  512, 512, 512, 512, 512]   # line-graph chain
output_dim = 3072  (concat of 6 × 512 hidden layers)
```

**PW's forward differs from MC-GearNet-Edge's.** Verified against
`proteinworkshop/models/graph_encoders/gear_net.py` on GitHub. The loop is:

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
1. `layers[i].edge_linear` in the checkpoint has `in_features = 89` at **every** layer →
   node layers consume the **original** edge feature (`f_ji`), not the updated
   `edge_hidden`. Passing `edge_hidden` instead caused `mat1 (2672×43) × mat2 (89×43)`
   dim mismatch (job 28307906).
2. `layers[i].linear` is used **twice** per layer: once for the standard
   `(num_relation × input_dim) → output_dim` aggregation, and once more for the
   edge-to-node scatter update. Works because edge_hidden's dim matches
   `layer.linear`'s input dim at every layer (43 at L0, 512 at L1+).

Forgetting the edge-to-node scatter step (job 28308226) produced a drastically
degraded representation — eff rank dropped to 3.3, projector saturation collapsed
to ~0.94, and 3D sensitivity disappeared (cos=0.995 at 0.5 Å). The complete
forward raises eff rank to 11.2 and restores 3D sensitivity (cos=0.91 at 0.5 Å).

## Per-variant summary

| Metric                          | Torsional | Structure | CA-GearNet | MC-GN-Edge | ESM2 L33 |
|---------------------------------|----------:|----------:|-----------:|-----------:|---------:|
| Encoder dim                     |      3072 |      3072 |        512 |       3072 |     1280 |
| Effective rank                  |      11.2 |      19.9 |         82 |        1.1 |      354 |
| Participation ratio             |       5.1 |       5.7 |         — |        — |       — |
| Dims for 99% var                |       119 |       409 |         — |        — |       — |
| Mean L2 norm                    |      37.1 |      22.7 |         — |  ~550,000 |        — |
| Dead dims (std<1e-6)            |         0 |        22 |          0 |        513 |        — |
| 3D cos @ 0.5 Å                  |     0.914 |     0.913 |       0.36 |       — |       — |
| Residue-shuffle cos             |     0.994 |     0.995 |         — |      0.984 |       — |
| AA probe accuracy               |      0.93 |      0.96 |       0.15 |          — |       — |
| Mean AA centroid cos            |     0.975 |     0.977 |         — |        — |       — |
| Within-protein - between-prot.  |     0.097 |     0.082 |         — |    -0.002 |       — |
| **Projector mean-dir baseline** | **0.698** | **0.781** |      0.419 |      0.864 |    0.664 |
| **Projector test (onehot)**     | **0.709** | **0.794** |      0.426 |      0.864 |    0.719 |
| Estimated REPA headroom        | ~0.07–0.15 |~0.03–0.08 |       ~0.35 |      ~0.003 |    ~0.08 |

### Layer-wise effective rank

| Layer | Torsional | Structure |
|------:|----------:|----------:|
| 0     |       5.2 |       6.8 |
| 1     |       6.8 |       7.4 |
| 2     |      10.6 |      15.3 |
| 3     |      14.3 |      12.8 |
| 4     |      10.0 |      13.6 |
| 5     |       8.2 |      14.0 |

Mid-layers carry the most rank. For the concat_hidden output, this means ~60% of
the 3072 output dims come from slabs with <15 eff rank each.

### Structural context (SS) sensitivity — within-SS minus between-SS

| AA  | Torsional Δ | Structure Δ |
|-----|------------:|------------:|
| ALA |       0.109 |       0.074 |
| GLY |       0.021 |       0.038 |
| LEU |       0.113 |       0.031 |
| VAL |       0.086 |       0.028 |

Torsional clearly dominates on SS-within minus SS-between context discrimination,
likely because its pretraining task (denoising Cα torsion angles) directly
depends on SS-specific local geometry.

## What's wrong with these encoders as REPA targets

Same root cause as MC-GearNet-Edge but less severe: `concat_hidden=True` over 6
layers of `linear + relu + BN` without a final LayerNorm produces a concat whose
last slab dominates in norm and whose per-slab effective rank is small. Mid-layer
slabs actually have healthy variance, but they get drowned by the layer-5 slab in
cosine similarity after concat.

PW adds the edge-to-node update (`layer.linear(scatter(edge_hidden))`) which
injects rank from the line-graph chain — hence eff rank 11.2 vs MC's 1.1. But the
post-concat projector saturation is still > 0.7 even with random input, which
means a REPA student trained to cos-sim against this target will hit ~0.7–0.8
easily regardless of whether it's learning meaningful geometry.

## Recommendations

1. **Prefer CA-GearNet** as the REPA target. It remains the only 3D-aware encoder
   with genuine headroom (>0.35) on this codebase.
2. If a 3D + sequence-aware variant is needed, **PW torsional** is usable. Its
   ~0.1 headroom is comparable to ESM2 L33, and it's explicitly 3D-sensitive.
3. **Do not use PW structure_denoising** for now. Higher saturation (0.78) and
   weaker SS delta suggest the pretraining task compressed the hidden space
   harder than torsional_denoising did.
4. An interesting follow-up: try per-layer REPA alignment (pick layer 2 or 3 of
   PW torsional, where eff rank peaks at ~14). Our current wrapper returns the
   concat; a mid-layer projection may give cleaner targets.

## Files referenced

- Port: `src/proteina/proteinfoundation/metrics/gearnet_utils.py` — classes
  `_PWLayer`, `PWGearNetEdge`, `NoTrainPWGearNetEdge`.
- Wrapper: `src/proteina/proteinfoundation/repa/gearnet_encoder.py` —
  `PWGearNetEdgePerResidueEncoder`.
- Fetch: `hpc-scripts/proteina/data_prep/fetch_pw_gearnet.sh`.
- Training configs (`encoder.type: pw_gearnet`):
  `src/proteina/configs/experiment_config/training/{256,256/afdb,128/afdb}/pw_gearnet/per_residue/`.
