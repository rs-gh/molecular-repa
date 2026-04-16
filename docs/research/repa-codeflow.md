# REPA Alignment Code Flow

## Overview

REPA (Representation Alignment) trains the flow matching model to produce intermediate hidden states that align with a frozen pretrained encoder's representations. The idea: if the model's internal representations match those of a structure-aware encoder, it should learn better structural features and converge faster.

## Config

```yaml
# training_repa.yaml
repa:
  gearnet_ckpt_path: .../gearnet_ca.pth  # frozen encoder weights
  layers: [4]                              # which transformer layers to align
  lambda_repa: 0.5                         # REPA loss weight
  combination_mode: additive               # fm_loss + 0.5 * repa_loss
  similarity_type: cosine                  # negative cosine similarity
  projector_hidden_dim: 512
  projector_num_layers: 2
```

## Code Flow

### 1. Model Setup (`proteina_repa.py:27-65`)

```
ProteinaREPA.__init__():
  1. Replace self.nn with ProteinTransformerAF3WithHiddenStates
     - Same architecture, but forward() can capture intermediate layer outputs
  2. Create frozen GearNetPerResidueEncoder (loaded from gearnet_ca.pth)
  3. Create trainable Projector MLP (one per aligned layer)
     - Linear(512 -> 512) -> SiLU -> Linear(512 -> 512)
     - Maps transformer token_dim to GearNet's encoder_dim
  4. Wrap in ProteinaREPALoss module
```

### 2. Training Step (`proteina_repa.py:75-206`)

```
ProteinaREPA.training_step(batch):
  1. Extract clean coordinates x_1, sample time t, noise x_0
  2. Interpolate: x_t = (1-t)*x_0 + t*x_1
  3. Optional self-conditioning pass (no hidden states needed)
  4. Main forward pass WITH hidden states:
       x_1_pred, nn_out = self.predict_clean(batch, return_hidden_states=True)
  5. Compute flow matching loss: ||x_1_pred - x_1||
  6. Compute auxiliary distogram loss
  7. Compute REPA loss from nn_out["hidden_states"] and x_1
  8. Combine: total_loss = fm_loss + aux_loss + 0.5 * repa_loss
```

### 3. Hidden State Extraction (`protein_transformer_repa.py:70-81`)

```
ProteinTransformerAF3WithHiddenStates.forward():
  for i in range(self.nlayers):     # 10 layers
      seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)

      if return_hidden_states and i in self.repa_layers:  # i == 4
          h = seqs[:, num_registers:, :]   # strip register tokens
          hidden_states.append(h)          # [b, n, 512]

  nn_out["hidden_states"] = hidden_states  # list of 1 tensor
```

The key: after transformer layer 4 (the 5th layer, mid-network), we grab the per-residue sequence representation `seqs`. Register tokens (10 extra tokens used for global attention) are stripped so dimensions match the protein length.

### 4. REPA Loss Computation (`repa_loss.py:75-115`)

```
ProteinaREPALoss.forward(hidden_states, x_1_nm, mask):

  Step A: Compute target representations (FROZEN, no gradients)
    target_repr = self.encoder(x_1_nm, mask)  # [b, n, 512]

  Step B: For each aligned layer (just layer 4):
    h = hidden_states[0]              # [b, n, 512] from transformer layer 4
    projected = projector(h)           # [b, n, 512] MLP maps to encoder space

    cos_sim = cosine_similarity(
        projected[mask],               # only real residues, not padding
        target_repr[mask],
        dim=-1
    )
    layer_loss = -cos_sim.mean()       # negative because we maximize similarity

  return repa_loss, {"repa/cos_sim_layer_4": cos_sim.mean()}
```

### 5. GearNet Encoder (`gearnet_encoder.py:96-130`)

The encoder converts dense protein coordinates into per-residue structural features:

```
GearNetPerResidueEncoder.forward(ca_coords_nm, mask):

  1. Convert nm -> Angstroms (GearNet was trained in Angstrom space)

  2. Dense -> sparse: flatten [b, n, 3] to atom-level tensors
     - coords: [total_atoms, 3]
     - atom_seq_pos: [total_atoms] sequential residue indices (1D)
     - atom2batch: [total_atoms] batch assignment

  3. Build graph (GearNet uses two edge types):
     a. Sequential graph: radius_graph(atom_seq_pos, max_distance=2.1)
        - Connects residues within 2 positions in sequence
     b. Spatial graph: radius_graph(coords, radius=5.0, max_num_neighbors=64)
        - Connects residues within 5 Angstroms in 3D space

  4. Run GearNet message passing layers:
     for layer in self.gearnet.layers:
         h_v = layer(h_v, edge_list, h_e)
     # h_v: [total_atoms, 512]

  5. Scatter back to dense [b, n, 512] (zero for padded positions)
```

## What Flows Where (Tensor Shapes)

```
Training batch:
  x_1:  [b, n, 3]  clean CA coordinates (nm)
  mask: [b, n]      True for real residues, False for padding

Transformer (10 layers, token_dim=512):
  Input:  x_t [b, n, 3] -> embedded to [b, n, 512]
  Layer 4 output: hidden_states[0] = [b, n, 512]
  Layer 10 output: x_1_pred [b, n, 3]

GearNet encoder (frozen):
  Input:  x_1 [b, n, 3] (clean coords, not noisy)
  Output: target_repr [b, n, 512]

Projector (trainable MLP):
  Input:  hidden_states[0] [b, n, 512]
  Output: projected [b, n, 512]

REPA loss:
  cosine_similarity(projected[mask], target_repr[mask]) -> scalar
```

## Gradient Flow

```
                    FROZEN (no gradients)
                    +------------------+
  x_1 (clean) ---> | GearNet encoder  | ---> target_repr
                    +------------------+

                    TRAINABLE (gradients flow back)
                    +------------------+     +------------------+
  x_t (noisy) ---> | Transformer L0-4 | --> | Projector MLP    | --> projected
                    +------------------+     +------------------+
                           |                         |
                    (also continues to L5-10          |
                     for flow matching loss)          v
                                              cosine_similarity(projected, target_repr)
                                                       |
                                                  REPA loss
```

Gradients from the REPA loss flow back through:
1. The Projector MLP (trainable)
2. Transformer layers 0-4 (trainable)

They do NOT flow through:
- GearNet encoder (frozen, @torch.no_grad)
- Transformer layers 5-10 (REPA doesn't touch these; they only get gradients from the flow matching loss)

## Why Layer 4?

Layer 4 is the middle of the 10-layer transformer. The intuition from the REPA paper (Yu et al., 2024) is that:
- Early layers learn low-level features (positions, distances)
- Middle layers learn structural motifs (secondary structure, contacts)
- Late layers specialize for the generation task

Aligning at the middle encourages the model to develop good structural representations early, which the later layers can build on for generation. Multiple layers can be aligned (e.g., `layers: [2, 4, 6]`), each with its own projector.

## Audit vs Reference Implementation (2026-04-16)

Audited against the original REPA paper (arXiv 2410.06940) and its reference code at
https://github.com/sihyun-yu/REPA (image-domain DiT).

### Divergences Found and Fixed

**1. Per-sample vs per-residue averaging (FIXED)**

The reference computes `mean_flat` per sample (averaging over tokens), then averages over the batch — each sample contributes equally regardless of length. Our original code flattened all unmasked tokens globally and took one `.mean()`, causing longer proteins to dominate the gradient.

This also created an inconsistency with the flow matching loss, which is per-sample: a 200-residue protein got the same FM loss weight as a 50-residue protein, but 4x the REPA weight.

Fix: Added `averaging` parameter to `ProteinaREPALoss` with options `"per_sample"` (paper default) and `"per_residue"` (legacy). Wired through config as `repa.averaging`.

**2. Projector depth (FIXED)**

Reference uses a 3-layer MLP (`Linear→SiLU→Linear→SiLU→Linear`). All our configs used `projector_num_layers: 2` (2-layer MLP). Updated all configs to `projector_num_layers: 3`. Cost: +262K parameters (0.44% of 60M model).

### Verified Correct

- `F.cosine_similarity` is mathematically equivalent to the reference's explicit `normalize→dot` approach
- Encoder targets from clean data (x_1), hidden states from noisy input (x_t) — matches paper
- Additive loss combination `fm_loss + λ * repa_loss` with default λ=0.5 — matches reference
- GearNet frozen via `@torch.no_grad()` + `requires_grad=False`
- Register tokens correctly stripped before alignment
- Self-conditioning pass skips hidden state extraction
- nm→Angstrom conversion correct (×10.0)

### Known Maintenance Risk

`ProteinTransformerAF3WithHiddenStates.forward()` duplicates the parent's `forward()` entirely. Future changes to `ProteinTransformerAF3.forward()` will not propagate. A regression test (`TestParentSubclassForwardEquivalence`) catches this.

### Test Coverage (tests/proteina/test_repa_components.py)

| Test | What it verifies |
|------|-----------------|
| `TestReferenceNumericalEquivalence` | Our loss matches the paper's exact computation |
| `TestPerSampleVsPerResidueAveraging` | Both modes work; they agree at equal lengths, diverge otherwise |
| `TestParentSubclassForwardEquivalence` | Subclass produces identical output to parent class |
| `TestProjectorArchitecture` | MLP structure matches expected layer counts |
| `TestMSESimilarityMode` | MSE path produces finite loss with gradients |
| `TestTradeoffCombinationMode` | Both additive and tradeoff formulas correct |

### Tabasco (tests/tabasco/test_repa_integration.py)

Same two divergences fixed. The `averaging` parameter uses `"per_atom"` (instead of proteina's `"per_residue"`) as the legacy option name. 12 tabasco configs updated to `num_layers: 3` for the projector.

| Test | What it verifies |
|------|-----------------|
| `TestTabascoReferenceEquivalence` | Loss matches paper's exact computation (equal + variable lengths) |
| `TestTabascoPerSampleVsPerAtomAveraging` | Both modes work; agree at equal lengths, diverge otherwise |
| `TestTabascoProjectorArchitecture` | MLP structure matches expected layer counts |
| `TestTabascoMSEMode` | MSE path produces finite loss with gradients |
