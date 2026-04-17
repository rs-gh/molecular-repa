# Tabasco Model Architecture

This document provides a detailed walkthrough of how the Tabasco molecular generation model works, with exact file and line references. All paths are relative to the tabasco submodule root (`src/tabasco/` in the `molecular-repa` project).

## Overview

Tabasco uses **flow matching** to generate 3D molecules jointly as coordinates + atom types. The core idea:
- Start with random coordinate noise + uniformly random one-hot atom types at t=0
- Iteratively denoise over ~100 Euler steps
- End with a clean molecule (coords + atom types) at t=1

At each step the network predicts, for every token, (i) the clean 3D coordinate `x_1_pred` and (ii) a logit distribution over atom types. Two domain-specific **interpolants** convert these predictions into a single update step: a continuous (Euclidean / SDE) step for coordinates and a discrete (categorical) step for atom types.

Unlike Proteina, Tabasco adds an optional **REPA alignment loss** that pulls the network's hidden states toward the embeddings of a frozen molecular encoder (ChemProp/CheMeleon, MACE-OFF, or a dummy MLP).

---

## High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         FLOW MATCHING LOOP                                │
│   for t in [0 → 1] over a schedule (linear / power / log):                │
│       x_t = (noisy coords, noisy one-hot atomics)                         │
│       pred = TransformerModule(coords_t, atomics_t, pad_mask, t)         │
│            → { "coords": x_1_pred,   # predicted clean coords            │
│                "atomics": atom_logits }                                   │
│       coords_t  = CenteredMetricInterpolant.step(coords_t, x_1_pred, t)  │
│                   # Euler (ODE) or Langevin+noise (SDE)                   │
│       atomics_t = DiscreteInterpolant.step(atomics_t, logits, t)         │
│                   # categorical jump proportional to (dt/(1-t))·softmax   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          TransformerModule                                │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │ (1) INPUT EMBEDDING                                                │   │
│  │     • coords_t [b,n,3]  → Linear        → embed_coords [b,n,H]     │   │
│  │     • atomics_t.argmax  → Embedding     → embed_atoms  [b,n,H]     │   │
│  │     • positions         → Sinusoid PE   → embed_posenc [b,n,H]     │   │
│  │     • t                 → Fourier time  → embed_time   [b,1,H]     │   │
│  │     h_in = sum(...) (or concat+Linear)  × (1 - padding_mask)       │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│                                    ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │ (2) TRANSFORMER TRUNK — plain self-attention, no pair bias         │   │
│  │     num_layers × TransformerEncoderLayer(pre-norm, SiLU, H·4 FFN)  │   │
│  │     → h_out [b,n,H]                                                │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│                                    ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │ (3) OUTPUT HEADS                                                   │   │
│  │     if cross_attention=True (default):                             │   │
│  │         h_coord = TransformerDecoderLayer(q=h_out, kv=h_in)        │   │
│  │         h_atom  = TransformerDecoderLayer(q=h_out, kv=h_in)        │   │
│  │     else:                                                          │   │
│  │         h_coord = h_atom = h_out                                   │   │
│  │     coords_pred     = LayerNorm→Linear(H→3)        (h_coord)       │   │
│  │     atomics_logits  = Linear→SiLU→Linear(H→A=9)    (h_atom)        │   │
│  │                                                                    │   │
│  │     For REPA training, also return h_coord and h_atom separately.  │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Note**: the network predicts `x_1_pred` **directly** (not a velocity). The velocity used for the ODE/SDE step is recovered inside the interpolant: `v = (x_1_pred - x_t) / (1 - t)` (`src/tabasco/flow/interpolate.py:344`).

---

## Detailed Data Flow Trace

### Entry Point: Sampling Script

**File**: `src/sample.py`
```
Line 24-53: def sample_batch(lightning_module, batch_size, num_steps, ...)
    │
    │ # Wrapper that places the LightningTabasco in eval mode and calls .sample()
    Line 105-138: def main()  # CLI entrypoint — loads checkpoint, exports pickle
```

### Step 1: Lightning Wrapper

**File**: `src/tabasco/models/lightning_tabasco.py`
```
Line 80-94: def training_step(self, batch, batch_idx)
    │
    │ # Single call — loss + stats
    Line 83: loss, stats_dict = self.model(batch, compute_stats=True)
    │                          └── FlowMatchingModel.forward
    │
    Line 112-118: def validation_step(...) — same loss on val/ split
```

### Step 2: Flow Matching Loop (Sampling)

**File**: `src/tabasco/models/flow_model.py`
```
Line 301: def sample(self, batch, num_steps, batch_size, return_trajectories)
    │
    Line 319: x_t = self._sample_noise_like_batch(batch, batch_size)
    │         └── Gaussian coord noise (masked, zero-COM) + random one-hot atomics
    │
    Line 323-325: T = self._get_sample_schedule(num_steps)
    │             # "linear" | "power" (t²) | "log" — see _get_sample_schedule
    │
    Line 327: for i in range(1, len(T)):  ◀── MAIN LOOP
    │    │
    │    Line 328-329: t = T[i - 1]; dt = T[i] - T[i - 1]
    │    │
    │    Line 331: x_t = self._step(x_t, t, dt)  ◀── CALLS MODEL + INTERPOLANT
    │
    Line 338: return x_t  # final (coords, atomics, padding_mask) TensorDict

Line 340: def _step(self, x_t, t, step_size)
    │
    Line 343: out_batch = self._call_net(x_t, t)         # forward pass (no_grad)
    Line 345: x_t["coords"]  = coords_interpolant.step(x_t, out_batch, t, step_size)
    Line 346: x_t["atomics"] = atomics_interpolant.step(x_t, out_batch, t, step_size)
    Line 347: return x_t
```

### Step 3: Model Forward Wrapper

**File**: `src/tabasco/models/flow_model.py`
```
Line 93: def _call_net(self, batch, t, return_hidden_states=False)
    │
    Line 95-101: net_output = self.net(coords, atomics, padding_mask, t, return_hidden_states)
    │            # TransformerModule.forward — see Step 4
    │
    Line 103-116: if return_hidden_states and cross_attention:
    │                 returns TensorDict with keys
    │                 { "coords", "atomics", "padding_mask",
    │                   "hidden_states_coord", "hidden_states_atom" }
    │
    Line 117-128: elif return_hidden_states:  # cross_attention=False
    │                 { "coords", "atomics", "padding_mask", "hidden_states_coord" }
    │
    Line 129-138: else: { "coords", "atomics", "padding_mask" }   # sampling default
```

Note: only `hidden_states_*` keys appearing in the `pred` TensorDict are consumed by REPA. The loss aggregator matches them by prefix (`losses.py:236`).

### Step 4: TransformerModule Forward

**File**: `src/tabasco/models/components/transformer_module.py`
```
Line 212: def forward(self, coords, atomics, padding_mask, t,
                       return_hidden_states=False, need_weights=False)
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 1: INPUT EMBEDDING
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 234-236: h_in, real_mask, embed_components = self._compute_input_embedding(...)
    │             │
    │             └── Line 155-210: _compute_input_embedding
    │                 Line 166: embed_coords     = Linear(3 → H)(coords)
    │                 Line 167: embed_atom_types = Embedding(A=9 → H)(atomics.argmax(-1))
    │                 Line 170: embed_posenc     = SinusoidEncoding(H, max_len=90)
    │                 Line 178: embed_time       = TimeFourierEncoding(H, max_len=200)(t)
    │                 Line 207: h_in = sum(embed_coords, atoms, posenc, time)
    │                           # or concat→Linear if concat_combine_input=True
    │                 Line 208: h_in *= real_mask.unsqueeze(-1)
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 2: TRANSFORMER TRUNK
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 242-269: if implementation == "pytorch":
    │    Line 267: h_out = self.transformer(h_in, src_key_padding_mask=padding_mask)
    │             # nn.TransformerEncoder: num_layers × TransformerEncoderLayer
    │             # pre-norm, batch_first, dim_feedforward = 4*H, activation=SiLU
    │             # STANDARD multi-head self-attention — no pair bias
    │
    Line 270-281: elif implementation == "reimplemented":
    │                 h_out = Transformer(h_in, padding_mask)  # custom in transformer.py
    │
    Line 283: h_out = h_out * real_mask.unsqueeze(-1)
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 3: OUTPUT HEADS
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 285-294: if cross_attention:
    │                 h_coord = coord_cross_attention(h_out, h_in, ...)
    │                 coords  = out_coord_linear(h_coord)
    │             else:
    │                 coords  = out_coord_linear(h_out)
    │
    Line 296-305: if cross_attention:
    │                 h_atom  = atom_cross_attention(h_out, h_in, ...)
    │                 atom_logits = out_atom_type_linear(h_atom)
    │             else:
    │                 atom_logits = out_atom_type_linear(h_out)
    │
    Line 307-321: return shape depends on return_hidden_states flag:
                  • False      → (coords, atom_logits)
                  • True       → (coords, atom_logits, h_coord[, h_atom])
                  • "all"      → (coords, atom_logits, analysis_dict)
                                 for trajectory / mech-interp capture
```

### Step 5: Output Heads Detail

**File**: `src/tabasco/models/components/transformer_module.py`
```
Line 99-102: out_coord_linear = Sequential(
    │            LayerNorm(H),
    │            Linear(H → 3, bias=False)
    │        )
    │
Line 104-108: out_atom_type_linear = Sequential(
    │            Linear(H → H),
    │            SiLU(),
    │            Linear(H → A=9)
    │        )
    │
Line 111-126: if cross_attention:
    │            coord_cross_attention = TransformerDecoderLayer(H, heads, 4H, pre-norm)
    │            atom_cross_attention  = TransformerDecoderLayer(H, heads, 4H, pre-norm)
    │        # Each decoder layer uses h_out as query, h_in (pre-trunk) as key/value.
    │        # This produces two distinct hidden-state heads for REPA alignment and
    │        # decouples the coord and atom-type predictions.
```

### Step 6: Attention (Standard PyTorch MHA)

Tabasco does **not** use pair-bias attention. Each `TransformerEncoderLayer` uses `nn.MultiheadAttention` with:
- `dim_head = hidden_dim / num_heads` (default 128/8 = 16)
- Pre-norm layout (`norm_first=True`)
- `key_padding_mask = padding_mask` — padded atoms contribute neither as queries (masked via `real_mask` on `h_in`) nor as keys/values (via `key_padding_mask`)
- No bias tensor added to logits — attention is content-only

The `reimplemented` branch (`src/tabasco/models/components/transformer.py`) provides a custom Transformer with the same interface but supports intermediate capture for mechanistic-interpretability analysis (`transformer_module.py:270-281`).

---

## Training vs Inference

### Training Loop

**File**: `src/tabasco/models/flow_model.py`
```
Line 140: def forward(self, batch, compute_stats=True)
    │
    Line 143-146: if num_random_augmentations:
    │                 batch = apply_random_rotation(batch, n_augmentations=...)
    │                 # Repeats batch (n+1)× with random 3D rotations — see
    │                 # src/tabasco/data/transforms.py:49-106
    │
    Line 148: path = self._create_path(batch)
    │         │
    │         └── Line 159-218: _create_path
    │             Line 170-171: t ~ time_distribution              # uniform | beta | histogram
    │             Line 174-175: noise_batch = _sample_noise_like_batch(x_1)
    │             Line 177-179: coords_interpolant.create_path
    │                           # x_0=Gaussian (zero-COM), x_t=(1-t)x_0 + t·x_1
    │             Line 180-182: atomics_interpolant.create_path
    │                           # x_0=uniform one-hot, x_t=random mask-uncorrupt per token
    │             Line 218: FlowPath(x_0, x_t, dx_t, x_1, t)
    │
    Line 150-154: return_hidden_states = self.repa_loss is not None
    │             pred = self._call_net(path.x_t, path.t, return_hidden_states)
    │
    Line 156: loss, stats_dict = self._compute_loss(path, pred, compute_stats)
    │
    Line 157: return loss, stats_dict
```

At training time the network sees **one random timestep per molecule** and is trained to predict the clean `x_1` directly. No self-conditioning.

### Inference Loop

**File**: `src/tabasco/models/flow_model.py:301-338` (see Step 2 above).

At inference, the model iterates `num_steps` times over a schedule in [0, 1]. Each iteration performs a forward pass and applies **two independent interpolant steps** — continuous (Euclidean Euler / SDE) for coordinates, discrete (categorical jump) for atomics.

### Key Differences

| Aspect               | Training                                     | Inference                                  |
|----------------------|----------------------------------------------|--------------------------------------------|
| Input x_t            | `(1-t)·x_0 + t·x_1`  (interpolation)         | Iterative from pure noise                  |
| Time t               | Random single sample per molecule            | Sequential 0→1 over schedule               |
| Forward passes       | 1                                            | `num_steps` (default 100)                  |
| Atomics noise        | random bitwise corruption-uncorruption       | categorical jump `∝ dt/(1-t) × softmax`    |
| Coords noise         | noise-free `x_t`                             | optional Langevin score + Wiener noise (SDE) |
| Rotation augmentation | `apply_random_rotation` × (1 + n_aug)        | none                                       |
| Gradients            | Yes (backprop)                               | No (`torch.no_grad`)                       |
| REPA loss            | Yes (if configured)                          | N/A                                        |
| Self-conditioning    | **No** (unlike Proteina)                     | **No**                                     |

---

## The Denoised Estimate (x_1_pred)

At every step, `TransformerModule` outputs `coords` directly in `x_1` space — this is the network's best guess at the final clean coordinates given `x_t` and `t`. Unlike Proteina, tabasco does **not** output a velocity.

### Coord velocity recovery (inside the interpolant)

**File**: `src/tabasco/flow/interpolate.py`
```python
# CenteredMetricInterpolant.step (Line 334-353)
Line 343: x1_pred  = pred[self.key]
Line 344: velocity = (x1_pred - batch_t[self.key]) / (1 - t)
Line 346: x_new    = batch_t[self.key] + velocity * dt
Line 347: x_new    = mask_and_zero_com(x_new, padding_mask)
```

### SDE variant (default for tabasco)

**File**: `src/tabasco/flow/interpolate.py:356-417` (`SDEMetricInterpolant`)

Adds a Langevin component and Wiener noise to the Euler step:

```
score         = (t * velocity - x_t) / (1 - t + 1e-6)               # Line 380-382
component     = langevin_schedule(t) * score                         # Line 398
wiener_noise  = √(2 · langevin_schedule(t) · σ_w²) · η,  η ~ N(0, I) # Line 400-402
x_new         = x_t + (velocity + component) · dt + wiener_noise · dt # Line 408-410
```

### Atomics update

**File**: `src/tabasco/flow/interpolate.py:199-232` (`DiscreteInterpolant.step`)

Atomics are one-hot. The network predicts class logits; the step draws a categorical jump:

```python
Line 217: x1_probs    = softmax(pred["atomics"])
Line 218: curr_state  = batch_t["atomics"].argmax(-1)
Line 220: step_probs  = clamp((dt / (1 - t)) · x1_probs, max=1)    # jump prob per class
Line 221-223: step_probs[curr_state] ← 1 - sum(other_jumps)         # stay prob
Line 229-230: x_next  = one_hot(Categorical(step_probs).sample())
```

This is a discrete-time CTMC-style update: the expected fraction of tokens that flip per step is proportional to `dt/(1-t)`, so near `t=1` most tokens commit to their current state.

---

## Loss Function

### Total training loss

**File**: `src/tabasco/models/flow_model.py:220-274` (`_compute_loss`)

```
Line 225-226: atomics_loss, _ = atomics_interpolant.compute_loss(path, pred)   # CE
Line 228-229: coords_loss, _  = coords_interpolant.compute_loss(path, pred)    # MSE
Line 231-234: dists_loss       = optional InterDistancesLoss
Line 237-240: repa_loss        = optional REPALoss

Line 264: diffusion_loss = atomics_loss + coords_loss + dists_loss
Line 266-270: if repa_loss:
                  if combination_mode == "tradeoff":
                      total = (1 - λ) * diffusion + λ * repa
                  else:  # "additive" (default)
                      total = diffusion + λ * repa
              else:
                  total = diffusion
```

### Coordinate loss

**File**: `src/tabasco/flow/interpolate.py:307-332` (`CenteredMetricInterpolant.compute_loss`)

```python
err  = (pred["coords"] - x_1["coords"]) * real_mask[..., None]      # Line 315
loss = sum(err**2, dim=(-1,-2)) / (n_atoms * 3)                     # Line 316
```

This is MSE in the `x_1` parameterization — mathematically equivalent to `MSE(v_pred, v_target)` with a `1/(1-t)²` time weighting (see the Proteina doc's derivation; same argument applies here). An optional `time_factor` callable multiplies the per-molecule loss to implement that reweighting explicitly.

### Atomics loss

**File**: `src/tabasco/flow/interpolate.py:168-197` (`DiscreteInterpolant.compute_loss`)

Standard per-token cross-entropy:

```python
loss        = CrossEntropyLoss(reduction="none")(logits.transpose(1,2), x_1.argmax(-1))  # Line 185-187
per_mol     = (loss * real_mask).sum(-1) / n_atoms                                        # Line 188
total       = per_mol.mean() * loss_weight                                                # Line 193
```

### Inter-atomic distance loss (optional)

**File**: `src/tabasco/models/components/losses.py:11-111` (`InterDistancesLoss`)

Masked MSE between predicted and true pairwise distance matrices. Used as an auxiliary geometric loss in some experiment configs.

---

## REPA Alignment (Tabasco-Specific)

This is the main architectural departure from Proteina. A pre-trained **frozen** molecular encoder produces a target representation of the clean molecule `x_1`; the diffusion network's **hidden states** are projected into that space and aligned via cosine similarity (or MSE).

### REPALoss

**File**: `src/tabasco/models/components/losses.py:114-284`

```
Line 127: def __init__(encoder, projector, lambda_repa, time_weighting,
                       similarity_type, combination_mode, averaging)
    │
    Line 161-163: self.encoder, self.projector, self.lambda_repa
    Line 170-171: freeze all encoder params

Line 224: def forward(path, pred, compute_stats)
    │
    Line 236: hs_keys = sorted(k for k in pred.keys() if k.startswith("hidden_states_"))
    │         # With cross_attention=True: ["hidden_states_atom", "hidden_states_coord"]
    │         # With cross_attention=False: ["hidden_states_coord"] only
    │
    Line 248-256: smiles    = path.x_1.get_non_tensor("smiles")    (if present)
    │             lmdb_keys = path.x_1.get_non_tensor("lmdb_key")  (if present)
    │
    Line 258-265: with torch.no_grad():
    │                 target_repr = self.encoder(
    │                     path.x_1["coords"],          # CLEAN coords
    │                     path.x_1["atomics"],
    │                     padding_mask,
    │                     smiles=smiles,
    │                     lmdb_keys=lmdb_keys,
    │                 )   # [B, N, encoder_dim]
    │
    Line 270: h_fused   = torch.cat([pred[k] for k in hs_keys], dim=-1)
    │         # cross_attention=True → [B, N, 2·hidden_dim] (coord + atom heads)
    │         # cross_attention=False → [B, N, hidden_dim]
    │
    Line 271: projected = self.projector(h_fused)        # [B, N, encoder_dim]
    │
    Line 276-279: loss = _cosine_loss | _mse_loss (projected, target_repr, real_mask, t)
    │             # per_atom  averaging (project default): global mean over unmasked atoms
    │             # per_sample averaging:                   each molecule weighted equally
```

### Projector

**File**: `src/tabasco/models/components/encoders.py:729-770`

MLP with a `LazyLinear` first layer so the fused hidden-state width is inferred automatically:

```
LazyLinear(hidden_dim) → SiLU
  → [Linear(hidden_dim, hidden_dim) → SiLU] × (num_layers - 2)
  → Linear(hidden_dim, encoder_dim)
```

Typical config: `hidden_dim=128`, `encoder_dim` matches the encoder (e.g. 2048 for CheMeleon, 192 for MACE-OFF small), `num_layers=3`.

### Encoders

| Encoder | File | Lines | Notes |
|---------|------|-------|-------|
| `MolecularEncoder` (base) | `src/tabasco/models/components/encoders.py` | 9-24 | `forward(coords, atomics, padding_mask, smiles=None, lmdb_keys=None)` |
| `DummyEncoder` | encoders.py | 27-71 | MLP(coords) — for testing |
| `ChemPropEncoder` | encoders.py | 74-308 | 2D graph MP with pretrained CheMeleon (2048-D), SMILES cache |
| `CachedChemPropEncoder` | encoders.py | 311-408 | Pre-computed ChemProp embeddings keyed by SMILES in LMDB |
| `MACEEncoder` | encoders.py | 411-621 | 3D equivariant MACE-OFF, `invariants_only=True` extracts l=0 features |
| `CachedMACEEncoder` | encoders.py | 624-726 | Pre-computed MACE embeddings keyed by conformer `lmdb_key` |

All subclasses accept `smiles=None` and `lmdb_keys=None` so the REPA pipeline can pass them through uniformly.

### SMILES Pipeline

SMILES (and `lmdb_key`) travel through the data pipeline as non-tensor fields on the per-sample `TensorDict`:

1. `UnconditionalLMDBDataset.__getitem__` computes `Chem.MolToSmiles(mol)` and calls `sample.set_non_tensor("smiles", smi)` (see `src/tabasco/data/components/lmdb_unconditional.py`).
2. `TensorDictCollator` (`src/tabasco/data/utils.py:14-28`) stacks samples preserving `NonTensorStack`.
3. `apply_random_rotation` (`src/tabasco/data/transforms.py:49-106`) repeats the batch `n_aug+1` times and must **manually** propagate the non-tensor fields (Line 94-104) — otherwise SMILES are silently dropped.
4. `REPALoss.forward` reads them via `path.x_1.get_non_tensor("smiles")` (losses.py:249) and hands them to the encoder.

The ChemProp fast path caches the expensive `MolGraph` construction by SMILES, which is ~6× faster than rebuilding connectivity from coordinates every step.

### combination_mode

Two ways to blend the REPA loss with the diffusion loss (`flow_model.py:266-270`):

- **additive** (default): `total = diffusion + λ · repa`. REPA acts as a regularizer; λ is a free hyperparameter.
- **tradeoff**: `total = (1 - λ) · diffusion + λ · repa`. Convex combination; λ ∈ [0, 1] is a mixing weight.

---

## Data Pipeline

**File**: `src/tabasco/data/lmdb_datamodule.py`
```
Line 33-127: LmdbDataModule
    • train_dataset / val_dataset: UnconditionalLMDBDataset
    • collate_fn: TensorDictCollator (stacks per-sample TensorDicts)
    • Batch keys (tensor): "coords" [B, N, 3], "atomics" [B, N, A=9],
                            "padding_mask" [B, N]
    • Batch keys (non-tensor): "smiles", "lmdb_key"
```

**File**: `src/tabasco/data/components/lmdb_unconditional.py`
```
Line 20-116: UnconditionalLMDBDataset
    • Stores molecules in LMDB; keys are byte-sorted strings
    • Per-sample options: random_rotation, permute_atoms,
      reorder_to_smiles_order, remove_hydrogens
    • Populates statistics: max_num_atoms, num_atoms_histogram, all_smiles
```

**File**: `src/tabasco/data/transforms.py`
```
Line  7-12: random_rotation                — single random 3D rotation
Line 15-32: permute_atoms                  — shuffle non-padded atom indices
Line 35-46: sample_uniform_rotation        — Haar-uniform SO(3) matrices
Line 49-106: apply_random_rotation         — training augmentation:
             repeats batch (n_aug+1)× with fresh rotations; must propagate
             non-tensor fields (smiles, lmdb_key) explicitly (Line 94-104)
```

---

## Key Components Reference

| Component                       | File                                               | Lines     | Purpose                                                   |
|---------------------------------|----------------------------------------------------|-----------|-----------------------------------------------------------|
| Flow Matching Model             | `src/tabasco/models/flow_model.py`                 | 18-482    | `FlowMatchingModel`: training + sampling orchestration    |
| Sampling loop                   | `src/tabasco/models/flow_model.py`                 | 301-347   | `.sample()` + `._step()`: Euler loop over schedule        |
| Training forward                | `src/tabasco/models/flow_model.py`                 | 140-157   | `.forward()`: path + net + loss                           |
| Path construction               | `src/tabasco/models/flow_model.py`                 | 159-218   | `_create_path`: samples `t`, noise, builds `FlowPath`     |
| Loss aggregation                | `src/tabasco/models/flow_model.py`                 | 220-274   | `_compute_loss`: sums coord+atom+dist+REPA                |
| Schedule                        | `src/tabasco/models/flow_model.py`                 | 276-299   | `_get_sample_schedule`: linear / power / log              |
| Coord interpolant               | `src/tabasco/flow/interpolate.py`                  | 235-353   | `CenteredMetricInterpolant`: linear interp, ODE Euler     |
| SDE coord interpolant (default) | `src/tabasco/flow/interpolate.py`                  | 356-417   | `SDEMetricInterpolant`: Langevin score + Wiener noise     |
| Atomics interpolant             | `src/tabasco/flow/interpolate.py`                  | 118-232   | `DiscreteInterpolant`: corrupt/uncorrupt + categorical step |
| Network                         | `src/tabasco/models/components/transformer_module.py` | 16-321 | `TransformerModule`: embed + trunk + heads                |
| Input embedding                 | `src/tabasco/models/components/transformer_module.py` | 155-210 | coord Linear + atom Embedding + posenc + time Fourier     |
| Output heads                    | `src/tabasco/models/components/transformer_module.py` | 99-126  | coord (LayerNorm+Linear), atom (Linear-SiLU-Linear), optional cross-attention |
| Positional encodings            | `src/tabasco/models/components/positional_encoder.py` | —       | `SinusoidEncoding`, `TimeFourierEncoding`                 |
| REPA loss                       | `src/tabasco/models/components/losses.py`          | 114-284   | Alignment to frozen encoder                               |
| Inter-distance loss             | `src/tabasco/models/components/losses.py`          | 11-111    | Optional pairwise-distance MSE                            |
| Projector                       | `src/tabasco/models/components/encoders.py`        | 729-770   | LazyLinear MLP projecting hidden states → encoder space   |
| ChemProp encoder                | `src/tabasco/models/components/encoders.py`        | 74-308    | CheMeleon 2D message passing                              |
| MACE encoder                    | `src/tabasco/models/components/encoders.py`        | 411-621   | MACE-OFF 3D equivariant                                   |
| Lightning wrapper               | `src/tabasco/models/lightning_tabasco.py`          | 23-136    | `training_step`, `validation_step`, `sample`              |
| LMDB dataset                    | `src/tabasco/data/components/lmdb_unconditional.py` | 20-116   | `UnconditionalLMDBDataset`                                |
| DataModule                      | `src/tabasco/data/lmdb_datamodule.py`              | 33-127    | `LmdbDataModule` (batching, workers, augmentation)        |
| Collator                        | `src/tabasco/data/utils.py`                        | 14-28     | `TensorDictCollator` (preserves non-tensor fields)        |
| Augmentation                    | `src/tabasco/data/transforms.py`                   | 49-106    | `apply_random_rotation` (propagates SMILES)               |

---

## Model Dimensions (Defaults)

| Dimension         | Value (QM9 baseline)      | Description                                                |
|-------------------|---------------------------|------------------------------------------------------------|
| `spatial_dim`     | 3                         | 3D Cartesian coordinates                                   |
| `atom_dim` (A)    | 9                         | Atom type classes: C, N, O, F, S, Cl, Br, I, * (padding)   |
| `hidden_dim` (H)  | 128 (QM9) / 256 (GEOM)    | Token representation dimension                             |
| `num_layers`      | 16                        | Transformer encoder layers                                 |
| `num_heads`       | 8                         | Attention heads (dim per head = H / num_heads)             |
| `dim_feedforward` | 4·H                       | TransformerEncoderLayer FFN hidden                         |
| `cross_attention` | True                      | Separate decoder layers for coord vs atom heads            |
| `max_len` (posenc)| 90                        | Sinusoid PE max sequence length                            |
| `max_len` (time)  | 200                       | Time Fourier encoding max index                            |
| `num_random_augmentations` | 7 (→ 8 total)    | Rotation augmentation multiplier                           |
| `sample_schedule` | log                       | Denser steps near `t=0`                                    |
| `time_distribution` | beta (α = 1.8)          | Training time sampling biased toward `t≈1` (clean end)     |

### REPA-specific (when enabled)

| Dimension         | Value (default)           | Description                                                |
|-------------------|---------------------------|------------------------------------------------------------|
| `encoder_dim`     | 2048 (ChemProp) / 192 (MACE) / 256 (Dummy) | Frozen encoder output width                   |
| `projector.hidden_dim` | 128                  | MLP intermediate width                                     |
| `projector.num_layers` | 3                    | MLP depth (audit-recommended)                              |
| `lambda_repa`     | 0.5                       | REPA weight                                                |
| `combination_mode` | additive                 | `additive` or `tradeoff`                                   |
| `similarity_type` | cosine                    | `cosine` or `mse`                                          |
| `averaging`       | per_atom                  | `per_atom` (project default) or `per_sample` (each molecule weighted equally) |

---

## Key Differences vs Proteina

| Aspect                     | Proteina                                              | Tabasco                                               |
|----------------------------|-------------------------------------------------------|-------------------------------------------------------|
| Domain                     | Protein backbones (Cα only)                           | Small molecules (coords + atom types)                 |
| Network output             | Velocity `v_pred`                                     | Direct `x_1_pred` + atom logits                       |
| Attention                  | Pair-bias attention with triangular pair updates      | Standard multi-head self-attention (no pair features) |
| Conditioning               | AdaLN from time + CATH fold codes                     | Additive time embedding (summed into tokens)          |
| Register tokens            | 10 learnable tokens prepended                         | None                                                  |
| Self-conditioning          | 50% chance during training, every inference step      | None                                                  |
| Discrete outputs           | None (continuous coords only)                         | Atom types via `DiscreteInterpolant` (categorical)    |
| SDE sampling               | Optional (Langevin + Wiener)                          | Default (`SDEMetricInterpolant`)                      |
| REPA alignment             | Not present                                           | Optional — ChemProp/CheMeleon or MACE encoders        |
| Output heads               | Single coord head (+ optional distogram aux)          | Separate coord + atom heads, optional cross-attention |
