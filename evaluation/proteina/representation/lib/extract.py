"""Hidden-state extraction from Proteina trunks + frozen GearNet encoder.

Timestep convention (flow matching): ``x_t = (1-t)·x_0 + t·x_1`` where
``x_0`` is the reference-noise sample and ``x_1`` is the clean structure.
``t=1.0`` is clean, ``t=0.0`` is pure noise. At ``t=1.0`` we skip noise
sampling entirely (fast path); for any ``t<1.0`` we sample x_0 from the
model's flow-matching reference distribution and interpolate — matching the
training-time noisy-input regime used in REPA paper Fig 7.
"""

from __future__ import annotations

from typing import Dict, List

import torch


def enable_hidden_states(model, layers: List[int]):
    """Ensure ``model.nn`` returns hidden states at the requested layers.

    Works for both trained Proteina checkpoints (``ProteinTransformerAF3``) and
    freshly-constructed Proteina models. For REPA checkpoints the trunk is
    already a ``ProteinTransformerAF3WithHiddenStates`` instance.

    The subclass-swap is safe because ``ProteinTransformerAF3WithHiddenStates``
    adds no new state — same weights, richer forward.
    """
    from proteinfoundation.repa.protein_transformer_repa import (
        ProteinTransformerAF3WithHiddenStates,
    )

    if not isinstance(model.nn, ProteinTransformerAF3WithHiddenStates):
        model.nn.__class__ = ProteinTransformerAF3WithHiddenStates
    model.nn.repa_layers = list(layers)
    return model


@torch.no_grad()
def extract_model_hidden_states_multilayer(
    model,
    batch: Dict,
    layers: List[int],
    chunk_size: int = 8,
    t_value: float = 1.0,
    noise_seed: int = 42,
) -> Dict[int, torch.Tensor]:
    """Run a forward pass at timestep ``t_value`` and return one ``[B, N, D]`` tensor per layer.

    Extracting all layers in one pass is nearly free: the transformer runs the
    same forward regardless of how many layers we capture, so we amortize the
    expensive checkpoint-load over all probe layers.

    Inputs:
        model:       Proteina / ProteinaREPA LightningModule (eval)
        batch:       dict with
                       coords  [B, N, 37, 3] Å (CA at idx 1)
                       mask    [B, N] bool
                       residue_type, chain_break_per_res, ... (per-residue)
        layers:      which trunk layer outputs to capture, e.g. [0, 4, 9]
        chunk_size:  sub-batch size for memory — default 8
        t_value:     flow-matching timestep. 1.0 = clean (fast path, no noise);
                     values <1.0 sample x_0 from the model's reference distribution
                     and interpolate x_t = (1-t)·x_0 + t·x_1 — matches training.
        noise_seed:  RNG seed for reproducible x_0 sampling. Fixed per sweep so
                     the same proteins see the same noise across (run, step, layer)
                     probes, making cross-checkpoint comparisons fair.

    Flow per chunk (clean, t=1.0):
        sub["x_t"] = ang_to_nm(coords[:, :, 1, :])   # [b, n, 3]
        sub["t"]   = torch.ones(b)                    # clean endpoint

    Flow per chunk (noisy, t<1.0):
        x_1    = mask_and_zero_com(ang_to_nm(coords[CA]))
        x_0    = model.fm.sample_reference(..., mask)
        x_t    = model.fm.interpolate(x_0, x_1, t_vec)
        sub["x_t"] = x_t
        sub["t"]   = t_vec

    Returns:
        {layer: [B, N, D]} all on CPU for downstream probes.
    """
    from proteinfoundation.utils.coors_utils import ang_to_nm

    enable_hidden_states(model, layers)
    device = next(model.parameters()).device
    model.eval()
    fm = getattr(model, "fm", None)
    if t_value < 1.0 and fm is None:
        raise RuntimeError(
            f"Cannot probe at t={t_value}: model has no .fm attribute for "
            "noise sampling / interpolation."
        )

    B = batch["coords"].shape[0]
    per_layer: Dict[int, List[torch.Tensor]] = {lyr: [] for lyr in layers}
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        sub = {k: v[s:e] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        sub = {k: v.to(device) for k, v in sub.items()}

        ca = sub["coords"][:, :, 1, :]  # [b, n, 3] Å
        x_1_nm = ang_to_nm(ca)  # [b, n, 3] nm
        b_here = e - s
        if t_value >= 1.0:
            sub["x_t"] = x_1_nm
            sub["t"] = torch.ones(b_here, device=device, dtype=x_1_nm.dtype)
        else:
            mask_b = sub["mask"].bool()
            x_1_c = fm._mask_and_zero_com(x_1_nm, mask_b)
            n_res = x_1_c.shape[1]
            # Deterministic x_0: seed offset per chunk, stable across (run, step, layer)
            # so every checkpoint sees the same noise for the same proteins.
            torch.manual_seed(noise_seed + s)
            # sample_reference concatenates `shape + (n, 3)` → pass only batch dim
            x_0 = fm.sample_reference(
                n=n_res,
                shape=(b_here,),
                device=device,
                dtype=x_1_c.dtype,
                mask=mask_b,
            )
            t_vec = torch.full(
                (b_here,), float(t_value), device=device, dtype=x_1_c.dtype
            )
            x_t = fm.interpolate(x_0, x_1_c, t_vec, mask=mask_b)
            sub["x_t"] = x_t
            sub["t"] = t_vec

        nn_out = model.nn(sub, return_hidden_states=True)
        hs = nn_out["hidden_states"]  # list of [b, n, D], aligned to `layers`
        for lyr, h in zip(layers, hs):
            per_layer[lyr].append(h.detach().cpu())

    return {lyr: torch.cat(v, dim=0) for lyr, v in per_layer.items()}  # {L: [B, N, D]}


@torch.no_grad()
def extract_model_hidden_states(
    model, batch: Dict, layer: int, chunk_size: int = 8
) -> torch.Tensor:
    """Backward-compat single-layer shim used by run_all.py."""
    out = extract_model_hidden_states_multilayer(
        model, batch, [layer], chunk_size=chunk_size
    )
    return out[layer]


def model_num_layers(model) -> int:
    """Return the transformer trunk's layer count (e.g. 10 for proteina 60M)."""
    return int(model.nn.nlayers)


@torch.no_grad()
def extract_gearnet_embeddings(
    encoder, batch: Dict, chunk_size: int = 8
) -> torch.Tensor:
    """Run frozen GearNet encoder over CA coords.

    Inputs:
        encoder:     GearNetPerResidueEncoder (frozen, eval)
        batch:       must contain coords [B, N, 37, 3] Å and mask [B, N] bool
        chunk_size:  sub-batch size for memory

    Flow per chunk:
        ca        = coords[:, :, 1, :]       # [b, n, 3] Å, CA only
        ca_nm     = ang_to_nm(ca)            # [b, n, 3] nm
        reps      = encoder(ca_nm, mask)     # [b, n, 512] — last GearNet layer

    Returns:
        [B, N, 512] on CPU.
    """
    from proteinfoundation.utils.coors_utils import ang_to_nm

    encoder.eval()
    device = next(encoder.parameters()).device

    B = batch["coords"].shape[0]
    outs = []
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        ca = batch["coords"][s:e, :, 1, :].to(device)  # [b, n, 3] Å
        mask_bool = batch["mask"][s:e].bool().to(device)  # [b, n]
        ca_nm = ang_to_nm(ca)  # [b, n, 3] nm
        out = encoder(ca_nm, mask_bool)  # [b, n, 512]
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)  # [B, N, 512]
