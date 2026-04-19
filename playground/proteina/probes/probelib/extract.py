"""Hidden-state extraction from Proteina trunks + frozen GearNet encoder.

Clean-endpoint probing convention: ``x_t = x_1`` (clean CA coords in nm) and
``t = 1.0``. Matches the setting in which REPA itself is evaluated — we probe
what the student has learned about real structures, not about noise.
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
) -> Dict[int, torch.Tensor]:
    """Run a clean forward pass and return one ``[B, N, D]`` tensor per layer.

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

    Flow per chunk:
        sub["x_t"] = ang_to_nm(coords[:, :, 1, :])   # [b, n, 3]
        sub["t"]   = torch.ones(b)                    # clean endpoint
        nn_out     = model.nn(sub, return_hidden_states=True)
        hs         = nn_out["hidden_states"]          # list aligned to `layers`

    Returns:
        {layer: [B, N, D]} all on CPU for downstream probes.
    """
    from proteinfoundation.utils.coors_utils import ang_to_nm

    enable_hidden_states(model, layers)
    device = next(model.parameters()).device
    model.eval()

    B = batch["coords"].shape[0]
    per_layer: Dict[int, List[torch.Tensor]] = {lyr: [] for lyr in layers}
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        sub = {k: v[s:e] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        sub = {k: v.to(device) for k, v in sub.items()}

        ca = sub["coords"][:, :, 1, :]  # [b, n, 3] Å
        x_1_nm = ang_to_nm(ca)  # [b, n, 3] nm
        sub["x_t"] = x_1_nm
        sub["t"] = torch.ones(e - s, device=device, dtype=x_1_nm.dtype)

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
