"""Metric functions for trajectory analysis.

All functions operate on raw tensors and have no model dependency.
Inspired by the crystallization metrics in proteina's analysis framework.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def attention_entropy(
    attn_weights: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
    eps: float = 1e-10,
) -> Tensor:
    """Shannon entropy of attention distribution per head.

    Args:
        attn_weights: Post-softmax attention [B, H, N, N] or [B, N, N].
        mask: Optional sequence mask [B, N] (True = valid).
        normalize: If True, divide by log(N) so result is in [0, 1].
        eps: Numerical stability.

    Returns:
        Entropy per batch and head [B, H] (or [B] if input had no head dim).
    """
    squeezed = False
    if attn_weights.dim() == 3:
        attn_weights = attn_weights.unsqueeze(1)
        squeezed = True

    p = attn_weights.clamp(min=eps)
    H_per_query = -(p * torch.log(p)).sum(dim=-1)  # [B, H, N]

    if mask is not None:
        seq_mask = mask.unsqueeze(1).float()  # [B, 1, N]
        H_per_query = H_per_query * seq_mask
        H = H_per_query.sum(dim=-1) / (seq_mask.sum(dim=-1) + eps)  # [B, H]
    else:
        H = H_per_query.mean(dim=-1)  # [B, H]

    if normalize:
        N = attn_weights.shape[-1]
        H = H / (np.log(N) + eps)

    if squeezed:
        H = H.squeeze(1)
    return H


def bond_precision(
    attn_weights: Tensor,
    bond_matrix: Tensor,
    k: Optional[int] = None,
    apply_apc: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Precision@k of attention-predicted bonds.

    Analog of proteina's contact precision: fraction of top-k
    attention-ranked pairs that are true molecular bonds.

    Args:
        attn_weights: [B, H, N, N] post-softmax attention.
        bond_matrix: [B, N, N] binary bond adjacency (1=bonded).
        k: Number of top pairs. Default: N (number of atoms).
        apply_apc: Apply Average Product Correction before ranking.
        eps: Numerical stability.

    Returns:
        Precision per batch and head [B, H].
    """
    B, H, N, _ = attn_weights.shape
    device = attn_weights.device

    if k is None:
        k = N

    # Upper triangle excluding diagonal (avoid self-attention)
    j_idx = torch.arange(N, device=device).unsqueeze(0)
    i_idx = torch.arange(N, device=device).unsqueeze(1)
    eval_mask = j_idx > i_idx  # [N, N]

    precision = torch.zeros(B, H, device=device)

    for bi in range(B):
        bonds_flat = bond_matrix[bi][eval_mask].float()
        num_valid = eval_mask.sum().item()

        if num_valid < k:
            continue

        for hi in range(H):
            A = attn_weights[bi, hi].float()
            A_sym = (A + A.T) / 2

            if apply_apc:
                row_mean = A_sym.mean(dim=-1, keepdim=True)
                col_mean = A_sym.mean(dim=-2, keepdim=True)
                total_mean = A_sym.mean()
                A_sym = A_sym - (row_mean * col_mean) / (total_mean + eps)

            scores = A_sym[eval_mask]
            actual_k = min(k, num_valid)
            _, top_idx = scores.topk(actual_k)
            precision[bi, hi] = bonds_flat[top_idx].mean()

    return precision


def distance_correlation(
    attn_weights: Tensor,
    dist_matrix: Tensor,
    mask: Optional[Tensor] = None,
    invert: bool = True,
) -> Tensor:
    """Pearson correlation between attention weights and distance matrix.

    Same computation as proteina's spatial alignment metric.

    Args:
        attn_weights: [B, H, N, N] post-softmax attention.
        dist_matrix: [B, N, N] pairwise distance matrix.
        mask: Optional [B, N, N] pair mask (True=valid).
        invert: If True, negate so positive = good (high attn ↔ low distance).

    Returns:
        Correlation per batch and head [B, H].
    """
    B, H, N, _ = attn_weights.shape
    device = attn_weights.device

    if mask is not None:
        valid_mask = mask.bool()
    else:
        valid_mask = torch.ones(B, N, N, dtype=torch.bool, device=device)

    # Exclude diagonal
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    valid_mask = valid_mask & diag_mask

    correlations = torch.zeros(B, H, device=device)

    for bi in range(B):
        valid = valid_mask[bi]
        gt_flat = dist_matrix[bi][valid].float()

        if gt_flat.numel() < 3:
            continue

        gt_mean = gt_flat.mean()
        gt_centered = gt_flat - gt_mean
        gt_std = gt_centered.std()

        if gt_std < 1e-8:
            continue

        for hi in range(H):
            attn_flat = attn_weights[bi, hi][valid].float()
            attn_mean = attn_flat.mean()
            attn_centered = attn_flat - attn_mean
            attn_std = attn_centered.std()

            if attn_std < 1e-8:
                continue

            rho = (attn_centered * gt_centered).mean() / (attn_std * gt_std)

            if invert:
                rho = -rho

            correlations[bi, hi] = rho

    return correlations


def structure_lens_rmsd(
    decoded_coords: Tensor,
    reference_coords: Tensor,
    mask: Tensor,
) -> float:
    """RMSD between decoded coordinates and reference, after Kabsch alignment.

    Args:
        decoded_coords: [B, N, 3] coordinates from intermediate layer.
        reference_coords: [B, N, 3] coordinates from final layer or ground truth.
        mask: [B, N] boolean mask (True=real atom).

    Returns:
        Mean RMSD across the batch (scalar).
    """
    rmsds = []
    B = decoded_coords.shape[0]

    for bi in range(B):
        valid = mask[bi].bool()
        P = decoded_coords[bi, valid].float()  # [n, 3]
        Q = reference_coords[bi, valid].float()  # [n, 3]

        if P.shape[0] < 3:
            continue

        # Center
        P_centered = P - P.mean(dim=0)
        Q_centered = Q - Q.mean(dim=0)

        # Kabsch: find optimal rotation via SVD
        H = P_centered.T @ Q_centered  # [3, 3]
        U, S, Vt = torch.linalg.svd(H)

        # Correct for reflection
        d = torch.det(Vt.T @ U.T)
        sign_matrix = torch.diag(torch.tensor([1.0, 1.0, d.sign()], device=P.device))
        R = Vt.T @ sign_matrix @ U.T

        P_aligned = P_centered @ R.T
        rmsd = torch.sqrt(((P_aligned - Q_centered) ** 2).sum(dim=-1).mean())
        rmsds.append(rmsd.item())

    return float(np.mean(rmsds)) if rmsds else float("nan")


def atom_type_accuracy(
    decoded_logits: Tensor,
    gt_atomics: Tensor,
    mask: Tensor,
) -> float:
    """Fraction of correctly predicted atom types from intermediate layer.

    Args:
        decoded_logits: [B, N, atom_dim] logits from intermediate decoder.
        gt_atomics: [B, N, atom_dim] one-hot ground truth atom types.
        mask: [B, N] boolean mask (True=real atom).

    Returns:
        Accuracy (scalar).
    """
    pred_types = decoded_logits.argmax(dim=-1)  # [B, N]
    gt_types = gt_atomics.argmax(dim=-1)  # [B, N]

    correct = (pred_types == gt_types) & mask
    total = mask.sum()

    if total == 0:
        return float("nan")

    return (correct.sum().float() / total.float()).item()


def alignment_cosine_sim(
    hidden_states: Tensor,
    target_repr: Tensor,
    projector: torch.nn.Module,
    mask: Tensor,
) -> float:
    """Mean cosine similarity between projected hidden states and encoder embeddings.

    Args:
        hidden_states: [B, N, D] hidden states from a transformer layer.
        target_repr: [B, N, encoder_dim] frozen encoder embeddings.
        projector: nn.Module that maps hidden_states to encoder_dim.
        mask: [B, N] boolean mask (True=real atom).

    Returns:
        Mean cosine similarity (scalar).
    """
    with torch.no_grad():
        projected = projector(hidden_states)  # [B, N, encoder_dim]

    real_mask = mask.bool()
    cos_sim = F.cosine_similarity(projected[real_mask], target_repr[real_mask], dim=-1)
    return cos_sim.mean().item()
