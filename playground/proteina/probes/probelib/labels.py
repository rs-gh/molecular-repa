"""Ground-truth constructors for the probes.

contact_labels          CA-CA contacts, long-range only (|i-j| ≥ 24, d < 8 Å)
cath_labels_from_raw    per-protein CATH class label at C/A/T level
mean_pool_by_mask       mask-aware mean pool for CATH probe input
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


def contact_labels(
    ca_coords_ang: torch.Tensor,  # [B, N, 3]
    mask: torch.Tensor,  # [B, N]
    threshold_ang: float = 8.0,
    min_seq_sep: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build long-range binary contact labels + pair mask.

    Flow:
        dist      = cdist(ca, ca)            # [B, N, N] Å
        seqsep    = |i - j|                  # [N, N]
        pair_mask = real_i & real_j & (seqsep ≥ min_seq_sep)
        labels    = (dist < threshold_ang) & pair_mask

    Inputs:
        ca_coords_ang:  [B, N, 3] CA coords in Å
        mask:           [B, N] bool, True = real residue
        threshold_ang:  contact cutoff in Å (default 8.0, standard)
        min_seq_sep:    ignore short-range pairs (default 24, long-range only)

    Returns:
        labels:    [B, N, N] float — 1 where (i, j) in contact AND valid, else 0
        pair_mask: [B, N, N] bool  — True where the pair is valid (both real, long-range)
    """
    B, N, _ = ca_coords_ang.shape
    dist = torch.cdist(ca_coords_ang, ca_coords_ang)  # [B, N, N] Å
    idx = torch.arange(N, device=ca_coords_ang.device)
    seqsep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [N, N]
    long_range = seqsep >= min_seq_sep  # [N, N]
    real = mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1)  # [B, N, N]
    pair_mask = real & long_range.unsqueeze(0)  # [B, N, N]
    labels = (dist < threshold_ang).float() * pair_mask.float()  # [B, N, N]
    return labels, pair_mask


def cath_labels_from_raw(
    raw: List[Data],
    level: str = "T",
) -> Tuple[np.ndarray, List[str]]:
    """Extract one coarse CATH label per protein (first domain, masked to ``level``).

    Level meaning:
        "C" — class only          (e.g. "3")
        "A" — class.arch          (e.g. "3.40")
        "T" — class.arch.topology (e.g. "3.40.50")

    Handles several historical shapes defensively:
        g.cath_code = "1.10.10.10"              (plain string)
        g.cath_code = ["1.10.10.10", ...]        (list — expected)
        g.cath_code = [["1.10.10.10"], ...]      (nested — seen on some builds)
        g.cath_code = None / []                  (unlabelled)

    Returns:
        labels:  [B] int64 — index into uniq, or -1 for unlabelled
        uniq:    list of unique label strings at this level (label index = position)
    """
    mapping = {"C": 0, "A": 1, "T": 2, "H": 3}
    k = mapping[level]
    strs: List[str | None] = []
    n_missing = 0
    for g in raw:
        cc = getattr(g, "cath_code", None)
        if cc is None or (hasattr(cc, "__len__") and len(cc) == 0):
            strs.append(None)
            n_missing += 1
            continue
        # Unwrap nested list / find first string.
        first = cc
        for _ in range(3):  # unwrap up to 3 levels
            if isinstance(first, (list, tuple)):
                if not first:
                    first = None
                    break
                first = first[0]
            else:
                break
        if not isinstance(first, str):
            strs.append(None)
            n_missing += 1
            continue
        parts = first.split(".")
        key = ".".join(parts[: k + 1]) if len(parts) > k else None
        strs.append(key)

    uniq = sorted(set(s for s in strs if s is not None))
    idx_of = {s: i for i, s in enumerate(uniq)}
    labels = np.asarray([idx_of.get(s, -1) for s in strs], dtype=np.int64)
    return labels, uniq


def mean_pool_by_mask(reps: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean-pool ``[B, N, D]`` reps with a ``[B, N]`` real-residue mask.

    Flow:
        summed = (reps * mask[..., None]).sum(dim=1)   # [B, D]
        n      = mask.sum(dim=1).clamp(min=1)          # [B] — avoid div-by-0
        pooled = summed / n                             # [B, D]

    Returns a float32 numpy array suitable for sklearn.
    """
    real = mask.float().unsqueeze(-1).cpu()  # [B, N, 1]
    summed = (reps * real).sum(dim=1)  # [B, D]
    n = real.sum(dim=1).clamp(min=1.0)  # [B, 1]
    return (summed / n).float().numpy()  # [B, D]
