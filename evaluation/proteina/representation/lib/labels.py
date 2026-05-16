"""Ground-truth constructors for the probes.

contact_labels          CA-CA contacts, long-range only (|i-j| ≥ 24, d < 8 Å)
cath_labels_from_raw    per-protein CATH class label at C/A/T level
mean_pool_by_mask       mask-aware mean pool for CATH probe input
dihedral_labels         per-residue (sinφ, cosφ, sinψ, cosψ) + valid mask
local_frame_features    per-residue 24-d trivial-geometric feature for
                        the dihedral analytical-floor baseline
knn_distance_features   per-residue 8-d distances to nearest CAs for
                        the inverse-folding local-geometry baseline
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


# Atom-37 indexing convention used throughout this codebase: N=0, CA=1, C=2.
_ATOM37_N, _ATOM37_CA, _ATOM37_C = 0, 1, 2


def _dihedral_angle(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
) -> torch.Tensor:
    """Standard 4-point dihedral via atan2 of cross products.

    Inputs are [..., 3] tensors of point coords; output is [...] in radians ∈ (-π, π].
    Numerically stable form:
        b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
        n1 = b1 × b2; n2 = b2 × b3
        m1 = n1 × (b2 / |b2|)
        x  = n1 · n2;  y = m1 · n2
        return atan2(y, x)
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2_norm = b2 / (b2.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    m1 = torch.cross(n1, b2_norm, dim=-1)
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def dihedral_labels(
    coords37: torch.Tensor,  # [B, N, 37, 3] Å
    mask: torch.Tensor,  # [B, N] bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-residue backbone (φ, ψ) targets for the dihedral regression probe.

    φ_i defined by atoms (C_{i-1}, N_i, CA_i, C_i) — undefined at i=0 (chain start).
    ψ_i defined by atoms (N_i, CA_i, C_i, N_{i+1}) — undefined at i=L-1 (chain end).

    Output uses (sin, cos) parameterisation so MSE loss handles the angular
    wrap-around cleanly:
        sincos[..., 0:2] = (sin φ, cos φ)
        sincos[..., 2:4] = (sin ψ, cos ψ)

    The validity tensor distinguishes per-angle availability — φ may be valid
    while ψ is not (and vice versa) at chain boundaries.

    Returns:
        sincos:  [B, N, 4] float; zeros at invalid positions (caller must mask)
        valid:   [B, N, 2] bool; column 0 = φ valid, column 1 = ψ valid
    """
    B, N, _, _ = coords37.shape
    device = coords37.device
    n_atoms = coords37[..., _ATOM37_N, :]  # [B, N, 3]
    ca = coords37[..., _ATOM37_CA, :]
    c = coords37[..., _ATOM37_C, :]
    real = mask.bool()  # [B, N]

    sincos = torch.zeros(B, N, 4, device=device, dtype=coords37.dtype)
    valid = torch.zeros(B, N, 2, device=device, dtype=torch.bool)

    if N >= 2:
        # φ_i (i = 1..N-1): C_{i-1}, N_i, CA_i, C_i
        phi = _dihedral_angle(
            c[:, :-1], n_atoms[:, 1:], ca[:, 1:], c[:, 1:]
        )  # [B, N-1]
        phi_valid = real[:, :-1] & real[:, 1:]
        sincos[:, 1:, 0] = torch.sin(phi)
        sincos[:, 1:, 1] = torch.cos(phi)
        valid[:, 1:, 0] = phi_valid

        # ψ_i (i = 0..N-2): N_i, CA_i, C_i, N_{i+1}
        psi = _dihedral_angle(n_atoms[:, :-1], ca[:, :-1], c[:, :-1], n_atoms[:, 1:])
        psi_valid = real[:, :-1] & real[:, 1:]
        sincos[:, :-1, 2] = torch.sin(psi)
        sincos[:, :-1, 3] = torch.cos(psi)
        valid[:, :-1, 1] = psi_valid

    # Zero out (sin, cos) at invalid positions so accidental unmasked use
    # produces a finite, obviously-wrong number rather than NaNs from the cross
    # product on padded zero-coords.
    sincos[..., 0:2] = sincos[..., 0:2] * valid[..., 0:1].float()
    sincos[..., 2:4] = sincos[..., 2:4] * valid[..., 1:2].float()
    return sincos, valid


def local_frame_features(
    coords37: torch.Tensor,  # [B, N, 37, 3] Å
    mask: torch.Tensor,  # [B, N] bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trivial-geometric baseline feature for the dihedral probe.

    For each residue i with both i-1 and i+1 present, build a 24-d feature
    consisting of the 8 backbone atoms (N, C of i-1; N, CA, C of i; N, CA of
    i+1 — wait, we need exactly the atoms that determine (φ_i, ψ_i)):

        atoms = [C_{i-1}, N_i, CA_i, C_i, N_{i+1}]   # 5 atoms × 3 coords = 15 dims

    These five atoms span both φ_i and ψ_i; an expressive head can recover the
    angles exactly. We rotate into a local frame centred at CA_i with x̂ along
    (N_i - CA_i) and z normal to the (N_i, CA_i, C_i) plane to remove rigid
    SE(3) variance — otherwise random global orientation noise dominates.

    Returns:
        feats:  [B, N, 15] float; zeros at invalid positions
        valid:  [B, N] bool; True where i-1 and i+1 are both real
    """
    B, N, _, _ = coords37.shape
    device = coords37.device
    n_atoms = coords37[..., _ATOM37_N, :]
    ca = coords37[..., _ATOM37_CA, :]
    c = coords37[..., _ATOM37_C, :]
    real = mask.bool()

    feats = torch.zeros(B, N, 15, device=device, dtype=coords37.dtype)
    valid = torch.zeros(B, N, dtype=torch.bool, device=device)

    if N < 3:
        return feats, valid

    # Indexable slices for i in [1..N-2]
    c_prev = c[:, :-2]  # i-1
    n_i = n_atoms[:, 1:-1]
    ca_i = ca[:, 1:-1]
    c_i = c[:, 1:-1]
    n_next = n_atoms[:, 2:]  # i+1

    # Build local frame per residue i:
    # x̂ = unit(N_i - CA_i)
    # z̃ = (N_i - CA_i) × (C_i - CA_i)   (out-of-plane normal)
    # ẑ = unit(z̃)
    # ŷ = ẑ × x̂
    v_n = n_i - ca_i
    v_c = c_i - ca_i
    x_hat = v_n / v_n.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z_tmp = torch.cross(v_n, v_c, dim=-1)
    z_hat = z_tmp / z_tmp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    y_hat = torch.cross(z_hat, x_hat, dim=-1)
    R = torch.stack(
        [x_hat, y_hat, z_hat], dim=-1
    )  # [B, N-2, 3, 3] columns are basis vectors

    def _to_local(p: torch.Tensor) -> torch.Tensor:
        # p: [B, N-2, 3] in world coords; subtract CA_i then project onto frame.
        d = p - ca_i  # [B, N-2, 3]
        # local = R^T · d  (since R has world-basis vectors as columns)
        return torch.einsum("bnij,bnj->bni", R.transpose(-1, -2), d)

    feats_inner = torch.cat(
        [
            _to_local(c_prev),
            _to_local(n_i),
            _to_local(
                ca_i
            ),  # by construction this is (0,0,0); kept for fixed shape / readability
            _to_local(c_i),
            _to_local(n_next),
        ],
        dim=-1,
    )  # [B, N-2, 15]

    feats[:, 1:-1] = feats_inner
    inner_valid = real[:, :-2] & real[:, 1:-1] & real[:, 2:]  # [B, N-2]
    valid[:, 1:-1] = inner_valid
    feats = feats * valid.float().unsqueeze(-1)
    return feats, valid


def knn_distance_features(
    ca_coords_ang: torch.Tensor,  # [B, N, 3] Å
    mask: torch.Tensor,  # [B, N] bool
    k: int = 8,
    fill_value: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trivial-geometric baseline feature for the inverse-folding probe.

    For each residue i, return the sorted distances to its k nearest non-self
    real CAs. This isolates "what does the local geometry alone tell you about
    AA identity" — no chemistry, just a bag of inter-residue distances.

    Padding handling:
        - Self-distance is masked out before topk.
        - Distances to padded residues are set to +inf before topk.
        - Residues with fewer than k real neighbours have the missing slots
          filled with ``fill_value`` (default 50 Å, well outside any real
          local environment) and the per-residue valid flag is True iff i
          itself is a real residue with ≥1 real neighbour.

    Returns:
        feats:  [B, N, k] float
        valid:  [B, N] bool
    """
    B, N, _ = ca_coords_ang.shape
    device = ca_coords_ang.device
    real = mask.bool()
    dist = torch.cdist(ca_coords_ang, ca_coords_ang)  # [B, N, N]

    # Mask self and padded targets to +inf so they sort to the back.
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
    target_pad = ~real.unsqueeze(1).expand(B, N, N)
    invalid = eye | target_pad
    dist = dist.masked_fill(invalid, float("inf"))

    # k might exceed N; clamp.
    k_eff = min(k, N - 1) if N > 1 else 0
    if k_eff <= 0:
        feats = torch.full(
            (B, N, k), fill_value, device=device, dtype=ca_coords_ang.dtype
        )
        valid = torch.zeros(B, N, dtype=torch.bool, device=device)
        return feats, valid

    topk_vals, _ = dist.topk(k_eff, dim=-1, largest=False)  # [B, N, k_eff]
    # Replace +inf (came from rows with too few real neighbours) with fill_value.
    topk_vals = torch.where(
        torch.isfinite(topk_vals), topk_vals, torch.full_like(topk_vals, fill_value)
    )
    if k_eff < k:
        pad = torch.full(
            (B, N, k - k_eff), fill_value, device=device, dtype=ca_coords_ang.dtype
        )
        topk_vals = torch.cat([topk_vals, pad], dim=-1)

    # Valid iff i is real and has at least one real neighbour.
    has_neighbor = (~target_pad & ~eye).any(dim=-1)  # [B, N]
    valid = real & has_neighbor
    feats = topk_vals * valid.float().unsqueeze(-1)
    return feats, valid


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
