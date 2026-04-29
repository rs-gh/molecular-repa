"""Geometry sanity metrics for generated protein backbones.

All functions accept atom37 tensors `[nres, 37, 3]` unless noted otherwise.
Atom37 layout: N=0, CA=1, C=2, O=4 (see src/proteina/openfold/np/residue_constants.py).
"""

from __future__ import annotations

import numpy as np
import torch

CA_IDX = 1  # atom37 index for CA
EXPECTED_CA_CA_BOND = 3.80  # A, idealized virtual bond
CLASH_THRESHOLD = 3.6  # A, non-adjacent CA pair below this counts as clash


def _to_np_ca(atom37: torch.Tensor | np.ndarray) -> np.ndarray:
    """Extract CA coords from atom37 tensor `[nres, 37, 3]` -> `[nres, 3]` numpy."""
    if isinstance(atom37, torch.Tensor):
        atom37 = atom37.detach().cpu().numpy()
    return atom37[:, CA_IDX, :]


def radius_of_gyration(atom37: torch.Tensor | np.ndarray) -> float:
    """Root-mean-square distance of CA atoms from centroid."""
    ca = _to_np_ca(atom37)
    centroid = ca.mean(axis=0)
    return float(np.sqrt(((ca - centroid) ** 2).sum(axis=1).mean()))


def ca_bond_lengths(atom37: torch.Tensor | np.ndarray) -> np.ndarray:
    """Consecutive CA-CA distances. Returns array of length N-1."""
    ca = _to_np_ca(atom37)
    return np.linalg.norm(ca[1:] - ca[:-1], axis=1)


def ca_ca_ca_angles(atom37: torch.Tensor | np.ndarray) -> np.ndarray:
    """Virtual bond angles between consecutive CA-CA-CA triplets in degrees.
    Returns array of length N-2.
    """
    ca = _to_np_ca(atom37)
    v1 = ca[:-2] - ca[1:-1]  # CA_{i} - CA_{i+1}
    v2 = ca[2:] - ca[1:-1]  # CA_{i+2} - CA_{i+1}
    # Guard against zero-length vectors
    n1 = np.linalg.norm(v1, axis=1) + 1e-8
    n2 = np.linalg.norm(v2, axis=1) + 1e-8
    cos = (v1 * v2).sum(axis=1) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def ca_clash_rate(
    atom37: torch.Tensor | np.ndarray, threshold: float = CLASH_THRESHOLD
) -> float:
    """Fraction of non-adjacent (|i-j|>=2) CA pairs within `threshold` A."""
    ca = _to_np_ca(atom37)
    n = ca.shape[0]
    if n < 3:
        return 0.0
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    # Build a mask: i<j, |i-j|>=2
    i, j = np.triu_indices(n, k=2)
    pair_dists = d[i, j]
    return float((pair_dists < threshold).mean())


def end_to_end_distance(atom37: torch.Tensor | np.ndarray) -> float:
    ca = _to_np_ca(atom37)
    return float(np.linalg.norm(ca[-1] - ca[0]))


def per_sample_summary(atom37: torch.Tensor | np.ndarray) -> dict:
    """One-shot compute of all scalar metrics for a single sample."""
    bonds = ca_bond_lengths(atom37)
    angles = ca_ca_ca_angles(atom37)
    return {
        "n_residues": int(atom37.shape[0]) if hasattr(atom37, "shape") else len(atom37),
        "rg": radius_of_gyration(atom37),
        "bond_mean": float(bonds.mean()),
        "bond_std": float(bonds.std()),
        "bond_max_dev": float(np.max(np.abs(bonds - EXPECTED_CA_CA_BOND))),
        "angle_mean": float(angles.mean()),
        "angle_std": float(angles.std()),
        "clash_rate": ca_clash_rate(atom37),
        "end_to_end": end_to_end_distance(atom37),
    }


# ----- Cross-mode RMSD (same-seed comparison) -----


def kabsch_rmsd(ca_a: np.ndarray, ca_b: np.ndarray) -> float:
    """Best-aligned RMSD between two equal-length CA trajectories via Kabsch."""
    assert ca_a.shape == ca_b.shape, f"shape mismatch {ca_a.shape} vs {ca_b.shape}"
    a = ca_a - ca_a.mean(axis=0)
    b = ca_b - ca_b.mean(axis=0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    a_aligned = a @ R.T
    return float(np.sqrt(((a_aligned - b) ** 2).sum(axis=1).mean()))


def cross_mode_ca_rmsd(
    atom37_a: torch.Tensor | np.ndarray, atom37_b: torch.Tensor | np.ndarray
) -> float:
    return kabsch_rmsd(_to_np_ca(atom37_a), _to_np_ca(atom37_b))
