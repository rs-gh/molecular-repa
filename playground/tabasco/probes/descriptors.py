"""P4 — per-molecule RDKit descriptor regression probe (tabasco).

Mean-pools per-atom representations over unmasked atoms, then runs linear
ridge regression against four standard RDKit descriptors:
  MolWt               — overall size / mass
  MolLogP             — lipophilicity
  NumRings            — topological complexity
  NumRotatableBonds   — conformational flexibility

These are cheap to compute, require zero external data, and cover distinct
axes of molecular structure. High R² on all four is a strong signal that the
representation captures molecule-level chemistry.

Usage:
  from descriptors import run_descriptor_probe
  results = run_descriptor_probe(reps, mols, batch["padding_mask"])
  for r in results: print(r.target, r.pearson, r.r2)
"""

from __future__ import annotations

from typing import List

import torch
from rdkit import Chem

from utils import (
    RegResult,
    descriptor_labels,
    linear_regress,
    mean_pool,
)


def run_descriptor_probe(
    reps: torch.Tensor,
    mols: List[Chem.Mol],
    padding_mask: torch.Tensor,
) -> List[RegResult]:
    """Linear regression probe on per-molecule descriptors. `reps`: [B, N, D] CPU."""
    X = mean_pool(reps, padding_mask.cpu())  # [B, D]
    Y, names = descriptor_labels(mols)  # [B, 4]
    return linear_regress(X, Y, target_names=names)


def describe(results: List[RegResult]) -> str:
    parts = []
    for r in results:
        parts.append(f"{r.target}: r={r.pearson:.3f} R²={r.r2:.3f}")
    return "descriptors: " + " | ".join(parts)
