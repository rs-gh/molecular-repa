"""P3 — atom-type classification probe (tabasco).

Trains a linear classifier to predict atomic number from per-atom
representations. This is the most direct test of whether a representation
discriminates atom identity — the finest-grained molecular property.

Known reference numbers (frozen encoders, GEOM training set):
  CheMeleon (2048-d, 2D graph)   : 1.000
  MACE-OFF small (192-d, 3D)     : > 0.95 (see playground/tabasco/mace/FINDINGS.md)
  GearNet (protein ref)          : 0.154

Usage:
  from atom_type import run_atom_type_probe
  result = run_atom_type_probe(reps, mols, max_atoms)
  print(result.accuracy, result.macro_f1)
"""

from __future__ import annotations

from typing import List

import torch
from rdkit import Chem

from utils import (
    ClassResult,
    atom_type_labels,
    flatten_unmasked,
    linear_classify,
)


def run_atom_type_probe(
    reps: torch.Tensor,
    mols: List[Chem.Mol],
    max_atoms: int,
) -> ClassResult:
    """Linear probe on atom type. `reps` is [B, N, D] on CPU."""
    labels = atom_type_labels(mols, max_atoms=max_atoms)  # [B, N], -1 for pad/'*'
    X, y = flatten_unmasked(reps, labels)
    if X.shape[0] == 0:
        raise ValueError("No atoms left after masking — check input mols.")
    return linear_classify(X, y)


def describe(result: ClassResult) -> str:
    return (
        f"atom-type: acc={result.accuracy:.4f}  f1={result.macro_f1:.4f}  "
        f"n_train={result.n_train} n_test={result.n_test} n_classes={result.n_classes}"
    )


if __name__ == "__main__":
    # Smoke test on MACE frozen encoder.
    import os
    from utils import (
        PROJECT_ROOT,
        _default_device,
        load_molecules,
        build_batch,
        extract_encoder_embeddings,
    )
    import sys

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from tabasco.models.components.encoders import MACEEncoder

    mols = load_molecules(200, dataset="qm9")
    max_atoms = max(m.GetNumAtoms() for m in mols)
    batch = build_batch(mols, max_atoms=max_atoms, device=_default_device())
    enc = MACEEncoder(model_name="small").to(_default_device())
    reps = extract_encoder_embeddings(enc, batch)
    print("rep shape:", reps.shape)
    res = run_atom_type_probe(reps, mols, max_atoms=max_atoms)
    print(describe(res))
