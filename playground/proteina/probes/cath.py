"""P2 — CATH fold classification probe (proteina).

Linear classifier on mean-pooled per-residue representations → coarse CATH
class. Defaults to T-level (class.arch.topology) which gives ~dozens of
classes depending on what's present in the subset.

For probes that barely have enough samples per class, falls back to A-level
(class.arch) or C-level (class only) — reports which was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

from utils import cath_labels_from_raw, mean_pool_by_mask


@dataclass
class CATHResult:
    level: str
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    n_classes: int


def run_cath_probe(
    reps: torch.Tensor,
    mask: torch.Tensor,
    raw: List[Data],
    preferred_level: str = "T",
    min_per_class: int = 3,
    seed: int = 42,
) -> CATHResult:
    """Linear probe on CATH class labels.

    Tries `preferred_level` first. If fewer than 2 classes survive
    the `min_per_class` filter, falls back to coarser levels until
    enough classes remain or we run out.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    X = mean_pool_by_mask(reps, mask.cpu())  # [B, D]

    for level in [preferred_level, "A", "C"]:
        y, uniq = cath_labels_from_raw(raw, level=level)
        # Filter unlabelled
        keep = y != -1
        Xi = X[keep]
        yi = y[keep]
        if len(Xi) < 10 or len(uniq) < 2:
            continue
        # Drop rare classes
        vals, cnts = np.unique(yi, return_counts=True)
        valid = vals[cnts >= min_per_class]
        mask2 = np.isin(yi, valid)
        Xi, yi = Xi[mask2], yi[mask2]
        if len(np.unique(yi)) < 2:
            continue

        Xtr, Xte, ytr, yte = train_test_split(
            Xi, yi, test_size=0.25, random_state=seed, stratify=yi
        )
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        return CATHResult(
            level=level,
            accuracy=float(accuracy_score(yte, pred)),
            macro_f1=float(f1_score(yte, pred, average="macro")),
            n_train=len(Xtr),
            n_test=len(Xte),
            n_classes=int(len(np.unique(yi))),
        )

    # Not enough labelled proteins or classes — return a sentinel row rather
    # than raising. Lets the sweep record contact-probe results and move on.
    return CATHResult(
        level="none",
        accuracy=float("nan"),
        macro_f1=float("nan"),
        n_train=0,
        n_test=0,
        n_classes=0,
    )
