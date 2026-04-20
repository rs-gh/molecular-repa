"""P2 — CATH fold classification probe.

Logistic regression on mean-pooled per-residue representations → coarse CATH
class. Defaults to T-level (class.arch.topology); falls back to A- or C-level
when too few samples per class survive.

Head is a true linear model (sklearn LogisticRegression), not an MLP — CATH is
simple enough that nonlinearity doesn't buy much and a linear probe cleanly
isolates "is the class label linearly decodable from the mean-pooled rep?".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

from lib.labels import cath_labels_from_raw, mean_pool_by_mask


@dataclass
class CATHResult:
    level: str
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    n_classes: int


def run_cath_probe(
    reps: torch.Tensor,  # [B, N, D]
    mask: torch.Tensor,  # [B, N]
    raw: List[Data],
    preferred_level: str = "T",
    min_per_class: int = 3,
    seed: int = 42,
) -> CATHResult:
    """Linear probe on CATH class labels.

    Flow:
        X          = mean_pool_by_mask(reps, mask)       # [B, D] per-protein
        for level in [preferred, A, C]:
            y      = cath_labels_from_raw(raw, level)    # [B] int
            drop   unlabelled (y == -1) and rare classes (< min_per_class)
            split  stratified 75/25
            clf    = LogisticRegression(max_iter=2000)
            fit    on Xtr, ytr; predict on Xte
            return CATHResult(...)
        # If no level has ≥ 2 classes with enough samples, return a sentinel
        # row (level="none", accuracy=NaN) so the sweep still records P1.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    X = mean_pool_by_mask(reps, mask.cpu())  # [B, D]

    for level in [preferred_level, "A", "C"]:
        y, uniq = cath_labels_from_raw(raw, level=level)  # [B], list
        # Filter unlabelled
        keep = y != -1
        Xi = X[keep]  # [B', D]
        yi = y[keep]  # [B']
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

    # Fallback sentinel — let the sweep record P1 even if P2 is undefined for this subset.
    return CATHResult(
        level="none",
        accuracy=float("nan"),
        macro_f1=float("nan"),
        n_train=0,
        n_test=0,
        n_classes=0,
    )
