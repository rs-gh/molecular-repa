"""P2 — CATH fold classification probe.

Logistic regression on mean-pooled per-residue representations → coarse CATH
class at a fixed level (T by default; caller picks).

Head is a true linear model (sklearn LogisticRegression), not an MLP — CATH is
simple enough that nonlinearity doesn't buy much and a linear probe cleanly
isolates "is the class label linearly decodable from the mean-pooled rep?".

The previous version of this probe silently fell back from T → A → C when too
few classes survived. That was removed because it broke cross-row comparison:
a curve over checkpoints could mix T-level and A-level rows without the reader
knowing. Now if the requested level cannot be fit (< 2 surviving classes after
``min_per_class``), the row is NaN at the requested level. For larger sample
sizes and a no-fallback OOV-aware design see ``cath_pretrained.py``.
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
    """Linear probe on CATH class labels at a fixed level.

    Data funnel (with default test_size=0.25):
        B proteins in                          e.g. B=200
          └─ drop unlabelled (no CATH code)        ~180 proteins  (varies by dataset)
          └─ drop rare classes (< min_per_class)   ~160 proteins  (varies by n_classes)
               └─ 75% train split                  ~120 proteins  → logistic regression fits here
               └─ 25% test split                    ~40 proteins  → accuracy/F1 reported here

    Note: the contact probe uses an 80/20 split; this probe uses 75/25. The
    difference is historical and not intentional — both are reasonable choices
    and the comparison across probes is relative (REPA vs baseline), so the
    split ratio doesn't affect interpretation.

    Flow:
        X          = mean_pool_by_mask(reps, mask)       # [B, D] per-protein
        y          = cath_labels_from_raw(raw, preferred_level)
        drop       unlabelled (y == -1) and rare classes (< min_per_class)
        if < 2 classes survive: NaN row at preferred_level
        split      stratified 75/25
        clf        = LogisticRegression(max_iter=2000)
        fit / predict / report accuracy + macro_f1

    Unlike the historical version, there is no T → A → C fallback: the level
    field on the returned row is always ``preferred_level``. NaN rows at small
    n are the honest answer; the heavy lifting is in ``cath_pretrained.py``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    X = mean_pool_by_mask(reps, mask.cpu())  # [B, D]

    nan_row = CATHResult(
        level=preferred_level,
        accuracy=float("nan"),
        macro_f1=float("nan"),
        n_train=0,
        n_test=0,
        n_classes=0,
    )

    y, uniq = cath_labels_from_raw(raw, level=preferred_level)
    keep = y != -1
    Xi = X[keep]
    yi = y[keep]
    if len(Xi) < 10 or len(uniq) < 2:
        return nan_row

    vals, cnts = np.unique(yi, return_counts=True)
    valid = vals[cnts >= min_per_class]
    sel = np.isin(yi, valid)
    Xi, yi = Xi[sel], yi[sel]
    if len(np.unique(yi)) < 2:
        return nan_row

    Xtr, Xte, ytr, yte = train_test_split(
        Xi, yi, test_size=0.25, random_state=seed, stratify=yi
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return CATHResult(
        level=preferred_level,
        accuracy=float(accuracy_score(yte, pred)),
        macro_f1=float(f1_score(yte, pred, average="macro")),
        n_train=len(Xtr),
        n_test=len(Xte),
        n_classes=int(len(np.unique(yi))),
    )
