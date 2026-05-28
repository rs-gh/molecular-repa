"""Pretrained-split CATH classification probe.

The in-place-split probe in ``cath.py`` trains a logistic regression on 75% of
a small pool (~200 val proteins, ~160 after dropping unlabelled and rare
classes) and evaluates on the remaining 25%. That leaves ~40 test proteins,
both halves come from the same val.lmdb, and a silent T → A → C fallback
makes cross-row comparisons across runs and checkpoints unreliable.

This module separates train/eval and pins the level:

    train_cath_probe(reps_train, mask_train, raw_train, level)
        -> (head, train_meta)

    eval_cath_probe(head, train_meta, reps_eval, mask_eval, raw_eval)
        -> CATHPretrainedResult

Typical use: train the head on features from a large train.lmdb sample (~1000+
proteins), evaluate on features from a fixed val.lmdb sample. The level (C, A,
T, or H) is fixed by the caller — to report all four levels, call once per
level. Unlike ``cath.py`` there is no fallback; an unfit row is a NaN row.

Eval-set proteins whose label was not seen at train time are reported as
``n_eval_oov`` rather than silently dropped — the in-place probe's "drop rare
classes" filter applied to a tiny pool was hiding this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from lib.labels import cath_labels_from_raw, mean_pool_by_mask


@dataclass
class CATHPretrainedResult:
    level: str  # "C" | "A" | "T" | "H"
    accuracy: float  # over n_eval_in_vocab; NaN if no in-vocab proteins
    macro_f1: float  # over n_eval_in_vocab
    top1_conf_mean: float  # mean of max-class softmax prob across ALL eval (incl. OOV)
    n_train: int  # train proteins after filtering rare classes
    n_eval: int  # total eval proteins with a label at this level
    n_classes: int  # vocab size at train time (after min_per_class)
    n_eval_in_vocab: int  # eval proteins whose true label is in the train vocab
    n_eval_oov: int  # n_eval - n_eval_in_vocab


def _pool_if_needed(
    reps: torch.Tensor,
    mask: torch.Tensor | None,
) -> np.ndarray:
    """Return [B, D] numpy. Accepts already-pooled [B, D] or per-residue [B, N, D]."""
    if reps.dim() == 2:
        return reps.detach().cpu().float().numpy()
    if reps.dim() == 3:
        if mask is None:
            raise ValueError("mask is required when reps are per-residue [B, N, D]")
        return mean_pool_by_mask(reps, mask.cpu())
    raise ValueError(f"reps must be 2D or 3D, got shape {tuple(reps.shape)}")


def _labels_at_level(raw: List[Data], level: str, vocab: List[str] | None = None):
    """Extract per-protein labels at ``level``.

    If ``vocab`` is given, return label indices into that vocab (or -1 if a
    protein's class is OOV / unlabelled). Otherwise build a fresh vocab from
    ``raw`` and return indices into it.

    Returns ``(labels: np.int64[B], vocab: list[str])``.
    """
    # Reuse the canonical extractor (it handles all the historical cath_code shapes).
    fresh_idx, fresh_uniq = cath_labels_from_raw(raw, level=level)

    if vocab is None:
        return fresh_idx, fresh_uniq

    # Re-key fresh_idx through the supplied vocab.
    vocab_to_idx = {s: i for i, s in enumerate(vocab)}
    out = np.full_like(fresh_idx, fill_value=-1)
    for i, j in enumerate(fresh_idx):
        if j < 0:
            continue
        s = fresh_uniq[j]
        out[i] = vocab_to_idx.get(s, -1)
    return out, vocab


def train_cath_probe(
    reps: torch.Tensor,  # [B, N, D] or [B, D]
    mask: torch.Tensor | None,  # [B, N] or None if reps are pooled
    raw: List[Data],
    level: str = "T",
    head_type: Literal["linear", "mlp"] = "linear",
    min_per_class: int = 3,
    seed: int = 42,
    mlp_epochs: int = 50,
    mlp_lr: float = 1e-3,
    mlp_hidden: int = 256,
    device: str | None = None,
) -> Tuple[Any, Dict]:
    """Fit a CATH classifier head at the requested level.

    Flow:
        X     = mean_pool_by_mask(reps, mask)            # [B, D]
        y, uniq = cath_labels_from_raw(raw, level)       # [B], list[str]
        keep  = (y != -1) and class count >= min_per_class
        head  = LogisticRegression / MLP
        fit   on (X[keep], y[keep])

    Returns:
        head:        sklearn LogisticRegression (linear) or torch.nn.Module (mlp)
        train_meta:  dict with keys
            level         the requested level
            vocab         list[str] — class names; index = head's output class
            head_type     "linear" | "mlp"
            in_dim        D
            n_train       len after filtering
            min_per_class echoed back
            seed          echoed back

    If too few classes survive (< 2), returns ``(None, train_meta)`` with
    ``vocab=[]`` so callers can emit a NaN row deterministically.
    """
    X = _pool_if_needed(reps, mask)  # [B, D] np.float32
    y, uniq = cath_labels_from_raw(raw, level=level)  # [B], list

    train_meta: Dict[str, Any] = {
        "level": level,
        "vocab": [],
        "head_type": head_type,
        "in_dim": int(X.shape[1]),
        "n_train": 0,
        "min_per_class": int(min_per_class),
        "seed": int(seed),
    }

    keep = y != -1
    if keep.sum() < 10:
        return None, train_meta
    Xk = X[keep]
    yk = y[keep]

    # Drop rare classes — the head can't learn classes with too few examples
    # and they pollute the macro-F1 if they survive into eval.
    vals, cnts = np.unique(yk, return_counts=True)
    valid = vals[cnts >= min_per_class]
    if len(valid) < 2:
        return None, train_meta
    sel = np.isin(yk, valid)
    Xk = Xk[sel]
    yk = yk[sel]

    # Compact the label space to the surviving classes.
    surviving_uniq = [uniq[v] for v in valid]
    old_to_new = {int(v): i for i, v in enumerate(valid)}
    yk_new = np.asarray([old_to_new[int(v)] for v in yk], dtype=np.int64)

    if head_type == "linear":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(Xk, yk_new)
        head: Any = clf
    elif head_type == "mlp":
        head = _train_mlp_head(
            Xk,
            yk_new,
            n_classes=len(surviving_uniq),
            hidden=mlp_hidden,
            epochs=mlp_epochs,
            lr=mlp_lr,
            seed=seed,
            device=device,
        )
    else:
        raise ValueError(f"unknown head_type {head_type!r}")

    train_meta["vocab"] = surviving_uniq
    train_meta["n_train"] = int(len(yk_new))
    return head, train_meta


def eval_cath_probe(
    head: Any,
    train_meta: Dict,
    reps: torch.Tensor,  # [B, N, D] or [B, D]
    mask: torch.Tensor | None,  # [B, N] or None
    raw: List[Data],
    device: str | None = None,
) -> CATHPretrainedResult:
    """Score a CATH head on a held-out set.

    Flow:
        X     = mean_pool_by_mask(reps, mask)
        y_idx = labels at train_meta['level'], keyed against train vocab
        for proteins with a ground-truth label:
            top-1 prob = max(predict_proba(X))     ← always recorded
            in-vocab proteins (y_idx >= 0) contribute to accuracy and macro-F1
            OOV proteins (y_idx == -1) are reported under n_eval_oov

    Confidence is reported across ALL labeled eval proteins, not just in-vocab,
    because the generation suite consumes top1_conf_mean as a "does this look
    like a recognizable fold" signal that should not silently exclude rare
    folds the train set didn't cover.
    """
    level = train_meta["level"]
    vocab = train_meta.get("vocab") or []
    head_type = train_meta.get("head_type", "linear")
    n_classes = len(vocab)

    if head is None or n_classes < 2:
        return CATHPretrainedResult(
            level=level,
            accuracy=float("nan"),
            macro_f1=float("nan"),
            top1_conf_mean=float("nan"),
            n_train=int(train_meta.get("n_train", 0)),
            n_eval=0,
            n_classes=n_classes,
            n_eval_in_vocab=0,
            n_eval_oov=0,
        )

    X = _pool_if_needed(reps, mask)  # [B, D]
    y_vocab, _ = _labels_at_level(
        raw, level=level, vocab=vocab
    )  # [B]; -1 = OOV/unlabelled

    # Mark unlabelled (no cath_code at all) vs OOV-but-labelled separately.
    y_fresh, fresh_uniq = cath_labels_from_raw(raw, level=level)
    labelled = y_fresh != -1  # has any cath_code at this level
    in_vocab_mask = (y_vocab != -1) & labelled  # label exists in train vocab
    # oov_mask = labelled & (~in_vocab_mask)  # label exists but not in train vocab — reserved for future OOV reporting

    n_labelled = int(labelled.sum())
    if n_labelled == 0:
        return CATHPretrainedResult(
            level=level,
            accuracy=float("nan"),
            macro_f1=float("nan"),
            top1_conf_mean=float("nan"),
            n_train=int(train_meta.get("n_train", 0)),
            n_eval=0,
            n_classes=n_classes,
            n_eval_in_vocab=0,
            n_eval_oov=0,
        )

    # Predict over all labelled proteins so confidence covers OOV too.
    X_lab = X[labelled]
    if head_type == "linear":
        proba = head.predict_proba(X_lab)  # [B_lab, n_classes]
        pred_lab = head.classes_[proba.argmax(axis=1)]
    else:
        proba, pred_lab = _predict_mlp(head, X_lab, device=device)

    top1_conf = proba.max(axis=1)  # [B_lab]
    top1_conf_mean = float(np.mean(top1_conf))

    # Accuracy / F1 only over in-vocab labelled proteins. We need to re-index
    # from the full-batch in_vocab_mask to the contiguous X_lab index space.
    lab_idx = np.where(labelled)[0]
    in_vocab_lab = np.array([in_vocab_mask[i] for i in lab_idx], dtype=bool)
    n_in_vocab = int(in_vocab_lab.sum())
    n_oov = int(n_labelled - n_in_vocab)

    if n_in_vocab > 0:
        from sklearn.metrics import accuracy_score, f1_score

        y_true_iv = y_vocab[lab_idx[in_vocab_lab]]
        y_pred_iv = pred_lab[in_vocab_lab]
        acc = float(accuracy_score(y_true_iv, y_pred_iv))
        f1 = float(f1_score(y_true_iv, y_pred_iv, average="macro", zero_division=0))
    else:
        acc = float("nan")
        f1 = float("nan")

    return CATHPretrainedResult(
        level=level,
        accuracy=acc,
        macro_f1=f1,
        top1_conf_mean=top1_conf_mean,
        n_train=int(train_meta.get("n_train", 0)),
        n_eval=n_labelled,
        n_classes=n_classes,
        n_eval_in_vocab=n_in_vocab,
        n_eval_oov=n_oov,
    )


def run_pretrained_cath_probe(
    reps_train: torch.Tensor,
    train_batch: Dict,
    raw_train: List[Data],
    reps_eval: torch.Tensor,
    eval_batch: Dict,
    raw_eval: List[Data],
    level: str = "T",
    head_type: Literal["linear", "mlp"] = "linear",
    min_per_class: int = 3,
    seed: int = 42,
    device: str | None = None,
) -> Dict:
    """End-to-end: fit on (reps_train, raw_train), score on (reps_eval, raw_eval).

    Returns a flat dict suitable for the JSONL row schema used by
    ``pretrain_probe_sweep.py``::

        {
            "cath_level": "T",
            "cath_accuracy": ...,
            "cath_macro_f1": ...,
            "cath_top1_conf_mean": ...,
            "cath_n_classes": ...,
            "cath_n_train": ...,
            "cath_n_eval": ...,
            "cath_n_eval_in_vocab": ...,
            "cath_n_eval_oov": ...,
        }
    """
    head, train_meta = train_cath_probe(
        reps_train,
        train_batch.get("mask") if isinstance(train_batch, dict) else None,
        raw_train,
        level=level,
        head_type=head_type,
        min_per_class=min_per_class,
        seed=seed,
        device=device,
    )
    result = eval_cath_probe(
        head,
        train_meta,
        reps_eval,
        eval_batch.get("mask") if isinstance(eval_batch, dict) else None,
        raw_eval,
        device=device,
    )
    return {
        "cath_level": result.level,
        "cath_accuracy": result.accuracy,
        "cath_macro_f1": result.macro_f1,
        "cath_top1_conf_mean": result.top1_conf_mean,
        "cath_n_classes": result.n_classes,
        "cath_n_train": result.n_train,
        "cath_n_eval": result.n_eval,
        "cath_n_eval_in_vocab": result.n_eval_in_vocab,
        "cath_n_eval_oov": result.n_eval_oov,
    }


# --------------------------------------------------------------------------- #
# Optional MLP head (used when head_type="mlp"). Linear is the default.
# --------------------------------------------------------------------------- #


class _MLPHead(torch.nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_mlp_head(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str | None,
) -> _MLPHead:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = _MLPHead(X.shape[1], n_classes, hidden=hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).long().to(device)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(Xt)
        loss = torch.nn.functional.cross_entropy(logits, yt)
        loss.backward()
        opt.step()
    head.eval()
    # Stash classes_ so eval can mirror sklearn's prediction interface.
    head.classes_ = np.arange(n_classes, dtype=np.int64)
    return head


@torch.no_grad()
def _predict_mlp(
    head: _MLPHead,
    X: np.ndarray,
    device: str | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(head.parameters()).device
    Xt = torch.from_numpy(X).float().to(device)
    logits = head(Xt)
    proba = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = proba.argmax(axis=1).astype(np.int64)
    return proba, pred
