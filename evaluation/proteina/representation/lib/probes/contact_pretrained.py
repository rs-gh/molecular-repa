"""Pretrained-split contact probe — REPA-paper-style probing.

The in-place-split probe in ``contact.py`` trains the head on 80% of a small
pool (~200 val proteins) and evaluates on the remaining 20%. That matches
quick experimentation but leaves only ~40 test proteins, and train/test both
come from the same val.lmdb.

This module separates the two halves:

    train_contact_probe(reps_train, ca_train, mask_train, lengths_train, ...)
        -> trained probe head (nn.Module)

    eval_contact_probe(head, reps_eval, ca_eval, mask_eval, lengths_eval)
        -> ContactResult (P@L/k precision, n_proteins_test)

Typical use: train the head on features from a large train.lmdb sample,
evaluate on features from a fixed val.lmdb sample. Both halves can come from
any LMDB; the contract is just that features have the same [B, N, D] shape
and the CA coords live in Å.

Architecture of the probe head is shared with the in-place probe (same
``_build_head`` helper, same ``_balanced_pairs`` sampler) so numbers live on
the same scale.
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import torch
import torch.nn.functional as F

from lib.labels import contact_labels
from lib.probes.contact import (
    ContactResult,
    _balanced_pairs,
    _build_head,
)


def train_contact_probe(
    reps: torch.Tensor,  # [B, N, D] CPU — train features
    ca_coords_ang: torch.Tensor,  # [B, N, 3] CPU
    mask: torch.Tensor,  # [B, N] CPU
    lengths: torch.Tensor,  # [B] CPU
    head_type: Literal["mlp", "linear"] = "mlp",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 4,
    n_pairs_per_protein: int = 200,
    seed: int = 42,
    device: str = None,
) -> torch.nn.Module:
    """Fit the probe head on all provided train features.

    Flow:
        labels, pair_mask = contact_labels(ca, mask)       # [B, N, N] each
        head = _build_head(3D, device, head_type)
        for epoch in range(epochs):
            for minibatch of `batch_size` proteins:
                for each protein: sample `n_pairs_per_protein` balanced pairs
                concat feats + y across minibatch
                BCE step

    Returns the trained head (in eval mode, on device). Caller is responsible
    for keeping the head around — it is NOT saved to disk here.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    B, N, D = reps.shape
    labels, pair_mask = contact_labels(ca_coords_ang, mask)  # [B, N, N] each

    head = _build_head(3 * D, device, head_type=head_type)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    train_idx = list(range(B))
    head.train()
    for _epoch in range(epochs):
        np.random.shuffle(train_idx)
        for b_start in range(0, len(train_idx), batch_size):
            sub = train_idx[b_start : b_start + batch_size]
            feats_all, y_all = [], []
            for i in sub:
                out = _balanced_pairs(
                    reps[i], labels[i], pair_mask[i], n_samples=n_pairs_per_protein
                )
                if out is None:
                    continue
                f, y = out
                feats_all.append(f)
                y_all.append(y)
            if not feats_all:
                continue
            f = torch.cat(feats_all, dim=0).to(device)
            y = torch.cat(y_all, dim=0).to(device)
            logits = head(f).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    return head


@torch.no_grad()
def eval_contact_probe(
    head: torch.nn.Module,
    reps: torch.Tensor,  # [B, N, D] CPU — eval features
    ca_coords_ang: torch.Tensor,  # [B, N, 3] CPU
    mask: torch.Tensor,  # [B, N] CPU
    lengths: torch.Tensor,  # [B] CPU
    min_length: int = 50,
    device: str = None,
) -> ContactResult:
    """Score every eval protein's upper-tri long-range pairs, compute P@L/k.

    Flow:
        labels, pair_mask = contact_labels(ca, mask)
        head.eval()
        for each eval protein of length L ≥ min_length:
            pm = pair_mask[i, :L, :L]
            lab = labels[i, :L, :L]
            iu = triu_indices(L, L, offset=1)
            ii, jj = iu[0][pm[iu]], iu[1][pm[iu]]
            feats = [h_i ‖ h_j ‖ |h_i - h_j|]
            scores = head(feats)
            for k in {L, L/2, L/5}:
                hits = labels[top-k].sum()
                p_at_k.append(hits / k)
        return mean(p_at_k) over proteins

    Proteins shorter than ``min_length`` are skipped (too short for a
    meaningful long-range stat). ``n_proteins_test`` reflects the actual
    count contributing to the metric, not ``B``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    B, N, D = reps.shape
    labels, pair_mask = contact_labels(ca_coords_ang, mask)

    head = head.to(device)
    head.eval()

    p_L, p_L2, p_L5 = [], [], []
    for i in range(B):
        L = int(lengths[i].item())
        if L < min_length:
            continue
        h = reps[i, :L]  # [L, D]
        pm = pair_mask[i, :L, :L]
        lab = labels[i, :L, :L]
        iu = torch.triu_indices(L, L, offset=1)
        keep = pm[iu[0], iu[1]]
        ii, jj = iu[0][keep], iu[1][keep]
        hi = h[ii]
        hj = h[jj]
        feats = torch.cat([hi, hj, (hi - hj).abs()], dim=-1).to(device)
        scores = head(feats).squeeze(-1).cpu()
        y = lab[ii, jj]
        if scores.numel() == 0:
            continue
        order = torch.argsort(scores, descending=True)
        for topk, target in [(L, p_L), (L // 2, p_L2), (L // 5, p_L5)]:
            k = max(1, min(topk, scores.numel()))
            hits = y[order[:k]].sum().item()
            target.append(hits / k)

    return ContactResult(
        p_at_L=float(np.mean(p_L)) if p_L else 0.0,
        p_at_L_2=float(np.mean(p_L2)) if p_L2 else 0.0,
        p_at_L_5=float(np.mean(p_L5)) if p_L5 else 0.0,
        n_proteins_test=len(p_L5),
    )


def run_pretrained_contact_probe(
    reps_train: torch.Tensor,
    train_batch: Dict,
    reps_eval: torch.Tensor,
    eval_batch: Dict,
    head_type: Literal["mlp", "linear"] = "mlp",
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict:
    """End-to-end: train the head on ``reps_train``, evaluate on ``reps_eval``.

    Returns a dict matching the JSONL schema used by the in-place probe so
    downstream plotting code can consume both regimes interchangeably:

        {
            "<head_type>": { "p_at_L_5": ..., "p_at_L_2": ..., "p_at_L": ...,
                             "n_proteins_test": ..., "seed": ... },
            "p_at_L_5": ...,  # flat legacy fields, for back-compat
            "p_at_L_2": ...,
            "p_at_L": ...,
            "n_proteins_test": ...,
        }
    """
    ca_train = train_batch["coords"][:, :, 1, :].cpu()
    mask_train = train_batch["mask"].cpu()
    lengths_train = train_batch["lengths"].cpu()

    ca_eval = eval_batch["coords"][:, :, 1, :].cpu()
    mask_eval = eval_batch["mask"].cpu()
    lengths_eval = eval_batch["lengths"].cpu()

    head = train_contact_probe(
        reps_train,
        ca_train,
        mask_train,
        lengths_train,
        head_type=head_type,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )
    result = eval_contact_probe(head, reps_eval, ca_eval, mask_eval, lengths_eval)

    out = {
        head_type: {
            "p_at_L": result.p_at_L,
            "p_at_L_2": result.p_at_L_2,
            "p_at_L_5": result.p_at_L_5,
            "n_proteins_test": result.n_proteins_test,
            "seed": int(seed),
        },
        "p_at_L": result.p_at_L,
        "p_at_L_2": result.p_at_L_2,
        "p_at_L_5": result.p_at_L_5,
        "n_proteins_test": result.n_proteins_test,
    }
    return out
