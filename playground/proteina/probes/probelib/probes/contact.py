"""P1 — long-range contact probe (P@L/k).

Named "linear probe" by convention (backbone frozen), but the default head is
actually a two-layer MLP: ``Linear(3D, 256) -> SiLU -> Linear(256, 1)``.

Input per pair:
    [h_i ‖ h_j ‖ |h_i - h_j|]  ∈ R^{3D}

Output: one scalar logit per pair. Trained via BCE; evaluated via
precision-at-top-L/k ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from probelib.labels import contact_labels


@dataclass
class ContactResult:
    p_at_L: float  # mean precision @ L predictions
    p_at_L_5: float  # mean precision @ L/5 predictions (headline metric)
    p_at_L_2: float  # mean precision @ L/2 predictions
    n_proteins_test: int


def _balanced_pairs(
    h: torch.Tensor,  # [N, D]
    lab: torch.Tensor,  # [N, N] binary contact labels
    pmask: torch.Tensor,  # [N, N] bool, True where pair is valid
    n_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """Sample up to ``n_samples`` positive + ``n_samples`` negative pairs from one protein.

    Flow:
        positives  = (lab == 1) & pmask               # [N, N] bool
        negatives  = (lab == 0) & pmask
        pos_ij     = nonzero(positives)                # [P_pos, 2] index pairs
        neg_ij     = nonzero(negatives)                # [P_neg, 2]
        sel        = min(available, n_samples) each
        ij         = concat(pos_sel, neg_sel)           # [P_sampled, 2]
        feats      = [h[i] ‖ h[j] ‖ |h[i]-h[j]|]       # [P_sampled, 3D]
        y          = [1, ..., 1, 0, ..., 0]             # [P_sampled]

    Returns ``(feats, y)`` or ``None`` if no positives or no negatives exist.
    """
    pos = (lab == 1) & pmask
    neg = (lab == 0) & pmask
    pos_ij = pos.nonzero(as_tuple=False)  # [P_pos, 2]
    neg_ij = neg.nonzero(as_tuple=False)  # [P_neg, 2]
    if pos_ij.numel() == 0 or neg_ij.numel() == 0:
        return None
    npos = min(len(pos_ij), n_samples)
    nneg = min(len(neg_ij), n_samples)
    pos_sel = pos_ij[torch.randperm(len(pos_ij))[:npos]]
    neg_sel = neg_ij[torch.randperm(len(neg_ij))[:nneg]]
    ij = torch.cat([pos_sel, neg_sel], dim=0)  # [npos+nneg, 2]
    y = torch.cat([torch.ones(npos), torch.zeros(nneg)])  # [npos+nneg]
    hi = h[ij[:, 0]]  # [P, D]
    hj = h[ij[:, 1]]  # [P, D]
    feats = torch.cat([hi, hj, (hi - hj).abs()], dim=-1)  # [P, 3D]
    return feats, y


def _build_head(in_dim: int, device: str) -> torch.nn.Module:
    """Default head: 2-layer MLP. Caller can swap for a pure Linear later (Axis 3)."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 256),  # (3D -> 256)
        torch.nn.SiLU(),
        torch.nn.Linear(256, 1),  # (256 -> 1)
    ).to(device)


def linear_probe_contacts(
    reps: torch.Tensor,  # [B, N, D] on CPU
    ca_coords_ang: torch.Tensor,  # [B, N, 3] on CPU (for labels)
    mask: torch.Tensor,  # [B, N] on CPU
    lengths: torch.Tensor,  # [B] on CPU, unmasked lengths
    test_frac: float = 0.2,
    seed: int = 42,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 4,
) -> ContactResult:
    """Train a small MLP on pair features, evaluate top-L/k precision.

    Inputs:
        reps:           [B, N, D] frozen-backbone hidden states
        ca_coords_ang:  [B, N, 3] CA coords in Å (used only for label construction)
        mask:           [B, N] bool
        lengths:        [B] — unmasked residue counts

    Flow (train):
        for each epoch:
          for each minibatch of `batch_size` proteins:
            for each protein: sample balanced pair feats [P, 3D]
            concat across proteins, BCE step
    Flow (eval):
        for each test protein of length L (≥ 50):
          build feats for ALL valid long-range upper-tri pairs  # [P, 3D]
          score via head                                         # [P]
          rank descending, take top-k, compute hits/k for k∈{L, L/2, L/5}
        average P@L/k across test proteins.

    Returns:
        ContactResult with p_at_L, p_at_L_2, p_at_L_5, n_proteins_test.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    B, N, D = reps.shape
    labels, pair_mask = contact_labels(ca_coords_ang, mask)  # [B,N,N] each

    perm = torch.randperm(B)
    n_test = max(1, int(B * test_frac))
    test_idx = perm[:n_test].tolist()
    train_idx = perm[n_test:].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = _build_head(3 * D, device)  # (3D -> 256 -> 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    # --- Train ---
    head.train()
    for _epoch in range(epochs):
        np.random.shuffle(train_idx)
        for b_start in range(0, len(train_idx), batch_size):
            sub = train_idx[b_start : b_start + batch_size]
            feats_all, y_all = [], []
            for i in sub:
                out = _balanced_pairs(reps[i], labels[i], pair_mask[i], n_samples=200)
                if out is None:
                    continue
                f, y = out
                feats_all.append(f)
                y_all.append(y)
            if not feats_all:
                continue
            f = torch.cat(feats_all, dim=0).to(device)  # [P_batch, 3D]
            y = torch.cat(y_all, dim=0).to(device)  # [P_batch]
            logits = head(f).squeeze(-1)  # [P_batch]
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # --- Eval: P@L/k per protein, averaged ---
    head.eval()
    p_L, p_L2, p_L5 = [], [], []
    with torch.no_grad():
        for i in test_idx:
            L = int(lengths[i].item())
            if L < 50:  # too short for meaningful long-range contact stats
                continue
            h = reps[i, :L]  # [L, D]
            pm = pair_mask[i, :L, :L]  # [L, L]
            lab = labels[i, :L, :L]  # [L, L]
            # Score upper triangle only — contact is symmetric, avoid double count.
            iu = torch.triu_indices(L, L, offset=1)  # [2, L*(L-1)/2]
            keep = pm[iu[0], iu[1]]
            ii, jj = iu[0][keep], iu[1][keep]  # [P_test] each
            hi = h[ii]  # [P_test, D]
            hj = h[jj]  # [P_test, D]
            feats = torch.cat([hi, hj, (hi - hj).abs()], dim=-1).to(
                device
            )  # [P_test, 3D]
            scores = head(feats).squeeze(-1).cpu()  # [P_test]
            y = lab[ii, jj]  # [P_test]
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


def run_contact_probe(
    reps: torch.Tensor,  # [B, N, D] CPU
    batch: Dict,  # must contain 'coords' [B,N,37,3] Å, 'mask' [B,N], 'lengths' [B]
) -> ContactResult:
    """Thin orchestrator — pulls CA coords + mask + lengths out of the batch dict."""
    ca = batch["coords"][:, :, 1, :].cpu()  # [B, N, 3] Å
    mask = batch["mask"].cpu()  # [B, N]
    lengths = batch["lengths"].cpu()  # [B]
    return linear_probe_contacts(reps, ca, mask, lengths)
