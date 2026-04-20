"""P1 — long-range contact probe (P@L/k).

Named "linear probe" by convention (backbone frozen), but the default head is
actually a two-layer MLP: ``Linear(3D, 256) -> SiLU -> Linear(256, 1)``.

Input per pair:
    [h_i ‖ h_j ‖ |h_i - h_j|]  ∈ R^{3D}

Output: one scalar logit per pair. Trained via BCE; evaluated via
precision-at-top-L/k ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

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


@dataclass
class MultiSeedResult:
    """Mean + std over multiple train/test splits (different seeds).

    Populated when ``linear_probe_contacts_multi`` is called with len(seeds) ≥ 2.
    Preserves the per-seed raw arrays so the caller can emit them for length
    bucketing / error bars.
    """

    p_at_L_mean: float
    p_at_L_std: float
    p_at_L_2_mean: float
    p_at_L_2_std: float
    p_at_L_5_mean: float
    p_at_L_5_std: float
    n_proteins_test: int
    seeds: List[int] = field(default_factory=list)
    p_at_L_5_per_seed: List[float] = field(default_factory=list)


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


def _build_head(
    in_dim: int, device: str, head_type: Literal["mlp", "linear"] = "mlp"
) -> torch.nn.Module:
    """Probe head.

    head_type="mlp":    Linear(3D, 256) -> SiLU -> Linear(256, 1)  (~393k params for D=512)
    head_type="linear": Linear(3D, 1)                              (~1.5k params for D=512)

    The linear variant matters for isolating representation quality — if the
    linear probe tracks the MLP probe closely, the MLP isn't adding
    memorization capacity and the reported number genuinely reflects the
    backbone's rep quality. If the linear probe is much worse, the MLP is
    finding nonlinear pair-relationships the backbone didn't expose directly.
    """
    if head_type == "linear":
        return torch.nn.Linear(in_dim, 1).to(device)
    elif head_type == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),  # (3D -> 256)
            torch.nn.SiLU(),
            torch.nn.Linear(256, 1),  # (256 -> 1)
        ).to(device)
    raise ValueError(f"unknown head_type: {head_type}")


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
    head_type: Literal["mlp", "linear"] = "mlp",
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
    head = _build_head(3 * D, device, head_type=head_type)
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


def linear_probe_contacts_multi(
    reps: torch.Tensor,
    ca_coords_ang: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    seeds: List[int],
    head_type: Literal["mlp", "linear"] = "mlp",
    **kwargs,
) -> MultiSeedResult:
    """Run the contact probe across multiple seeds; report mean ± std.

    Each seed gets its own train/test split and fresh head. ``reps`` are
    reused across seeds (only the probe retrains), so marginal cost per
    extra seed is small — dominated by 15 epochs of MLP training on ~1600
    sampled pairs, not by the backbone forward pass.

    Returns MultiSeedResult with p_at_L{,_2,_5}_mean/std + per-seed raw values.
    """
    per_seed: List[ContactResult] = []
    for s in seeds:
        r = linear_probe_contacts(
            reps,
            ca_coords_ang,
            mask,
            lengths,
            seed=s,
            head_type=head_type,
            **kwargs,
        )
        per_seed.append(r)
    p_L = np.array([r.p_at_L for r in per_seed])
    p_L2 = np.array([r.p_at_L_2 for r in per_seed])
    p_L5 = np.array([r.p_at_L_5 for r in per_seed])
    # n_proteins_test should be the same across seeds (test_frac is fixed, only
    # the identity of the held-out set changes). Take any one.
    n_test = per_seed[0].n_proteins_test if per_seed else 0
    return MultiSeedResult(
        p_at_L_mean=float(p_L.mean()),
        p_at_L_std=float(p_L.std(ddof=0)),
        p_at_L_2_mean=float(p_L2.mean()),
        p_at_L_2_std=float(p_L2.std(ddof=0)),
        p_at_L_5_mean=float(p_L5.mean()),
        p_at_L_5_std=float(p_L5.std(ddof=0)),
        n_proteins_test=n_test,
        seeds=list(seeds),
        p_at_L_5_per_seed=p_L5.tolist(),
    )


def run_contact_probe_full(
    reps: torch.Tensor,
    batch: Dict,
    heads: List[Literal["mlp", "linear"]] = ("mlp",),
    seeds: List[int] = (42,),
) -> Dict:
    """Rich contact-probe orchestrator with head + seed options.

    Output schema (ready to drop into the JSONL ``contact`` field):

        {
            "mlp":    { "p_at_L_5_mean": ..., "p_at_L_5_std": ..., ...
                        "seeds": [42, 43, 44],
                        "p_at_L_5_per_seed": [...] },
            "linear": { ... },              # present only if "linear" in heads
            # Legacy flat fields — also populated for back-compat with old code
            # that reads r["contact"]["p_at_L_5"] directly. Taken from the MLP
            # head (or linear if mlp wasn't requested) at the first seed.
            "p_at_L":   ..., "p_at_L_2":   ..., "p_at_L_5":   ...,
            "n_proteins_test": ...,
        }
    """
    ca = batch["coords"][:, :, 1, :].cpu()  # [B, N, 3]
    m = batch["mask"].cpu()  # [B, N]
    lengths = batch["lengths"].cpu()  # [B]

    out: Dict = {}
    for head in heads:
        multi = linear_probe_contacts_multi(
            reps, ca, m, lengths, seeds=list(seeds), head_type=head
        )
        out[head] = {
            "p_at_L_mean": multi.p_at_L_mean,
            "p_at_L_std": multi.p_at_L_std,
            "p_at_L_2_mean": multi.p_at_L_2_mean,
            "p_at_L_2_std": multi.p_at_L_2_std,
            "p_at_L_5_mean": multi.p_at_L_5_mean,
            "p_at_L_5_std": multi.p_at_L_5_std,
            "n_proteins_test": multi.n_proteins_test,
            "seeds": multi.seeds,
            "p_at_L_5_per_seed": multi.p_at_L_5_per_seed,
        }

    # Back-compat flat fields. Prefer MLP if available (historical default),
    # otherwise fall back to the first requested head.
    preferred = "mlp" if "mlp" in out else next(iter(out))
    out["p_at_L"] = out[preferred]["p_at_L_mean"]
    out["p_at_L_2"] = out[preferred]["p_at_L_2_mean"]
    out["p_at_L_5"] = out[preferred]["p_at_L_5_mean"]
    out["n_proteins_test"] = out[preferred]["n_proteins_test"]
    return out


def evaluate_contact_from_scores(
    scores: torch.Tensor,  # [B, N, N] per-pair scores (higher = more likely contact)
    batch: Dict,
    test_frac: float = 0.2,
    seed: int = 42,
) -> ContactResult:
    """P@L/k evaluation for analytic scorers (random_rank, distance_only, ...).

    No training. Uses the same train/test split as ``linear_probe_contacts`` so
    the reported n_proteins_test matches the MLP probe — apples-to-apples.

    Inputs:
        scores:  [B, N, N] pair scores (arbitrary scale; only ranking matters)
        batch:   must contain coords, mask, lengths

    Flow:
        labels, pair_mask = contact_labels(ca, mask)        # [B, N, N] each
        test_idx          = 20% held out, deterministic seed
        for each test protein of length L:
            flatten upper-tri long-range pairs with scores[i, :L, :L]
            rank, take top-k, compute hits/k for k ∈ {L, L/2, L/5}
        average P@L/k across test proteins.

    Returns ContactResult (p_at_L, p_at_L_2, p_at_L_5, n_proteins_test).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ca = batch["coords"][:, :, 1, :].cpu()  # [B, N, 3] Å
    mask = batch["mask"].cpu()  # [B, N] bool
    lengths = batch["lengths"].cpu()  # [B]

    B, N, _ = ca.shape
    labels, pair_mask = contact_labels(ca, mask)  # [B, N, N] each

    # Match the MLP probe's deterministic 80/20 split so n_proteins_test
    # reported here lines up with the trained-model rows on the same batch.
    perm = torch.randperm(B)
    n_test = max(1, int(B * test_frac))
    test_idx = perm[:n_test].tolist()

    p_L, p_L2, p_L5 = [], [], []
    for i in test_idx:
        L = int(lengths[i].item())
        if L < 50:
            continue
        pm = pair_mask[i, :L, :L]  # [L, L]
        lab = labels[i, :L, :L]  # [L, L]
        iu = torch.triu_indices(L, L, offset=1)
        keep = pm[iu[0], iu[1]]
        ii, jj = iu[0][keep], iu[1][keep]  # [P_test] each
        pair_scores = scores[i, :L, :L][ii, jj]  # [P_test]
        y = lab[ii, jj]
        if pair_scores.numel() == 0:
            continue
        order = torch.argsort(pair_scores, descending=True)
        for topk, target in [(L, p_L), (L // 2, p_L2), (L // 5, p_L5)]:
            k = max(1, min(topk, pair_scores.numel()))
            hits = y[order[:k]].sum().item()
            target.append(hits / k)

    return ContactResult(
        p_at_L=float(np.mean(p_L)) if p_L else 0.0,
        p_at_L_2=float(np.mean(p_L2)) if p_L2 else 0.0,
        p_at_L_5=float(np.mean(p_L5)) if p_L5 else 0.0,
        n_proteins_test=len(p_L5),
    )
