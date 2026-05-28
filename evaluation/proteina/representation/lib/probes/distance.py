"""P5 — Pair-distance regression probe.

Per-pair Cα–Cα distance prediction in Å from `[h_i ‖ h_j ‖ |h_i - h_j|]`.
A strict generalisation of the long-range contact probe: contact is the same
pair feature thresholded at 8 Å; here we regress the actual distance and
report MAE bucketed by sequence separation.

The baseline-via-feature-only short-circuit ("seqsep_pair") is implemented
here rather than as a `RepSource` because the input is a pair feature
(``|i-j|``), not a per-residue tensor — see ``train_distance_probe_seqsep``.

Citations:
    Zhang et al., ICLR 2023  (GearNet §3.2 — masked distance prediction SSL)
    Jamasb et al., ICLR 2024 (ProteinWorkshop §D.7 — adopted as benchmark)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lib.probes.contact import _build_head

# Sequence-separation bucket boundaries: short / medium / long.
SEP_SHORT_MAX = 6  # |i-j| < 6
SEP_MEDIUM_MAX = 24  # 6 <= |i-j| < 24; long is everything ≥ 24


@dataclass
class DistanceResult:
    mae_total: float
    mae_short: float
    mae_medium: float
    mae_long: float
    n_pairs_test: int


def _stratified_pairs(
    h: torch.Tensor,  # [N, D]
    ca: torch.Tensor,  # [N, 3]
    real: torch.Tensor,  # [N] bool
    n_per_bucket: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """Sample up to ``n_per_bucket`` pairs from each |i-j| bucket on one protein.

    Returns (feats [P, 3D], dist [P]) or None if the protein is too short / all-padded.
    """
    N = h.shape[0]
    if int(real.sum().item()) < 2:
        return None
    idx = torch.arange(N)
    seqsep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [N, N]
    # Upper triangle only (i < j) — pair is symmetric.
    iu = torch.triu_indices(N, N, offset=1)
    sep = seqsep[iu[0], iu[1]]
    valid = real[iu[0]] & real[iu[1]]
    sub_iu = iu[:, valid]
    sub_sep = sep[valid]
    if sub_iu.shape[1] == 0:
        return None

    short_mask = sub_sep < SEP_SHORT_MAX
    med_mask = (sub_sep >= SEP_SHORT_MAX) & (sub_sep < SEP_MEDIUM_MAX)
    long_mask = sub_sep >= SEP_MEDIUM_MAX
    chosen = []
    for bm in (short_mask, med_mask, long_mask):
        cands = sub_iu[:, bm]
        K = cands.shape[1]
        if K == 0:
            continue
        take = min(n_per_bucket, K)
        sel = torch.randperm(K)[:take]
        chosen.append(cands[:, sel])
    if not chosen:
        return None
    pairs = torch.cat(chosen, dim=1)  # [2, P]
    hi = h[pairs[0]]
    hj = h[pairs[1]]
    feats = torch.cat([hi, hj, (hi - hj).abs()], dim=-1)  # [P, 3D]
    dist = (ca[pairs[0]] - ca[pairs[1]]).norm(dim=-1)  # [P]
    return feats, dist


def train_distance_probe(
    reps: torch.Tensor,  # [B, N, D]
    ca_coords_ang: torch.Tensor,  # [B, N, 3]
    mask: torch.Tensor,  # [B, N]
    head_type: Literal["mlp", "linear"] = "linear",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 4,
    n_per_bucket: int = 100,
    seed: int = 42,
    device: str = None,
) -> torch.nn.Module:
    """Fit a 1-output Huber regression head on stratified pair features.

    ``n_per_bucket=100`` (lowered 2026-05-14 from 200): per-protein pair budget
    of 300 (100 × 3 buckets). At n_train=5000 / 8 epochs that's ~12M training
    pair-forwards — saturates the linear head while keeping per-layer wall
    time under ~50s and the full per-checkpoint sweep under the 15-min array
    slot. The eval pass still uses every valid upper-tri pair so the reported
    MAE is unbiased.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    B, N, D = reps.shape
    real = mask.bool()
    head = _build_head(3 * D, device, head_type=head_type, out_dim=1)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    train_idx = list(range(B))
    head.train()
    for _epoch in range(epochs):
        np.random.shuffle(train_idx)
        for b_start in range(0, len(train_idx), batch_size):
            sub = train_idx[b_start : b_start + batch_size]
            feats_all, dist_all = [], []
            for i in sub:
                out = _stratified_pairs(
                    reps[i], ca_coords_ang[i], real[i], n_per_bucket=n_per_bucket
                )
                if out is None:
                    continue
                f, d = out
                feats_all.append(f)
                dist_all.append(d)
            if not feats_all:
                continue
            f = torch.cat(feats_all, dim=0).to(device)
            t = torch.cat(dist_all, dim=0).to(device)
            pred = head(f).squeeze(-1)
            loss = F.huber_loss(pred, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    return head


@torch.no_grad()
def eval_distance_probe(
    head: torch.nn.Module,
    reps: torch.Tensor,
    ca_coords_ang: torch.Tensor,
    mask: torch.Tensor,
    device: str = None,
) -> DistanceResult:
    """MAE in Å, bucketed by |i-j| ∈ {<6, 6-24, ≥24}.

    Eval samples *all* valid upper-triangle pairs per protein (no cap), so
    the long bucket dominates by count; we report bucket means separately.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.to(device)
    head.eval()

    B, N, D = reps.shape
    real = mask.bool()
    short_errs: List[float] = []
    med_errs: List[float] = []
    long_errs: List[float] = []

    idx = torch.arange(N)
    seqsep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    iu = torch.triu_indices(N, N, offset=1)
    sep_all = seqsep[iu[0], iu[1]]

    for i in range(B):
        ri = real[i]
        valid = ri[iu[0]] & ri[iu[1]]
        sub_iu = iu[:, valid]
        sub_sep = sep_all[valid]
        if sub_iu.shape[1] == 0:
            continue
        h = reps[i]
        hi = h[sub_iu[0]]
        hj = h[sub_iu[1]]
        feats = torch.cat([hi, hj, (hi - hj).abs()], dim=-1).to(device)
        # Forward in chunks to bound memory.
        preds = []
        for start in range(0, feats.shape[0], 16384):
            preds.append(head(feats[start : start + 16384]).squeeze(-1).cpu())
        pred = torch.cat(preds, dim=0)
        true = (ca_coords_ang[i, sub_iu[0]] - ca_coords_ang[i, sub_iu[1]]).norm(dim=-1)
        ae = (pred - true).abs()
        short_errs.extend(ae[sub_sep < SEP_SHORT_MAX].tolist())
        med_errs.extend(
            ae[(sub_sep >= SEP_SHORT_MAX) & (sub_sep < SEP_MEDIUM_MAX)].tolist()
        )
        long_errs.extend(ae[sub_sep >= SEP_MEDIUM_MAX].tolist())

    n_total = len(short_errs) + len(med_errs) + len(long_errs)
    if n_total == 0:
        return DistanceResult(0.0, 0.0, 0.0, 0.0, 0)
    mae_short = float(np.mean(short_errs)) if short_errs else 0.0
    mae_med = float(np.mean(med_errs)) if med_errs else 0.0
    mae_long = float(np.mean(long_errs)) if long_errs else 0.0
    mae_total = float(np.sum(short_errs + med_errs + long_errs) / n_total)
    return DistanceResult(
        mae_total=mae_total,
        mae_short=mae_short,
        mae_medium=mae_med,
        mae_long=mae_long,
        n_pairs_test=int(n_total),
    )


# --------------------------------------------------------------------------- #
# Trivial seqsep-only baseline: head fed |i-j| as the only feature.
# --------------------------------------------------------------------------- #


def train_distance_probe_seqsep(
    ca_coords_ang: torch.Tensor,  # [B, N, 3]
    mask: torch.Tensor,  # [B, N]
    head_type: Literal["mlp", "linear"] = "linear",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 4,
    n_per_bucket: int = 200,
    seed: int = 42,
    device: str = None,
) -> torch.nn.Module:
    """Baseline head fed only ``|i - j|`` (1-d feature).

    Tests how much of the distance MAE win is just the chain-prior. Padded as
    a real `nn.Module` so the eval path is identical (head(feature) → scalar).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = _build_head(1, device, head_type=head_type, out_dim=1)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    B, N, _ = ca_coords_ang.shape
    real = mask.bool()
    idx_t = torch.arange(N)

    train_idx = list(range(B))
    head.train()
    for _epoch in range(epochs):
        np.random.shuffle(train_idx)
        for b_start in range(0, len(train_idx), batch_size):
            sub = train_idx[b_start : b_start + batch_size]
            seps_all, dist_all = [], []
            for i in sub:
                ri = real[i]
                if int(ri.sum().item()) < 2:
                    continue
                iu = torch.triu_indices(N, N, offset=1)
                v = ri[iu[0]] & ri[iu[1]]
                sub_iu = iu[:, v]
                sub_sep = (idx_t[sub_iu[0]] - idx_t[sub_iu[1]]).abs()
                # Stratified subsample per bucket.
                short_mask = sub_sep < SEP_SHORT_MAX
                med_mask = (sub_sep >= SEP_SHORT_MAX) & (sub_sep < SEP_MEDIUM_MAX)
                long_mask = sub_sep >= SEP_MEDIUM_MAX
                chosen_idx = []
                for bm in (short_mask, med_mask, long_mask):
                    where = bm.nonzero(as_tuple=False).squeeze(-1)
                    if where.numel() == 0:
                        continue
                    take = min(n_per_bucket, where.numel())
                    chosen_idx.append(where[torch.randperm(where.numel())[:take]])
                if not chosen_idx:
                    continue
                sel = torch.cat(chosen_idx, dim=0)
                pair_sep = sub_sep[sel].float()
                pair_ij = sub_iu[:, sel]
                dist = (
                    ca_coords_ang[i, pair_ij[0]] - ca_coords_ang[i, pair_ij[1]]
                ).norm(dim=-1)
                seps_all.append(pair_sep)
                dist_all.append(dist)
            if not seps_all:
                continue
            f = torch.cat(seps_all, dim=0).unsqueeze(-1).to(device)  # [P, 1]
            t = torch.cat(dist_all, dim=0).to(device)
            pred = head(f).squeeze(-1)
            loss = F.huber_loss(pred, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    return head


@torch.no_grad()
def eval_distance_probe_seqsep(
    head: torch.nn.Module,
    ca_coords_ang: torch.Tensor,
    mask: torch.Tensor,
    device: str = None,
) -> DistanceResult:
    """Same eval interface as ``eval_distance_probe`` but the head reads
    ``|i-j|`` only — the per-residue ``reps`` are not consulted at all."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.to(device)
    head.eval()

    B, N, _ = ca_coords_ang.shape
    real = mask.bool()
    idx = torch.arange(N)
    iu = torch.triu_indices(N, N, offset=1)
    sep_all = (idx[iu[0]] - idx[iu[1]]).abs()

    short_errs: List[float] = []
    med_errs: List[float] = []
    long_errs: List[float] = []
    for i in range(B):
        ri = real[i]
        v = ri[iu[0]] & ri[iu[1]]
        sub_iu = iu[:, v]
        sub_sep = sep_all[v].float()
        if sub_iu.shape[1] == 0:
            continue
        f = sub_sep.unsqueeze(-1).to(device)
        preds = []
        for start in range(0, f.shape[0], 16384):
            preds.append(head(f[start : start + 16384]).squeeze(-1).cpu())
        pred = torch.cat(preds, dim=0)
        true = (ca_coords_ang[i, sub_iu[0]] - ca_coords_ang[i, sub_iu[1]]).norm(dim=-1)
        ae = (pred - true).abs()
        short_errs.extend(ae[sub_sep < SEP_SHORT_MAX].tolist())
        med_errs.extend(
            ae[(sub_sep >= SEP_SHORT_MAX) & (sub_sep < SEP_MEDIUM_MAX)].tolist()
        )
        long_errs.extend(ae[sub_sep >= SEP_MEDIUM_MAX].tolist())

    n_total = len(short_errs) + len(med_errs) + len(long_errs)
    if n_total == 0:
        return DistanceResult(0.0, 0.0, 0.0, 0.0, 0)
    return DistanceResult(
        mae_total=float(np.sum(short_errs + med_errs + long_errs) / n_total),
        mae_short=float(np.mean(short_errs)) if short_errs else 0.0,
        mae_medium=float(np.mean(med_errs)) if med_errs else 0.0,
        mae_long=float(np.mean(long_errs)) if long_errs else 0.0,
        n_pairs_test=int(n_total),
    )


def run_pretrained_distance_probe(
    reps_train: torch.Tensor,
    train_batch: Dict,
    reps_eval: torch.Tensor,
    eval_batch: Dict,
    head_type: Literal["mlp", "linear"] = "linear",
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = None,
) -> Dict:
    ca_train = train_batch["coords"][:, :, 1, :].cpu()
    m_train = train_batch["mask"].cpu()
    ca_eval = eval_batch["coords"][:, :, 1, :].cpu()
    m_eval = eval_batch["mask"].cpu()

    head = train_distance_probe(
        reps_train,
        ca_train,
        m_train,
        head_type=head_type,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )
    res = eval_distance_probe(head, reps_eval, ca_eval, m_eval, device=device)
    return {
        "dist_mae_total": res.mae_total,
        "dist_mae_short": res.mae_short,
        "dist_mae_medium": res.mae_medium,
        "dist_mae_long": res.mae_long,
        "dist_n_pairs_test": res.n_pairs_test,
    }
