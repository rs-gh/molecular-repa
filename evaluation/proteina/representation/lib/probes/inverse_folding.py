"""P3 — Inverse-folding probe.

Per-residue 20-way amino-acid classification from a frozen per-residue
representation `h_i ∈ R^D`. The REPA-style residue-level analogue of the
ImageNet-class linear probe: if the diffusion model's hidden states encode
the local chemical environment that determines the amino acid, a linear
head should recover it.

Citations:
    Ingraham et al., NeurIPS 2019  (formulation; CATH-topology splits)
    Jamasb et al., ICLR 2024       (ProteinWorkshop §D.8 — downstream task)

Pipeline B only (train on a large train.lmdb sample, evaluate on a fixed
val.lmdb sample). Mirrors the contact_pretrained module's split into
``train_*`` / ``eval_*`` / ``run_pretrained_*`` for harness consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.nn.functional as F

from lib.probes.contact import _build_head

# OpenFold AA index range; padded positions can hold garbage indices and must
# be filtered via ``mask`` before constructing CE targets.
N_AA = 20


@dataclass
class InverseFoldingResult:
    top1_acc: float
    macro_f1: float
    n_residues_test: int
    per_class_acc: List[float]


def _flatten_valid(
    reps: torch.Tensor,  # [B, N, D]
    residue_type: torch.Tensor,  # [B, N] long
    mask: torch.Tensor,  # [B, N] bool
):
    """Gather `(reps, labels)` over real residues, dropping pad / OOV."""
    real = mask.bool() & (residue_type >= 0) & (residue_type < N_AA)
    h = reps[real]  # [P, D]
    y = residue_type[real].long()  # [P]
    return h, y


def train_inverse_folding_probe(
    reps: torch.Tensor,  # [B, N, D] CPU
    residue_type: torch.Tensor,  # [B, N] CPU long
    mask: torch.Tensor,  # [B, N] CPU bool
    head_type: Literal["mlp", "linear"] = "linear",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 4096,
    seed: int = 42,
    device: str = None,
) -> torch.nn.Module:
    """Fit an AA classifier head on flattened per-residue features.

    Flow:
        h, y     = flatten over real residues               [P, D], [P]
        head     = _build_head(D, device, head_type, out_dim=20)
        for epoch in range(epochs):
            shuffle indices, iterate minibatches of `batch_size`
            CE step
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    h, y = _flatten_valid(reps, residue_type, mask)
    D = reps.shape[-1]
    head = _build_head(D, device, head_type=head_type, out_dim=N_AA)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    P = h.shape[0]
    if P == 0:
        head.eval()
        return head

    head.train()
    for _epoch in range(epochs):
        perm = torch.randperm(P)
        for start in range(0, P, batch_size):
            idx = perm[start : start + batch_size]
            f = h[idx].to(device)
            t = y[idx].to(device)
            logits = head(f)
            loss = F.cross_entropy(logits, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    return head


@torch.no_grad()
def eval_inverse_folding_probe(
    head: torch.nn.Module,
    reps: torch.Tensor,
    residue_type: torch.Tensor,
    mask: torch.Tensor,
    device: str = None,
) -> InverseFoldingResult:
    """Top-1 accuracy + macro-F1 over real eval residues."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.to(device)
    head.eval()

    h, y = _flatten_valid(reps, residue_type, mask)
    P = h.shape[0]
    if P == 0:
        return InverseFoldingResult(
            top1_acc=0.0, macro_f1=0.0, n_residues_test=0, per_class_acc=[0.0] * N_AA
        )

    # Forward in chunks to avoid blowing GPU memory on long batches.
    preds_chunks = []
    for start in range(0, P, 8192):
        f = h[start : start + 8192].to(device)
        preds_chunks.append(head(f).argmax(dim=-1).cpu())
    preds = torch.cat(preds_chunks, dim=0)  # [P]
    y_cpu = y.cpu()

    correct = (preds == y_cpu).float()
    top1 = float(correct.mean().item())

    # Per-class accuracy + macro-F1 (computed manually to avoid sklearn dep).
    per_class_acc: List[float] = []
    f1s: List[float] = []
    for c in range(N_AA):
        true_c = y_cpu == c
        pred_c = preds == c
        tp = float((true_c & pred_c).sum().item())
        fp = float((~true_c & pred_c).sum().item())
        fn = float((true_c & ~pred_c).sum().item())
        n_c = float(true_c.sum().item())
        per_class_acc.append((tp / n_c) if n_c > 0 else float("nan"))
        denom_p = tp + fp
        denom_r = tp + fn
        prec = (tp / denom_p) if denom_p > 0 else 0.0
        rec = (tp / denom_r) if denom_r > 0 else 0.0
        if prec + rec > 0:
            f1s.append(2 * prec * rec / (prec + rec))
        elif n_c == 0:
            # Class absent from eval set — skip rather than penalise.
            continue
        else:
            f1s.append(0.0)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return InverseFoldingResult(
        top1_acc=top1,
        macro_f1=macro_f1,
        n_residues_test=int(P),
        per_class_acc=per_class_acc,
    )


def run_pretrained_inverse_folding_probe(
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
    """End-to-end: train the head on train features, evaluate on eval features.

    Output dict mirrors the JSONL row schema used by the harness so the caller
    can drop fields straight into a row.
    """
    rt_train = train_batch["residue_type"].cpu()
    m_train = train_batch["mask"].cpu()
    rt_eval = eval_batch["residue_type"].cpu()
    m_eval = eval_batch["mask"].cpu()

    head = train_inverse_folding_probe(
        reps_train,
        rt_train,
        m_train,
        head_type=head_type,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )
    res = eval_inverse_folding_probe(head, reps_eval, rt_eval, m_eval, device=device)
    return {
        "if_top1_acc": res.top1_acc,
        "if_macro_f1": res.macro_f1,
        "if_n_residues_test": res.n_residues_test,
        "if_per_class_acc": res.per_class_acc,
    }
