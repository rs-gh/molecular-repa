"""P4 — Backbone dihedral regression probe.

Per-residue (φ, ψ) prediction from a frozen per-residue representation
`h_i ∈ R^D`. The most direct test of whether the diffusion model's hidden
states encode the local backbone geometry the model is trained to denoise.

Targets are parameterised as (sin φ, cos φ, sin ψ, cos ψ) so MSE handles
the angular wrap-around cleanly. Reported metric is mean *circular* angular
error in degrees (computed via atan2 of the renormalised (sin, cos) pair).

Citation: Jamasb et al., ICLR 2024 (ProteinWorkshop §D.7 — adds per-residue
backbone dihedral prediction beyond GearNet's quadruplet variant).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import torch
import torch.nn.functional as F

from lib.labels import dihedral_labels
from lib.probes.contact import _build_head


@dataclass
class DihedralResult:
    mae_phi_deg: float
    mae_psi_deg: float
    mae_total_deg: float
    n_residues_test: int  # # of residues contributing to mae_total_deg


def _flatten_with_targets(
    reps: torch.Tensor,  # [B, N, D]
    coords37: torch.Tensor,  # [B, N, 37, 3]
    mask: torch.Tensor,  # [B, N]
):
    """Return (h, sincos, valid_phi, valid_psi) as flat per-residue tensors."""
    sincos, valid = dihedral_labels(coords37, mask)  # [B, N, 4], [B, N, 2]
    real = mask.bool()
    # Keep every real residue; per-angle validity is handled in the loss.
    flat_real = real.flatten()
    h = reps.reshape(-1, reps.shape[-1])[flat_real]
    sc = sincos.reshape(-1, 4)[flat_real]
    v_phi = valid[..., 0].flatten()[flat_real]
    v_psi = valid[..., 1].flatten()[flat_real]
    return h, sc, v_phi, v_psi


def train_dihedral_probe(
    reps: torch.Tensor,
    coords37: torch.Tensor,
    mask: torch.Tensor,
    head_type: Literal["mlp", "linear"] = "linear",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 4096,
    seed: int = 42,
    device: str = None,
) -> torch.nn.Module:
    """Fit a 4-output regression head on per-residue features.

    Loss is masked MSE on (sin, cos) of φ and ψ separately — invalid positions
    (chain ends or padded neighbours) contribute zero gradient.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    h, sc, v_phi, v_psi = _flatten_with_targets(reps, coords37, mask)
    D = reps.shape[-1]
    head = _build_head(D, device, head_type=head_type, out_dim=4)
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
            t = sc[idx].to(device)
            mphi = v_phi[idx].to(device).float().unsqueeze(-1)  # [b, 1]
            mpsi = v_psi[idx].to(device).float().unsqueeze(-1)
            pred = head(f)  # [b, 4]
            # Per-pair masked MSE on (sin, cos).
            err_phi = ((pred[:, 0:2] - t[:, 0:2]) ** 2 * mphi).sum() / mphi.sum().clamp(
                min=1.0
            )
            err_psi = ((pred[:, 2:4] - t[:, 2:4]) ** 2 * mpsi).sum() / mpsi.sum().clamp(
                min=1.0
            )
            loss = err_phi + err_psi
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    return head


@torch.no_grad()
def eval_dihedral_probe(
    head: torch.nn.Module,
    reps: torch.Tensor,
    coords37: torch.Tensor,
    mask: torch.Tensor,
    device: str = None,
) -> DihedralResult:
    """Mean circular angular error in degrees, separately for φ and ψ + total."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.to(device)
    head.eval()

    h, sc, v_phi, v_psi = _flatten_with_targets(reps, coords37, mask)
    P = h.shape[0]
    if P == 0:
        return DihedralResult(
            mae_phi_deg=0.0, mae_psi_deg=0.0, mae_total_deg=0.0, n_residues_test=0
        )

    preds = []
    for start in range(0, P, 8192):
        f = h[start : start + 8192].to(device)
        preds.append(head(f).cpu())
    pred = torch.cat(preds, dim=0)  # [P, 4]

    def _angular_err(
        pred_sc: torch.Tensor, true_sc: torch.Tensor, valid: torch.Tensor
    ) -> float:
        # pred_sc, true_sc: [P, 2] columns (sin, cos); valid: [P] bool
        pn = F.normalize(pred_sc, dim=-1, eps=1e-8)
        tn = F.normalize(true_sc, dim=-1, eps=1e-8)
        # Circular distance: |angle(pred) - angle(true)| via dot product clamped.
        cos_diff = (pn * tn).sum(dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos_diff)  # [P], radians, ∈ [0, π]
        if valid.sum().item() == 0:
            return 0.0
        return float((ang[valid].mean() * 180.0 / math.pi).item())

    mae_phi = _angular_err(pred[:, 0:2], sc[:, 0:2], v_phi)
    mae_psi = _angular_err(pred[:, 2:4], sc[:, 2:4], v_psi)

    # Total: mean over the union of valid positions, weighting each angle once.
    n_phi = int(v_phi.sum().item())
    n_psi = int(v_psi.sum().item())
    if n_phi + n_psi > 0:
        mae_total = (mae_phi * n_phi + mae_psi * n_psi) / (n_phi + n_psi)
    else:
        mae_total = 0.0

    return DihedralResult(
        mae_phi_deg=mae_phi,
        mae_psi_deg=mae_psi,
        mae_total_deg=float(mae_total),
        n_residues_test=int(P),
    )


def run_pretrained_dihedral_probe(
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
    coords_train = train_batch["coords"].cpu()
    m_train = train_batch["mask"].cpu()
    coords_eval = eval_batch["coords"].cpu()
    m_eval = eval_batch["mask"].cpu()

    head = train_dihedral_probe(
        reps_train,
        coords_train,
        m_train,
        head_type=head_type,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )
    res = eval_dihedral_probe(head, reps_eval, coords_eval, m_eval, device=device)
    return {
        "dih_mae_phi_deg": res.mae_phi_deg,
        "dih_mae_psi_deg": res.mae_psi_deg,
        "dih_mae_total_deg": res.mae_total_deg,
        "dih_n_residues_test": res.n_residues_test,
    }
