"""CKNNA (Centered Kernel Nearest-Neighbor Alignment) — Huh et al. 2024.

Used by the REPA paper (Yu et al. 2024 Fig 2b/3b/7/8) as the canonical
representation-alignment metric between two feature spaces evaluated on the
same N samples.

This is a faithful port of the reference implementation in
``github.com/minyoungg/platonic-rep/metrics.py`` (Huh et al., 2024) — the
"unbiased" variant, which zeros out kernel diagonals and uses Song et al.'s
unbiased HSIC. Restricting the masked HSIC to the mutual-kNN intersection of
the two kernels makes the metric concentrate on agreement of *local*
neighbourhoods, mitigating sensitivity to global geometry mismatches that
plagues vanilla CKA.

Sanity reference: CKNNA(x, x) = 1; CKNNA(x, perm(x)) ≪ 1; CKNNA(x, random) ≈ 0.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _hsic_unbiased(K: Tensor, L: Tensor) -> Tensor:
    """Unbiased HSIC (Song et al. 2012, Eq. 5).

    Zeros the diagonals of K, L, then computes the unbiased estimator. Matches
    `hsic_unbiased` in platonic-rep/metrics.py exactly.
    """
    m = K.shape[0]
    K_tilde = K.clone()
    K_tilde.fill_diagonal_(0.0)
    L_tilde = L.clone()
    L_tilde.fill_diagonal_(0.0)
    val = (
        (K_tilde * L_tilde.T).sum()
        + (K_tilde.sum() * L_tilde.sum()) / ((m - 1) * (m - 2))
        - 2.0 * (K_tilde @ L_tilde).sum() / (m - 2)
    )
    return val / (m * (m - 3))


def _similarity(K: Tensor, L: Tensor, topk: int) -> Tensor:
    """Masked unbiased HSIC restricted to the mutual k-NN intersection."""
    n = K.shape[0]
    device = K.device
    K_hat = K.clone()
    K_hat.fill_diagonal_(float("-inf"))
    L_hat = L.clone()
    L_hat.fill_diagonal_(float("-inf"))
    _, idx_K = K_hat.topk(k=topk, dim=1)
    _, idx_L = L_hat.topk(k=topk, dim=1)
    mask_K = torch.zeros(n, n, device=device, dtype=K.dtype).scatter_(1, idx_K, 1.0)
    mask_L = torch.zeros(n, n, device=device, dtype=L.dtype).scatter_(1, idx_L, 1.0)
    mask = mask_K * mask_L
    return _hsic_unbiased(mask * K, mask * L)


def cknna(phi: Tensor, psi: Tensor, k: int = 10) -> float:
    """Compute CKNNA between two feature matrices on the same N samples.

    Args:
        phi: (N, D_a) features from model A.
        psi: (N, D_b) features from model B (same N, possibly different D).
        k:   k for the mutual-kNN restriction (paper default 10).

    Returns:
        Scalar float in roughly [0, 1]. CKNNA(x, x) = 1.

    Notes:
        - All N×N matrices live on whichever device phi/psi are on. For
          N ≈ 10k that is ~400 MB per matrix in fp32, fine on a single GPU.
        - Computed in float32 for numerical stability.
    """
    if phi.shape[0] != psi.shape[0]:
        raise ValueError(
            f"phi and psi must have same N; got {phi.shape[0]} vs {psi.shape[0]}"
        )
    if phi.shape[0] <= k + 1:
        # +1 because diagonal is masked out before the topk
        raise ValueError(f"Need N > k + 1; got N={phi.shape[0]}, k={k}")

    phi = phi.float()
    psi = psi.float()

    K = phi @ phi.T
    L = psi @ psi.T

    sim_kl = _similarity(K, L, k)
    sim_kk = _similarity(K, K, k)
    sim_ll = _similarity(L, L, k)

    denom = (sim_kk * sim_ll).clamp_min(1e-30).sqrt()
    return float((sim_kl / (denom + 1e-6)).item())


def cknna_bootstrap(
    phi: Tensor,
    psi: Tensor,
    k: int = 10,
    n_boot: int = 50,
    n_sample: int | None = None,
    seed: int = 0,
) -> dict:
    """CKNNA with subsample-based confidence interval over residues.

    **Subsampling is without replacement.** Bootstrap-with-replacement
    catastrophically inflates CKNNA: when row i is sampled twice, the two
    copies are each other's perfect nearest neighbour in BOTH kernels
    (similarity = ||x||² in K and L). Those duplicate pairs slip into the
    mutual-kNN intersection ``alpha`` and dominate the masked HSIC, so the
    bootstrap distribution drifts far above the full-data point estimate.

    Without replacement at n_sample = 0.8·N is a standard kernel-method
    fix: each draw is a valid CKNNA on a smaller-but-distinct subset, and
    the spread across draws gives a defensible 5–95% interval.

    Args:
        phi, psi: as in ``cknna``. Same N samples in matching order.
        k:        kNN neighborhood size.
        n_boot:   number of subsample draws.
        n_sample: rows per draw; default = max(k+2, floor(0.8 * N)).
        seed:     RNG seed for reproducibility.

    Returns:
        {"point": float, "lo5": float, "hi95": float, "median": float}
        ``point`` is the CKNNA on the full N. ``lo5``/``hi95`` are the
        5th and 95th percentiles of the subsample distribution.
    """
    N = phi.shape[0]
    if n_sample is None:
        n_sample = max(k + 2, int(0.8 * N))
    if n_sample > N:
        raise ValueError(
            f"n_sample={n_sample} > N={N}; cannot subsample without replacement."
        )

    point = cknna(phi, psi, k=k)

    g = torch.Generator(device="cpu").manual_seed(seed)
    boots = []
    for _ in range(n_boot):
        idx = torch.randperm(N, generator=g)[:n_sample]
        boots.append(cknna(phi[idx], psi[idx], k=k))
    boots_t = torch.tensor(boots)
    return {
        "point": point,
        "lo5": float(boots_t.quantile(0.05).item()),
        "hi95": float(boots_t.quantile(0.95).item()),
        "median": float(boots_t.median().item()),
        "n_sample": int(n_sample),
    }
