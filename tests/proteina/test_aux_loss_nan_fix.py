"""Tests that the auxiliary-loss NaN fix in `Proteina.compute_auxiliary_loss`
preserves numerical equivalence on non-pathological inputs and eliminates
the NaN source (extreme `pair_pred` at padded positions under bf16) on
pathological ones.

Context: with length-bucketed training, N is padded to the bucket upper edge
(up to 512), and the pre-fix code computed CE over all bs*n*n pair positions
— including padded ones — then masked the result. Under bf16 a single
extreme logit at a padded position could NaN log_softmax; autograd's
`0 * NaN = NaN` then corrupted the gradient. See chat thread 2026-04-21.

This test avoids importing the full `Proteina` Lightning module (which drags
in torch_scatter and fails CXXABI on our login node) by reproducing both the
pre-fix and post-fix algorithms as pure functions.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure-function mirrors of the two code paths
# ---------------------------------------------------------------------------
def _aux_reference(
    x_1,
    x_1_pred,
    mask,
    pair_pred,
    thres_aux_2d_loss,
    num_dist_buckets,
    max_dist_boundary,
):
    """Pre-fix code path — verbatim from proteina.py before the NaN fix."""
    bs, n, _ = x_1.shape
    nres = mask.sum(-1)

    gt_ca = x_1 * mask[..., None]
    pred_ca = x_1_pred * mask[..., None]
    pair_mask = mask[..., None, :] * mask[..., None]

    gt_pd = torch.linalg.norm(gt_ca[:, :, None] - gt_ca[:, None], dim=-1) * pair_mask
    pred_pd = (
        torch.linalg.norm(pred_ca[:, :, None] - pred_ca[:, None], dim=-1) * pair_mask
    )

    max_dist = thres_aux_2d_loss if thres_aux_2d_loss is not None else 1e10
    total_pair_mask = pair_mask * (gt_pd < max_dist)

    den = total_pair_mask.sum(dim=(-1, -2)) - nres
    dist_mat = torch.sum((gt_pd - pred_pd) ** 2 * total_pair_mask, dim=(-1, -2))
    dist_mat = dist_mat / den  # unsafe: 0/0 when no off-diagonal pair < thres

    boundaries = torch.linspace(
        0.0, max_dist_boundary, num_dist_buckets - 1, device=pair_pred.device
    )
    gt_bucket = torch.bucketize(gt_pd, boundaries)

    pair_pred_v = pair_pred.view(bs * n * n, num_dist_buckets)
    gt_bucket_v = gt_bucket.view(bs * n * n)
    distogram = F.cross_entropy(pair_pred_v, gt_bucket_v, reduction="none")
    distogram = distogram.view(bs, n, n)
    distogram = torch.sum(distogram * pair_mask, dim=(-1, -2))  # mask AFTER CE
    distogram = distogram / (pair_mask.sum(dim=(-1, -2)) + 1e-10)

    return dist_mat, distogram


def _aux_fixed(
    x_1,
    x_1_pred,
    mask,
    pair_pred,
    thres_aux_2d_loss,
    num_dist_buckets,
    max_dist_boundary,
):
    """Post-fix code path — mirror of the current proteina.py.

    Changes vs reference:
      - den.clamp(min=1.0) to avoid 0/0 in dist_mat
      - Gather real pairs BEFORE cross_entropy so padded logits never enter
        the computational graph (kills NaN source at padded positions)
    """
    bs, n, _ = x_1.shape
    nres = mask.sum(-1)

    gt_ca = x_1 * mask[..., None]
    pred_ca = x_1_pred * mask[..., None]
    pair_mask = mask[..., None, :] * mask[..., None]

    gt_pd = torch.linalg.norm(gt_ca[:, :, None] - gt_ca[:, None], dim=-1) * pair_mask
    pred_pd = (
        torch.linalg.norm(pred_ca[:, :, None] - pred_ca[:, None], dim=-1) * pair_mask
    )

    max_dist = thres_aux_2d_loss if thres_aux_2d_loss is not None else 1e10
    total_pair_mask = pair_mask * (gt_pd < max_dist)

    den = total_pair_mask.sum(dim=(-1, -2)) - nres
    dist_mat = torch.sum((gt_pd - pred_pd) ** 2 * total_pair_mask, dim=(-1, -2))
    dist_mat = dist_mat / den.clamp(min=1.0)  # fix #1: safe denominator

    boundaries = torch.linspace(
        0.0, max_dist_boundary, num_dist_buckets - 1, device=pair_pred.device
    )
    gt_bucket = torch.bucketize(gt_pd, boundaries)

    pair_pred_flat = pair_pred.view(bs * n * n, num_dist_buckets)
    gt_bucket_flat = gt_bucket.view(bs * n * n)
    pair_mask_flat = pair_mask.view(bs * n * n).bool()
    # fix #2: CE only on real pairs
    ce_real = F.cross_entropy(
        pair_pred_flat[pair_mask_flat],
        gt_bucket_flat[pair_mask_flat],
        reduction="none",
    )
    distogram_flat = pair_pred.new_zeros(bs * n * n)
    distogram_flat[pair_mask_flat] = ce_real
    distogram = distogram_flat.view(bs, n, n)
    distogram = torch.sum(distogram, dim=(-1, -2))
    distogram = distogram / (pair_mask.sum(dim=(-1, -2)) + 1e-10)

    return dist_mat, distogram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_batch(
    bs, n, nres_list, *, seed=0, extreme_pad_logits=False, dtype=torch.float32
):
    torch.manual_seed(seed)
    x_1 = torch.randn(bs, n, 3, dtype=dtype) * 0.3
    x_1_pred = x_1 + torch.randn_like(x_1) * 0.05
    mask = torch.zeros(bs, n, dtype=torch.bool)
    for b, k in enumerate(nres_list):
        mask[b, :k] = True
    num_buckets = 64
    pair_pred = torch.randn(bs, n, n, num_buckets, dtype=dtype) * 0.5
    if extreme_pad_logits:
        pad_pair = ~(
            mask[:, None, :] & mask[:, :, None]
        )  # [bs, n, n], True where padded
        # Inject NaN directly at padded positions — this models what bf16
        # overflow can produce inside log_softmax's max-stabilization when
        # inputs contain values near the exponent limit.
        pair_pred[pad_pair] = float("nan")
    return x_1, x_1_pred, mask.to(dtype), pair_pred


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("thres", [0.6, 1e10])
def test_equivalence_no_padding_well_behaved(thres):
    """No padding, well-behaved logits: both implementations agree."""
    bs, n = 2, 32
    x_1, x_1_pred, mask, pair_pred = _make_batch(bs, n, [n, n], seed=42)
    ref_dm, ref_dg = _aux_reference(x_1, x_1_pred, mask, pair_pred, thres, 64, 1.0)
    fix_dm, fix_dg = _aux_fixed(x_1, x_1_pred, mask, pair_pred, thres, 64, 1.0)
    assert torch.isfinite(ref_dm).all()
    assert torch.isfinite(ref_dg).all()
    torch.testing.assert_close(ref_dm, fix_dm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ref_dg, fix_dg, rtol=1e-5, atol=1e-6)


def test_reference_produces_nan_on_padded_nan_logits():
    """Sanity: inject NaN at padded positions → reference propagates NaN via
    `pair_mask * NaN = NaN`. Confirms the failure mode the fix addresses."""
    bs, n = 2, 64
    x_1, x_1_pred, mask, pair_pred = _make_batch(
        bs, n, [30, 50], seed=11, extreme_pad_logits=True
    )
    ref_dm, ref_dg = _aux_reference(x_1, x_1_pred, mask, pair_pred, 0.6, 64, 1.0)
    assert torch.isnan(ref_dg).any(), f"reference should NaN but got {ref_dg}"


def test_fix_survives_padded_nan_logits_forward():
    """With fix, NaN at padded positions never enters CE → finite output.
    Compares against the clean-logits result: should be identical because
    padded positions contribute nothing either way."""
    bs, n = 2, 64
    # Build once with clean logits, once with NaN padded logits — same seed
    x_1_c, x_1p_c, mask_c, pp_clean = _make_batch(
        bs, n, [30, 50], seed=11, extreme_pad_logits=False
    )
    x_1_p, x_1p_p, mask_p, pp_poison = _make_batch(
        bs, n, [30, 50], seed=11, extreme_pad_logits=True
    )
    # Sanity: real-pair logits should be identical between clean/poison.
    # Compare only at real-pair positions to avoid NaN*0=NaN artifacts.
    pm = (mask_c[:, None, :] * mask_c[:, :, None]).bool()  # [bs, n, n]
    idx = pm.view(-1)  # [bs*n*n]
    real_clean = pp_clean.view(-1, pp_clean.shape[-1])[idx]
    real_poison = pp_poison.view(-1, pp_poison.shape[-1])[idx]
    torch.testing.assert_close(real_clean, real_poison)

    fix_dm_c, fix_dg_c = _aux_fixed(x_1_c, x_1p_c, mask_c, pp_clean, 0.6, 64, 1.0)
    fix_dm_p, fix_dg_p = _aux_fixed(x_1_p, x_1p_p, mask_p, pp_poison, 0.6, 64, 1.0)

    assert torch.isfinite(fix_dg_p).all(), f"fix NaN'd on padded NaN logits: {fix_dg_p}"
    assert torch.isfinite(fix_dm_p).all()
    torch.testing.assert_close(fix_dg_c, fix_dg_p, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fix_dm_c, fix_dm_p, rtol=1e-5, atol=1e-6)


def test_fix_survives_padded_nan_logits_backward():
    """With fix, NaN at padded positions must not contaminate gradients.
    Reference backward gives NaN grads via autograd's `0 * NaN = NaN`;
    the gather-before-CE pattern keeps padded positions out of the graph."""
    bs, n = 2, 48
    x_1, x_1_pred, mask, pair_pred = _make_batch(
        bs, n, [20, 35], seed=5, extreme_pad_logits=True
    )
    pair_pred = pair_pred.clone().requires_grad_(True)
    _, fix_dg = _aux_fixed(x_1, x_1_pred, mask, pair_pred, 0.6, 64, 1.0)
    loss = fix_dg.mean()
    assert torch.isfinite(loss), f"forward NaN: {loss}"
    loss.backward()
    assert torch.isfinite(pair_pred.grad).all(), (
        f"backward produced NaN in pair_pred.grad at {torch.isnan(pair_pred.grad).sum()} "
        f"positions (likely from '0 * NaN' through masked positions)"
    )
    # Padded positions should have exactly zero gradient (they weren't in the graph)
    pad_pair = ~(mask[:, None, :].bool() & mask[:, :, None].bool())  # [bs, n, n]
    pad_grads = pair_pred.grad[pad_pair]
    assert (
        pad_grads == 0
    ).all(), (
        f"padded positions have non-zero gradient: max abs = {pad_grads.abs().max()}"
    )


def test_fix_den_clamp_prevents_divide_by_zero():
    """Sparse batch where no off-diagonal pair is under thres_aux_2d_loss.
    Reference returns NaN (0/0); fix returns 0 (numerator also 0)."""
    bs, n = 1, 8
    # Spread residues so all off-diagonal pair distances > 0.6
    x_1 = torch.zeros(bs, n, 3)
    for i in range(n):
        x_1[0, i] = torch.tensor([float(i) * 10.0, 0.0, 0.0])
    x_1_pred = x_1.clone()
    mask = torch.ones(bs, n)
    pair_pred = torch.randn(bs, n, n, 64) * 0.5

    ref_dm, _ = _aux_reference(x_1, x_1_pred, mask, pair_pred, 0.6, 64, 1.0)
    fix_dm, _ = _aux_fixed(x_1, x_1_pred, mask, pair_pred, 0.6, 64, 1.0)

    assert torch.isnan(ref_dm).any() or torch.isinf(ref_dm).any()
    assert torch.isfinite(fix_dm).all()
    assert (fix_dm == 0).all(), f"expected 0 on den=0 case, got {fix_dm}"
