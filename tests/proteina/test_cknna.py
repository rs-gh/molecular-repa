"""Unit tests for the CKNNA implementation in evaluation/proteina/alignment/lib/cknna.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ALIGN_ROOT = (
    Path(__file__).resolve().parents[2] / "evaluation" / "proteina" / "alignment"
)
if str(ALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(ALIGN_ROOT))

from lib.cknna import cknna, cknna_bootstrap  # noqa: E402


def test_self_cknna_is_one():
    torch.manual_seed(0)
    phi = torch.randn(200, 64)
    val = cknna(phi, phi, k=10)
    assert val == pytest.approx(1.0, abs=1e-5), f"CKNNA(x, x) should be 1.0, got {val}"


def test_cknna_invariant_to_orthogonal_rotation():
    """CKNNA is kernel-based — rotating one feature space leaves it invariant."""
    torch.manual_seed(1)
    phi = torch.randn(200, 64)
    Q, _ = torch.linalg.qr(torch.randn(64, 64))
    psi = phi @ Q
    val = cknna(phi, psi, k=10)
    assert val == pytest.approx(1.0, abs=1e-4)


def test_cknna_independent_features_is_low():
    """Two independent random feature sets should have ~zero CKNNA."""
    torch.manual_seed(2)
    phi = torch.randn(500, 64)
    psi = torch.randn(500, 64)
    val = cknna(phi, psi, k=10)
    assert (
        abs(val) < 0.05
    ), f"CKNNA between independent randoms should be near 0; got {val}"


def test_cknna_row_permutation_breaks_alignment():
    """Permuting rows of one matrix breaks the sample correspondence."""
    torch.manual_seed(3)
    phi = torch.randn(300, 64)
    perm = torch.randperm(300)
    val = cknna(phi, phi[perm], k=10)
    # Permuted should be far from 1.0 (chance correspondence only)
    assert val < 0.2


def test_cknna_handles_different_dims():
    """phi and psi can have different D — only N must match."""
    torch.manual_seed(4)
    phi = torch.randn(200, 64)
    psi = torch.randn(200, 128)
    val = cknna(phi, psi, k=10)
    # Independent randoms → near 0
    assert abs(val) < 0.1


def test_cknna_mismatched_N_raises():
    phi = torch.randn(100, 32)
    psi = torch.randn(200, 32)
    with pytest.raises(ValueError, match="same N"):
        cknna(phi, psi)


def test_cknna_too_few_samples_raises():
    phi = torch.randn(5, 32)
    psi = torch.randn(5, 32)
    with pytest.raises(ValueError, match="N > k"):
        cknna(phi, psi, k=10)


def test_cknna_signal_above_noise():
    """When psi shares signal with phi vs random, CKNNA should be much higher."""
    torch.manual_seed(5)
    phi = torch.randn(500, 64)
    # psi = phi + noise (shared signal)
    psi_signal = phi + 0.5 * torch.randn(500, 64)
    psi_noise = torch.randn(500, 64)

    v_signal = cknna(phi, psi_signal, k=10)
    v_noise = cknna(phi, psi_noise, k=10)
    assert v_signal > 0.3, f"shared-signal CKNNA too low: {v_signal}"
    assert v_noise < 0.1, f"noise CKNNA too high: {v_noise}"
    assert v_signal > v_noise + 0.3


def test_bootstrap_returns_expected_keys():
    torch.manual_seed(6)
    phi = torch.randn(200, 64)
    psi = phi + 0.3 * torch.randn(200, 64)
    out = cknna_bootstrap(phi, psi, k=10, n_boot=10, seed=0)
    assert {"point", "lo5", "hi95", "median"} <= set(out.keys())
    assert out["lo5"] <= out["median"] <= out["hi95"]
    assert 0.0 < out["point"] < 1.01
