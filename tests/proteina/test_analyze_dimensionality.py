"""Tests for the encoder-profiling spectral diagnostics.

Validates the RankMe (Garrido et al. 2023) and participation-ratio
(Gao et al. 2017) implementations in
``encoder_profiling/proteina/_probes/lib.py:analyze_dimensionality``.
"""

import os
import sys

import numpy as np
import torch

probes_root = os.path.join(
    os.path.dirname(__file__), "../../encoder_profiling/proteina"
)
sys.path.insert(0, probes_root)

from _probes.lib import analyze_dimensionality  # noqa: E402


def test_orthonormal_columns_give_full_rank():
    # Q = N x d with orthonormal columns -> singular values all equal -> RankMe = d
    torch.manual_seed(0)
    n, d = 200, 16
    A = torch.randn(n, d)
    Q, _ = torch.linalg.qr(A, mode="reduced")  # n x d, orthonormal cols

    out = analyze_dimensionality(Q)

    # All d singular values equal => entropy is log(d) => exp(H) = d
    assert abs(out["rankme"] - d) < 1e-3
    # Centered Q is no longer orthonormal but PR should still be close to d-1
    # (one direction lost to mean removal). Don't assert exact; just sanity.
    assert out["participation_ratio"] > d - 2


def test_rank_one_collapse():
    # rank-1 matrix: all rows are scalar multiples of one vector
    # -> single nonzero singular value -> RankMe = 1
    torch.manual_seed(0)
    u = torch.randn(500, 1)
    v = torch.randn(1, 64)
    X = u @ v  # rank-1

    out = analyze_dimensionality(X)

    # Roy-Vetterli erank of a rank-1 matrix is exactly 1
    assert abs(out["rankme"] - 1.0) < 1e-3
    assert out["participation_ratio"] < 1.5
    assert out["dims_for_99pct_var"] == 1


def test_low_rank_plus_noise():
    # 15 informative directions + noise. RankMe should sit near the true rank,
    # PR likewise; dims_for_99pct_var should recover the rank tightly.
    torch.manual_seed(0)
    true_rank = 15
    n, d = 1000, 128
    U = torch.randn(n, true_rank)
    V = torch.randn(true_rank, d)
    X = U @ V + 0.05 * torch.randn(n, d)

    out = analyze_dimensionality(X)

    # RankMe is upper-bounded by rank for clean low-rank, but slight noise
    # leaks into the tail. Allow a generous band.
    assert true_rank <= out["rankme"] <= 2 * true_rank + 5
    assert true_rank <= out["dims_for_99pct_var"] <= true_rank + 5


def test_keys_match_schema():
    # Guards against accidental key drops in the JSON schema (collate.py and
    # external consumers depend on these names).
    torch.manual_seed(0)
    out = analyze_dimensionality(torch.randn(100, 32))
    expected = {
        "rankme",
        "participation_ratio",
        "dim_total",
        "dims_for_90pct_var",
        "dims_for_95pct_var",
        "dims_for_99pct_var",
        "top_singular_value",
        "condition_number",
        "singular_values_uncentered",
        "singular_values_centered",
        "cumulative_variance",
    }
    assert set(out.keys()) == expected
    # Old keys must be gone — anything reading them is broken and should fail loudly.
    assert "effective_rank" not in out
    assert "singular_values" not in out


def test_subsampling_above_max_residues():
    # Above max_residues, function still returns valid output and dim_total
    # tracks the full embedding dim (not the subsample).
    torch.manual_seed(0)
    big = torch.randn(35000, 64)
    out = analyze_dimensionality(big, max_residues=30000)
    assert out["dim_total"] == 64
    assert 1.0 < out["rankme"] <= 64
    assert len(out["singular_values_uncentered"]) == 64


def test_rankme_unchanged_by_scaling():
    # Roy-Vetterli erank is scale-invariant: X and 7*X have identical RankMe.
    torch.manual_seed(0)
    X = torch.randn(500, 32)
    out_a = analyze_dimensionality(X)
    out_b = analyze_dimensionality(7.5 * X)
    assert abs(out_a["rankme"] - out_b["rankme"]) < 1e-4


def test_singular_values_sorted_descending():
    # Both spectra must come back in non-increasing order; downstream plots
    # rely on this.
    torch.manual_seed(0)
    out = analyze_dimensionality(torch.randn(300, 24))
    s_unc = np.array(out["singular_values_uncentered"])
    s_c = np.array(out["singular_values_centered"])
    assert np.all(np.diff(s_unc) <= 1e-6)
    assert np.all(np.diff(s_c) <= 1e-6)
