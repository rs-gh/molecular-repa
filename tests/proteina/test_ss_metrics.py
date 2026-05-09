"""Tests for SS metrics in evaluation/proteina/generation/scripts/utils/ss_metrics.py."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `utils.ss_metrics` importable the same way evaluate.py imports it
# (via PYTHONPATH = scripts/).
SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2] / "evaluation/proteina/generation/scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.ss_metrics import (  # noqa: E402
    SS_COLS,
    _annotate_one,
    compute_per_sample_ss_fractions,
    compute_ss_jsd,
    compute_ss_metrics,
)

# Use existing generated PDBs (no fixture creation needed).
SAMPLE_RUN_DIR = (
    Path("/home/sr2173/git/molecular-repa/eval_output/")
    / "inference_paper_inference_fid_60m_n128_paper_sweep_repa_l4_128_pw_structure_step100k_step_100000"
    / "samples_fid"
)


def _have_samples() -> bool:
    return SAMPLE_RUN_DIR.exists() and len(list(SAMPLE_RUN_DIR.glob("*_fid.pdb"))) > 0


pytestmark = pytest.mark.skipif(
    not _have_samples(), reason="no generated PDBs available"
)


def _few_pdbs(n: int = 5) -> list[str]:
    return sorted(glob.glob(str(SAMPLE_RUN_DIR / "*_fid.pdb")))[:n]


def test_annotate_one_returns_simplex_point():
    f = _annotate_one(_few_pdbs(1)[0])
    assert f is not None
    assert f.shape == (3,)
    assert np.isclose(f.sum(), 1.0, atol=1e-6)
    assert (f >= 0).all() and (f <= 1).all()


def test_annotate_one_missing_file_returns_none():
    assert _annotate_one("/tmp/__not_a_pdb__.pdb") is None


def test_per_sample_fractions_shape_and_normalisation():
    paths = _few_pdbs(5)
    fracs, kept = compute_per_sample_ss_fractions(paths)
    assert fracs.shape[1] == 3
    assert fracs.shape[0] == len(kept)
    assert len(kept) >= 1
    assert np.allclose(fracs.sum(axis=1), 1.0, atol=1e-6)


def test_per_sample_fractions_caching(tmp_path):
    paths = _few_pdbs(3)
    cache = tmp_path / "ss_fractions.npz"
    fracs1, _ = compute_per_sample_ss_fractions(paths, cache_path=cache)
    assert cache.exists()
    # Second call must return identical numbers without re-running annotate_sse
    # (we can't easily intercept the call, but matching arrays is good enough).
    fracs2, _ = compute_per_sample_ss_fractions(paths, cache_path=cache)
    assert np.array_equal(fracs1, fracs2)


def test_jsd_self_is_zero():
    rng = np.random.default_rng(0)
    x = rng.dirichlet([1.0, 1.0, 1.0], size=20)
    assert compute_ss_jsd(x, x) == pytest.approx(0.0, abs=1e-12)


def test_jsd_orthogonal_distributions_are_large():
    # Pure-helix vs pure-sheet distributions: JSD should be 1 (max).
    helix = np.tile([1.0, 0.0, 0.0], (10, 1))
    sheet = np.tile([0.0, 1.0, 0.0], (10, 1))
    js = compute_ss_jsd(helix, sheet)
    assert 0.99 < js <= 1.0


def test_jsd_empty_returns_nan():
    nan_val = compute_ss_jsd(np.zeros((0, 3)), np.ones((5, 3)) / 3)
    assert np.isnan(nan_val)


def test_compute_ss_metrics_end_to_end(tmp_path):
    paths = _few_pdbs(8)
    # synthetic reference: dirichlet sample ≈ PDB-like (~36/22/42)
    rng = np.random.default_rng(0)
    ref = rng.dirichlet([3.6, 2.2, 4.2], size=200).astype(np.float32)
    import torch

    pdb_ref = tmp_path / "ss_ref_pdb.pt"
    afdb_ref = tmp_path / "ss_ref_afdb.pt"
    torch.save(torch.from_numpy(ref), pdb_ref)
    torch.save(torch.from_numpy(ref), afdb_ref)

    designable = paths[:4]  # pretend half are designable
    out = compute_ss_metrics(
        list_of_pdbs=paths,
        designable_pdbs=designable,
        ss_reference_pdb_path=str(pdb_ref),
        ss_reference_afdb_path=str(afdb_ref),
        cache_dir=tmp_path,
    )

    # All-samples columns
    for col in SS_COLS:
        assert f"_res_ss_frac_{col}" in out
    assert "_res_ss_jsd_pdb" in out
    assert "_res_ss_jsd_afdb" in out
    assert out["_res_ss_n"] == len(paths)

    # Designable subset columns
    for col in SS_COLS:
        assert f"_res_ss_frac_{col}_designable" in out
    assert "_res_ss_jsd_pdb_designable" in out
    assert out["_res_ss_n_designable"] == 4

    # Fractions sum to ~1
    s = sum(out[f"_res_ss_frac_{c}"] for c in SS_COLS)
    assert s == pytest.approx(1.0, abs=1e-6)
