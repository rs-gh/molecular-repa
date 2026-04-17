"""Smoke tests for the new protein generation metrics in evaluate.py.

Tests build_length_to_pdbs, compute_diversity_metrics, compute_novelty_metrics,
and the centroid precomputation script using real generated PDBs from
eval_output/inference_fid_60m_baseline/.

Designability is NOT tested here (requires ProteinMPNN + ESMFold on GPU).
"""

import glob
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

# Set up import paths for proteina
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../src/proteina/proteinfoundation")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src/proteina"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../evaluation/proteina/scripts")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../hpc-scripts/proteina/data_prep")
)

# Patch broken torch_scatter/torch_cluster CUDA extensions (same as hpc-scripts/proteina/evaluation/eval_fid.sh)
import proteinfoundation.repa.pyg_compat  # noqa: F401

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")
EVAL_DIR = os.path.join(REPO_ROOT, "eval_output/inference_fid_60m_baseline")
SAMPLES_DIR = os.path.join(EVAL_DIR, "samples_fid")
TENSORS_DIR = os.path.join(EVAL_DIR, "tensors")

HAS_EVAL_DATA = (
    os.path.isdir(SAMPLES_DIR)
    and len(glob.glob(os.path.join(SAMPLES_DIR, "*_fid.pdb"))) > 0
)
HAS_LMDB = os.path.isfile(
    "/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb/train.lmdb"
)


@pytest.fixture
def sample_pdbs():
    """Return a small list of PDB paths from eval output."""
    pdbs = sorted(
        glob.glob(os.path.join(SAMPLES_DIR, "*_fid.pdb")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    return pdbs[:10]


@pytest.fixture
def sample_coords():
    """Load atom37 coords from a few PDBs."""
    from proteinfoundation.metrics.designability import load_pdb

    pdbs = sorted(
        glob.glob(os.path.join(SAMPLES_DIR, "*_fid.pdb")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    coords = []
    for p in pdbs[:5]:
        prot = load_pdb(p)
        coords.append(np.array(prot.atom_positions))
    return coords


# ── build_length_to_pdbs ──


@pytest.mark.skipif(not HAS_EVAL_DATA, reason="No eval output data")
class TestBuildLengthToPdbs:
    def test_returns_dict_with_int_keys(self):
        from evaluate import build_length_to_pdbs

        result = build_length_to_pdbs(TENSORS_DIR, SAMPLES_DIR)
        assert isinstance(result, dict)
        assert len(result) > 0
        for k in result:
            assert isinstance(k, int), f"Key {k} is {type(k)}, expected int"

    def test_lengths_match_tensor_filenames(self):
        from evaluate import build_length_to_pdbs
        import re

        result = build_length_to_pdbs(TENSORS_DIR, SAMPLES_DIR)

        # Parse expected lengths from tensor filenames
        expected_lengths = set()
        for tf in glob.glob(os.path.join(TENSORS_DIR, "batch_*.pt")):
            match = re.search(r"_n(\d+)_x", os.path.basename(tf))
            if match:
                expected_lengths.add(int(match.group(1)))

        assert set(result.keys()) == expected_lengths

    def test_total_pdbs_matches_directory(self):
        from evaluate import build_length_to_pdbs

        result = build_length_to_pdbs(TENSORS_DIR, SAMPLES_DIR)

        total_mapped = sum(len(v) for v in result.values())
        total_on_disk = len(glob.glob(os.path.join(SAMPLES_DIR, "*_fid.pdb")))
        assert (
            total_mapped == total_on_disk
        ), f"Mapped {total_mapped} PDBs but {total_on_disk} exist on disk"

    def test_all_paths_exist(self):
        from evaluate import build_length_to_pdbs

        result = build_length_to_pdbs(TENSORS_DIR, SAMPLES_DIR)
        for length, pdbs in result.items():
            for p in pdbs[:3]:  # Spot-check first 3 per bin
                assert os.path.exists(p), f"PDB not found: {p}"


# ── TM-score and diversity ──


@pytest.mark.skipif(not HAS_EVAL_DATA, reason="No eval output data")
class TestTMScore:
    def test_tm_score_self_is_one(self, sample_coords):
        from proteinfoundation.metrics.tm_score import compute_tm_score

        coords = sample_coords[0]
        score = compute_tm_score(coords, coords)
        assert abs(score - 1.0) < 0.01, f"Self TM-score should be ~1.0, got {score}"

    def test_tm_score_range(self, sample_coords):
        from proteinfoundation.metrics.tm_score import compute_tm_score

        score = compute_tm_score(sample_coords[0], sample_coords[1])
        assert 0.0 <= score <= 1.0, f"TM-score out of range: {score}"


@pytest.mark.skipif(not HAS_EVAL_DATA, reason="No eval output data")
class TestDiversity:
    def test_compute_diversity_returns_positive(self, sample_coords):
        from proteinfoundation.metrics.tm_score import compute_diversity

        n_clusters = compute_diversity(sample_coords, tm_threshold=0.5)
        assert n_clusters >= 1
        assert n_clusters <= len(sample_coords)

    def test_diversity_metrics_function(self):
        from evaluate import compute_diversity_metrics

        result = compute_diversity_metrics(
            TENSORS_DIR,
            SAMPLES_DIR,
            subset_per_bin=5,
            tm_threshold=0.5,
        )
        assert isinstance(result, dict)
        assert len(result) > 0
        assert "_res_diversity_clusters_mean" in result
        assert "_res_diversity_n_bins" in result
        assert result["_res_diversity_clusters_mean"] > 0

    def test_diversity_skip_when_zero(self):
        from evaluate import compute_diversity_metrics

        result = compute_diversity_metrics(TENSORS_DIR, SAMPLES_DIR, subset_per_bin=0)
        assert result == {}


# ── Novelty ──


@pytest.mark.skipif(not HAS_EVAL_DATA, reason="No eval output data")
class TestNovelty:
    def test_novelty_with_self_centroids(self, sample_pdbs):
        """If centroids ARE the evaluated structures, novelty should be ~0."""
        from evaluate import compute_novelty_metrics
        from proteinfoundation.metrics.designability import load_pdb

        # Use the exact same PDBs as both centroids and evaluation targets.
        # compute_novelty_metrics uses RandomState(42) to subsample, so we
        # provide exactly max_eval PDBs to guarantee 1:1 correspondence.
        eval_pdbs = sample_pdbs[:3]
        centroid_coords = []
        for p in eval_pdbs:
            prot = load_pdb(p)
            centroid_coords.append(np.array(prot.atom_positions))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            centroid_path = f.name
            torch.save(centroid_coords, centroid_path)

        try:
            result = compute_novelty_metrics(
                eval_pdbs,
                centroid_path=centroid_path,
                tm_threshold=0.5,
                max_eval=3,
            )
            assert "_res_novelty_rate" in result
            # Self-comparison: TM ≈ 1.0 for each, so novelty_rate ≈ 0.0
            assert (
                result["_res_novelty_rate"] == 0.0
            ), f"Expected 0 novelty with self-centroids, got {result['_res_novelty_rate']}"
            assert result["_res_novelty_max_tm_mean"] > 0.9
        finally:
            os.unlink(centroid_path)

    def test_novelty_skip_when_no_centroids(self, sample_pdbs):
        from evaluate import compute_novelty_metrics

        result = compute_novelty_metrics(sample_pdbs, centroid_path=None)
        assert result == {}

    def test_novelty_skip_when_missing_file(self, sample_pdbs):
        from evaluate import compute_novelty_metrics

        result = compute_novelty_metrics(
            sample_pdbs, centroid_path="/nonexistent/centroids.pt"
        )
        assert result == {}


# ── Centroid precomputation ──


@pytest.mark.skipif(not HAS_LMDB, reason="Training LMDB not available")
class TestPrecomputeCentroids:
    def test_centroid_script_smoke(self):
        """Run centroid computation on a tiny subset."""
        from precompute_centroids import load_training_coords, greedy_tm_cluster

        entries = load_training_coords(
            "/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb/train.lmdb",
            max_entries=20,
        )
        assert len(entries) == 20

        # Group a few same-length entries and cluster
        coords_50_70 = [c for _, c in entries if 50 <= c.shape[0] <= 70]
        if len(coords_50_70) >= 2:
            centers = greedy_tm_cluster(coords_50_70[:5], tm_threshold=0.5)
            assert len(centers) >= 1
            assert len(centers) <= 5

    def test_end_to_end_centroid_save_load(self):
        """Run full pipeline on tiny subset and verify output format."""
        from precompute_centroids import load_training_coords

        entries = load_training_coords(
            "/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb/train.lmdb",
            max_entries=10,
        )

        # Just save coords directly as centroids (skip clustering for speed)
        centroid_coords = [c for _, c in entries[:5]]

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            centroid_path = f.name
            torch.save(centroid_coords, centroid_path)

        try:
            loaded = torch.load(centroid_path, map_location="cpu", weights_only=False)
            assert isinstance(loaded, list)
            assert len(loaded) == 5
            for c in loaded:
                assert isinstance(c, np.ndarray)
                assert c.ndim == 3
                assert c.shape[1] == 37
                assert c.shape[2] == 3
        finally:
            os.unlink(centroid_path)
