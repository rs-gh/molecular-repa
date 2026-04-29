"""Tests for LMDB protein dataset: conversion and read-back equivalence.

Verifies that converting .pt files to LMDB and reading them back produces
identical PyG Data objects, ensuring the LMDB pipeline is a safe drop-in
replacement for the per-file loading path.
"""

import os
import pickle

import lmdb
import pytest
import torch
from torch_geometric.data import Data

from proteinfoundation.datasets.lmdb_utils import convert_pt_to_lmdb
from proteinfoundation.datasets.lmdb_dataset import ProteinLMDBDataset
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_graph():
    """Create a minimal PyG Data object mimicking a processed protein."""
    n_residues = 50
    return Data(
        coords=torch.randn(n_residues, 37, 3),
        coord_mask=torch.ones(n_residues, 37, dtype=torch.bool),
        residue_type=torch.randint(0, 20, (n_residues,)),
        bfactor_avg=torch.randn(n_residues),
        residue_pdb_idx=torch.arange(n_residues),
        seq_pos=torch.arange(n_residues).unsqueeze(1),
        id="test_protein",
    )


@pytest.fixture
def pt_dir_with_samples(sample_graph, tmp_path):
    """Create a temp directory with .pt files for conversion testing."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    names = []
    for i in range(5):
        g = sample_graph.clone()
        g.id = f"protein_{i}"
        g.coords = torch.randn_like(g.coords)  # unique per sample
        torch.save(g, processed_dir / f"protein_{i}.pt")
        names.append(f"protein_{i}")

    return processed_dir, names


# ---------------------------------------------------------------------------
# Tests: Conversion
# ---------------------------------------------------------------------------


class TestConvertPtToLmdb:
    """Test the .pt -> LMDB conversion utility."""

    def test_basic_conversion(self, pt_dir_with_samples, tmp_path):
        """Convert .pt files to LMDB and verify entry count."""
        processed_dir, names = pt_dir_with_samples
        lmdb_path = str(tmp_path / "test.lmdb")

        n_written = convert_pt_to_lmdb(
            processed_dir=str(processed_dir),
            file_names=names,
            output_path=lmdb_path,
            apply_coord_reorder=False,
        )

        assert n_written == 5
        assert os.path.exists(lmdb_path)

    def test_lmdb_contains_all_keys(self, pt_dir_with_samples, tmp_path):
        """Verify LMDB has sequential string keys for all samples."""
        processed_dir, names = pt_dir_with_samples
        lmdb_path = str(tmp_path / "test.lmdb")

        convert_pt_to_lmdb(
            processed_dir=str(processed_dir),
            file_names=names,
            output_path=lmdb_path,
            apply_coord_reorder=False,
        )

        db = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False)
        with db.begin() as txn:
            keys = [k.decode() for k, _ in txn.cursor()]
        db.close()

        assert keys == ["0", "1", "2", "3", "4"]

    def test_round_trip_equivalence(self, pt_dir_with_samples, tmp_path):
        """Data read from LMDB should match the original .pt file."""
        processed_dir, names = pt_dir_with_samples
        lmdb_path = str(tmp_path / "test.lmdb")

        convert_pt_to_lmdb(
            processed_dir=str(processed_dir),
            file_names=names,
            output_path=lmdb_path,
            apply_coord_reorder=False,
        )

        db = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False)
        with db.begin() as txn:
            for i, name in enumerate(names):
                original = torch.load(processed_dir / f"{name}.pt", weights_only=False)
                from_lmdb = pickle.loads(txn.get(str(i).encode()))

                assert torch.equal(original.coords, from_lmdb.coords)
                assert torch.equal(original.coord_mask, from_lmdb.coord_mask)
                assert torch.equal(original.residue_type, from_lmdb.residue_type)
                assert original.id == from_lmdb.id
        db.close()

    def test_coord_reorder_applied(self, pt_dir_with_samples, tmp_path):
        """When apply_coord_reorder=True, coords should be reordered."""
        processed_dir, names = pt_dir_with_samples
        lmdb_path = str(tmp_path / "test.lmdb")

        convert_pt_to_lmdb(
            processed_dir=str(processed_dir),
            file_names=names,
            output_path=lmdb_path,
            apply_coord_reorder=True,
        )

        db = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False)
        with db.begin() as txn:
            original = torch.load(processed_dir / f"{names[0]}.pt", weights_only=False)
            from_lmdb = pickle.loads(txn.get(b"0"))

            expected = original.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
            assert torch.equal(expected, from_lmdb.coords)
        db.close()

    def test_skips_missing_files(self, pt_dir_with_samples, tmp_path):
        """Missing .pt files should be skipped, not crash."""
        processed_dir, names = pt_dir_with_samples
        lmdb_path = str(tmp_path / "test.lmdb")

        names_with_missing = names + ["nonexistent_protein"]
        n_written = convert_pt_to_lmdb(
            processed_dir=str(processed_dir),
            file_names=names_with_missing,
            output_path=lmdb_path,
            apply_coord_reorder=False,
        )

        assert n_written == 5  # missing file skipped


# ---------------------------------------------------------------------------
# Tests: Dataset
# ---------------------------------------------------------------------------


class TestProteinLMDBDataset:
    """Test the LMDB-backed dataset class."""

    @pytest.fixture
    def lmdb_dataset(self, pt_dir_with_samples, tmp_path):
        """Create an LMDB file and return a dataset over it."""
        processed_dir, names = pt_dir_with_samples
        lmdb_path = str(tmp_path / "test.lmdb")
        convert_pt_to_lmdb(
            processed_dir=str(processed_dir),
            file_names=names,
            output_path=lmdb_path,
            apply_coord_reorder=False,
        )
        return ProteinLMDBDataset(lmdb_path=lmdb_path)

    def test_len(self, lmdb_dataset):
        assert len(lmdb_dataset) == 5

    def test_getitem_returns_data(self, lmdb_dataset):
        graph = lmdb_dataset[0]
        assert isinstance(graph, Data)
        assert hasattr(graph, "coords")
        assert hasattr(graph, "coord_mask")
        assert hasattr(graph, "residue_type")

    def test_getitem_all_indices(self, lmdb_dataset):
        """Every index should return valid data."""
        for i in range(len(lmdb_dataset)):
            graph = lmdb_dataset[i]
            assert graph.coords.shape[1] == 37
            assert graph.coords.shape[2] == 3

    def test_transform_applied(self, pt_dir_with_samples, tmp_path):
        """Transforms should be applied to each sample."""
        processed_dir, names = pt_dir_with_samples
        lmdb_path = str(tmp_path / "test.lmdb")
        convert_pt_to_lmdb(
            processed_dir=str(processed_dir),
            file_names=names,
            output_path=lmdb_path,
            apply_coord_reorder=False,
        )

        call_count = {"n": 0}

        def counting_transform(data):
            call_count["n"] += 1
            return data

        ds = ProteinLMDBDataset(lmdb_path=lmdb_path, transform=counting_transform)
        _ = ds[0]
        _ = ds[1]
        assert call_count["n"] == 2

    def test_lazy_connection(self, pt_dir_with_samples, tmp_path):
        """DB should not connect until first access."""
        processed_dir, names = pt_dir_with_samples
        lmdb_path = str(tmp_path / "test.lmdb")
        convert_pt_to_lmdb(
            processed_dir=str(processed_dir),
            file_names=names,
            output_path=lmdb_path,
            apply_coord_reorder=False,
        )

        ds = ProteinLMDBDataset(lmdb_path=lmdb_path)
        assert ds.db is None
        _ = len(ds)
        assert ds.db is not None
