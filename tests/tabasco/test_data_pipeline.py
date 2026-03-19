"""Integration tests for the data loading pipeline.

These tests verify that:
  A. MoleculeConverter produces correctly shaped and encoded tensors
  B. QM9 LMDB items have correct shapes and valid one-hot encodings
  C. SMILES in QM9 items match the underlying molecule (regression for the
     all_smiles[index] alignment bug — indices ≥10 are the critical cases)
  D. Batch collation preserves SMILES in the correct order
  E. GEOM LMDB (conditional — skipped when not available on this machine)

Key regression:
    The old code used `self.all_smiles[index]` in __getitem__.  Because
    BaseLMDBDataset.keys comes from txn.cursor().iternext() which returns
    keys in BYTE-SORTED order (b"0","b1","b10","b100","b11","b2",...),
    all_smiles[10] refers to the SMILES of the molecule that was inserted
    10th during _process(), but keys[10] is molecule b"18" (or similar).
    The fix: compute Chem.MolToSmiles(data_dict["molecule"]) on-the-fly.
"""

from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tensordict import TensorDict
from torch.utils.data import DataLoader

from tabasco.chem.constants import ATOM_NAMES
from tabasco.chem.convert import MoleculeConverter
from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
from tabasco.data.utils import TensorDictCollator

# -- data paths ---------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "src" / "tabasco" / "data"

QM9_PT = DATA_DIR / "processed_qm9_train.pt"
QM9_LMDB = DATA_DIR / "lmdb_qm9"

GEOM_PT = DATA_DIR / "processed_geom_train.pt"
GEOM_LMDB = DATA_DIR / "lmdb_geom"

ATOM_DIM = len(ATOM_NAMES)  # 9


# =============================================================================
#  A. MoleculeConverter correctness  (no disk data needed)
# =============================================================================


class TestMoleculeConverter:
    @pytest.fixture(scope="class")
    def converter(self):
        return MoleculeConverter()

    @pytest.fixture(scope="class")
    def ethanol_tensor(self, converter):
        """Ethanol (CCO) — 3 heavy atoms, padded to 10."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveAllHs(mol)
        return converter.to_tensor(mol, pad_to_size=10)

    def test_coords_shape_and_dtype(self, ethanol_tensor):
        coords = ethanol_tensor["coords"]
        assert coords.shape == (10, 3)
        assert coords.dtype == torch.float32

    def test_atomics_shape_and_dtype(self, ethanol_tensor):
        atomics = ethanol_tensor["atomics"]
        assert atomics.shape == (10, ATOM_DIM)
        assert atomics.dtype == torch.float32

    def test_padding_mask_shape_and_dtype(self, ethanol_tensor):
        mask = ethanol_tensor["padding_mask"]
        assert mask.shape == (10,)
        assert mask.dtype == torch.bool

    def test_correct_number_of_real_atoms(self, ethanol_tensor):
        """Ethanol has 3 heavy atoms; the rest should be padded."""
        real_atoms = (~ethanol_tensor["padding_mask"]).sum().item()
        assert real_atoms == 3

    def test_atom_encoding_carbon(self, ethanol_tensor):
        """First two atoms of ethanol (CCO) are carbon."""
        atomics = ethanol_tensor["atomics"]
        c_idx = ATOM_NAMES.index("C")
        assert atomics[0].argmax().item() == c_idx
        assert atomics[1].argmax().item() == c_idx

    def test_atom_encoding_oxygen(self, ethanol_tensor):
        """Third atom of ethanol (CCO) is oxygen."""
        atomics = ethanol_tensor["atomics"]
        o_idx = ATOM_NAMES.index("O")
        assert atomics[2].argmax().item() == o_idx

    def test_real_atoms_one_hot_valid(self, ethanol_tensor):
        """Real atom rows must sum to exactly 1.0."""
        atomics = ethanol_tensor["atomics"]
        mask = ethanol_tensor["padding_mask"]
        real_sums = atomics[~mask].sum(dim=-1)
        assert torch.allclose(real_sums, torch.ones_like(real_sums))

    def test_padded_atoms_encoded_as_dummy(self, ethanol_tensor):
        """Padded atom rows are one-hot for the '*' dummy atom, not all-zero.

        MoleculeConverter._pad_to_size sets dummy_atoms[:, dummy_atom_idx] = 1.
        """
        atomics = ethanol_tensor["atomics"]
        mask = ethanol_tensor["padding_mask"]
        dummy_idx = ATOM_NAMES.index("*")
        # Each padded row sums to 1 (valid one-hot) ...
        padded_sums = atomics[mask].sum(dim=-1)
        assert torch.allclose(padded_sums, torch.ones_like(padded_sums))
        # ... and the hot index is the dummy atom
        assert (atomics[mask].argmax(dim=-1) == dummy_idx).all()


# =============================================================================
#  QM9 fixtures
# =============================================================================


@pytest.fixture(scope="module")
def qm9_dataset():
    if not QM9_LMDB.exists():
        pytest.skip("QM9 LMDB not available")
    return UnconditionalLMDBDataset(
        data_dir=str(QM9_PT),
        split="train",
        lmdb_dir=str(QM9_LMDB),
        add_random_rotation=False,
        add_random_permutation=False,
    )


# =============================================================================
#  B. QM9 LMDB — shapes and tensor validity
# =============================================================================


class TestQM9ItemShapes:
    def test_item_coords_shape(self, qm9_dataset):
        item = qm9_dataset[0]
        N = qm9_dataset.max_mol_num_atoms
        assert item["coords"].shape == (N, 3)

    def test_item_atomics_shape(self, qm9_dataset):
        item = qm9_dataset[0]
        N = qm9_dataset.max_mol_num_atoms
        assert item["atomics"].shape == (N, ATOM_DIM)

    def test_item_padding_mask_shape(self, qm9_dataset):
        item = qm9_dataset[0]
        N = qm9_dataset.max_mol_num_atoms
        assert item["padding_mask"].shape == (N,)

    def test_item_has_real_atoms(self, qm9_dataset):
        item = qm9_dataset[0]
        assert (~item["padding_mask"]).any()

    def test_item_real_atoms_one_hot_valid(self, qm9_dataset):
        item = qm9_dataset[0]
        atomics = item["atomics"]
        mask = item["padding_mask"]
        real_sums = atomics[~mask].sum(dim=-1)
        assert torch.allclose(real_sums, torch.ones_like(real_sums))

    def test_item_padded_atoms_encoded_as_dummy(self, qm9_dataset):
        """Padded rows are one-hot for '*' dummy atom (same convention as MoleculeConverter)."""
        item = qm9_dataset[0]
        atomics = item["atomics"]
        mask = item["padding_mask"]
        if mask.any():
            dummy_idx = ATOM_NAMES.index("*")
            padded_sums = atomics[mask].sum(dim=-1)
            assert torch.allclose(padded_sums, torch.ones_like(padded_sums))
            assert (atomics[mask].argmax(dim=-1) == dummy_idx).all()

    def test_coords_dtype(self, qm9_dataset):
        assert qm9_dataset[0]["coords"].dtype == torch.float32

    def test_atomics_dtype(self, qm9_dataset):
        assert qm9_dataset[0]["atomics"].dtype == torch.float32

    def test_padding_mask_dtype(self, qm9_dataset):
        assert qm9_dataset[0]["padding_mask"].dtype == torch.bool


# =============================================================================
#  C. QM9 LMDB — SMILES correctness
#     (regression for the all_smiles[index] alignment bug)
# =============================================================================


class TestQM9SMILESCorrectness:
    def _check_smiles_matches_mol(self, qm9_dataset, idx):
        item = qm9_dataset[idx]
        data_dict = qm9_dataset.get_data_dict(idx)
        expected = Chem.MolToSmiles(data_dict["molecule"])
        actual = item.get_non_tensor("smiles")
        assert actual == expected, f"Index {idx}: got '{actual}', expected '{expected}'"

    def test_smiles_is_set_on_item(self, qm9_dataset):
        item = qm9_dataset[0]
        smi = item.get_non_tensor("smiles")
        assert isinstance(smi, str) and len(smi) > 0

    @pytest.mark.parametrize("idx", [0, 1, 5, 9])
    def test_smiles_matches_molecule_small_indices(self, qm9_dataset, idx):
        """Indices < 10: even the old all_smiles[index] happened to be correct."""
        self._check_smiles_matches_mol(qm9_dataset, idx)

    @pytest.mark.parametrize("idx", [10, 11, 99, 100, 101, 999, 1000])
    def test_smiles_matches_molecule_large_indices(self, qm9_dataset, idx):
        """Indices ≥ 10: byte-sorted LMDB keys diverge from numeric insertion order.

        This is the key regression test — the old `all_smiles[index]` code
        would silently return the SMILES for a different molecule at these
        indices (b"10" sorts before b"2", so keys[2] != molecule 2).
        """
        self._check_smiles_matches_mol(qm9_dataset, idx)

    def test_smiles_is_canonical(self, qm9_dataset):
        """Stored SMILES should be canonical RDKit output."""
        for idx in [0, 10, 100]:
            item = qm9_dataset[idx]
            smi = item.get_non_tensor("smiles")
            # A canonical SMILES round-trips to itself
            assert Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == smi


# =============================================================================
#  D. QM9 LMDB — batch collation
# =============================================================================


class TestQM9BatchCollation:
    def test_batch_smiles_list_length(self, qm9_dataset):
        batch_size = 4
        loader = DataLoader(
            qm9_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=TensorDictCollator(),
        )
        batch = next(iter(loader))
        smiles_list = list(batch.get_non_tensor("smiles"))
        assert len(smiles_list) == batch_size

    def test_batch_smiles_all_non_empty(self, qm9_dataset):
        loader = DataLoader(
            qm9_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=TensorDictCollator(),
        )
        batch = next(iter(loader))
        for smi in batch.get_non_tensor("smiles"):
            assert isinstance(smi, str) and len(smi) > 0

    def test_batch_smiles_match_individual_items(self, qm9_dataset):
        """Collating items [0,1,2,3] should give the same SMILES in the same order."""
        individual_smiles = [qm9_dataset[i].get_non_tensor("smiles") for i in range(4)]
        items = [qm9_dataset[i] for i in range(4)]
        batch = TensorDict.stack(items, dim=0)
        batch_smiles = list(batch.get_non_tensor("smiles"))
        assert batch_smiles == individual_smiles

    def test_batch_tensor_shapes(self, qm9_dataset):
        loader = DataLoader(
            qm9_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=TensorDictCollator(),
        )
        batch = next(iter(loader))
        N = qm9_dataset.max_mol_num_atoms
        assert batch["coords"].shape == (4, N, 3)
        assert batch["atomics"].shape == (4, N, ATOM_DIM)
        assert batch["padding_mask"].shape == (4, N)


# =============================================================================
#  E. GEOM LMDB — conditional (skipped if not available locally)
# =============================================================================


@pytest.fixture(scope="module")
def geom_dataset():
    if not GEOM_LMDB.exists():
        pytest.skip("GEOM LMDB not available locally")
    return UnconditionalLMDBDataset(
        data_dir=str(GEOM_PT),
        split="train",
        lmdb_dir=str(GEOM_LMDB),
        add_random_rotation=False,
        add_random_permutation=False,
    )


class TestGEOMSMILESCorrectness:
    def _check_smiles_matches_mol(self, geom_dataset, idx):
        item = geom_dataset[idx]
        data_dict = geom_dataset.get_data_dict(idx)
        expected = Chem.MolToSmiles(data_dict["molecule"])
        actual = item.get_non_tensor("smiles")
        assert actual == expected, f"Index {idx}: got '{actual}', expected '{expected}'"

    def test_smiles_is_set(self, geom_dataset):
        smi = geom_dataset[0].get_non_tensor("smiles")
        assert isinstance(smi, str) and len(smi) > 0

    @pytest.mark.parametrize("idx", [10, 11, 99, 100, 1000])
    def test_smiles_matches_molecule_large_indices(self, geom_dataset, idx):
        """Same alignment regression test as QM9, but for GEOM."""
        self._check_smiles_matches_mol(geom_dataset, idx)

    def test_batch_high_smiles_reuse(self, geom_dataset):
        """GEOM has multiple conformers per molecule.

        A batch of 256 should contain significantly fewer than 256 unique SMILES,
        demonstrating that the MolGraph cache will get real hits during training.
        """
        batch_size = 256
        loader = DataLoader(
            geom_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=TensorDictCollator(),
        )
        batch = next(iter(loader))
        smiles_list = list(batch.get_non_tensor("smiles"))
        unique_count = len(set(smiles_list))
        assert unique_count < batch_size, (
            f"Expected fewer unique SMILES than batch size ({batch_size}), "
            f"got {unique_count} unique out of {batch_size}"
        )
