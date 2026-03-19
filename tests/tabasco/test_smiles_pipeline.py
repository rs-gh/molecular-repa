"""Tests for SMILES threading through the training pipeline.

These tests use synthetic batches (no dataset on disk required) and verify
that SMILES strings are correctly propagated at each step:
  1. apply_random_rotation preserves SMILES in the augmented batch
  2. ChemPropEncoder fast path (MolFromSmiles + cache) matches slow path
  3. MolGraph cache grows correctly and handles duplicates
  4. SMILES survive rotation augmentation and reach the encoder's cache
     in a full FlowMatchingModel forward pass
"""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tensordict import TensorDict

from tabasco.chem.convert import MoleculeConverter
from tabasco.data.transforms import apply_random_rotation
from tabasco.flow.interpolate import CenteredMetricInterpolant, DiscreteInterpolant
from tabasco.models.components.encoders import ChemPropEncoder, Projector
from tabasco.models.components.losses import REPALoss
from tabasco.models.flow_model import FlowMatchingModel


# -- shared test molecules ----------------------------------------------------
SAMPLE_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN",
]  # ethanol, acetic acid, benzene, ethylamine
MAX_ATOMS = 10


def _smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return Chem.RemoveAllHs(mol)


def _make_batch_with_smiles(smiles_list: list[str]) -> TensorDict:
    """Build a batched TensorDict with SMILES non-tensor fields set."""
    converter = MoleculeConverter()
    individual = []
    for smi in smiles_list:
        td = converter.to_tensor(_smiles_to_mol(smi), pad_to_size=MAX_ATOMS)
        td.set_non_tensor("smiles", smi)
        individual.append(td)
    return TensorDict.stack(individual, dim=0)


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def encoder():
    return ChemPropEncoder(pretrained="chemeleon")


@pytest.fixture
def fresh_encoder():
    """New encoder with empty cache each test."""
    return ChemPropEncoder(pretrained="chemeleon")


# =============================================================================
#  1. apply_random_rotation propagates SMILES
# =============================================================================


class TestRotationAugmentation:
    def test_smiles_propagated_after_augmentation(self):
        batch = _make_batch_with_smiles(SAMPLE_SMILES)
        n_aug = 3
        aug = apply_random_rotation(batch, n_augmentations=n_aug)

        aug_smiles = list(aug.get_non_tensor("smiles"))

        assert len(aug_smiles) == len(SAMPLE_SMILES) * (n_aug + 1)
        # coords.repeat uses block repetition: [s0,s1,...,sB-1] × naug
        assert aug_smiles == SAMPLE_SMILES * (n_aug + 1)

    def test_smiles_order_matches_tensor_order(self):
        """coords[i] and smiles[i] must refer to the same molecule after augmentation."""
        batch = _make_batch_with_smiles(SAMPLE_SMILES)
        n_aug = 1
        aug = apply_random_rotation(batch, n_augmentations=n_aug)

        aug_smiles = list(aug.get_non_tensor("smiles"))
        B = len(SAMPLE_SMILES)
        naug = n_aug + 1

        # Block-wise: first B entries are aug 0, next B are aug 1, ...
        for aug_idx in range(naug):
            for mol_idx in range(B):
                flat = aug_idx * B + mol_idx
                assert aug_smiles[flat] == SAMPLE_SMILES[mol_idx]

    def test_augmentation_without_smiles_does_not_raise(self):
        converter = MoleculeConverter()
        td = converter.to_tensor(_smiles_to_mol("CCO"), pad_to_size=MAX_ATOMS)
        batch = TensorDict.stack([td, td], dim=0)

        aug = apply_random_rotation(batch, n_augmentations=3)
        assert aug["coords"].shape[0] == 8

        with pytest.raises(KeyError):
            aug.get_non_tensor("smiles")

    def test_augmented_coords_shape(self):
        batch = _make_batch_with_smiles(SAMPLE_SMILES)
        n_aug = 7
        aug = apply_random_rotation(batch, n_augmentations=n_aug)
        assert aug["coords"].shape == (len(SAMPLE_SMILES) * (n_aug + 1), MAX_ATOMS, 3)


# =============================================================================
#  2. ChemPropEncoder fast path correctness
# =============================================================================


class TestChemPropEncoderFastPath:
    def test_fast_path_produces_valid_embeddings(self, fresh_encoder):
        """Fast path (MolFromSmiles) produces finite embeddings of correct shape.

        Note: the fast path is NOT numerically equivalent to the slow path
        (DetermineConnectivity from 3D geometry).  For aromatic molecules like
        benzene, MolFromSmiles gives proper aromatic bonds while
        DetermineConnectivity may give alternating single/double bonds, leading
        to different ChemProp features.  The fast path is expected to be more
        accurate, not identical.
        """
        batch = _make_batch_with_smiles(SAMPLE_SMILES)
        coords = batch["coords"]
        atomics = batch["atomics"]
        padding_mask = batch["padding_mask"]

        with torch.no_grad():
            emb_fast = fresh_encoder(
                coords, atomics, padding_mask, smiles=SAMPLE_SMILES
            )

        B, N = len(SAMPLE_SMILES), MAX_ATOMS
        assert emb_fast.shape[0] == B
        assert emb_fast.shape[1] == N
        assert torch.isfinite(emb_fast).all(), "Fast-path embeddings contain NaN/Inf"

    def test_molgraph_cache_populates(self, fresh_encoder):
        assert len(fresh_encoder._molgraph_cache) == 0

        batch = _make_batch_with_smiles(SAMPLE_SMILES)
        with torch.no_grad():
            fresh_encoder(
                batch["coords"],
                batch["atomics"],
                batch["padding_mask"],
                smiles=SAMPLE_SMILES,
            )

        assert len(fresh_encoder._molgraph_cache) == len(set(SAMPLE_SMILES))

    def test_molgraph_cache_stabilises_on_second_call(self, fresh_encoder):
        batch = _make_batch_with_smiles(SAMPLE_SMILES)
        kwargs = dict(smiles=SAMPLE_SMILES)

        with torch.no_grad():
            fresh_encoder(
                batch["coords"], batch["atomics"], batch["padding_mask"], **kwargs
            )
        size_after_first = len(fresh_encoder._molgraph_cache)

        with torch.no_grad():
            fresh_encoder(
                batch["coords"], batch["atomics"], batch["padding_mask"], **kwargs
            )

        assert len(fresh_encoder._molgraph_cache) == size_after_first

    def test_duplicate_smiles_single_cache_entry(self, fresh_encoder):
        # Two identical SMILES in a batch → only one cache entry
        duped = ["CCO", "CCO"]
        batch = _make_batch_with_smiles(duped)
        with torch.no_grad():
            fresh_encoder(
                batch["coords"], batch["atomics"], batch["padding_mask"], smiles=duped
            )

        assert len(fresh_encoder._molgraph_cache) == 1


# =============================================================================
#  3. End-to-end: SMILES survive augmentation and reach the encoder
# =============================================================================


class TestEndToEndSMILESFlow:
    @pytest.fixture
    def flow_model_with_chemprop(self):
        enc = ChemPropEncoder(pretrained="chemeleon")
        proj = Projector(hidden_dim=128, encoder_dim=enc.encoder_dim, num_layers=2)
        repa = REPALoss(encoder=enc, projector=proj, lambda_repa=0.5)

        from tabasco.models.components.transformer_module import TransformerModule

        net = TransformerModule(
            spatial_dim=3,
            atom_dim=9,
            num_heads=4,
            num_layers=2,
            hidden_dim=128,
            implementation="pytorch",
        )
        return FlowMatchingModel(
            net=net,
            coords_interpolant=CenteredMetricInterpolant(
                key="coords", key_pad_mask="padding_mask"
            ),
            atomics_interpolant=DiscreteInterpolant(
                key="atomics", key_pad_mask="padding_mask"
            ),
            repa_loss=repa,
            num_random_augmentations=3,
        )

    def test_smiles_reach_encoder_through_augmentation(self, flow_model_with_chemprop):
        model = flow_model_with_chemprop
        encoder = model.repa_loss.encoder
        encoder._molgraph_cache.clear()

        batch = _make_batch_with_smiles(SAMPLE_SMILES)

        loss, _ = model(batch, compute_stats=False)
        loss.backward()

        assert len(encoder._molgraph_cache) > 0, (
            "Cache is empty — SMILES did not reach the encoder. "
            "Check apply_random_rotation propagates non-tensor fields."
        )

    def test_cache_size_bounded_by_unique_smiles(self, flow_model_with_chemprop):
        """Cache should not grow beyond the number of unique SMILES in the batch."""
        model = flow_model_with_chemprop
        encoder = model.repa_loss.encoder
        encoder._molgraph_cache.clear()

        batch = _make_batch_with_smiles(SAMPLE_SMILES)
        model(batch, compute_stats=False)

        assert len(encoder._molgraph_cache) <= len(set(SAMPLE_SMILES))
