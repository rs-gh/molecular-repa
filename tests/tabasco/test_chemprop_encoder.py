"""Tests for ChemPropEncoder integration."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tensordict import TensorDict

from tabasco.models.components.encoders import ChemPropEncoder
from tabasco.chem.convert import MoleculeConverter


class TestChemPropEncoder:
    """Tests for ChemPropEncoder with CheMeleon pretrained weights."""

    @pytest.fixture
    def converter(self):
        return MoleculeConverter()

    @pytest.fixture
    def encoder(self):
        return ChemPropEncoder(pretrained="chemeleon")

    @pytest.fixture
    def sample_smiles(self):
        return ["CCO", "CC(=O)O", "c1ccccc1", "CCN"]

    def _smiles_to_mol(self, smiles: str):
        """Convert SMILES to RDKit mol with 3D coordinates."""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        return Chem.RemoveAllHs(mol)

    def _create_batch(
        self, smiles_list: list[str], converter: MoleculeConverter, max_atoms: int = 10
    ):
        """Create a batched TensorDict from SMILES."""
        tensordicts = [
            converter.to_tensor(self._smiles_to_mol(s), pad_to_size=max_atoms)
            for s in smiles_list
        ]
        return TensorDict(
            {
                "coords": torch.stack([td["coords"] for td in tensordicts]),
                "atomics": torch.stack([td["atomics"] for td in tensordicts]),
                "padding_mask": torch.stack([td["padding_mask"] for td in tensordicts]),
            },
            batch_size=[len(smiles_list)],
        )

    def test_chemeleon_model_loads(self, encoder):
        """Test that CheMeleon pretrained weights load correctly."""
        assert encoder.encoder_dim > 0
        assert hasattr(encoder, "message_passing")
        assert hasattr(encoder, "featurizer")

        # Verify model has parameters
        params = list(encoder.message_passing.parameters())
        assert len(params) > 0
        assert all(p.dtype == torch.float32 for p in params)

    def test_encoder_produces_embeddings(self, encoder, converter, sample_smiles):
        """Test that encoder produces valid embeddings for known molecules."""
        batch = self._create_batch(sample_smiles, converter, max_atoms=10)
        B, N = batch["coords"].shape[:2]

        with torch.no_grad():
            embeddings = encoder(
                batch["coords"],
                batch["atomics"],
                batch["padding_mask"],
            )

        # Check output shape
        assert embeddings.shape == (B, N, encoder.encoder_dim)

        # Check that real atoms have non-zero embeddings
        for i in range(B):
            real_mask = ~batch["padding_mask"][i]
            real_embeddings = embeddings[i, real_mask]
            assert real_embeddings.shape[0] > 0
            assert not torch.allclose(
                real_embeddings, torch.zeros_like(real_embeddings)
            )

        # Check that padded positions are zeroed
        for i in range(B):
            pad_mask = batch["padding_mask"][i]
            if pad_mask.any():
                padded_embeddings = embeddings[i, pad_mask]
                assert torch.allclose(
                    padded_embeddings, torch.zeros_like(padded_embeddings)
                )
