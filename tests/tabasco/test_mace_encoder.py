"""Tests for MACEEncoder integration."""

import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tensordict import TensorDict

from tabasco.models.components.encoders import MACEEncoder, Projector
from tabasco.models.components.losses import REPALoss
from tabasco.models.components.transformer_module import TransformerModule
from tabasco.models.flow_model import FlowMatchingModel
from tabasco.flow.path import FlowPath
from tabasco.flow.interpolate import CenteredMetricInterpolant, DiscreteInterpolant
from tabasco.chem.convert import MoleculeConverter


@pytest.fixture(scope="module")
def mace_encoder():
    """Load MACEEncoder once for all tests (slow to init)."""
    torch.set_default_dtype(torch.float32)
    return MACEEncoder(model_name="small")


@pytest.fixture
def converter():
    return MoleculeConverter()


@pytest.fixture
def sample_smiles():
    return ["CCO", "CC(=O)O", "c1ccccc1", "CCN"]


def _smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit mol with 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return Chem.RemoveAllHs(mol)


def _create_batch(
    smiles_list: list[str], converter: MoleculeConverter, max_atoms: int = 10
):
    """Create a batched TensorDict from SMILES."""
    tensordicts = [
        converter.to_tensor(_smiles_to_mol(s), pad_to_size=max_atoms)
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


class TestMACEEncoder:
    """Tests for MACEEncoder loading and embedding extraction."""

    def test_model_loads(self, mace_encoder):
        """Test that MACE model loads and has correct encoder_dim."""
        assert mace_encoder.encoder_dim == 192  # small: 96/layer x 2 layers
        assert hasattr(mace_encoder, "mace_model")
        assert mace_encoder.model_name == "small"

    def test_parameters_frozen(self, mace_encoder):
        """All MACE parameters should be frozen."""
        for param in mace_encoder.mace_model.parameters():
            assert not param.requires_grad

    def test_encoder_produces_embeddings(self, mace_encoder, converter, sample_smiles):
        """Test that encoder produces valid embeddings for known molecules."""
        batch = _create_batch(sample_smiles, converter, max_atoms=10)
        B, N = batch["coords"].shape[:2]

        with torch.no_grad():
            embeddings = mace_encoder(
                batch["coords"],
                batch["atomics"],
                batch["padding_mask"],
            )

        assert embeddings.shape == (B, N, mace_encoder.encoder_dim)
        assert embeddings.dtype == torch.float32

    def test_real_atoms_nonzero(self, mace_encoder, converter, sample_smiles):
        """Real atoms should have non-zero embeddings."""
        batch = _create_batch(sample_smiles, converter, max_atoms=10)
        B = batch["coords"].shape[0]

        with torch.no_grad():
            embeddings = mace_encoder(
                batch["coords"],
                batch["atomics"],
                batch["padding_mask"],
            )

        for i in range(B):
            real_mask = ~batch["padding_mask"][i]
            real_embeddings = embeddings[i, real_mask]
            assert real_embeddings.shape[0] > 0
            assert not torch.allclose(
                real_embeddings, torch.zeros_like(real_embeddings)
            )

    def test_padded_positions_zeroed(self, mace_encoder, converter, sample_smiles):
        """Padded positions should have zero embeddings."""
        batch = _create_batch(sample_smiles, converter, max_atoms=10)
        B = batch["coords"].shape[0]

        with torch.no_grad():
            embeddings = mace_encoder(
                batch["coords"],
                batch["atomics"],
                batch["padding_mask"],
            )

        for i in range(B):
            pad_mask = batch["padding_mask"][i]
            if pad_mask.any():
                padded_embeddings = embeddings[i, pad_mask]
                assert torch.allclose(
                    padded_embeddings, torch.zeros_like(padded_embeddings)
                )

    def test_smiles_kwarg_accepted(self, mace_encoder, converter):
        """MACEEncoder should accept smiles= without error (for interface compat)."""
        batch = _create_batch(["CCO", "CCN"], converter, max_atoms=10)
        with torch.no_grad():
            embeddings = mace_encoder(
                batch["coords"],
                batch["atomics"],
                batch["padding_mask"],
                smiles=["CCO", "CCN"],
            )
        assert embeddings.shape[-1] == mace_encoder.encoder_dim

    def test_no_sparsity(self, mace_encoder, converter, sample_smiles):
        """MACE embeddings should have no exact zeros (unlike CheMeleon's 93%)."""
        batch = _create_batch(sample_smiles, converter, max_atoms=10)
        with torch.no_grad():
            embeddings = mace_encoder(
                batch["coords"],
                batch["atomics"],
                batch["padding_mask"],
            )

        # Check only real atoms
        for i in range(batch["coords"].shape[0]):
            real = embeddings[i, ~batch["padding_mask"][i]]
            sparsity = (real == 0).float().mean().item()
            assert sparsity < 0.01, f"MACE should not be sparse, got {sparsity:.1%}"

    def test_conformer_sensitivity(self, mace_encoder, converter):
        """MACE should give different embeddings for different 3D conformers."""
        smiles = "CCCCCC"  # hexane — many rotamers
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMultipleConfs(mol, numConfs=2, params=params)
        AllChem.MMFFOptimizeMoleculeConfs(mol)

        embeddings = []
        for conf_id in range(mol.GetNumConformers()):
            mol_copy = Chem.RWMol(mol)
            # Keep only this conformer
            for cid in range(mol_copy.GetNumConformers()):
                if cid != conf_id:
                    mol_copy.RemoveConformer(cid)
            mol_noh = Chem.RemoveAllHs(mol_copy)
            n = mol_noh.GetNumAtoms()
            td = converter.to_tensor(mol_noh, pad_to_size=n)
            coords = td["coords"].unsqueeze(0)
            atomics = td["atomics"].unsqueeze(0)
            padding = td["padding_mask"].unsqueeze(0)

            with torch.no_grad():
                emb = mace_encoder(coords, atomics, padding)
            embeddings.append(emb.squeeze(0))

        # Embeddings should differ (cosine < 1.0)
        cos = torch.nn.functional.cosine_similarity(
            embeddings[0], embeddings[1], dim=-1
        ).mean()
        assert cos < 0.9999, f"MACE should distinguish conformers, got cos={cos:.6f}"

    def test_empty_batch(self, mace_encoder):
        """All-padding batch should return zeros."""
        B, N = 2, 5
        coords = torch.randn(B, N, 3)
        atomics = torch.zeros(B, N, 9)
        atomics[:, :, 8] = 1.0  # all dummy atoms
        padding = torch.ones(B, N, dtype=torch.bool)

        with torch.no_grad():
            out = mace_encoder(coords, atomics, padding)

        assert out.shape == (B, N, mace_encoder.encoder_dim)
        assert torch.allclose(out, torch.zeros_like(out))


class TestMACEREPAIntegration:
    """Tests for MACE encoder integrated with REPA loss and FlowMatchingModel."""

    @pytest.fixture
    def hidden_dim(self):
        return 64

    @pytest.fixture
    def atom_dim(self):
        return 9

    @pytest.fixture
    def transformer(self, hidden_dim, atom_dim):
        return TransformerModule(
            spatial_dim=3,
            atom_dim=atom_dim,
            num_heads=4,
            num_layers=2,
            hidden_dim=hidden_dim,
            implementation="pytorch",
        )

    @pytest.fixture
    def projector(self, hidden_dim, mace_encoder):
        return Projector(
            hidden_dim=hidden_dim, encoder_dim=mace_encoder.encoder_dim, num_layers=2
        )

    @pytest.fixture
    def repa_loss(self, mace_encoder, projector):
        return REPALoss(
            encoder=mace_encoder,
            projector=projector,
            lambda_repa=0.5,
            time_weighting=False,
        )

    @pytest.fixture
    def molecular_batch(self, converter):
        return _create_batch(["CCO", "CC(=O)O"], converter, max_atoms=10)

    @pytest.fixture
    def batch_data(self, hidden_dim, molecular_batch):
        coords = molecular_batch["coords"]
        atomics = molecular_batch["atomics"]
        padding_mask = molecular_batch["padding_mask"]
        batch_size = coords.shape[0]
        num_atoms = coords.shape[1]

        t = torch.rand(batch_size)
        hidden_states = torch.randn(
            batch_size, num_atoms, hidden_dim, requires_grad=True
        )

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=batch_size,
        )
        path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)
        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": hidden_states,
                "padding_mask": padding_mask,
            },
            batch_size=batch_size,
        )

        return {
            "coords": coords,
            "atomics": atomics,
            "padding_mask": padding_mask,
            "t": t,
            "hidden_states": hidden_states,
            "path": path,
            "pred": pred,
        }

    @pytest.fixture
    def flow_model(self, transformer, repa_loss):
        coords_interpolant = CenteredMetricInterpolant(
            key="coords", key_pad_mask="padding_mask"
        )
        atomics_interpolant = DiscreteInterpolant(
            key="atomics", key_pad_mask="padding_mask"
        )
        return FlowMatchingModel(
            net=transformer,
            coords_interpolant=coords_interpolant,
            atomics_interpolant=atomics_interpolant,
            repa_loss=repa_loss,
        )

    def test_encoder_params_frozen(self, repa_loss):
        """MACE encoder parameters should not require gradients."""
        for param in repa_loss.encoder.parameters():
            assert not param.requires_grad

    def test_repa_loss_computes(self, repa_loss, batch_data):
        """REPA loss should compute a scalar loss with MACE encoder."""
        loss, stats = repa_loss(
            batch_data["path"], batch_data["pred"], compute_stats=True
        )
        assert loss.dim() == 0
        assert "repa_loss" in stats
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_projector_receives_gradients(self, repa_loss, batch_data):
        """Projector should be trainable; MACE encoder should not receive gradients."""
        loss, _ = repa_loss(batch_data["path"], batch_data["pred"], compute_stats=True)
        loss.backward()

        # Hidden states receive gradients
        assert batch_data["hidden_states"].grad is not None
        assert batch_data["hidden_states"].grad.abs().sum() > 0

        # Projector receives gradients
        projector_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in repa_loss.projector.parameters()
        )
        assert projector_has_grad

        # Encoder does NOT receive gradients
        encoder_has_grad = any(
            p.grad is not None for p in repa_loss.encoder.parameters()
        )
        assert not encoder_has_grad

    def test_flow_model_end_to_end(self, flow_model, molecular_batch):
        """Full forward + backward through FlowMatchingModel with MACE encoder."""
        loss, stats = flow_model(molecular_batch, compute_stats=True)
        assert "repa_loss" in stats
        assert not torch.isnan(loss)

        loss.backward()

        # Transformer backbone receives gradients
        net_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in flow_model.net.parameters()
        )
        assert net_has_grad

        # Projector receives gradients
        projector_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in flow_model.repa_loss.projector.parameters()
        )
        assert projector_has_grad

        # Encoder frozen
        encoder_has_grad = any(
            p.grad is not None for p in flow_model.repa_loss.encoder.parameters()
        )
        assert not encoder_has_grad

    def test_time_weighting(self, mace_encoder, projector, hidden_dim, molecular_batch):
        """Time weighting should scale loss: higher t -> higher |loss|."""
        repa_with_weight = REPALoss(
            encoder=mace_encoder,
            projector=projector,
            lambda_repa=1.0,
            time_weighting=True,
        )

        coords = molecular_batch["coords"]
        atomics = molecular_batch["atomics"]
        padding_mask = molecular_batch["padding_mask"]
        batch_size = coords.shape[0]
        num_atoms = coords.shape[1]
        hidden_states = torch.randn(batch_size, num_atoms, hidden_dim)

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=batch_size,
        )
        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": hidden_states,
                "padding_mask": padding_mask,
            },
            batch_size=batch_size,
        )

        t_low = torch.tensor([0.1, 0.1])
        path_low = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t_low)
        loss_low, _ = repa_with_weight(path_low, pred, compute_stats=False)

        t_high = torch.tensor([0.9, 0.9])
        path_high = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t_high)
        loss_high, _ = repa_with_weight(path_high, pred, compute_stats=False)

        assert abs(loss_high.item()) > abs(loss_low.item())

        expected_ratio = t_high.mean().item() / t_low.mean().item()
        actual_ratio = abs(loss_high.item()) / abs(loss_low.item())
        assert abs(actual_ratio - expected_ratio) < 0.1
