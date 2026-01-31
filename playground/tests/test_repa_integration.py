"""Tests for REPA integration into TABASCO."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tensordict import TensorDict

from tabasco.models.components.transformer_module import TransformerModule
from tabasco.models.components.encoders import ChemPropEncoder, DummyEncoder, Projector
from tabasco.models.components.losses import REPALoss
from tabasco.models.flow_model import FlowMatchingModel
from tabasco.flow.path import FlowPath
from tabasco.flow.interpolate import CenteredMetricInterpolant, DiscreteInterpolant
from tabasco.chem.convert import MoleculeConverter


class TestREPAIntegration:
    """Tests for REPA loss integration."""

    @pytest.fixture
    def hidden_dim(self):
        return 64

    @pytest.fixture
    def atom_dim(self):
        return 9  # Matches len(ATOM_NAMES) in tabasco.chem.constants

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
    def dummy_encoder(self):
        return DummyEncoder(input_dim=3, hidden_dim=64, encoder_dim=128)

    @pytest.fixture
    def chemprop_encoder(self):
        return ChemPropEncoder(pretrained="chemeleon")

    @pytest.fixture
    def encoder(self, chemprop_encoder):
        """Default encoder for tests."""
        return chemprop_encoder

    @pytest.fixture
    def projector(self, hidden_dim, encoder):
        return Projector(
            hidden_dim=hidden_dim, encoder_dim=encoder.encoder_dim, num_layers=2
        )

    @pytest.fixture
    def repa_loss(self, encoder, projector):
        return REPALoss(
            encoder=encoder,
            projector=projector,
            lambda_repa=0.5,
            time_weighting=False,
        )

    @pytest.fixture
    def molecular_batch(self):
        """Create a batch of real molecular data from SMILES."""
        converter = MoleculeConverter()
        smiles_list = ["CCO", "CC(=O)O"]  # Ethanol, Acetic acid
        max_atoms = 10

        tensordicts = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol = Chem.RemoveAllHs(mol)
            td = converter.to_tensor(mol, pad_to_size=max_atoms)
            tensordicts.append(td)

        return TensorDict(
            {
                "coords": torch.stack([td["coords"] for td in tensordicts]),
                "atomics": torch.stack([td["atomics"] for td in tensordicts]),
                "padding_mask": torch.stack([td["padding_mask"] for td in tensordicts]),
            },
            batch_size=[len(smiles_list)],
        )

    @pytest.fixture
    def batch_data(self, hidden_dim, molecular_batch):
        """Create batch data for testing with real molecules."""
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
                "hidden_states": hidden_states,
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
        """Create a FlowMatchingModel with REPA loss."""
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

    def test_return_hidden_states(self, transformer, hidden_dim, atom_dim):
        """Test that TransformerModule returns hidden states when requested."""
        batch_size, num_atoms = 2, 10
        coords = torch.randn(batch_size, num_atoms, 3)
        atomics = torch.nn.functional.softmax(
            torch.randn(batch_size, num_atoms, atom_dim), dim=-1
        )
        padding_mask = torch.zeros(batch_size, num_atoms, dtype=torch.bool)
        padding_mask[:, 8:] = True
        t = torch.rand(batch_size)

        # Without hidden states
        output = transformer(
            coords, atomics, padding_mask, t, return_hidden_states=False
        )
        assert len(output) == 2

        # With hidden states
        output = transformer(
            coords, atomics, padding_mask, t, return_hidden_states=True
        )
        assert len(output) == 3
        _, _, hidden_states = output
        assert hidden_states.shape == (batch_size, num_atoms, hidden_dim)

    def test_encoder_params_frozen(self, repa_loss):
        """Test that encoder parameters are frozen (requires_grad=False)."""
        for param in repa_loss.encoder.parameters():
            assert not param.requires_grad

    def test_projector_params_trainable_with_gradients(self, repa_loss, batch_data):
        """Test that projector parameters are trainable and receive gradients."""
        # Verify projector params are trainable
        for param in repa_loss.projector.parameters():
            assert param.requires_grad

        # Compute loss and backprop
        loss, _ = repa_loss(batch_data["path"], batch_data["pred"], compute_stats=True)
        loss.backward()

        # Verify hidden states receive gradients
        assert batch_data["hidden_states"].grad is not None
        assert batch_data["hidden_states"].grad.abs().sum() > 0

        # Verify projector receives gradients
        projector_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in repa_loss.projector.parameters()
        )
        assert projector_has_grad

        # Verify encoder does NOT receive gradients
        encoder_has_grad = any(
            p.grad is not None for p in repa_loss.encoder.parameters()
        )
        assert not encoder_has_grad

    def test_time_weighting(self, encoder, projector, hidden_dim, molecular_batch):
        """Test that time weighting scales loss correctly.

        At t~1 (clean molecules), loss magnitude should be higher than at t~0 (noisy).
        The ratio of |loss| values should match the ratio of time values.
        """
        repa_with_weight = REPALoss(
            encoder, projector, lambda_repa=1.0, time_weighting=True
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
                "hidden_states": hidden_states,
                "padding_mask": padding_mask,
            },
            batch_size=batch_size,
        )

        # Low time (noisy)
        t_low = torch.tensor([0.1, 0.1])
        path_low = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t_low)
        loss_low, _ = repa_with_weight(path_low, pred, compute_stats=False)

        # High time (clean)
        t_high = torch.tensor([0.9, 0.9])
        path_high = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t_high)
        loss_high, _ = repa_with_weight(path_high, pred, compute_stats=False)

        # |loss| at t~1 should be greater than at t~0
        assert abs(loss_high.item()) > abs(loss_low.item())

        # Ratio should match time ratio
        expected_ratio = t_high.mean().item() / t_low.mean().item()
        actual_ratio = abs(loss_high.item()) / abs(loss_low.item())
        assert abs(actual_ratio - expected_ratio) < 0.1

    def test_flow_model_gradients_with_repa(self, flow_model, molecular_batch):
        """Test that flow model params are trainable and encoder is frozen."""
        # Forward pass
        loss, _ = flow_model(molecular_batch, compute_stats=True)
        loss.backward()

        # Core network should have gradients
        net_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in flow_model.net.parameters()
        )
        assert net_has_grad

        # Projector should have gradients
        projector_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in flow_model.repa_loss.projector.parameters()
        )
        assert projector_has_grad

        # Encoder should not have gradients
        encoder_has_grad = any(
            p.grad is not None for p in flow_model.repa_loss.encoder.parameters()
        )
        assert not encoder_has_grad
