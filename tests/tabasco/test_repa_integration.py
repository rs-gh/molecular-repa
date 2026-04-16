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
            encoder=encoder, projector=projector, lambda_repa=1.0, time_weighting=True
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

    def test_invalid_combination_mode_raises(self, dummy_encoder, hidden_dim):
        """Test that an invalid combination_mode raises ValueError."""
        projector = Projector(
            hidden_dim=hidden_dim, encoder_dim=dummy_encoder.encoder_dim
        )
        with pytest.raises(ValueError, match="combination_mode"):
            REPALoss(dummy_encoder, projector, combination_mode="bad_mode")

    def test_default_combination_mode_is_additive(self, dummy_encoder, hidden_dim):
        """Test that the default combination_mode is 'additive'."""
        projector = Projector(
            hidden_dim=hidden_dim, encoder_dim=dummy_encoder.encoder_dim
        )
        loss = REPALoss(dummy_encoder, projector)
        assert loss.combination_mode == "additive"

    def test_combination_mode_formula(
        self, transformer, dummy_encoder, hidden_dim, molecular_batch
    ):
        """Test that additive and tradeoff modes implement their formulas correctly.

        With shared net/encoder/projector and the same random seed, component losses
        are identical across both modes. Therefore:
            loss_additive - loss_tradeoff = lam * diffusion_loss
        """
        projector = Projector(
            hidden_dim=hidden_dim, encoder_dim=dummy_encoder.encoder_dim
        )
        lam = 0.5

        def make_model(mode):
            rl = REPALoss(
                dummy_encoder, projector, lambda_repa=lam, combination_mode=mode
            )
            return FlowMatchingModel(
                net=transformer,
                coords_interpolant=CenteredMetricInterpolant(
                    key="coords", key_pad_mask="padding_mask"
                ),
                atomics_interpolant=DiscreteInterpolant(
                    key="atomics", key_pad_mask="padding_mask"
                ),
                repa_loss=rl,
            )

        additive_model = make_model("additive")
        tradeoff_model = make_model("tradeoff")

        torch.manual_seed(42)
        loss_a, stats_a = additive_model(molecular_batch, compute_stats=True)
        torch.manual_seed(42)
        loss_t, _ = tradeoff_model(molecular_batch, compute_stats=True)

        # No interdist_loss in either model, so diffusion = atomics + coords
        diffusion = stats_a["atomics_loss"] + stats_a["coords_loss"]
        expected_diff = lam * diffusion
        actual_diff = loss_a.item() - loss_t.item()
        assert abs(actual_diff - expected_diff) < 1e-5

    def test_tradeoff_lambda_weighting(
        self, transformer, dummy_encoder, hidden_dim, molecular_batch
    ):
        """Verify tradeoff formula: total = (1-λ)·D + λ·R, for λ=0.8 and λ=0.2.

        Higher λ shifts weight from diffusion to REPA. The direction of the difference
        in total loss depends on sign(R - D), so we assert the formula directly.
        Note: R (repa_loss) may be negative when representations are positively aligned
        (cosine similarity), so directional assertions on the totals are not meaningful.
        """
        projector = Projector(
            hidden_dim=hidden_dim, encoder_dim=dummy_encoder.encoder_dim
        )

        def make_model(lam):
            rl = REPALoss(
                dummy_encoder, projector, lambda_repa=lam, combination_mode="tradeoff"
            )
            return FlowMatchingModel(
                net=transformer,
                coords_interpolant=CenteredMetricInterpolant(
                    key="coords", key_pad_mask="padding_mask"
                ),
                atomics_interpolant=DiscreteInterpolant(
                    key="atomics", key_pad_mask="padding_mask"
                ),
                repa_loss=rl,
            )

        model_08 = make_model(0.8)
        model_02 = make_model(0.2)

        torch.manual_seed(42)
        total_08, stats_08 = model_08(molecular_batch, compute_stats=True)
        torch.manual_seed(42)
        total_02, _ = model_02(molecular_batch, compute_stats=True)

        D = stats_08["atomics_loss"] + stats_08["coords_loss"]
        R = stats_08["repa_loss"]  # negative when representations are aligned (cosine)

        assert abs(total_08.item() - (0.2 * D + 0.8 * R)) < 1e-5
        assert abs(total_02.item() - (0.8 * D + 0.2 * R)) < 1e-5
        # Difference isolates the weighting shift: 0.6*(R - D)
        assert abs((total_08.item() - total_02.item()) - 0.6 * (R - D)) < 1e-5

    def test_additive_lambda_weighting(
        self, transformer, dummy_encoder, hidden_dim, molecular_batch
    ):
        """Verify additive formula: total = D + λ·R, for λ=0.8 and λ=0.2.

        REPA loss (R) may be negative (negative cosine similarity), so the total is not
        guaranteed to increase with λ. The difference total_08 - total_02 = 0.6·R exactly,
        regardless of sign.
        """
        projector = Projector(
            hidden_dim=hidden_dim, encoder_dim=dummy_encoder.encoder_dim
        )

        def make_model(lam):
            rl = REPALoss(
                dummy_encoder, projector, lambda_repa=lam, combination_mode="additive"
            )
            return FlowMatchingModel(
                net=transformer,
                coords_interpolant=CenteredMetricInterpolant(
                    key="coords", key_pad_mask="padding_mask"
                ),
                atomics_interpolant=DiscreteInterpolant(
                    key="atomics", key_pad_mask="padding_mask"
                ),
                repa_loss=rl,
            )

        model_08 = make_model(0.8)
        model_02 = make_model(0.2)

        torch.manual_seed(42)
        total_08, stats_08 = model_08(molecular_batch, compute_stats=True)
        torch.manual_seed(42)
        total_02, _ = model_02(molecular_batch, compute_stats=True)

        D = stats_08["atomics_loss"] + stats_08["coords_loss"]
        R = stats_08["repa_loss"]  # negative when representations are aligned (cosine)

        assert abs(total_08.item() - (D + 0.8 * R)) < 1e-5
        assert abs(total_02.item() - (D + 0.2 * R)) < 1e-5
        # Difference is exactly 0.6·R, regardless of sign
        assert abs((total_08.item() - total_02.item()) - 0.6 * R) < 1e-5


class TestCrossAttentionFusion:
    """Tests for cross_attention=True: fused hidden states through single projector."""

    @pytest.fixture
    def hidden_dim(self):
        return 64

    @pytest.fixture
    def atom_dim(self):
        return 9

    @pytest.fixture
    def transformer_cross_attn(self, hidden_dim, atom_dim):
        return TransformerModule(
            spatial_dim=3,
            atom_dim=atom_dim,
            num_heads=4,
            num_layers=2,
            hidden_dim=hidden_dim,
            implementation="pytorch",
            cross_attention=True,
        )

    @pytest.fixture
    def dummy_encoder(self):
        return DummyEncoder(input_dim=3, hidden_dim=64, encoder_dim=128)

    @pytest.fixture
    def fused_projector(self, hidden_dim, dummy_encoder):
        """Projector for fused input. LazyLinear infers input dim automatically."""
        return Projector(hidden_dim=hidden_dim, encoder_dim=dummy_encoder.encoder_dim)

    @pytest.fixture
    def repa_loss_fused(self, dummy_encoder, fused_projector):
        return REPALoss(
            encoder=dummy_encoder,
            projector=fused_projector,
            lambda_repa=0.5,
            time_weighting=False,
        )

    @pytest.fixture
    def molecular_batch(self):
        """Create a batch of real molecular data from SMILES."""
        converter = MoleculeConverter()
        smiles_list = ["CCO", "CC(=O)O"]
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

    def test_cross_attn_returns_four_outputs(
        self, transformer_cross_attn, hidden_dim, atom_dim
    ):
        """TransformerModule with cross_attention=True returns h_coord and h_atom."""
        batch_size, num_atoms = 2, 10
        coords = torch.randn(batch_size, num_atoms, 3)
        atomics = torch.nn.functional.softmax(
            torch.randn(batch_size, num_atoms, atom_dim), dim=-1
        )
        padding_mask = torch.zeros(batch_size, num_atoms, dtype=torch.bool)
        padding_mask[:, 8:] = True
        t = torch.rand(batch_size)

        # Without hidden states — still 2 outputs
        output = transformer_cross_attn(
            coords, atomics, padding_mask, t, return_hidden_states=False
        )
        assert len(output) == 2

        # With hidden states — 4 outputs (coords, atom_logits, h_coord, h_atom)
        output = transformer_cross_attn(
            coords, atomics, padding_mask, t, return_hidden_states=True
        )
        assert len(output) == 4
        _, _, h_coord, h_atom = output
        assert h_coord.shape == (batch_size, num_atoms, hidden_dim)
        assert h_atom.shape == (batch_size, num_atoms, hidden_dim)

    def test_fusion_concatenation_in_repa_loss(
        self, dummy_encoder, fused_projector, hidden_dim, molecular_batch
    ):
        """REPALoss concatenates hidden_states_coord and hidden_states_atom."""
        coords = molecular_batch["coords"]
        atomics = molecular_batch["atomics"]
        padding_mask = molecular_batch["padding_mask"]
        batch_size = coords.shape[0]
        num_atoms = coords.shape[1]

        h_coord = torch.randn(batch_size, num_atoms, hidden_dim)
        h_atom = torch.randn(batch_size, num_atoms, hidden_dim)

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=batch_size,
        )
        t = torch.rand(batch_size)
        path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)

        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": h_coord,
                "hidden_states_atom": h_atom,
                "padding_mask": padding_mask,
            },
            batch_size=batch_size,
        )

        repa = REPALoss(
            encoder=dummy_encoder,
            projector=fused_projector,
            lambda_repa=0.5,
        )
        loss, stats = repa(path, pred, compute_stats=True)

        assert loss.dim() == 0  # scalar
        assert "repa_loss" in stats

    def test_gradient_flow_through_both_heads(
        self, dummy_encoder, fused_projector, hidden_dim, molecular_batch
    ):
        """Gradients flow from REPA loss back through both hidden state heads."""
        coords = molecular_batch["coords"]
        atomics = molecular_batch["atomics"]
        padding_mask = molecular_batch["padding_mask"]
        batch_size = coords.shape[0]
        num_atoms = coords.shape[1]

        h_coord = torch.randn(batch_size, num_atoms, hidden_dim, requires_grad=True)
        h_atom = torch.randn(batch_size, num_atoms, hidden_dim, requires_grad=True)

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=batch_size,
        )
        t = torch.rand(batch_size)
        path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)

        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": h_coord,
                "hidden_states_atom": h_atom,
                "padding_mask": padding_mask,
            },
            batch_size=batch_size,
        )

        repa = REPALoss(
            encoder=dummy_encoder,
            projector=fused_projector,
            lambda_repa=0.5,
        )
        loss, _ = repa(path, pred)
        loss.backward()

        # Both heads must receive gradients
        assert h_coord.grad is not None
        assert h_coord.grad.abs().sum() > 0
        assert h_atom.grad is not None
        assert h_atom.grad.abs().sum() > 0

        # Projector trainable, encoder frozen
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in repa.projector.parameters()
        )
        assert not any(p.grad is not None for p in repa.encoder.parameters())

    def test_flow_model_end_to_end_cross_attention(
        self, transformer_cross_attn, repa_loss_fused, molecular_batch
    ):
        """Full forward + backward through FlowMatchingModel with cross_attention=True."""
        model = FlowMatchingModel(
            net=transformer_cross_attn,
            coords_interpolant=CenteredMetricInterpolant(
                key="coords", key_pad_mask="padding_mask"
            ),
            atomics_interpolant=DiscreteInterpolant(
                key="atomics", key_pad_mask="padding_mask"
            ),
            repa_loss=repa_loss_fused,
        )

        loss, stats = model(molecular_batch, compute_stats=True)
        assert "repa_loss" in stats

        loss.backward()

        # Transformer backbone receives gradients
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.net.parameters()
        )

        # Both cross-attention heads receive gradients
        coord_ca_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.net.coord_cross_attention.parameters()
        )
        atom_ca_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.net.atom_cross_attention.parameters()
        )
        assert coord_ca_has_grad, "coord cross-attention head has no gradients"
        assert atom_ca_has_grad, "atom cross-attention head has no gradients"

        # Projector trainable, encoder frozen
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.repa_loss.projector.parameters()
        )
        assert not any(p.grad is not None for p in model.repa_loss.encoder.parameters())

    def test_single_head_no_fusion(self, dummy_encoder, hidden_dim, molecular_batch):
        """With only hidden_states_coord (no cross_attention), no fusion occurs."""
        # Projector with hidden_dim (not 2*hidden_dim) — single head
        projector = Projector(
            hidden_dim=hidden_dim, encoder_dim=dummy_encoder.encoder_dim
        )
        repa = REPALoss(encoder=dummy_encoder, projector=projector)

        coords = molecular_batch["coords"]
        atomics = molecular_batch["atomics"]
        padding_mask = molecular_batch["padding_mask"]
        batch_size = coords.shape[0]
        num_atoms = coords.shape[1]

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=batch_size,
        )
        t = torch.rand(batch_size)
        path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)

        # Only hidden_states_coord — same as cross_attention=False
        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": torch.randn(batch_size, num_atoms, hidden_dim),
                "padding_mask": padding_mask,
            },
            batch_size=batch_size,
        )

        loss, stats = repa(path, pred, compute_stats=True)
        assert loss.dim() == 0
        assert "repa_loss" in stats


# ═══════════════════════════════════════════════════════════════════════════
# REPA Pipeline Audit Tests
# Verify correctness against the original REPA paper (arXiv 2410.06940)
# Reference code: https://github.com/sihyun-yu/REPA
# ═══════════════════════════════════════════════════════════════════════════


def _reference_repa_loss(projected, target, mask):
    """Exact reimplementation of the original REPA paper's loss computation.

    From https://github.com/sihyun-yu/REPA/blob/main/loss.py:
        z_tilde_j = F.normalize(z_tilde_j, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

    Key: mean_flat averages over spatial dims PER SAMPLE, then averages over batch.
    """
    import torch.nn.functional as F

    b = projected.shape[0]
    total = torch.tensor(0.0, device=projected.device)
    for i in range(b):
        m = mask[i]
        if not m.any():
            continue
        z_tilde = F.normalize(projected[i, m], dim=-1)
        z = F.normalize(target[i, m], dim=-1)
        total = total + (-(z * z_tilde).sum(dim=-1)).mean()
    return total / b


class TestTabascoReferenceEquivalence:
    """Verify tabasco REPA loss matches the paper's exact computation."""

    @pytest.fixture
    def make_batch(self):
        """Create a synthetic FlowPath + pred TensorDict."""

        def _make(b, n, hidden_dim, encoder_dim, atom_dim=9, mask=None):
            coords = torch.randn(b, n, 3)
            atomics = torch.zeros(b, n, atom_dim)
            atomics[:, :, 0] = 1.0  # All carbon
            if mask is None:
                padding_mask = torch.zeros(b, n, dtype=torch.bool)
            else:
                padding_mask = ~mask  # REPALoss expects padding_mask (True=padded)

            x_1 = TensorDict(
                {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
                batch_size=b,
            )
            t = torch.rand(b)
            path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)

            pred = TensorDict(
                {
                    "coords": coords,
                    "atomics": atomics,
                    "hidden_states_coord": torch.randn(b, n, hidden_dim),
                    "padding_mask": padding_mask,
                },
                batch_size=b,
            )
            return path, pred

        return _make

    def test_per_sample_matches_reference_equal_lengths(self, make_batch):
        """With equal-length molecules, per_sample averaging matches reference."""
        torch.manual_seed(42)
        b, n, hidden_dim, encoder_dim = 4, 9, 64, 128

        encoder = DummyEncoder(encoder_dim=encoder_dim)
        torch.manual_seed(0)
        projector = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim)

        repa = REPALoss(
            encoder=encoder,
            projector=projector,
            averaging="per_sample",
            time_weighting=False,
        )

        path, pred = make_batch(b, n, hidden_dim, encoder_dim)
        loss, _ = repa(path, pred)

        # Manually compute reference
        with torch.no_grad():
            h_fused = pred["hidden_states_coord"]
            projected = projector(h_fused)
            target_repr = encoder(
                path.x_1["coords"], path.x_1["atomics"], pred["padding_mask"]
            )
            real_mask = ~pred["padding_mask"]

        ref_loss = _reference_repa_loss(projected, target_repr, real_mask)
        torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-5)

    def test_per_sample_matches_reference_variable_lengths(self, make_batch):
        """With variable-length molecules, per_sample should still match reference."""
        torch.manual_seed(123)
        b, n, hidden_dim, encoder_dim = 3, 15, 64, 128

        encoder = DummyEncoder(encoder_dim=encoder_dim)
        torch.manual_seed(0)
        projector = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim)

        # Variable lengths: 5, 10, 15
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[0, 5:] = False
        mask[1, 10:] = False

        repa = REPALoss(
            encoder=encoder,
            projector=projector,
            averaging="per_sample",
            time_weighting=False,
        )

        path, pred = make_batch(b, n, hidden_dim, encoder_dim, mask=mask)
        loss, _ = repa(path, pred)

        with torch.no_grad():
            h_fused = pred["hidden_states_coord"]
            projected = projector(h_fused)
            target_repr = encoder(
                path.x_1["coords"], path.x_1["atomics"], pred["padding_mask"]
            )
            real_mask = ~pred["padding_mask"]

        ref_loss = _reference_repa_loss(projected, target_repr, real_mask)
        torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-5)


class TestTabascoPerSampleVsPerAtomAveraging:
    """Document the difference between per_sample and per_atom averaging."""

    def test_equal_lengths_same_result(self):
        """With equal-length molecules, both averaging modes should agree."""
        torch.manual_seed(99)
        b, n, hidden_dim, encoder_dim = 3, 9, 64, 128

        encoder = DummyEncoder(encoder_dim=encoder_dim)

        # Create one projector and materialize its LazyLinear, then copy weights
        torch.manual_seed(0)
        proj_ps = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim)
        proj_ps(torch.randn(1, 1, hidden_dim))  # materialize LazyLinear
        proj_pa = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim)
        proj_pa(torch.randn(1, 1, hidden_dim))  # materialize
        proj_pa.load_state_dict(proj_ps.state_dict())

        repa_ps = REPALoss(
            encoder=encoder,
            projector=proj_ps,
            averaging="per_sample",
            time_weighting=False,
        )
        repa_pa = REPALoss(
            encoder=encoder,
            projector=proj_pa,
            averaging="per_atom",
            time_weighting=False,
        )

        coords = torch.randn(b, n, 3)
        atomics = torch.zeros(b, n, 9)
        atomics[:, :, 0] = 1.0
        padding_mask = torch.zeros(b, n, dtype=torch.bool)

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=b,
        )
        t = torch.rand(b)
        path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)

        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": torch.randn(b, n, hidden_dim),
                "padding_mask": padding_mask,
            },
            batch_size=b,
        )

        l_ps, _ = repa_ps(path, pred)
        l_pa, _ = repa_pa(path, pred)
        torch.testing.assert_close(l_ps, l_pa, atol=1e-5, rtol=1e-5)

    def test_variable_lengths_different_result(self):
        """With different-length molecules, the two modes should diverge."""
        torch.manual_seed(99)
        b, n, hidden_dim, encoder_dim = 2, 20, 64, 128

        encoder = DummyEncoder(encoder_dim=encoder_dim)

        torch.manual_seed(0)
        proj_ps = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim)
        proj_ps(torch.randn(1, 1, hidden_dim))  # materialize
        proj_pa = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim)
        proj_pa(torch.randn(1, 1, hidden_dim))  # materialize
        proj_pa.load_state_dict(proj_ps.state_dict())

        repa_ps = REPALoss(
            encoder=encoder,
            projector=proj_ps,
            averaging="per_sample",
            time_weighting=False,
        )
        repa_pa = REPALoss(
            encoder=encoder,
            projector=proj_pa,
            averaging="per_atom",
            time_weighting=False,
        )

        coords = torch.randn(b, n, 3)
        atomics = torch.zeros(b, n, 9)
        atomics[:, :, 0] = 1.0
        # Lengths 3 vs 20 — very different
        padding_mask = torch.zeros(b, n, dtype=torch.bool)
        padding_mask[0, 3:] = True

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=b,
        )
        t = torch.rand(b)
        path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)

        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": torch.randn(b, n, hidden_dim),
                "padding_mask": padding_mask,
            },
            batch_size=b,
        )

        l_ps, _ = repa_ps(path, pred)
        l_pa, _ = repa_pa(path, pred)

        assert not torch.allclose(l_ps, l_pa, atol=1e-4), (
            f"per_sample ({l_ps.item():.6f}) and per_atom ({l_pa.item():.6f}) "
            "should differ when molecule lengths are unequal"
        )


class TestTabascoProjectorArchitecture:
    """Verify projector MLP structure matches expectations."""

    def test_three_layer_projector(self):
        """3-layer projector (paper default): 3 Linear, 2 SiLU."""
        proj = Projector(hidden_dim=128, encoder_dim=192, num_layers=3)
        # Materialize LazyLinear
        proj(torch.randn(1, 5, 64))

        linears = [m for m in proj.mlp if isinstance(m, torch.nn.Linear)]
        activations = [m for m in proj.mlp if isinstance(m, torch.nn.SiLU)]

        assert len(linears) == 3, f"Expected 3 Linear layers, got {len(linears)}"
        assert (
            len(activations) == 2
        ), f"Expected 2 SiLU activations, got {len(activations)}"

    def test_two_layer_projector(self):
        """2-layer projector: 2 Linear, 1 SiLU."""
        proj = Projector(hidden_dim=128, encoder_dim=192, num_layers=2)
        proj(torch.randn(1, 5, 64))

        linears = [m for m in proj.mlp if isinstance(m, torch.nn.Linear)]
        activations = [m for m in proj.mlp if isinstance(m, torch.nn.SiLU)]

        assert len(linears) == 2, f"Expected 2 Linear layers, got {len(linears)}"
        assert (
            len(activations) == 1
        ), f"Expected 1 SiLU activation, got {len(activations)}"


class TestTabascoMSEMode:
    """Verify the MSE similarity path works correctly."""

    def test_mse_loss_finite_and_positive(self):
        encoder = DummyEncoder(encoder_dim=128)
        projector = Projector(hidden_dim=64, encoder_dim=128)
        repa = REPALoss(
            encoder=encoder,
            projector=projector,
            similarity_type="mse",
            time_weighting=False,
        )

        b, n = 2, 9
        coords = torch.randn(b, n, 3)
        atomics = torch.zeros(b, n, 9)
        atomics[:, :, 0] = 1.0
        padding_mask = torch.zeros(b, n, dtype=torch.bool)

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=b,
        )
        t = torch.rand(b)
        path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)

        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": torch.randn(b, n, 64),
                "padding_mask": padding_mask,
            },
            batch_size=b,
        )

        loss, stats = repa(path, pred)
        assert torch.isfinite(loss)
        assert loss > 0
        assert "repa_loss" in stats

    def test_mse_gradient_flow(self):
        encoder = DummyEncoder(encoder_dim=128)
        projector = Projector(hidden_dim=64, encoder_dim=128)
        repa = REPALoss(
            encoder=encoder,
            projector=projector,
            similarity_type="mse",
            time_weighting=False,
        )

        b, n = 2, 9
        coords = torch.randn(b, n, 3)
        atomics = torch.zeros(b, n, 9)
        atomics[:, :, 0] = 1.0
        padding_mask = torch.zeros(b, n, dtype=torch.bool)
        hs = torch.randn(b, n, 64, requires_grad=True)

        x_1 = TensorDict(
            {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
            batch_size=b,
        )
        t = torch.rand(b)
        path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)

        pred = TensorDict(
            {
                "coords": coords,
                "atomics": atomics,
                "hidden_states_coord": hs,
                "padding_mask": padding_mask,
            },
            batch_size=b,
        )

        loss, _ = repa(path, pred)
        loss.backward()

        assert hs.grad is not None
        for p in projector.parameters():
            assert p.grad is not None
