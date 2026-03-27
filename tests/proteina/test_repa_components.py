"""Tests for Proteina REPA components.

Tests the individual REPA building blocks (encoder, transformer, loss, projector)
with synthetic data. Does NOT require real PDB data or GearNet checkpoint.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

# Add proteina to path
proteina_root = os.path.join(os.path.dirname(__file__), "../../src/proteina")
sys.path.insert(0, proteina_root)

# Patch torch_scatter with native PyTorch ops if C extensions don't work.
# This must happen before any proteina imports.
try:
    import proteinfoundation.repa.pyg_compat  # noqa: F401
except Exception:
    pass

# Check if proteina's heavy dependencies are available and functional.
try:
    from torch_scatter import scatter_mean  # noqa: F401 (may be patched)
    import einops  # noqa: F401

    HAS_PROTEINA_DEPS = True
except (ImportError, OSError):
    HAS_PROTEINA_DEPS = False


# ─── Test ProteinTransformerAF3WithHiddenStates ─────────────────────────────


needs_proteina_deps = pytest.mark.skipif(
    not HAS_PROTEINA_DEPS,
    reason="Proteina dependencies not installed (run: make setup-proteina)",
)


@needs_proteina_deps
class TestTransformerHiddenStates:
    """Test that the transformer subclass correctly extracts hidden states."""

    @pytest.fixture
    def nn_kwargs(self):
        """Minimal 60M-like config for testing (smaller dimensions)."""
        return dict(
            token_dim=64,
            nlayers=4,
            nheads=4,
            residual_mha=True,
            residual_transition=True,
            parallel_mha_transition=False,
            use_attn_pair_bias=False,
            strict_feats=False,
            feats_init_seq=["x_sc"],
            feats_cond_seq=["time_emb"],
            t_emb_dim=32,
            dim_cond=64,
            idx_emb_dim=32,
            fold_emb_dim=32,
            feats_pair_repr=[],
            feats_pair_cond=[],
            pair_repr_dim=32,
            xt_pair_dist_dim=16,
            xt_pair_dist_min=0.1,
            xt_pair_dist_max=3.0,
            x_sc_pair_dist_dim=16,
            x_sc_pair_dist_min=0.1,
            x_sc_pair_dist_max=3.0,
            x_motif_pair_dist_dim=16,
            x_motif_pair_dist_min=0.1,
            x_motif_pair_dist_max=3.0,
            seq_sep_dim=127,
            update_pair_repr=False,
            update_pair_repr_every_n=2,
            use_tri_mult=False,
            num_registers=0,
            use_qkln=False,
            num_buckets_predict_pair=None,
            cath_code_dir=".",
            multilabel_mode="sample",
        )

    @pytest.fixture
    def batch_nn(self):
        """Synthetic batch for testing."""
        b, n = 2, 16
        return {
            "x_t": torch.randn(b, n, 3),
            "t": torch.rand(b),
            "mask": torch.ones(b, n, dtype=torch.bool),
        }

    def test_without_hidden_states(self, nn_kwargs, batch_nn):
        """Standard forward should work without hidden states."""
        from proteinfoundation.repa.protein_transformer_repa import (
            ProteinTransformerAF3WithHiddenStates,
        )

        model = ProteinTransformerAF3WithHiddenStates(repa_layers=[1, 2], **nn_kwargs)
        nn_out = model(batch_nn, return_hidden_states=False)

        assert "coors_pred" in nn_out
        assert nn_out["coors_pred"].shape == (2, 16, 3)
        assert "hidden_states" not in nn_out

    def test_with_hidden_states(self, nn_kwargs, batch_nn):
        """Should return hidden states at specified layers."""
        from proteinfoundation.repa.protein_transformer_repa import (
            ProteinTransformerAF3WithHiddenStates,
        )

        repa_layers = [1, 3]
        model = ProteinTransformerAF3WithHiddenStates(
            repa_layers=repa_layers, **nn_kwargs
        )
        nn_out = model(batch_nn, return_hidden_states=True)

        assert "hidden_states" in nn_out
        assert len(nn_out["hidden_states"]) == 2
        for h in nn_out["hidden_states"]:
            assert h.shape == (2, 16, 64)  # [b, n, token_dim]

    def test_hidden_states_have_gradients(self, nn_kwargs, batch_nn):
        """Hidden states must retain gradients for REPA loss backprop."""
        from proteinfoundation.repa.protein_transformer_repa import (
            ProteinTransformerAF3WithHiddenStates,
        )

        model = ProteinTransformerAF3WithHiddenStates(repa_layers=[2], **nn_kwargs)
        nn_out = model(batch_nn, return_hidden_states=True)

        h = nn_out["hidden_states"][0]
        loss = h.sum()
        loss.backward()

        # Check gradients flow back to model parameters
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "Gradients should flow through hidden states"

    def test_coors_pred_unchanged(self, nn_kwargs, batch_nn):
        """Extracting hidden states should not change coordinate predictions."""
        from proteinfoundation.repa.protein_transformer_repa import (
            ProteinTransformerAF3WithHiddenStates,
        )

        model = ProteinTransformerAF3WithHiddenStates(repa_layers=[1], **nn_kwargs)
        model.eval()

        with torch.no_grad():
            out_no_hs = model(batch_nn, return_hidden_states=False)
            out_with_hs = model(batch_nn, return_hidden_states=True)

        torch.testing.assert_close(out_no_hs["coors_pred"], out_with_hs["coors_pred"])


# ─── Test Projector ─────────────────────────────────────────────────────────


class TestProjector:
    def test_output_shape(self):
        from proteinfoundation.repa.repa_loss import Projector

        proj = Projector(hidden_dim=256, encoder_dim=512, num_layers=2)
        x = torch.randn(2, 16, 64)  # LazyLinear infers input dim
        out = proj(x)
        assert out.shape == (2, 16, 512)

    def test_gradient_flow(self):
        from proteinfoundation.repa.repa_loss import Projector

        proj = Projector(hidden_dim=128, encoder_dim=256, num_layers=3)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = proj(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ─── Test REPA Loss (with mock encoder) ─────────────────────────────────────


class MockEncoder(nn.Module):
    """Mock encoder that returns random per-residue features."""

    def __init__(self, encoder_dim=512):
        super().__init__()
        self.encoder_dim = encoder_dim

    @torch.no_grad()
    def forward(self, ca_coords_nm, mask):
        b, n, _ = ca_coords_nm.shape
        features = torch.randn(b, n, self.encoder_dim, device=ca_coords_nm.device)
        features = features * mask[..., None].float()
        return features


class TestREPALoss:
    def test_loss_computation(self):
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        encoder_dim = 64
        token_dim = 32
        b, n = 2, 16
        repa_layers = [2, 4]

        encoder = MockEncoder(encoder_dim=encoder_dim)
        projectors = nn.ModuleList(
            [
                Projector(hidden_dim=token_dim, encoder_dim=encoder_dim)
                for _ in repa_layers
            ]
        )

        loss_fn = ProteinaREPALoss(
            encoder=encoder,
            projectors=projectors,
            repa_layers=repa_layers,
            lambda_repa=0.5,
            combination_mode="additive",
            similarity_type="cosine",
        )

        hidden_states = [torch.randn(b, n, token_dim) for _ in repa_layers]
        x_1_nm = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)

        repa_loss, stats = loss_fn(hidden_states, x_1_nm, mask)

        assert repa_loss.shape == ()
        assert torch.isfinite(repa_loss)
        assert "repa/loss" in stats
        assert "repa/cos_sim_layer_2" in stats

    def test_gradient_through_projectors(self):
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        encoder_dim = 64
        token_dim = 32
        b, n = 2, 16
        repa_layers = [1]

        encoder = MockEncoder(encoder_dim=encoder_dim)
        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=encoder_dim)]
        )

        loss_fn = ProteinaREPALoss(
            encoder=encoder,
            projectors=projectors,
            repa_layers=repa_layers,
        )

        hidden_states = [torch.randn(b, n, token_dim, requires_grad=True)]
        x_1_nm = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)

        repa_loss, _ = loss_fn(hidden_states, x_1_nm, mask)
        repa_loss.backward()

        # Gradients should flow through projectors
        assert hidden_states[0].grad is not None
        for p in projectors.parameters():
            assert p.grad is not None

    def test_masking(self):
        """REPA loss should respect the residue mask."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        encoder_dim = 32
        token_dim = 16
        b, n = 2, 16
        repa_layers = [0]

        encoder = MockEncoder(encoder_dim=encoder_dim)
        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=encoder_dim)]
        )

        loss_fn = ProteinaREPALoss(
            encoder=encoder,
            projectors=projectors,
            repa_layers=repa_layers,
        )

        hidden_states = [torch.randn(b, n, token_dim)]
        x_1_nm = torch.randn(b, n, 3)

        # Mask out half the residues
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[:, n // 2 :] = False

        repa_loss, _ = loss_fn(hidden_states, x_1_nm, mask)
        assert torch.isfinite(repa_loss)


# ─── Test GearNetPerResidueEncoder (format conversion only, no checkpoint) ──


class TestGearNetEncoderFormatConversion:
    """Test the dense-to-flat conversion logic without requiring a checkpoint.

    These tests replicate the conversion logic inline since importing
    GearNetPerResidueEncoder requires torch_geometric.
    """

    def test_dense_to_gearnet_inputs(self):
        b, n = 2, 20
        ca_coords_ang = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[0, 15:] = False  # First protein has 15 residues
        mask[1, 18:] = False  # Second protein has 18 residues

        # Replicate the conversion logic from the encoder
        device = ca_coords_ang.device
        batch_ids = torch.arange(b, device=device)[:, None].expand(b, n)
        flat_coords = ca_coords_ang.reshape(b * n, 3)
        flat_batch = batch_ids.reshape(b * n)
        flat_mask = mask.reshape(b * n)
        residue_idx = torch.arange(n, device=device)[None, :].expand(b, n)
        flat_residue_idx = residue_idx.reshape(b * n)

        valid = flat_mask.bool()
        coords = flat_coords[valid]
        atom2batch = flat_batch[valid]
        atom_seq_pos = flat_residue_idx[valid]

        # Should have 15 + 18 = 33 valid atoms
        assert coords.shape == (33, 3)
        assert atom2batch.shape == (33,)
        assert atom_seq_pos.shape == (33,)

        # First protein's residues should be batch 0
        assert (atom2batch[:15] == 0).all()
        # Second protein's residues should be batch 1
        assert (atom2batch[15:] == 1).all()

        # Residue positions should be sequential within each protein
        assert (atom_seq_pos[:15] == torch.arange(15)).all()
        assert (atom_seq_pos[15:] == torch.arange(18)).all()

    def test_scatter_to_dense(self):
        """Test scattering flat features back to dense format."""
        b, n, dim = 2, 20, 64
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[0, 15:] = False
        mask[1, 18:] = False

        total_atoms = 15 + 18
        h_v = torch.randn(total_atoms, dim)

        # Replicate scatter logic
        flat_mask = mask.reshape(b * n).bool()
        dense_idx = torch.arange(b * n)[flat_mask]
        batch_idx = dense_idx // n
        res_idx = dense_idx % n

        output = torch.zeros(b, n, dim)
        output[batch_idx, res_idx] = h_v

        # Masked positions should be zero
        assert (output[0, 15:] == 0).all()
        assert (output[1, 18:] == 0).all()
        # Valid positions should match h_v
        assert (output[0, :15] == h_v[:15]).all()
        assert (output[1, :18] == h_v[15:]).all()
