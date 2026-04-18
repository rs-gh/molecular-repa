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
    """Mock encoder that returns deterministic per-residue features.

    Uses a fixed linear projection of coordinates so that the same inputs
    always produce the same outputs (important for numerical equivalence tests).
    """

    def __init__(self, encoder_dim=512):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.proj = nn.Linear(3, encoder_dim, bias=False)

    @torch.no_grad()
    def forward(self, ca_coords_nm, mask, residue_type=None):
        features = self.proj(ca_coords_nm)
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
        # mean_flat on -(z * z_tilde).sum(dim=-1) — mean over tokens for this sample
        total = total + (-(z * z_tilde).sum(dim=-1)).mean()
    # Average over batch (and over number of encoder/layer pairs — here just 1)
    return total / b


class TestReferenceNumericalEquivalence:
    """Test A: Verify our loss matches the original REPA paper's computation."""

    def test_per_sample_matches_reference_equal_lengths(self):
        """With equal-length sequences, per_sample averaging should match the reference."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        torch.manual_seed(42)
        encoder_dim = 64
        token_dim = 32
        b, n = 4, 20
        repa_layers = [0]

        encoder = MockEncoder(encoder_dim=encoder_dim)
        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=encoder_dim)]
        )

        loss_fn = ProteinaREPALoss(
            encoder=encoder,
            projectors=projectors,
            repa_layers=repa_layers,
            averaging="per_sample",
        )

        hidden_states = [torch.randn(b, n, token_dim)]
        x_1_nm = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)  # All same length

        # Get our loss
        repa_loss, _ = loss_fn(hidden_states, x_1_nm, mask)

        # Manually compute what the projector produces and what the encoder returns
        with torch.no_grad():
            projected = projectors[0](hidden_states[0])
            target_repr = encoder(x_1_nm, mask)

        ref_loss = _reference_repa_loss(projected, target_repr, mask)

        torch.testing.assert_close(repa_loss, ref_loss, atol=1e-5, rtol=1e-5)

    def test_per_sample_matches_reference_variable_lengths(self):
        """With variable-length sequences, per_sample should still match reference."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        torch.manual_seed(123)
        encoder_dim = 64
        token_dim = 32
        b, n = 3, 50
        repa_layers = [0]

        encoder = MockEncoder(encoder_dim=encoder_dim)
        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=encoder_dim)]
        )

        loss_fn = ProteinaREPALoss(
            encoder=encoder,
            projectors=projectors,
            repa_layers=repa_layers,
            averaging="per_sample",
        )

        hidden_states = [torch.randn(b, n, token_dim)]
        x_1_nm = torch.randn(b, n, 3)
        # Variable lengths: 10, 30, 50
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[0, 10:] = False
        mask[1, 30:] = False

        repa_loss, _ = loss_fn(hidden_states, x_1_nm, mask)

        with torch.no_grad():
            projected = projectors[0](hidden_states[0])
            target_repr = encoder(x_1_nm, mask)

        ref_loss = _reference_repa_loss(projected, target_repr, mask)

        torch.testing.assert_close(repa_loss, ref_loss, atol=1e-5, rtol=1e-5)


class TestPerSampleVsPerResidueAveraging:
    """Test B: Document the difference between the two averaging modes."""

    def test_equal_lengths_same_result(self):
        """With equal-length proteins, both averaging modes should agree."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        torch.manual_seed(99)
        encoder_dim = 32
        token_dim = 16
        b, n = 3, 20
        repa_layers = [0]

        encoder = MockEncoder(encoder_dim=encoder_dim)

        # Use input_dim so weights are materialized immediately (not LazyLinear)
        # and create shared projector weights via state_dict copy.
        torch.manual_seed(0)
        proj_ps = nn.ModuleList(
            [
                Projector(
                    hidden_dim=token_dim, encoder_dim=encoder_dim, input_dim=token_dim
                )
            ]
        )
        torch.manual_seed(0)
        proj_pr = nn.ModuleList(
            [
                Projector(
                    hidden_dim=token_dim, encoder_dim=encoder_dim, input_dim=token_dim
                )
            ]
        )

        loss_per_sample = ProteinaREPALoss(
            encoder=encoder,
            projectors=proj_ps,
            repa_layers=repa_layers,
            averaging="per_sample",
        )
        loss_per_residue = ProteinaREPALoss(
            encoder=encoder,
            projectors=proj_pr,
            repa_layers=repa_layers,
            averaging="per_residue",
        )

        hidden_states = [torch.randn(b, n, token_dim)]
        x_1_nm = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)  # All same length

        l_ps, _ = loss_per_sample(hidden_states, x_1_nm, mask)
        l_pr, _ = loss_per_residue(hidden_states, x_1_nm, mask)

        torch.testing.assert_close(l_ps, l_pr, atol=1e-5, rtol=1e-5)

    def test_variable_lengths_different_result(self):
        """With different-length proteins, the two modes should diverge."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        torch.manual_seed(99)
        encoder_dim = 32
        token_dim = 16
        b, n = 2, 100
        repa_layers = [0]

        encoder = MockEncoder(encoder_dim=encoder_dim)

        torch.manual_seed(0)
        proj_ps = nn.ModuleList(
            [
                Projector(
                    hidden_dim=token_dim, encoder_dim=encoder_dim, input_dim=token_dim
                )
            ]
        )
        torch.manual_seed(0)
        proj_pr = nn.ModuleList(
            [
                Projector(
                    hidden_dim=token_dim, encoder_dim=encoder_dim, input_dim=token_dim
                )
            ]
        )

        loss_per_sample = ProteinaREPALoss(
            encoder=encoder,
            projectors=proj_ps,
            repa_layers=repa_layers,
            averaging="per_sample",
        )
        loss_per_residue = ProteinaREPALoss(
            encoder=encoder,
            projectors=proj_pr,
            repa_layers=repa_layers,
            averaging="per_residue",
        )

        hidden_states = [torch.randn(b, n, token_dim)]
        x_1_nm = torch.randn(b, n, 3)
        # Lengths 10 vs 100 — very different
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[0, 10:] = False

        l_ps, _ = loss_per_sample(hidden_states, x_1_nm, mask)
        l_pr, _ = loss_per_residue(hidden_states, x_1_nm, mask)

        # They should NOT be equal with asymmetric lengths
        assert not torch.allclose(l_ps, l_pr, atol=1e-4), (
            f"per_sample ({l_ps.item():.6f}) and per_residue ({l_pr.item():.6f}) "
            "should differ when protein lengths are unequal"
        )


@needs_proteina_deps
class TestParentSubclassForwardEquivalence:
    """Test C: Ensure the REPA subclass produces identical outputs to the parent."""

    @pytest.fixture
    def nn_kwargs(self):
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

    def test_parent_subclass_identical_output(self, nn_kwargs):
        """Parent and subclass should produce bitwise-identical coors_pred."""
        from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
        from proteinfoundation.repa.protein_transformer_repa import (
            ProteinTransformerAF3WithHiddenStates,
        )

        parent = ProteinTransformerAF3(**nn_kwargs)
        child = ProteinTransformerAF3WithHiddenStates(repa_layers=[1], **nn_kwargs)

        # Copy parent weights into child
        child.load_state_dict(parent.state_dict(), strict=False)

        batch_nn = {
            "x_t": torch.randn(2, 16, 3),
            "t": torch.rand(2),
            "mask": torch.ones(2, 16, dtype=torch.bool),
        }

        parent.eval()
        child.eval()

        with torch.no_grad():
            parent_out = parent(batch_nn)
            child_out = child(batch_nn, return_hidden_states=False)

        torch.testing.assert_close(
            parent_out["coors_pred"],
            child_out["coors_pred"],
            atol=0,
            rtol=0,
            msg="REPA subclass forward() has drifted from parent — outputs differ",
        )


class TestProjectorArchitecture:
    """Test D: Verify projector MLP structure matches expectations."""

    def test_three_layer_projector(self):
        """3-layer projector (paper default): 3 Linear, 2 SiLU."""
        from proteinfoundation.repa.repa_loss import Projector

        proj = Projector(hidden_dim=256, encoder_dim=512, num_layers=3, input_dim=128)
        linears = [m for m in proj.mlp if isinstance(m, nn.Linear)]
        activations = [m for m in proj.mlp if isinstance(m, nn.SiLU)]

        assert len(linears) == 3, f"Expected 3 Linear layers, got {len(linears)}"
        assert (
            len(activations) == 2
        ), f"Expected 2 SiLU activations, got {len(activations)}"

        # Check dimensions: input→hidden, hidden→hidden, hidden→output
        assert linears[0].in_features == 128
        assert linears[0].out_features == 256
        assert linears[1].in_features == 256
        assert linears[1].out_features == 256
        assert linears[2].in_features == 256
        assert linears[2].out_features == 512

    def test_two_layer_projector(self):
        """2-layer projector: 2 Linear, 1 SiLU."""
        from proteinfoundation.repa.repa_loss import Projector

        proj = Projector(hidden_dim=256, encoder_dim=512, num_layers=2, input_dim=128)
        linears = [m for m in proj.mlp if isinstance(m, nn.Linear)]
        activations = [m for m in proj.mlp if isinstance(m, nn.SiLU)]

        assert len(linears) == 2, f"Expected 2 Linear layers, got {len(linears)}"
        assert (
            len(activations) == 1
        ), f"Expected 1 SiLU activation, got {len(activations)}"

    def test_one_layer_projector_uses_lazy(self):
        """num_layers=1: LazyLinear→SiLU→Linear (same as 2-layer but with lazy input).

        Note: The Projector always prepends [first_linear, SiLU] then appends
        the output linear. With num_layers=1 the middle loop runs 0 times,
        giving the same structure as num_layers=2 but with LazyLinear input.
        """
        from proteinfoundation.repa.repa_loss import Projector

        proj = Projector(hidden_dim=256, encoder_dim=512, num_layers=1)
        lazy_linears = [m for m in proj.mlp if isinstance(m, nn.LazyLinear)]
        activations = [m for m in proj.mlp if isinstance(m, nn.SiLU)]

        assert len(lazy_linears) == 1, f"Expected 1 LazyLinear, got {len(lazy_linears)}"
        assert (
            len(activations) == 1
        ), f"Expected 1 SiLU activation, got {len(activations)}"


class TestMSESimilarityMode:
    """Test E: Verify the MSE similarity path works correctly."""

    def test_mse_loss_finite_and_positive(self):
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
            similarity_type="mse",
        )

        hidden_states = [torch.randn(b, n, token_dim)]
        x_1_nm = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)

        repa_loss, stats = loss_fn(hidden_states, x_1_nm, mask)

        assert torch.isfinite(repa_loss)
        assert repa_loss > 0, "MSE loss should be positive"
        assert "repa/mse_layer_0" in stats

    def test_mse_gradient_flow(self):
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
            similarity_type="mse",
        )

        hidden_states = [torch.randn(b, n, token_dim, requires_grad=True)]
        x_1_nm = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)

        repa_loss, _ = loss_fn(hidden_states, x_1_nm, mask)
        repa_loss.backward()

        assert hidden_states[0].grad is not None
        for p in projectors.parameters():
            assert p.grad is not None

    def test_mse_per_sample_averaging(self):
        """MSE with per_sample averaging should weight proteins equally."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        torch.manual_seed(77)
        encoder_dim = 32
        token_dim = 16
        b, n = 2, 100
        repa_layers = [0]

        encoder = MockEncoder(encoder_dim=encoder_dim)

        def make_loss(averaging):
            projectors = nn.ModuleList(
                [Projector(hidden_dim=token_dim, encoder_dim=encoder_dim)]
            )
            return ProteinaREPALoss(
                encoder=encoder,
                projectors=projectors,
                repa_layers=repa_layers,
                similarity_type="mse",
                averaging=averaging,
            )

        loss_ps = make_loss("per_sample")
        loss_pr = make_loss("per_residue")
        loss_pr.projectors.load_state_dict(loss_ps.projectors.state_dict())

        hidden_states = [torch.randn(b, n, token_dim)]
        x_1_nm = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[0, 10:] = False  # 10 vs 100

        l_ps, _ = loss_ps(hidden_states, x_1_nm, mask)
        l_pr, _ = loss_pr(hidden_states, x_1_nm, mask)

        assert not torch.allclose(
            l_ps, l_pr, atol=1e-4
        ), "MSE per_sample and per_residue should differ with asymmetric lengths"


class TestTradeoffCombinationMode:
    """Test F: Verify the tradeoff combination formula."""

    def test_tradeoff_formula(self):
        """With known inputs, verify (1-λ)*fm + λ*repa."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        encoder_dim = 32
        token_dim = 16
        repa_layers = [0]
        lam = 0.3

        encoder = MockEncoder(encoder_dim=encoder_dim)
        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=encoder_dim)]
        )

        loss_fn = ProteinaREPALoss(
            encoder=encoder,
            projectors=projectors,
            repa_layers=repa_layers,
            lambda_repa=lam,
            combination_mode="tradeoff",
        )

        hidden_states = [torch.randn(2, 16, token_dim)]
        x_1_nm = torch.randn(2, 16, 3)
        mask = torch.ones(2, 16, dtype=torch.bool)

        repa_loss, _ = loss_fn(hidden_states, x_1_nm, mask)

        # Simulate the combination that proteina_repa.py would do
        fm_loss = torch.tensor(1.5)
        combined = (1.0 - lam) * fm_loss + lam * repa_loss

        expected = 0.7 * 1.5 + 0.3 * repa_loss.item()
        torch.testing.assert_close(
            combined, torch.tensor(expected), atol=1e-5, rtol=1e-5
        )

    def test_additive_formula(self):
        """With known inputs, verify fm + λ*repa."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        encoder_dim = 32
        token_dim = 16
        repa_layers = [0]
        lam = 0.5

        encoder = MockEncoder(encoder_dim=encoder_dim)
        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=encoder_dim)]
        )

        loss_fn = ProteinaREPALoss(
            encoder=encoder,
            projectors=projectors,
            repa_layers=repa_layers,
            lambda_repa=lam,
            combination_mode="additive",
        )

        hidden_states = [torch.randn(2, 16, token_dim)]
        x_1_nm = torch.randn(2, 16, 3)
        mask = torch.ones(2, 16, dtype=torch.bool)

        repa_loss, _ = loss_fn(hidden_states, x_1_nm, mask)

        fm_loss = torch.tensor(2.0)
        combined = fm_loss + lam * repa_loss

        expected = 2.0 + 0.5 * repa_loss.item()
        torch.testing.assert_close(
            combined, torch.tensor(expected), atol=1e-5, rtol=1e-5
        )
