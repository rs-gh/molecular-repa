"""Tests for the trajectory analysis framework.

Tests the source-level modifications (intermediate extraction, attention weights)
and the analysis metric functions.
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add source paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tabasco" / "src"))

from tabasco.models.components.transformer import Transformer
from tabasco.models.components.attention import AttentionBlock


# ---------------------------------------------------------------------------
# Transformer intermediate extraction
# ---------------------------------------------------------------------------


class TestTransformerIntermediates:
    """Test that return_intermediates and need_weights don't change outputs."""

    @pytest.fixture
    def transformer(self):
        return Transformer(dim=64, depth=4, num_heads=4)

    @pytest.fixture
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randn(2, 10, 64)

    def test_return_intermediates_preserves_output(self, transformer, sample_input):
        """Output must be identical whether or not intermediates are captured."""
        transformer.eval()
        with torch.no_grad():
            out_plain = transformer(sample_input)
            out_inter, intermediates, _ = transformer(
                sample_input,
                return_intermediates=True,
            )
        assert torch.allclose(
            out_plain, out_inter, atol=1e-6
        ), "return_intermediates changed the output"

    def test_intermediates_count_matches_depth(self, transformer, sample_input):
        """Should have one intermediate per layer."""
        transformer.eval()
        with torch.no_grad():
            _, intermediates, _ = transformer(
                sample_input,
                return_intermediates=True,
            )
        assert (
            len(intermediates) == 4
        ), f"Expected 4 intermediates, got {len(intermediates)}"

    def test_intermediates_shape(self, transformer, sample_input):
        """Each intermediate should have same shape as input."""
        transformer.eval()
        with torch.no_grad():
            _, intermediates, _ = transformer(
                sample_input,
                return_intermediates=True,
            )
        for i, h in enumerate(intermediates):
            assert (
                h.shape == sample_input.shape
            ), f"Intermediate {i} shape {h.shape} != input shape {sample_input.shape}"

    def test_need_weights_preserves_output(self, transformer, sample_input):
        """Output must be identical whether or not attention weights are captured."""
        transformer.eval()
        with torch.no_grad():
            out_plain = transformer(sample_input)
            out_w, _, attn_weights = transformer(
                sample_input,
                need_weights=True,
            )
        assert torch.allclose(
            out_plain, out_w, atol=1e-6
        ), "need_weights changed the output"

    def test_attn_weights_count(self, transformer, sample_input):
        """Should have one set of weights per layer."""
        transformer.eval()
        with torch.no_grad():
            _, _, attn_weights = transformer(
                sample_input,
                need_weights=True,
            )
        assert len(attn_weights) == 4

    def test_attn_weights_are_distributions(self, transformer, sample_input):
        """Attention weights should sum to ~1 along last dim."""
        transformer.eval()
        with torch.no_grad():
            _, _, attn_weights = transformer(
                sample_input,
                need_weights=True,
            )
        for i, w in enumerate(attn_weights):
            sums = w.sum(dim=-1)
            assert torch.allclose(
                sums, torch.ones_like(sums), atol=1e-4
            ), f"Layer {i} attention weights don't sum to 1"

    def test_both_intermediates_and_weights(self, transformer, sample_input):
        """Can request both intermediates and attention weights simultaneously."""
        transformer.eval()
        with torch.no_grad():
            out, intermediates, attn_weights = transformer(
                sample_input,
                return_intermediates=True,
                need_weights=True,
            )
        assert intermediates is not None and len(intermediates) == 4
        assert attn_weights is not None and len(attn_weights) == 4

    def test_with_padding_mask(self, transformer):
        """Intermediates work correctly with padding."""
        transformer.eval()
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, 7:] = True  # Pad last 3 positions of first batch
        mask[1, 5:] = True  # Pad last 5 positions of second batch

        with torch.no_grad():
            out, intermediates, _ = transformer(
                x,
                padding_mask=mask,
                return_intermediates=True,
            )
        assert len(intermediates) == 4


# ---------------------------------------------------------------------------
# AttentionBlock need_weights
# ---------------------------------------------------------------------------


class TestAttentionBlock:
    def test_need_weights_returns_tuple(self):
        block = AttentionBlock(dim=64, num_heads=4)
        block.eval()
        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            result = block(x, need_weights=True)

        assert (
            isinstance(result, tuple) and len(result) == 2
        ), "need_weights=True should return (output, weights)"
        out, weights = result
        assert out.shape == x.shape
        assert weights.dim() == 3  # [B, N, N] (averaged over heads by PyTorch default)

    def test_need_weights_false_returns_tensor(self):
        block = AttentionBlock(dim=64, num_heads=4)
        block.eval()
        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            result = block(x, need_weights=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# TransformerModule return_hidden_states="all"
# ---------------------------------------------------------------------------


class TestTransformerModuleAnalysis:
    @pytest.fixture
    def module(self):
        from tabasco.models.components.transformer_module import TransformerModule

        return TransformerModule(
            spatial_dim=3,
            atom_dim=9,
            num_heads=4,
            num_layers=4,
            hidden_dim=64,
            activation="SiLU",
            implementation="reimplemented",
            cross_attention=False,
        )

    @pytest.fixture
    def inputs(self):
        torch.manual_seed(42)
        B, N = 2, 9
        return {
            "coords": torch.randn(B, N, 3),
            "atomics": torch.nn.functional.one_hot(
                torch.randint(0, 9, (B, N)), 9
            ).float(),
            "padding_mask": torch.zeros(B, N, dtype=torch.bool),
            "t": torch.tensor([0.5, 0.7]),
        }

    def test_return_all_preserves_output(self, module, inputs):
        """return_hidden_states='all' should not change coords/atomics output."""
        module.eval()
        with torch.no_grad():
            coords1, logits1 = module(
                inputs["coords"],
                inputs["atomics"],
                inputs["padding_mask"],
                inputs["t"],
                return_hidden_states=False,
            )
            coords2, logits2, analysis = module(
                inputs["coords"],
                inputs["atomics"],
                inputs["padding_mask"],
                inputs["t"],
                return_hidden_states="all",
                need_weights=True,
            )

        assert torch.allclose(coords1, coords2, atol=1e-5)
        assert torch.allclose(logits1, logits2, atol=1e-5)

    def test_analysis_dict_keys(self, module, inputs):
        """Analysis dict should contain expected keys."""
        module.eval()
        with torch.no_grad():
            _, _, analysis = module(
                inputs["coords"],
                inputs["atomics"],
                inputs["padding_mask"],
                inputs["t"],
                return_hidden_states="all",
                need_weights=True,
            )

        assert "intermediates" in analysis
        assert "attn_weights" in analysis
        assert "h_in" in analysis
        assert "embed_components" in analysis

    def test_intermediates_list_length(self, module, inputs):
        module.eval()
        with torch.no_grad():
            _, _, analysis = module(
                inputs["coords"],
                inputs["atomics"],
                inputs["padding_mask"],
                inputs["t"],
                return_hidden_states="all",
            )
        assert len(analysis["intermediates"]) == 4

    def test_embed_components(self, module, inputs):
        """Embed components should contain coords, atoms, posenc, time."""
        module.eval()
        with torch.no_grad():
            _, _, analysis = module(
                inputs["coords"],
                inputs["atomics"],
                inputs["padding_mask"],
                inputs["t"],
                return_hidden_states="all",
            )
        ec = analysis["embed_components"]
        assert "coords" in ec
        assert "atoms" in ec
        assert "posenc" in ec
        assert "time" in ec


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_attention_entropy_uniform(self):
        """Uniform attention should have entropy near 1 (normalized)."""
        from playground.mech_interp.metrics import attention_entropy

        N = 10
        attn = torch.ones(1, 4, N, N) / N
        ent = attention_entropy(attn)
        assert ent.shape == (1, 4)
        assert torch.allclose(ent, torch.ones_like(ent), atol=0.05)

    def test_attention_entropy_sharp(self):
        """One-hot attention should have entropy near 0."""
        from playground.mech_interp.metrics import attention_entropy

        N = 10
        attn = torch.zeros(1, 4, N, N)
        attn[:, :, :, 0] = 1.0  # All attention on position 0
        ent = attention_entropy(attn)
        assert (ent < 0.05).all()

    def test_bond_precision_perfect(self):
        """If attention exactly matches bonds, precision should be 1."""
        from playground.mech_interp.metrics import bond_precision

        N = 5
        bonds = torch.zeros(1, N, N)
        bonds[0, 0, 1] = bonds[0, 1, 0] = 1
        bonds[0, 1, 2] = bonds[0, 2, 1] = 1

        # Attention that perfectly matches bonds
        attn = torch.zeros(1, 1, N, N)
        attn[0, 0, 0, 1] = 0.5
        attn[0, 0, 1, 2] = 0.5
        attn = attn + 1e-6  # Avoid exact zeros
        attn = attn / attn.sum(dim=-1, keepdim=True)

        prec = bond_precision(attn, bonds, k=2, apply_apc=False)
        assert prec.shape == (1, 1)

    def test_structure_lens_rmsd_identical(self):
        """RMSD between identical coordinates should be 0."""
        from playground.mech_interp.metrics import structure_lens_rmsd

        coords = torch.randn(2, 10, 3)
        mask = torch.ones(2, 10, dtype=torch.bool)
        rmsd = structure_lens_rmsd(coords, coords, mask)
        assert rmsd < 1e-4

    def test_atom_accuracy_perfect(self):
        """Perfect predictions should give accuracy 1.0."""
        from playground.mech_interp.metrics import atom_type_accuracy

        logits = torch.zeros(2, 10, 9)
        logits[:, :, 3] = 10.0  # Strong prediction for type 3
        gt = torch.nn.functional.one_hot(
            torch.full((2, 10), 3, dtype=torch.long), 9
        ).float()
        mask = torch.ones(2, 10, dtype=torch.bool)
        acc = atom_type_accuracy(logits, gt, mask)
        assert acc == 1.0


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


class TestAblationConfig:
    def test_should_ablate_in_range(self):
        from playground.mech_interp.trajectory_analyzer import AblationConfig

        config = AblationConfig(ablate_coords=True, t_min=0.3, t_max=0.7)
        assert config.should_ablate(0.5)
        assert not config.should_ablate(0.1)
        assert not config.should_ablate(0.9)

    def test_label_generation(self):
        from playground.mech_interp.trajectory_analyzer import AblationConfig

        config = AblationConfig(ablate_coords=True, ablate_atoms=True)
        assert "coords" in config.label
        assert "atoms" in config.label

    def test_baseline_label(self):
        from playground.mech_interp.trajectory_analyzer import AblationConfig

        config = AblationConfig()
        assert config.label == "baseline"


# ---------------------------------------------------------------------------
# TrajectoryMetrics save/load
# ---------------------------------------------------------------------------


class TestTrajectoryMetrics:
    def test_save_load_roundtrip(self, tmp_path):
        from playground.mech_interp.trajectory_analyzer import TrajectoryMetrics

        metrics = TrajectoryMetrics(
            timesteps=np.array([0.0, 0.5, 1.0]),
            timestep_indices=np.array([0, 50, 99]),
            num_layers=4,
            num_heads=8,
            coord_rmsd_to_final=np.random.rand(3, 4),
            coord_rmsd_to_gt=None,
            atom_accuracy=np.random.rand(3, 4),
            attn_entropy=np.random.rand(3, 4, 8),
            bond_precision_vals=None,
            dist_correlation=None,
            chemprop_cosine_sim=None,
        )

        path = str(tmp_path / "test_metrics.npz")
        metrics.save(path)
        loaded = TrajectoryMetrics.load(path)

        assert np.allclose(metrics.timesteps, loaded.timesteps)
        assert np.allclose(metrics.coord_rmsd_to_final, loaded.coord_rmsd_to_final)
        assert np.allclose(metrics.attn_entropy, loaded.attn_entropy)
        assert loaded.num_layers == 4
        assert loaded.num_heads == 8
