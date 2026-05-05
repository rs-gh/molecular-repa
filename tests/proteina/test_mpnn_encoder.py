"""Tests for ProteinMPNNPerResidueEncoder.

CPU-only smoke tests using random weights. Real-checkpoint loading is gated
behind PROTEINMPNN_TEST=1 + presence of v_48_020.pt at the standard path.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

proteina_root = os.path.join(os.path.dirname(__file__), "../../src/proteina")
sys.path.insert(0, proteina_root)

try:
    import proteinfoundation.repa.pyg_compat  # noqa: F401
except Exception:
    pass

try:
    from proteinfoundation.repa.mpnn_encoder import ProteinMPNNPerResidueEncoder
    from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

    HAS_DEPS = True
except Exception as exc:
    HAS_DEPS = False
    _IMPORT_ERR = exc

needs_deps = pytest.mark.skipif(not HAS_DEPS, reason="proteinfoundation not importable")

_REAL_CKPT = os.environ.get(
    "PROTEINMPNN_CKPT",
    os.path.join(
        os.environ.get("PROTEINMPNN_WEIGHTS_DIR", ""),
        "ca_model_weights/v_48_020.pt",
    ),
)
_RUN_REAL = (
    os.environ.get("PROTEINMPNN_TEST", "0") == "1"
    and _REAL_CKPT
    and os.path.exists(_REAL_CKPT)
)


def _random_batch(b=2, n=24, device="cpu"):
    """CA coordinates in nm scale (~0.3 -> ~3 Angstrom). Mask trims protein 0."""
    coords = torch.randn(b, n, 3, device=device) * 0.3
    mask = torch.ones(b, n, dtype=torch.bool, device=device)
    mask[0, 18:] = False
    residue_type = torch.randint(0, 20, (b, n), device=device)
    return coords, mask, residue_type


@needs_deps
class TestProteinMPNNPerResidueEncoder:
    def test_output_shape(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        coords, mask, res = _random_batch(b=2, n=24)
        out = enc(coords, mask, residue_type=res)
        assert out.shape == (2, 24, 128)

    def test_encoder_dim_attribute(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        assert enc.encoder_dim == 128

    def test_masked_positions_zero(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        coords, mask, res = _random_batch(b=2, n=24)
        out = enc(coords, mask, residue_type=res)
        # mask[0, 18:] = False -> those rows must be exactly zero
        assert out[0, 18:].abs().sum() == 0
        assert out[0, :18].abs().sum() > 0

    def test_output_finite(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        coords, mask, res = _random_batch(b=2, n=24)
        out = enc(coords, mask, residue_type=res)
        assert torch.isfinite(out).all()

    def test_residue_type_ignored(self):
        """Structure-only encoder must produce identical output regardless of
        residue_type (it's never read)."""
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        coords, mask, res = _random_batch(b=2, n=24)
        out_a = enc(coords, mask, residue_type=res)
        out_b = enc(coords, mask, residue_type=torch.zeros_like(res))
        out_c = enc(coords, mask, residue_type=None)
        assert torch.allclose(out_a, out_b)
        assert torch.allclose(out_a, out_c)

    def test_deterministic_with_zero_augment(self):
        """augment_eps=0 -> two forwards on the same coords must be bitwise
        identical (no Gaussian noise injected by features layer)."""
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        coords, mask, res = _random_batch(b=2, n=24)
        out_a = enc(coords, mask)
        out_b = enc(coords, mask)
        assert torch.equal(out_a, out_b)

    def test_frozen_params(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        for name, p in enc.named_parameters():
            assert not p.requires_grad, f"{name} should be frozen"

    def test_train_mode_forced_eval(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        enc.train(True)
        assert not enc.training

    def test_no_grad_to_encoder(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        coords, mask, res = _random_batch(b=2, n=12)
        enc(coords, mask, residue_type=res)
        for p in enc.parameters():
            assert p.grad is None

    def test_coord_sensitivity(self):
        """Output must change when coordinates change (sanity: encoder is
        actually consuming the geometry)."""
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        coords, mask, res = _random_batch(b=1, n=20)
        out_a = enc(coords, mask)
        # Perturb the unmasked coords
        perturbed = coords.clone()
        perturbed[0, :15] = perturbed[0, :15] + 0.05
        out_b = enc(perturbed, mask)
        # Embeddings of perturbed positions must differ
        assert not torch.allclose(out_a[0, :15], out_b[0, :15], atol=1e-4)


@needs_deps
class TestMPNNREPALossIntegration:
    def test_repa_loss_forward_backward(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=None, random_init=True)
        token_dim = 32
        b, n = 2, 16
        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=enc.encoder_dim)]
        )
        loss_fn = ProteinaREPALoss(
            encoder=enc,
            projectors=projectors,
            repa_layers=[0],
        )
        hidden_states = [torch.randn(b, n, token_dim, requires_grad=True)]
        coords = torch.randn(b, n, 3) * 0.3
        mask = torch.ones(b, n, dtype=torch.bool)

        repa_loss, stats = loss_fn(hidden_states, coords, mask)
        assert torch.isfinite(repa_loss)
        repa_loss.backward()

        # Projector receives gradient, encoder stays frozen.
        for p in projectors.parameters():
            assert p.grad is not None
        for p in enc.parameters():
            assert p.grad is None


@pytest.mark.skipif(
    not _RUN_REAL,
    reason="Set PROTEINMPNN_TEST=1 and PROTEINMPNN_WEIGHTS_DIR with v_48_020.pt",
)
class TestProteinMPNNRealCheckpoint:
    def test_checkpoint_loads(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=_REAL_CKPT)
        assert enc.encoder_dim == 128

    def test_real_forward(self):
        enc = ProteinMPNNPerResidueEncoder(ckpt_path=_REAL_CKPT)
        coords, mask, res = _random_batch(b=1, n=32)
        out = enc(coords, mask, residue_type=res)
        assert out.shape == (1, 32, 128)
        assert torch.isfinite(out).all()
        # Real weights -> per-residue embeddings should not collapse to a
        # constant vector.
        assert out[0, :, :].std(dim=0).mean() > 1e-3
