"""Tests for ESMPerResidueEncoder.

Shape / mask / alphabet-mapping checks use a tiny stub ESM (no network).
The full 650M-parameter load is exercised only when ``ESM_TEST=1`` is set,
to keep the default test run lightweight on constrained disk/memory.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

proteina_root = os.path.join(os.path.dirname(__file__), "../../src/proteina")
sys.path.insert(0, proteina_root)

try:
    from proteinfoundation.repa.esm_encoder import ESMPerResidueEncoder  # noqa: F401
    from openfold.np.residue_constants import restypes

    HAS_ESM_DEPS = True
except Exception:
    HAS_ESM_DEPS = False


needs_deps = pytest.mark.skipif(
    not HAS_ESM_DEPS, reason="openfold / transformers not installed"
)

_RUN_REAL_ESM = os.environ.get("ESM_TEST", "0") == "1"


# --- Stub ESM so we can unit-test the wrapper without the 650M download -----


class _StubTokenizer:
    """Minimal stand-in for HF AutoTokenizer covering what the wrapper uses."""

    def __init__(self):
        # ESM-2's actual vocab ordering - doesn't need to be exact, just
        # self-consistent: every AA letter must round-trip to a unique id.
        specials = ["<cls>", "<pad>", "<eos>", "<unk>"]
        aa_letters = [
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "C",
            "W",
        ]
        self._vocab = {tok: i for i, tok in enumerate(specials + aa_letters)}
        self.cls_token_id = self._vocab["<cls>"]
        self.pad_token_id = self._vocab["<pad>"]
        self.eos_token_id = self._vocab["<eos>"]
        self.unk_token_id = self._vocab["<unk>"]

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._vocab.get(tok, self.unk_token_id)


class _StubESMConfig:
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class _StubESMOutput:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _StubESMModel(nn.Module):
    """Replaces AutoModel.from_pretrained - deterministic, tiny, same API."""

    def __init__(self, hidden_size=32, vocab_size=64, num_layers=3):
        super().__init__()
        self.config = _StubESMConfig(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        h = self.embedding(input_ids)
        hs = [h] if output_hidden_states else None
        for layer in self.layers:
            h = torch.tanh(layer(h))
            if output_hidden_states:
                hs.append(h)
        return _StubESMOutput(last_hidden_state=h, hidden_states=hs)


@pytest.fixture
def stubbed_esm(monkeypatch):
    """Patch `transformers.AutoModel` / `AutoTokenizer` with tiny stubs."""
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, _: _StubTokenizer()),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        classmethod(lambda cls, _: _StubESMModel()),
    )


# --- Shape / masking / alphabet tests (stubbed ESM) -------------------------


@needs_deps
class TestESMEncoderWithStub:
    def test_output_shape(self, stubbed_esm):
        enc = ESMPerResidueEncoder(model_id="stub")
        b, n = 2, 8
        coords = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        residue_type = torch.randint(0, 20, (b, n))

        out = enc(coords, mask, residue_type=residue_type)

        assert out.shape == (b, n, enc.encoder_dim)
        assert torch.isfinite(out).all()

    def test_masked_positions_zero(self, stubbed_esm):
        enc = ESMPerResidueEncoder(model_id="stub")
        b, n = 2, 8
        coords = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[0, 5:] = False
        mask[1, 6:] = False
        residue_type = torch.randint(0, 20, (b, n))

        out = enc(coords, mask, residue_type=residue_type)

        # Padded positions must be exactly zero (encoder multiplies by mask).
        assert out[0, 5:].abs().sum() == 0
        assert out[1, 6:].abs().sum() == 0
        # Valid positions should not all be zero.
        assert out[0, :5].abs().sum() > 0

    def test_alphabet_mapping_differs(self, stubbed_esm):
        """Different AA indices must produce different embeddings."""
        enc = ESMPerResidueEncoder(model_id="stub")
        b, n = 1, 4
        coords = torch.zeros(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)

        a_idx = restypes.index("A")
        g_idx = restypes.index("G")
        seq_a = torch.full((b, n), a_idx, dtype=torch.long)
        seq_g = torch.full((b, n), g_idx, dtype=torch.long)

        out_a = enc(coords, mask, residue_type=seq_a)
        out_g = enc(coords, mask, residue_type=seq_g)

        assert not torch.allclose(out_a, out_g)

    def test_identical_sequences_identical_output(self, stubbed_esm):
        """Determinism: same residue_type tensor -> same output (no noise)."""
        enc = ESMPerResidueEncoder(model_id="stub")
        b, n = 2, 6
        coords = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        residue_type = torch.randint(0, 20, (1, n)).expand(b, n).contiguous()

        out = enc(coords, mask, residue_type=residue_type)
        # Both batch elements see the same tokens + mask -> identical rows.
        torch.testing.assert_close(out[0], out[1])

    def test_padding_index_handled(self, stubbed_esm):
        """Dense collator pads with -1; encoder must not index-error on it."""
        enc = ESMPerResidueEncoder(model_id="stub")
        b, n = 2, 8
        coords = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        mask[0, 5:] = False
        residue_type = torch.randint(0, 20, (b, n))
        residue_type[0, 5:] = -1  # collator pad value for int tensors

        out = enc(coords, mask, residue_type=residue_type)
        assert torch.isfinite(out).all()

    def test_unk_index_handled(self, stubbed_esm):
        """Residue index 20 (UNK) must route to ESM's <unk> token."""
        enc = ESMPerResidueEncoder(model_id="stub")
        b, n = 1, 4
        coords = torch.zeros(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        residue_type = torch.full((b, n), 20, dtype=torch.long)

        out = enc(coords, mask, residue_type=residue_type)
        assert torch.isfinite(out).all()

    def test_residue_type_required(self, stubbed_esm):
        enc = ESMPerResidueEncoder(model_id="stub")
        coords = torch.randn(1, 4, 3)
        mask = torch.ones(1, 4, dtype=torch.bool)
        with pytest.raises(ValueError):
            enc(coords, mask)

    def test_frozen_params(self, stubbed_esm):
        enc = ESMPerResidueEncoder(model_id="stub")
        for name, p in enc.named_parameters():
            assert not p.requires_grad, f"{name} should be frozen"

    def test_train_mode_forced_eval(self, stubbed_esm):
        enc = ESMPerResidueEncoder(model_id="stub")
        enc.train(True)  # caller tries to flip to train mode
        assert not enc.training

    def test_layer_selection_returns_intermediate(self, stubbed_esm):
        """layer=N must return hidden_states[N], not last_hidden_state."""
        b, n = 1, 4
        coords = torch.zeros(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        residue_type = torch.randint(0, 20, (b, n))

        enc_last = ESMPerResidueEncoder(model_id="stub", layer=None)
        enc_mid = ESMPerResidueEncoder(model_id="stub", layer=1)

        # Force identical weights so any output difference comes from layer pick.
        enc_mid.load_state_dict(enc_last.state_dict())

        out_last = enc_last(coords, mask, residue_type=residue_type)
        out_mid = enc_mid(coords, mask, residue_type=residue_type)

        # Stub has 3 layers: hidden_states[1] (post-block-1) must differ from
        # last_hidden_state (post-block-3); same shape, different values.
        assert out_last.shape == out_mid.shape
        assert not torch.allclose(out_last, out_mid)


# --- Integration test: loss + residue_type plumbing (no real ESM) -----------


@needs_deps
class TestESMEncoderLossIntegration:
    def test_repa_loss_passes_residue_type(self, stubbed_esm):
        """ProteinaREPALoss.forward must thread residue_type into the encoder."""
        from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

        enc = ESMPerResidueEncoder(model_id="stub")
        token_dim = 16
        b, n = 2, 8
        repa_layers = [0]

        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=enc.encoder_dim)]
        )
        loss_fn = ProteinaREPALoss(
            encoder=enc,
            projectors=projectors,
            repa_layers=repa_layers,
        )

        hidden_states = [torch.randn(b, n, token_dim, requires_grad=True)]
        x_1_nm = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        residue_type = torch.randint(0, 20, (b, n))

        repa_loss, stats = loss_fn(
            hidden_states, x_1_nm, mask, residue_type=residue_type
        )
        assert torch.isfinite(repa_loss)
        repa_loss.backward()

        # Frozen ESM params stay None; projector gets gradient.
        for p in enc.parameters():
            assert p.grad is None
        for p in projectors.parameters():
            assert p.grad is not None


# --- Real ESM integration (gated on env var) --------------------------------


@pytest.mark.skipif(
    not (HAS_ESM_DEPS and _RUN_REAL_ESM),
    reason="Set ESM_TEST=1 to download + run real ESM-2 checkpoint",
)
class TestESMEncoderReal:
    def test_real_esm_forward(self):
        enc = ESMPerResidueEncoder(model_id="facebook/esm2_t33_650M_UR50D")
        assert enc.encoder_dim == 1280

        b, n = 1, 16
        coords = torch.randn(b, n, 3)
        mask = torch.ones(b, n, dtype=torch.bool)
        residue_type = torch.randint(0, 20, (b, n))

        out = enc(coords, mask, residue_type=residue_type)
        assert out.shape == (b, n, 1280)
        assert torch.isfinite(out).all()
