"""Tests for GearNetEdge and MCGearNetEdgePerResidueEncoder.

All tests run on CPU without a real checkpoint. Shape, mask, frozen-param,
and forward-pass correctness are verified using the randomly-initialised model.
Real checkpoint loading is gated on the MC_GEARNET_TEST env var.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

proteina_root = os.path.join(os.path.dirname(__file__), "../../src/proteina")
sys.path.insert(0, proteina_root)

# Patch torch_scatter with native PyTorch ops if CUDA extensions don't load.
# Must happen before any proteina imports.
try:
    import proteinfoundation.repa.pyg_compat  # noqa: F401
except Exception:
    pass

try:
    from torch_scatter import scatter_mean  # noqa: F401
    from proteinfoundation.metrics.gearnet_utils import GearNetEdge
    from proteinfoundation.repa.gearnet_encoder import MCGearNetEdgePerResidueEncoder
    from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss

    HAS_DEPS = True
except Exception as exc:
    HAS_DEPS = False
    _IMPORT_ERR = exc

needs_deps = pytest.mark.skipif(not HAS_DEPS, reason="proteinfoundation not importable")

_REAL_CKPT = os.environ.get(
    "MC_GEARNET_CKPT",
    os.path.join(
        os.environ.get("DATA_PATH", ""),
        "metric_factory/model_weights/mc_gearnet_edge.pth",
    ),
)
_RUN_REAL = os.environ.get("MC_GEARNET_TEST", "0") == "1" and os.path.exists(_REAL_CKPT)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _random_batch(b=2, n=16, device="cpu"):
    coords = torch.randn(b, n, 3, device=device) * 5.0  # ~Ångström scale
    mask = torch.ones(b, n, dtype=torch.bool, device=device)
    mask[0, 12:] = False  # first protein is shorter
    residue_type = torch.randint(0, 20, (b, n), device=device)
    return coords, mask, residue_type


def _flat_batch(b=2, n=16, device="cpu"):
    """Flat tensors as expected by GearNetEdge.forward directly."""
    lengths = [n - 2, n]
    coords_list, batch_list, res_list = [], [], []
    for i, seq_len in enumerate(lengths):
        coords_list.append(torch.randn(seq_len, 3, device=device) * 5.0)
        batch_list.append(torch.full((seq_len,), i, dtype=torch.long, device=device))
        res_list.append(torch.randint(0, 20, (seq_len,), device=device))
    return (
        torch.cat(coords_list),
        torch.cat(batch_list),
        torch.cat(res_list),
    )


# ── GearNetEdge (raw model) ───────────────────────────────────────────────────


@needs_deps
class TestGearNetEdgeArchitecture:
    def test_output_dim(self):
        m = GearNetEdge()
        assert m.output_dim == 3072

    def test_forward_shape(self):
        m = GearNetEdge()
        coords, atom2batch, res = _flat_batch(b=2, n=8)
        out = m(coords, res, atom2batch)
        assert out.shape == (coords.shape[0], 3072)

    def test_forward_finite(self):
        m = GearNetEdge()
        coords, atom2batch, res = _flat_batch(b=2, n=8)
        out = m(coords, res, atom2batch)
        assert torch.isfinite(out).all(), "GearNetEdge output contains NaN/Inf"

    def test_output_not_constant(self):
        """Different residue types must produce different embeddings."""
        m = GearNetEdge()
        n = 10
        coords = torch.zeros(n, 3)
        atom2batch = torch.zeros(n, dtype=torch.long)
        res_a = torch.zeros(n, dtype=torch.long)  # all Ala
        res_g = torch.full((n,), 5, dtype=torch.long)  # all Gly (index 5)
        out_a = m(coords, res_a, atom2batch)
        out_g = m(coords, res_g, atom2batch)
        assert not torch.allclose(
            out_a, out_g
        ), "Outputs should differ for different residue types"

    def test_unk_residue_handled(self):
        """Index 20 (UNK) must not cause an error."""
        m = GearNetEdge()
        coords, atom2batch, _ = _flat_batch(b=1, n=6)
        res = torch.full((coords.shape[0],), 20, dtype=torch.long)
        out = m(coords, res, atom2batch)
        assert torch.isfinite(out).all()

    def test_state_dict_key_prefixes(self):
        """Module names must match the actual mc_gearnet_edge.pth checkpoint key structure."""
        m = GearNetEdge()
        keys = set(m.state_dict().keys())
        # Node layers (each has self_loop + linear + batch_norm)
        assert "layers.0.self_loop.weight" in keys
        assert "layers.0.linear.weight" in keys
        assert "layers.0.batch_norm.weight" in keys
        assert "layers.5.linear.weight" in keys
        # Edge layers
        assert "edge_layers.0.self_loop.weight" in keys
        assert "edge_layers.0.linear.weight" in keys
        assert "edge_layers.5.linear.weight" in keys
        # Top-level batch norms
        assert "batch_norms.0.weight" in keys
        assert "batch_norms.5.weight" in keys
        # No IEConv, no initial embedding linear, no layer_norms
        assert not any("ieconv" in k for k in keys)
        assert "embedding_batch_norm.weight" not in keys
        assert "layer_norms.0.weight" not in keys

    def test_linear_shapes(self):
        m = GearNetEdge()
        sd = m.state_dict()
        # layers.0: node_in=21, num_relation=7 → linear input=7*21=147
        assert sd["layers.0.linear.weight"].shape == (512, 7 * 21)
        assert sd["layers.0.self_loop.weight"].shape == (512, 21)
        # layers.1+: node_in=512 → linear input=7*512=3584
        assert sd["layers.1.linear.weight"].shape == (512, 7 * 512)
        # edge_layers.0: edge_in=59, num_angle_bin=8 → linear input=8*59=472, output=21
        assert sd["edge_layers.0.linear.weight"].shape == (21, 8 * 59)
        assert sd["edge_layers.0.self_loop.weight"].shape == (21, 59)
        # edge_layers.1: edge_in=21 → linear input=8*21=168, output=512
        assert sd["edge_layers.1.linear.weight"].shape == (512, 8 * 21)
        # edge_layers.2+: 8*512=4096 → 512
        assert sd["edge_layers.2.linear.weight"].shape == (512, 8 * 512)


# ── MCGearNetEdgePerResidueEncoder ───────────────────────────────────────────


def _make_encoder_no_ckpt():
    """Instantiate encoder with random weights (bypasses checkpoint loading)."""
    enc = MCGearNetEdgePerResidueEncoder.__new__(MCGearNetEdgePerResidueEncoder)
    nn.Module.__init__(enc)
    from proteinfoundation.metrics.gearnet_utils import GearNetEdge

    enc.gearnet = GearNetEdge()
    enc.encoder_dim = enc.gearnet.output_dim
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@needs_deps
class TestMCGearNetEdgeEncoder:
    def test_output_shape(self):
        enc = _make_encoder_no_ckpt()
        coords, mask, res = _random_batch(b=2, n=16)
        out = enc(coords, mask, residue_type=res)
        assert out.shape == (2, 16, 3072)

    def test_masked_positions_zero(self):
        enc = _make_encoder_no_ckpt()
        coords, mask, res = _random_batch(b=2, n=16)
        out = enc(coords, mask, residue_type=res)
        # mask[0, 12:] = False  → those positions must be exactly zero
        assert out[0, 12:].abs().sum() == 0
        assert out[0, :12].abs().sum() > 0

    def test_output_finite(self):
        enc = _make_encoder_no_ckpt()
        coords, mask, res = _random_batch(b=2, n=16)
        out = enc(coords, mask, residue_type=res)
        assert torch.isfinite(out).all()

    def test_encoder_dim_attribute(self):
        enc = _make_encoder_no_ckpt()
        assert enc.encoder_dim == 3072

    def test_residue_type_required(self):
        enc = _make_encoder_no_ckpt()
        coords, mask, _ = _random_batch(b=1, n=8)
        with pytest.raises(ValueError, match="residue_type"):
            enc(coords, mask)

    def test_frozen_params(self):
        enc = _make_encoder_no_ckpt()
        for name, p in enc.named_parameters():
            assert not p.requires_grad, f"{name} should be frozen"

    def test_train_mode_forced_eval(self):
        enc = _make_encoder_no_ckpt()
        enc.train(True)
        assert not enc.training

    def test_no_grad_flow_to_encoder(self):
        enc = _make_encoder_no_ckpt()
        coords, mask, res = _random_batch(b=2, n=8)
        enc(coords, mask, residue_type=res)
        # Encoder is frozen — no grad should reach its params
        for p in enc.parameters():
            assert p.grad is None


# ── REPA loss integration ────────────────────────────────────────────────────


@needs_deps
class TestMCGearNetEdgeLossIntegration:
    def test_repa_loss_forward_backward(self):
        """End-to-end: loss computes and gradients reach projector but not encoder."""
        enc = _make_encoder_no_ckpt()
        token_dim = 32
        b, n = 2, 8
        projectors = nn.ModuleList(
            [Projector(hidden_dim=token_dim, encoder_dim=enc.encoder_dim)]
        )
        loss_fn = ProteinaREPALoss(
            encoder=enc,
            projectors=projectors,
            repa_layers=[0],
        )
        hidden_states = [torch.randn(b, n, token_dim, requires_grad=True)]
        coords = torch.randn(b, n, 3) * 0.3  # nm scale
        mask = torch.ones(b, n, dtype=torch.bool)
        residue_type = torch.randint(0, 20, (b, n))

        repa_loss, stats = loss_fn(
            hidden_states, coords, mask, residue_type=residue_type
        )
        assert torch.isfinite(repa_loss)
        repa_loss.backward()

        # Projector receives gradient
        for p in projectors.parameters():
            assert p.grad is not None
        # Encoder stays frozen
        for p in enc.parameters():
            assert p.grad is None


# ── Parity tests: refactored _build_edges vs reference loop ─────────────────
#
# The reference implementation below is the *original* loop-based _build_edges
# before the radius_graph refactor. It serves as ground truth for the new
# vectorised version. Tests run on CPU — no checkpoint needed.


def _reference_build_edges(model, coords, residue_types, atom2batch, local_idx):
    """Original loop-based _build_edges, kept here as a reference oracle."""
    import torch.nn.functional as F

    device = coords.device

    all_ni, all_no, all_rel = [], [], []
    for b in atom2batch.unique():
        g_idx = torch.where(atom2batch == b)[0]
        n_b = g_idx.shape[0]
        for offset in range(-model._SEQ_MAX_DIST, model._SEQ_MAX_DIST + 1):
            rel = offset + model._SEQ_MAX_DIST
            if offset > 0:
                ni, no = g_idx[: n_b - offset], g_idx[offset:]
            elif offset < 0:
                ni, no = g_idx[-offset:], g_idx[: n_b + offset]
            else:
                ni = no = g_idx
            all_ni.append(ni)
            all_no.append(no)
            all_rel.append(
                torch.full((ni.shape[0],), rel, device=device, dtype=torch.long)
            )

    sp_ni, sp_no, knn_ni, knn_no = [], [], [], []
    for b in atom2batch.unique():
        g_idx = torch.where(atom2batch == b)[0]
        n_b = g_idx.shape[0]
        sub_c = coords[g_idx].float()
        dist_mat = torch.cdist(sub_c, sub_c)
        seq_dist = (
            torch.arange(n_b, device=device).unsqueeze(0)
            - torch.arange(n_b, device=device).unsqueeze(1)
        ).abs()
        sp_mask = (dist_mat < model._SPATIAL_RADIUS) & (seq_dist >= model._SEQ_MIN_DIST)
        sp_mask.fill_diagonal_(False)
        si, so = sp_mask.nonzero(as_tuple=True)
        sp_ni.append(g_idx[si])
        sp_no.append(g_idx[so])
        d_knn = dist_mat.clone()
        d_knn[seq_dist < model._SEQ_MIN_DIST] = float("inf")
        d_knn.fill_diagonal_(float("inf"))
        k = min(model._KNN_K, n_b - 1)
        if k > 0:
            _, knn_j = d_knn.topk(k, dim=-1, largest=False)
            ki = (
                torch.arange(n_b, device=device)
                .unsqueeze(1)
                .expand_as(knn_j)
                .reshape(-1)
            )
            kj = knn_j.reshape(-1)
            valid = d_knn[ki, kj] < float("inf")
            knn_ni.append(g_idx[ki[valid]])
            knn_no.append(g_idx[kj[valid]])

    node_in = torch.cat(all_ni)
    node_out = torch.cat(all_no)
    rel = torch.cat(all_rel)
    if sp_ni:
        sni, sno = torch.cat(sp_ni), torch.cat(sp_no)
        node_in = torch.cat([node_in, sni])
        node_out = torch.cat([node_out, sno])
        rel = torch.cat(
            [
                rel,
                torch.full(
                    (sni.shape[0],), model._SPATIAL_REL, device=device, dtype=torch.long
                ),
            ]
        )
    if knn_ni:
        kni, kno = torch.cat(knn_ni), torch.cat(knn_no)
        node_in = torch.cat([node_in, kni])
        node_out = torch.cat([node_out, kno])
        rel = torch.cat(
            [
                rel,
                torch.full(
                    (kni.shape[0],), model._KNN_REL, device=device, dtype=torch.long
                ),
            ]
        )

    edge_index = torch.stack([node_in, node_out, rel], dim=1)
    ni, no = edge_index[:, 0], edge_index[:, 1]
    rt = edge_index[:, 2]
    res_i = F.one_hot(residue_types[ni].clamp(0, 20), 21).float()
    res_j = F.one_hot(residue_types[no].clamp(0, 20), 21).float()
    rel_oh = F.one_hot(rt, model._NUM_RELATION).float()
    d = (coords[no].float() - coords[ni].float()).norm(dim=-1)
    centers = torch.linspace(0, 50, 10, device=device)
    d_rbf = torch.exp(-0.5 * ((d.unsqueeze(-1) - centers) / 5.0) ** 2)
    edge_feat = torch.cat([res_i, res_j, rel_oh, d_rbf], dim=-1)
    return edge_index, edge_feat


def _canonical_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Sort edge_index rows for order-independent comparison."""
    idx = edge_index[:, 0] * 100000 + edge_index[:, 1] * 100 + edge_index[:, 2]
    return edge_index[idx.argsort()]


@needs_deps
class TestBuildEdgesParity:
    """New vectorised _build_edges must produce identical edges to the reference loop."""

    def _run(self, b, n, seed=0):
        torch.manual_seed(seed)
        model = GearNetEdge()
        model.eval()
        coords, atom2batch, res = _flat_batch(b=b, n=n)
        local_idx = model._local_idx(atom2batch)
        ref_ei, ref_ef = _reference_build_edges(
            model, coords, res, atom2batch, local_idx
        )
        new_ei, new_ef = model._build_edges(coords, res, atom2batch, local_idx)
        return ref_ei, ref_ef, new_ei, new_ef

    def test_edge_count_matches(self):
        ref_ei, _, new_ei, _ = self._run(b=2, n=12)
        assert (
            ref_ei.shape[0] == new_ei.shape[0]
        ), f"Edge count mismatch: ref={ref_ei.shape[0]}, new={new_ei.shape[0]}"

    def test_edge_set_identical(self):
        ref_ei, _, new_ei, _ = self._run(b=2, n=12)
        ref_sorted = _canonical_edges(ref_ei)
        new_sorted = _canonical_edges(new_ei)
        assert torch.equal(
            ref_sorted, new_sorted
        ), "Edge sets differ between reference and new implementation"

    def test_edge_features_match(self):
        ref_ei, ref_ef, new_ei, new_ef = self._run(b=2, n=12)
        # Align features by canonical edge order
        ref_order = (
            ref_ei[:, 0] * 100000 + ref_ei[:, 1] * 100 + ref_ei[:, 2]
        ).argsort()
        new_order = (
            new_ei[:, 0] * 100000 + new_ei[:, 1] * 100 + new_ei[:, 2]
        ).argsort()
        assert torch.allclose(
            ref_ef[ref_order], new_ef[new_order], atol=1e-5
        ), "Edge features differ between reference and new implementation"

    def test_parity_larger_batch(self):
        """Larger batch to stress the vectorised path."""
        ref_ei, _, new_ei, _ = self._run(b=4, n=20, seed=7)
        ref_sorted = _canonical_edges(ref_ei)
        new_sorted = _canonical_edges(new_ei)
        assert torch.equal(ref_sorted, new_sorted)

    def test_local_idx_correct(self):
        """_local_idx should return 0,1,...,n_i-1 for each batch element."""
        torch.manual_seed(0)
        coords, atom2batch, _ = _flat_batch(b=3, n=10)
        model = GearNetEdge()
        local = model._local_idx(atom2batch)
        for b in atom2batch.unique():
            sel = atom2batch == b
            expected = torch.arange(int(sel.sum()))
            assert torch.equal(local[sel], expected), f"local_idx wrong for batch {b}"


# ── Real checkpoint (gated) ──────────────────────────────────────────────────


@pytest.mark.skipif(
    not _RUN_REAL,
    reason="Set MC_GEARNET_TEST=1 and ensure mc_gearnet_edge.pth is present",
)
class TestMCGearNetEdgeRealCheckpoint:
    def test_checkpoint_loads(self):
        enc = MCGearNetEdgePerResidueEncoder(ckpt_path=_REAL_CKPT)
        assert enc.encoder_dim == 3072

    def test_real_forward(self):
        enc = MCGearNetEdgePerResidueEncoder(ckpt_path=_REAL_CKPT)
        coords, mask, res = _random_batch(b=1, n=32)
        out = enc(coords, mask, residue_type=res)
        assert out.shape == (1, 32, 3072)
        assert torch.isfinite(out).all()
        # Embeddings should not be constant across residues
        assert out[0, :, :].std(dim=0).mean() > 0.01
