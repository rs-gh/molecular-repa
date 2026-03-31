"""Tests for pyg_compat shim — scatter ops, radius_graph, and fake module behavior."""

import pytest
import torch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src/proteina"))

from proteinfoundation.repa.pyg_compat import (
    _scatter_mean_native,
    _scatter_sum_native,
    _radius_graph_native,
)


# ---------------------------------------------------------------------------
# scatter tests
# ---------------------------------------------------------------------------


class TestScatterOps:
    def test_scatter_sum_basic(self):
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = torch.tensor([0, 0, 1, 1])
        result = _scatter_sum_native(src, index, dim=0)
        assert torch.allclose(result, torch.tensor([3.0, 7.0]))

    def test_scatter_mean_basic(self):
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = torch.tensor([0, 0, 1, 1])
        result = _scatter_mean_native(src, index, dim=0)
        assert torch.allclose(result, torch.tensor([1.5, 3.5]))

    def test_scatter_sum_2d(self):
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = torch.tensor([0, 1, 0])
        result = _scatter_sum_native(src, index, dim=0)
        assert torch.allclose(result, torch.tensor([[6.0, 8.0], [3.0, 4.0]]))

    def test_scatter_sum_with_dim_size(self):
        src = torch.tensor([1.0, 2.0])
        index = torch.tensor([0, 0])
        result = _scatter_sum_native(src, index, dim=0, dim_size=3)
        assert result.shape[0] == 3
        assert torch.allclose(result, torch.tensor([3.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# radius_graph tests
# ---------------------------------------------------------------------------


class TestRadiusGraph:
    def test_basic_cluster(self):
        """Two pairs of close points — should get edges within each pair."""
        x = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0], [10.1, 0.0, 0.0]]
        )
        row, col = _radius_graph_native(x, r=1.0)
        edges = set(zip(row.tolist(), col.tolist()))
        # 0-1 and 2-3 should be connected (both directions)
        assert (0, 1) in edges
        assert (1, 0) in edges
        assert (2, 3) in edges
        assert (3, 2) in edges
        # 0-2 should NOT be connected
        assert (0, 2) not in edges
        assert (0, 3) not in edges

    def test_no_self_loops(self):
        """Default loop=False should not include self-loops."""
        x = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        row, col = _radius_graph_native(x, r=1.0, loop=False)
        for r_i, c_i in zip(row.tolist(), col.tolist()):
            assert r_i != c_i

    def test_self_loops(self):
        """loop=True should include self-loops for nodes within radius."""
        x = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        row, col = _radius_graph_native(x, r=1.0, loop=True)
        edges = set(zip(row.tolist(), col.tolist()))
        assert (0, 0) in edges
        assert (1, 1) in edges

    def test_batch_boundaries(self):
        """Edges should not cross batch boundaries."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],  # close to 0, same batch
                [0.2, 0.0, 0.0],
            ]
        )  # close to 0 and 1, different batch
        batch = torch.tensor([0, 0, 1])
        row, col = _radius_graph_native(x, r=1.0, batch=batch)
        edges = set(zip(row.tolist(), col.tolist()))
        assert (0, 1) in edges
        assert (1, 0) in edges
        # Node 2 is in batch 1 alone — no edges
        assert (0, 2) not in edges
        assert (2, 0) not in edges

    def test_single_atom_batch_element(self):
        """Single-atom batch elements should produce no edges, not crash."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # batch 0: single atom
                [1.0, 0.0, 0.0],  # batch 1: two atoms
                [1.1, 0.0, 0.0],
            ]
        )
        batch = torch.tensor([0, 1, 1])
        row, col = _radius_graph_native(x, r=0.5, batch=batch)
        edges = set(zip(row.tolist(), col.tolist()))
        # Batch 0 has one atom — no edges
        assert all(r_i != 0 and c_i != 0 for r_i, c_i in edges)
        # Batch 1 should have edges
        assert (1, 2) in edges
        assert (2, 1) in edges

    def test_all_single_atom_batches(self):
        """All batch elements have 1 atom — should return empty edge index."""
        x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        batch = torch.tensor([0, 1, 2])
        row, col = _radius_graph_native(x, r=0.5, batch=batch)
        assert row.shape[0] == 0
        assert col.shape[0] == 0

    def test_empty_input(self):
        """Empty input should return empty edge index."""
        x = torch.zeros(0, 3)
        batch = torch.zeros(0, dtype=torch.long)
        row, col = _radius_graph_native(x, r=1.0, batch=batch)
        assert row.shape[0] == 0
        assert col.shape[0] == 0

    def test_max_num_neighbors(self):
        """max_num_neighbors should limit edges per node."""
        # 5 points all within radius of each other
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.4, 0.0, 0.0],
            ]
        )
        row, col = _radius_graph_native(x, r=1.0, max_num_neighbors=2)
        # Each node should have at most 2 outgoing edges
        for node in range(5):
            n_neighbors = (row == node).sum().item()
            assert (
                n_neighbors <= 2
            ), f"Node {node} has {n_neighbors} neighbors, expected <= 2"

    def test_max_num_neighbors_keeps_closest(self):
        """When limited, should keep the closest neighbors."""
        x = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        row, col = _radius_graph_native(x, r=5.0, max_num_neighbors=1)
        # Node 0's closest neighbor is node 1
        node0_neighbors = col[row == 0].tolist()
        assert 1 in node0_neighbors
        assert len(node0_neighbors) == 1

    def test_no_batch_arg(self):
        """When batch=None, all points are in one batch."""
        x = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        row, col = _radius_graph_native(x, r=1.0, batch=None)
        edges = set(zip(row.tolist(), col.tolist()))
        assert (0, 1) in edges
        assert (1, 0) in edges

    def test_flow_target_to_source(self):
        """flow='target_to_source' should swap edge direction."""
        x = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]])
        row_st, col_st = _radius_graph_native(x, r=1.0, flow="source_to_target")
        row_ts, col_ts = _radius_graph_native(x, r=1.0, flow="target_to_source")
        # Edges should be the same but swapped
        edges_st = set(zip(row_st.tolist(), col_st.tolist()))
        edges_ts = set(zip(row_ts.tolist(), col_ts.tolist()))
        for r_i, c_i in edges_st:
            assert (c_i, r_i) in edges_ts

    def test_large_radius_no_edges(self):
        """Points far apart with small radius — no edges."""
        x = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        row, col = _radius_graph_native(x, r=1.0)
        assert row.shape[0] == 0

    def test_output_dtype_and_device(self):
        """Output should be long tensors on same device as input."""
        x = torch.randn(10, 3)
        batch = torch.zeros(10, dtype=torch.long)
        row, col = _radius_graph_native(x, r=5.0, batch=batch)
        assert row.dtype == torch.long
        assert col.dtype == torch.long
        assert row.device == x.device

    def test_1d_input(self):
        """1D position input (not 2D coords) should work — GearNet uses this."""
        x = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0])
        batch = torch.tensor([0, 0, 0, 1, 1])
        row, col = _radius_graph_native(x, r=1.5, batch=batch)
        edges = set(zip(row.tolist(), col.tolist()))
        assert (0, 1) in edges
        assert (1, 2) in edges
        assert (3, 4) in edges
        # No cross-batch
        assert (2, 3) not in edges

    def test_gearnet_sequential_graph_pattern(self):
        """Simulate GearNet's sequential graph: radius_graph on 1D positions.

        GearNet calls radius_graph(atom_seq_pos.float(), max_distance + 0.1, batch=atom2batch)
        where atom_seq_pos is a flat 1D tensor of sequential residue indices.
        """
        # 8 residues across 2 proteins (batch 0: 5 residues, batch 1: 3 residues)
        atom_seq_pos = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.float)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
        max_distance = 2
        row, col = _radius_graph_native(atom_seq_pos, r=max_distance + 0.1, batch=batch)
        edges = set(zip(row.tolist(), col.tolist()))
        # Within batch 0: residues within distance 2
        assert (0, 1) in edges  # dist=1
        assert (0, 2) in edges  # dist=2
        assert (0, 3) not in edges  # dist=3, too far
        # No cross-batch edges
        assert (4, 5) not in edges

    def test_gearnet_spatial_graph_pattern(self):
        """Simulate GearNet's spatial graph: radius_graph on 3D coords with max_num_neighbors=64."""
        # Small protein-like coords
        coords = torch.randn(20, 3) * 5.0  # 20 residues
        batch = torch.cat([torch.zeros(12), torch.ones(8)]).long()
        row, col = _radius_graph_native(
            coords, r=10.0, batch=batch, max_num_neighbors=64
        )
        # No cross-batch edges
        for r_i, c_i in zip(row.tolist(), col.tolist()):
            assert (
                batch[r_i] == batch[c_i]
            ), f"Cross-batch edge: {r_i} (batch {batch[r_i]}) -> {c_i} (batch {batch[c_i]})"


# ---------------------------------------------------------------------------
# Fake module tests
# ---------------------------------------------------------------------------


class TestFakeModules:
    def test_fake_torch_cluster_dunder_raises(self):
        """Dunder attributes should raise AttributeError, not return stubs."""
        tc = sys.modules.get("torch_cluster")
        if tc is None or not hasattr(type(tc), "__getattr__"):
            pytest.skip("torch_cluster is real or not patched")
        with pytest.raises(AttributeError):
            _ = tc.__file__
        with pytest.raises(AttributeError):
            _ = tc.__path__

    def test_fake_torch_cluster_stub_raises_on_call(self):
        """Non-dunder attributes should return stubs that raise NotImplementedError."""
        tc = sys.modules.get("torch_cluster")
        if tc is None or not hasattr(type(tc), "__getattr__"):
            pytest.skip("torch_cluster is real or not patched")
        with pytest.raises(NotImplementedError, match="knn"):
            tc.knn()

    def test_fake_torch_cluster_radius_graph_is_real(self):
        """radius_graph should be our native implementation, not a stub."""
        tc = sys.modules.get("torch_cluster")
        if tc is None:
            pytest.skip("torch_cluster not in sys.modules")
        assert tc.radius_graph is _radius_graph_native

    def test_inspect_works_with_fake_modules(self):
        """inspect should not crash on our fake modules (the original bug)."""
        import inspect

        tc = sys.modules.get("torch_cluster")
        if tc is None:
            pytest.skip("torch_cluster not in sys.modules")
        # This was the original crash: inspect.getsourcefile tried filename.endswith()
        # on our stub function returned by __getattr__('__file__')
        try:
            inspect.getfile(tc)
        except TypeError:
            pass  # Expected — fake module has no __file__
        # Should not raise AttributeError

    def test_torch_distributed_tensor_import(self):
        """torch.distributed.tensor should import without crashing."""
        import torch.distributed.tensor  # noqa: F401
