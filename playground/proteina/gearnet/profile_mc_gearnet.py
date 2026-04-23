"""Profile MC-GearNet-Edge forward pass to identify the step-time bottleneck.

Runs a realistic batch (bs=80, n=128 residues) through GearNetEdge and times
each section: _local_idx, _build_edges, _build_line_graph, and the 6-layer
GNN forward. Also compares against GearNet-CA at the same batch size.

Usage (on a GPU node via profile_mc_gearnet.sh):
    python -u playground/proteina/gearnet/profile_mc_gearnet.py
    python -u playground/proteina/gearnet/profile_mc_gearnet.py --bs 40
"""

import argparse
import os
import sys
import time

import torch
import torch.cuda

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src/proteina"))

# Must patch before any proteina import
try:
    import proteinfoundation.repa.pyg_compat  # noqa: F401
except Exception:
    pass

from proteinfoundation.metrics.gearnet_utils import (
    GearNetEdge,
)
from proteinfoundation.repa.gearnet_encoder import (
    GearNetPerResidueEncoder,
    MCGearNetEdgePerResidueEncoder,
)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        cuda_sync()
        self.elapsed = time.perf_counter() - self.t0
        print(f"  {self.label:<40s} {self.elapsed*1000:8.1f} ms")


def make_flat_batch(bs, n, device):
    """Simulate a realistic training batch: bs proteins each of length n."""
    # Vary lengths slightly as in real data (±10%)
    lengths = torch.randint(max(4, n - n // 10), n + 1, (bs,))
    coords_list, res_list, batch_list = [], [], []
    for i, seq_len in enumerate(lengths.tolist()):
        coords_list.append(torch.randn(seq_len, 3, device=device) * 5.0)
        res_list.append(torch.randint(0, 20, (seq_len,), device=device))
        batch_list.append(torch.full((seq_len,), i, dtype=torch.long, device=device))
    return torch.cat(coords_list), torch.cat(res_list), torch.cat(batch_list)


def make_dense_batch(bs, n, device):
    """Dense [b, n, 3] format for the encoder wrappers."""
    coords = torch.randn(bs, n, 3, device=device) * 5.0
    mask = torch.ones(bs, n, dtype=torch.bool, device=device)
    # Vary lengths: last 10% of residues in first half of batch are padding
    mask[: bs // 2, n - n // 10 :] = False
    res = torch.randint(0, 20, (bs, n), device=device)
    return coords, mask, res


def debug_line_graph(model, coords, res, atom2batch):
    """Quick sanity check: verify lg indices are in range before timing."""
    with torch.no_grad():
        local_idx = model._local_idx(atom2batch)
        edge_index, _ = model._build_edges(coords, res, atom2batch, local_idx)
        n_edges = edge_index.shape[0]
        lg_ei, n_lg_nodes = model._build_line_graph(coords, edge_index)
        print("\n--- Line graph sanity ---")
        print(
            f"  n_nodes={coords.shape[0]}, n_edges={n_edges}, n_lg_nodes={n_lg_nodes}"
        )
        if lg_ei.shape[0] > 0:
            lg_src, lg_dst = lg_ei[:, 0], lg_ei[:, 1]
            print(f"  lg_edges={lg_ei.shape[0]}")
            print(
                f"  lg_src: min={lg_src.min().item()}, max={lg_src.max().item()} (should be < {n_lg_nodes})"
            )
            print(
                f"  lg_dst: min={lg_dst.min().item()}, max={lg_dst.max().item()} (should be < {n_lg_nodes})"
            )
            ok = (lg_src.max() < n_lg_nodes) and (lg_dst.max() < n_lg_nodes)
            print(f"  indices in range: {ok}")
        else:
            print("  lg_edges=0 (empty line graph)")


def profile_mc_gearnet_internals(model, coords, res, atom2batch, n_warmup=3, n_rep=10):
    """Time each internal section of GearNetEdge.forward."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(coords, res, atom2batch)

    print(f"\n--- GearNetEdge internals (n_rep={n_rep}, mean over reps) ---")
    print(
        f"  Batch: {atom2batch.max().item()+1} proteins, "
        f"{coords.shape[0]} total residues"
    )

    times = {
        k: 0.0
        for k in ["local_idx", "build_edges", "build_line_graph", "gnn_layers", "total"]
    }

    with torch.no_grad():
        for _ in range(n_rep):
            cuda_sync()
            t_total = time.perf_counter()

            t0 = time.perf_counter()
            cuda_sync()
            local_idx = model._local_idx(atom2batch)
            cuda_sync()
            times["local_idx"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            cuda_sync()
            edge_index, edge_feat59 = model._build_edges(
                coords, res, atom2batch, local_idx
            )
            cuda_sync()
            times["build_edges"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            cuda_sync()
            lg_ei, n_lg_nodes = model._build_line_graph(coords, edge_index)
            cuda_sync()
            times["build_line_graph"] += time.perf_counter() - t0

            ni, no, rel = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
            lg_ni, lg_no, lg_rel = lg_ei[:, 0], lg_ei[:, 1], lg_ei[:, 2]
            h_v = torch.nn.functional.one_hot(res.clamp(0, 20), 21).float()
            edge_hidden = edge_feat59
            n_nodes = coords.shape[0]

            t0 = time.perf_counter()
            cuda_sync()
            hiddens = []
            for layer, edge_layer, bn in zip(
                model.layers, model.edge_layers, model.batch_norms
            ):
                edge_hidden = edge_layer(edge_hidden, lg_ni, lg_no, lg_rel, n_lg_nodes)
                h_new = layer(h_v, ni, no, rel, n_nodes, edge_input=edge_hidden)
                if h_new.shape == h_v.shape:
                    h_new = h_new + h_v
                h_new = bn(h_new)
                hiddens.append(h_new)
                h_v = h_new
            cuda_sync()
            times["gnn_layers"] += time.perf_counter() - t0

            cuda_sync()
            times["total"] += time.perf_counter() - t_total

    for k, v in times.items():
        mean_ms = v / n_rep * 1000
        pct = v / times["total"] * 100
        print(f"  {k:<40s} {mean_ms:8.1f} ms  ({pct:4.1f}%)")

    print(
        f"\n  Edge counts: {edge_index.shape[0]} edges, "
        f"{lg_ei.shape[0]} line-graph edges"
    )
    return times


def profile_encoder_wrappers(mc_ckpt, ca_ckpt, bs, n, device, n_warmup=3, n_rep=10):
    """Compare full encoder wrapper forward passes (dense in, dense out)."""
    coords_nm, mask, res = make_dense_batch(bs, n, device)
    # Convert to nm (training uses nm)
    coords_nm = coords_nm / 10.0

    print(f"\n--- Encoder wrapper comparison (bs={bs}, n={n}, n_rep={n_rep}) ---")

    if os.path.exists(mc_ckpt):
        enc_mc = MCGearNetEdgePerResidueEncoder(mc_ckpt).to(device)
        for _ in range(n_warmup):
            with torch.no_grad():
                enc_mc(coords_nm, mask, residue_type=res)
        t0 = time.perf_counter()
        for _ in range(n_rep):
            with torch.no_grad():
                enc_mc(coords_nm, mask, residue_type=res)
        cuda_sync()
        mc_ms = (time.perf_counter() - t0) / n_rep * 1000
        print(f"  MC-GearNet-Edge  (3072-dim): {mc_ms:8.1f} ms/call")
    else:
        print(f"  MC-GearNet-Edge: checkpoint not found at {mc_ckpt}")

    if os.path.exists(ca_ckpt):
        enc_ca = GearNetPerResidueEncoder(ca_ckpt).to(device)
        for _ in range(n_warmup):
            with torch.no_grad():
                enc_ca(coords_nm, mask)
        t0 = time.perf_counter()
        for _ in range(n_rep):
            with torch.no_grad():
                enc_ca(coords_nm, mask)
        cuda_sync()
        ca_ms = (time.perf_counter() - t0) / n_rep * 1000
        print(f"  GearNet-CA       ( 512-dim): {ca_ms:8.1f} ms/call")
        if os.path.exists(mc_ckpt):
            print(f"  MC/CA ratio: {mc_ms/ca_ms:.1f}x")
    else:
        print(f"  GearNet-CA: checkpoint not found at {ca_ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=80)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--n_rep", type=int, default=10)
    parser.add_argument("--n_warmup", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Report which radius_graph implementation is actually running
    from torch_geometric.nn import radius_graph as rg_used
    import torch_cluster

    print(f"radius_graph source module: {rg_used.__module__}")
    print(
        f"torch_cluster.radius_graph: {getattr(torch_cluster, 'radius_graph', 'N/A')}"
    )

    data_path = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
    mc_ckpt = os.path.join(
        data_path, "metric_factory/model_weights/mc_gearnet_edge.pth"
    )
    ca_ckpt = os.path.join(data_path, "metric_factory/model_weights/gearnet_ca.pth")

    # Profile MC-GearNet internals with random weights (no checkpoint needed)
    torch.manual_seed(0)
    model = GearNetEdge().to(device)
    model.eval()
    coords, res, atom2batch = make_flat_batch(args.bs, args.n, device)

    # Sanity check line graph indices before timing (OOM otherwise)
    debug_line_graph(model, coords, res, atom2batch)

    profile_mc_gearnet_internals(
        model,
        coords,
        res,
        atom2batch,
        n_warmup=args.n_warmup,
        n_rep=args.n_rep,
    )

    # Full encoder wrapper comparison (needs real checkpoints)
    profile_encoder_wrappers(
        mc_ckpt,
        ca_ckpt,
        args.bs,
        args.n,
        device,
        n_warmup=args.n_warmup,
        n_rep=args.n_rep,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
