"""Characterize MC-GearNet-Edge.

MC-GearNet-Edge differences vs CA-GearNet:
 - Output 3072-dim (concat of 6 hidden layers, each 512)
 - 6 layers (not 8)
 - Takes residue_type as a required input - node features are residue one-hot
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))
sys.path.insert(0, str(REPO_ROOT / "playground" / "proteina"))

import proteinfoundation.repa.pyg_compat  # noqa: F401, E402

from _encoder_probes import (  # noqa: E402
    EncoderProbe,
    load_proteins,
    make_embed_fn,
    run_pipeline,
)
from proteinfoundation.repa.gearnet_encoder import MCGearNetEdgePerResidueEncoder  # noqa: E402


DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_PATH = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")
MC_CKPT = os.path.join(DATA_PATH, "metric_factory/model_weights/mc_gearnet_edge.pth")


def setup_encoder(device):
    print(f"Loading MC-GearNet-Edge from {MC_CKPT}")
    encoder = MCGearNetEdgePerResidueEncoder(ckpt_path=MC_CKPT)
    encoder.eval().to(device)
    print(f"Encoder dim: {encoder.encoder_dim}")
    return encoder


@torch.no_grad()
def layerwise_fn(encoder, proteins, device):
    """The 3072-dim output is a concat of 6x512-dim layer outputs - these are
    the slabs that feed into the projector, so per-slab rank/norm tells us
    which slabs carry signal."""
    print("\n" + "=" * 70)
    print("MC-GearNet-Edge Layer-wise Representation")
    print("=" * 70)

    gn = encoder.gearnet
    n_layers = len(gn.layers)
    print(f"MC-GearNet-Edge has {n_layers} layers, output_dim={gn.output_dim}")

    n_test = min(30, len(proteins))
    layer_embs = [[] for _ in range(n_layers)]

    for graph in proteins[:n_test]:
        ca_ang = graph.coords[:, 1, :].float()
        mask = graph.coord_mask[:, 1].bool()
        if mask.sum() < 4:
            continue
        ca_ang_v = ca_ang[mask].to(device)
        rt_flat = graph.residue_type.long()[mask].clamp(0, 20).to(device)
        atom2batch = torch.zeros(ca_ang_v.shape[0], dtype=torch.long, device=device)

        n_nodes = ca_ang_v.shape[0]
        local_idx = gn._local_idx(atom2batch)
        h_v = F.one_hot(rt_flat, 21).float()
        edge_index, edge_feat59 = gn._build_edges(
            ca_ang_v, rt_flat, atom2batch, local_idx
        )
        ni, no, rel = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        lg_ei, n_lg_nodes = gn._build_line_graph(ca_ang_v, edge_index)
        lg_ni, lg_no, lg_rel = lg_ei[:, 0], lg_ei[:, 1], lg_ei[:, 2]

        edge_hidden = edge_feat59
        for i, (layer, edge_layer, bn) in enumerate(
            zip(gn.layers, gn.edge_layers, gn.batch_norms)
        ):
            edge_hidden = edge_layer(edge_hidden, lg_ni, lg_no, lg_rel, n_lg_nodes)
            h_new = layer(h_v, ni, no, rel, n_nodes, edge_input=edge_hidden)
            if h_new.shape == h_v.shape:
                h_new = h_new + h_v
            h_new = bn(h_new)
            layer_embs[i].append(h_new.float().cpu())
            h_v = h_new

    print(
        f"\n{'Layer':>5} | {'Dim':>4} | {'Eff Rank':>9} | {'Mean Norm':>10} | {'Sparsity':>9}"
    )
    print("-" * 58)
    rows = []
    for i in range(n_layers):
        if not layer_embs[i]:
            continue
        H = torch.cat(layer_embs[i], dim=0)
        Hs = H[torch.randperm(H.shape[0])[:15000]] if H.shape[0] > 15000 else H
        centered = Hs - Hs.mean(dim=0)
        S = torch.linalg.svdvals(centered).numpy()
        p = (S**2) / (S**2).sum()
        p = p[p > 0]
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        norm_mean = float(H.norm(dim=-1).mean())
        sparsity = float((H == 0).float().mean())
        print(
            f"  {i:>3d} | {H.shape[1]:>4d} | {eff_rank:>9.1f} | {norm_mean:>10.4f} | {sparsity:>8.4f}"
        )
        rows.append(
            {
                "layer": i,
                "dim": int(H.shape[1]),
                "eff_rank": eff_rank,
                "mean_norm": norm_mean,
                "sparsity": sparsity,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=200)
    ap.add_argument("--random-seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    proteins = load_proteins(LMDB_PATH, args.n_proteins, seed=args.random_seed)
    encoder = setup_encoder(device)

    import time as _time

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "results", _time.strftime("%Y%m%d_%H%M%S")
    )
    probe = EncoderProbe(
        name="mc-gearnet-edge",
        encoder=encoder,
        embed_fn=make_embed_fn(encoder, device),
        is_3d_aware=True,
        accepts_residue_type=True,
        context_mode="structural",
        layerwise_fn=layerwise_fn,
        output_dir=output_dir,
    )
    run_pipeline(probe, proteins, device)


if __name__ == "__main__":
    main()
