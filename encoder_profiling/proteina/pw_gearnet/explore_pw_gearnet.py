"""Characterize ProteinWorkshop GearNet-Edge.

PW GearNet-Edge is the 6-layer relational GNN variant pretrained by the
ProteinWorkshop benchmark (Zenodo 8287754). Architecture vs MC-GearNet-Edge:
  - Node input is 43-dim: AA one-hot (23) + seq pos enc (16) + alpha (2) + kappa (2).
    Alpha/kappa are CA-only dihedral and bend angles.
  - Edge input is 89-dim.
  - 6 layers x 512 -> concat 3072 output. Different graph construction.

Usage:
  python explore_pw_gearnet.py --ckpt path/to/pw_gearnet_*.ckpt --variant torsional
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
from proteinfoundation.repa.gearnet_encoder import PWGearNetEdgePerResidueEncoder  # noqa: E402


DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_PATH = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")


def setup_encoder(ckpt_path, device):
    print(f"Loading PW GearNet-Edge from {ckpt_path}")
    encoder = PWGearNetEdgePerResidueEncoder(ckpt_path=ckpt_path)
    encoder.eval().to(device)
    print(f"Encoder dim: {encoder.encoder_dim}")
    return encoder


@torch.no_grad()
def layerwise_fn(encoder, proteins, device):
    """6 hidden layer outputs that concatenate into the 3072-dim final embedding."""
    print("\n" + "=" * 70)
    print("PW GearNet-Edge Layer-wise Representation")
    print("=" * 70)
    from torch_scatter import scatter_sum

    gn = encoder.gearnet
    n_layers = len(gn.layers)
    print(f"PW GearNet-Edge has {n_layers} layers, output_dim={gn.output_dim}")

    n_test = min(30, len(proteins))
    layer_embs = [[] for _ in range(n_layers)]

    for graph in proteins[:n_test]:
        ca_coords = graph.coords[:, 1, :].float()
        mask = graph.coord_mask[:, 1].bool()
        if mask.sum() < 4:
            continue
        ca_ang = ca_coords[mask].to(device)
        rt_flat = graph.residue_type.long()[mask].clamp(0, 20).to(device)
        atom2batch = torch.zeros(ca_ang.shape[0], dtype=torch.long, device=device)

        n_nodes = ca_ang.shape[0]
        local_idx = gn._local_idx(atom2batch)
        h_v = gn._node_features(ca_ang, rt_flat, local_idx, atom2batch)
        ni, no, rel = gn._build_knn_graph(ca_ang, atom2batch)
        edge_feat = gn._edge_features(h_v, ca_ang, local_idx, ni, no, rel)
        lg_src, lg_dst, lg_rel, n_lg = gn._build_line_graph(ca_ang, ni, no)

        edge_input = edge_feat
        for i, (layer, edge_layer, bn) in enumerate(
            zip(gn.layers, gn.edge_layers, gn.batch_norms)
        ):
            h_new = layer(h_v, ni, no, rel, n_nodes, edge_feat=edge_feat)
            if h_new.shape == h_v.shape:
                h_new = h_new + h_v
            edge_hidden = edge_layer(edge_input, lg_src, lg_dst, lg_rel, n_lg)
            node_out_idx = no * gn._NUM_RELATION + rel
            update = scatter_sum(
                edge_hidden,
                node_out_idx,
                dim=0,
                dim_size=n_nodes * gn._NUM_RELATION,
            )
            update = update.view(n_nodes, gn._NUM_RELATION * edge_hidden.shape[1])
            update = F.relu(layer.linear(update))
            h_new = h_new + update
            h_new = bn(h_new)
            layer_embs[i].append(h_new.float().cpu())
            h_v = h_new
            edge_input = edge_hidden

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
    ap.add_argument(
        "--ckpt", type=str, required=True, help="Path to PW GearNet-Edge .ckpt"
    )
    ap.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Short label (e.g. 'torsional' or 'structure') for log header",
    )
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
    encoder = setup_encoder(args.ckpt, device)

    import time as _time

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__),
        "results",
        f"{args.variant}_{_time.strftime('%Y%m%d_%H%M%S')}",
    )
    probe = EncoderProbe(
        name=f"pw-gearnet-edge:{args.variant}",
        encoder=encoder,
        embed_fn=make_embed_fn(encoder, device),
        is_3d_aware=True,
        accepts_residue_type=True,
        context_mode="structural",
        layerwise_fn=layerwise_fn,
        notes=[f"ckpt: {args.ckpt}"],
        output_dir=output_dir,
    )
    run_pipeline(probe, proteins, device)


if __name__ == "__main__":
    main()
