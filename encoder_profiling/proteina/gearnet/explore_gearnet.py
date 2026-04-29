"""Characterize the CA-only GearNet encoder.

Delegates the standard battery of analyses to encoder_profiling.proteina._probes.lib;
keeps only the encoder-specific bits here (checkpoint loading, layerwise hook).

Run:
  source .venv/bin/activate
  python encoder_profiling/proteina/gearnet/explore_gearnet.py [--random-init]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))
sys.path.insert(0, str(REPO_ROOT / "encoder_profiling" / "proteina"))

import proteinfoundation.repa.pyg_compat  # noqa: F401, E402

from _probes import (  # noqa: E402
    EncoderProbe,
    load_proteins,
    make_embed_fn,
    run_pipeline,
)
from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder  # noqa: E402


DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_PATH = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")
GEARNET_CKPT = os.path.join(DATA_PATH, "metric_factory/model_weights/gearnet_ca.pth")


def setup_encoder(device, *, random_init=False, random_seed=0):
    if random_init:
        print(f"Initializing CA-GearNet with RANDOM weights (seed={random_seed})")
        encoder = GearNetPerResidueEncoder(
            ckpt_path=None, random_init=True, random_seed=random_seed
        )
    else:
        print(f"Loading CA-GearNet from {GEARNET_CKPT}")
        encoder = GearNetPerResidueEncoder(ckpt_path=GEARNET_CKPT)
    encoder.eval().to(device)
    print(f"Encoder dim: {encoder.encoder_dim}")
    return encoder


@torch.no_grad()
def layerwise_fn(encoder, proteins, device):
    """Walk CA-GearNet's 8 layers and report per-layer rank/norm/sparsity.

    Returns a list[dict] with one row per layer (consumed by run_pipeline).
    """
    print("\n" + "=" * 70)
    print("CA-GearNet Layer-wise Representation")
    print("=" * 70)

    n_layers = len(encoder.gearnet.layers)
    print(f"GearNet has {n_layers} layers")
    n_test = min(30, len(proteins))
    layer_embs = [[] for _ in range(n_layers)]

    for graph in proteins[:n_test]:
        ca_ang = graph.coords[:, 1, :].float()  # A
        mask = graph.coord_mask[:, 1].bool()
        coords, atom_type, atom_seq_pos, atom2batch = encoder._dense_to_gearnet_inputs(
            ca_ang.unsqueeze(0).to(device),
            mask.float().unsqueeze(0).to(device),
        )
        coords = coords.float()
        h_v = encoder.gearnet.node_feature(atom_type, atom_seq_pos)
        edge_list = encoder.gearnet.construct_graph(atom_seq_pos, coords, atom2batch)
        h_e = encoder.gearnet.edge_feature(edge_list, atom_seq_pos, coords, atom2batch)
        for i, layer in enumerate(encoder.gearnet.layers):
            h_v = layer(h_v, edge_list, h_e)
            layer_embs[i].append(h_v.float().cpu())

    rows = []
    print(f"\n{'Layer':>5} | {'Eff Rank':>9} | {'Mean Norm':>10} | {'Sparsity':>9}")
    print("-" * 50)
    for i in range(n_layers):
        H = torch.cat(layer_embs[i], dim=0)
        centered = H - H.mean(dim=0)
        S = torch.linalg.svdvals(centered).numpy()
        p = (S**2) / (S**2).sum()
        p = p[p > 0]
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        mean_norm = float(H.norm(dim=-1).mean())
        sparsity = float((H == 0).float().mean())
        print(f"  {i:>3d} | {eff_rank:>9.1f} | {mean_norm:>10.4f} | {sparsity:>8.4f}")
        rows.append(
            {
                "layer": i,
                "eff_rank": eff_rank,
                "mean_norm": mean_norm,
                "sparsity": sparsity,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=200)
    ap.add_argument(
        "--random-seed", type=int, default=None, help="Randomize protein selection."
    )
    ap.add_argument(
        "--random-init",
        action="store_true",
        help="Initialize GearNet weights randomly (architecture-only baseline).",
    )
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write results.json + layerwise.csv (default: ./results/<timestamp>).",
    )
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    proteins = load_proteins(LMDB_PATH, args.n_proteins, seed=args.random_seed)
    encoder = setup_encoder(
        device, random_init=args.random_init, random_seed=args.init_seed
    )

    import time as _time

    name = f"ca-gearnet{'-random' if args.random_init else ''}"
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "results", _time.strftime("%Y%m%d_%H%M%S")
    )
    probe = EncoderProbe(
        name=name,
        encoder=encoder,
        embed_fn=make_embed_fn(encoder, device),
        is_3d_aware=True,
        accepts_residue_type=False,  # CA-GearNet ignores residue_type
        context_mode="structural",
        layerwise_fn=layerwise_fn,
        output_dir=output_dir,
    )
    run_pipeline(probe, proteins, device)


if __name__ == "__main__":
    main()
