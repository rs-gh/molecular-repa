"""Characterize the ProteinMPNN CA-only structure encoder.

Mirrors encoder_profiling/proteina/gearnet/explore_gearnet.py: delegates the
standard battery to encoder_profiling.proteina._probes.lib and only keeps
encoder-specific bits (checkpoint loading, layerwise hook).

Run:
  source .venv/bin/activate
  python encoder_profiling/proteina/mpnn/explore_mpnn.py [--random-init]
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
from proteinfoundation.repa.mpnn_encoder import ProteinMPNNPerResidueEncoder  # noqa: E402


DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_PATH = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")
DEFAULT_MPNN_CKPT = os.path.join(
    os.environ.get(
        "PROTEINMPNN_WEIGHTS_DIR",
        "/home/sr2173/rds/hpc-work/proteina/ProteinMPNN",
    ),
    "ca_model_weights/v_48_020.pt",
)


def setup_encoder(device, *, ckpt_path, random_init=False, random_seed=0):
    if random_init:
        print(f"Initializing ProteinMPNN with RANDOM weights (seed={random_seed})")
        encoder = ProteinMPNNPerResidueEncoder(
            ckpt_path=None, random_init=True, random_seed=random_seed
        )
    else:
        print(f"Loading ProteinMPNN encoder from {ckpt_path}")
        encoder = ProteinMPNNPerResidueEncoder(ckpt_path=ckpt_path)
    encoder.eval().to(device)
    print(f"Encoder dim: {encoder.encoder_dim}")
    return encoder


@torch.no_grad()
def layerwise_fn(encoder, proteins, device):
    """Run each EncLayer in turn and report rank/norm/sparsity per layer.

    Mirrors the GearNet layerwise probe but uses the ProteinMPNN encoder
    pipeline (CA features -> KNN graph -> EncLayer stack).
    """
    print("\n" + "=" * 70)
    print("ProteinMPNN Layer-wise Representation")
    print("=" * 70)

    # protein_mpnn_utils ships without __init__; import-helper in the encoder
    # module already added it to sys.path.
    from protein_mpnn_utils import gather_nodes  # noqa: E402

    n_layers = len(encoder.mpnn.encoder_layers)
    print(f"ProteinMPNN has {n_layers} encoder layers")
    n_test = min(30, len(proteins))
    layer_embs = [[] for _ in range(n_layers)]

    for graph in proteins[:n_test]:
        # graphein protein -> CA-only nm coords + mask. Match the convention
        # used by graph_to_inputs / make_embed_fn in lib.
        ca_ang = graph.coords[:, 1, :].float()  # [n, 3] in Angstroms
        mask_b = graph.coord_mask[:, 1].bool()  # [n] valid CAs
        # The encoder expects nm; here we go straight to A inside the model.
        ca_nm = (ca_ang / 10.0).unsqueeze(0).to(device)  # [1, n, 3] in nm
        mask_f = mask_b.float().unsqueeze(0).to(device)  # [1, n] float
        n = ca_nm.shape[1]

        ca = ca_nm * 10.0  # back to A inside encoder math
        residue_idx = (
            torch.arange(n, device=device, dtype=torch.long).unsqueeze(0).expand(1, n)
        )
        chain_encoding_all = torch.zeros(1, n, dtype=torch.long, device=device)

        E, E_idx = encoder.mpnn.features(ca, mask_f, residue_idx, chain_encoding_all)
        h_V = torch.zeros(
            (E.shape[0], E.shape[1], E.shape[-1]), device=E.device, dtype=E.dtype
        )
        h_E = encoder.mpnn.W_e(E)
        mask_attend = gather_nodes(mask_f.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask_f.unsqueeze(-1) * mask_attend

        valid = mask_b.to(device)
        for i, layer in enumerate(encoder.mpnn.encoder_layers):
            h_V, h_E = layer(h_V, h_E, E_idx, mask_f, mask_attend)
            # Take only valid residues; flatten batch dim.
            layer_embs[i].append(h_V[0, valid].float().cpu())

    rows = []
    print(f"\n{'Layer':>5} | {'RankMe':>9} | {'Mean Norm':>10} | {'Sparsity':>9}")
    print("-" * 50)
    for i in range(n_layers):
        H = torch.cat(layer_embs[i], dim=0)
        S = torch.linalg.svdvals(H).numpy()
        p = S / S.sum()
        p = p[p > 0]
        rankme = float(np.exp(-np.sum(p * np.log(p))))
        mean_norm = float(H.norm(dim=-1).mean())
        sparsity = float((H == 0).float().mean())
        print(f"  {i:>3d} | {rankme:>9.1f} | {mean_norm:>10.4f} | {sparsity:>8.4f}")
        rows.append(
            {
                "layer": i,
                "rankme": rankme,
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
        help="Use random ProteinMPNN weights (architecture-only baseline).",
    )
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=DEFAULT_MPNN_CKPT,
        help="Path to ProteinMPNN .pt (default: v_48_020).",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    proteins = load_proteins(LMDB_PATH, args.n_proteins, seed=args.random_seed)
    encoder = setup_encoder(
        device,
        ckpt_path=args.ckpt,
        random_init=args.random_init,
        random_seed=args.init_seed,
    )

    import time as _time

    name = f"proteinmpnn{'-random' if args.random_init else ''}"
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "results", _time.strftime("%Y%m%d_%H%M%S")
    )
    probe = EncoderProbe(
        name=name,
        encoder=encoder,
        embed_fn=make_embed_fn(encoder, device),
        is_3d_aware=True,
        accepts_residue_type=False,  # CA-only ProteinMPNN ignores residue_type
        context_mode="structural",
        layerwise_fn=layerwise_fn,
        output_dir=output_dir,
    )
    run_pipeline(probe, proteins, device)


if __name__ == "__main__":
    main()
