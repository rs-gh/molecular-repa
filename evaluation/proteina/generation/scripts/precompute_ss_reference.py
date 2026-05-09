"""Precompute reference (helix, sheet, coil) fraction distributions for SS metrics.

Streams a sample of structures from a training LMDB, runs biotite's P-SEA
implementation (`annotate_sse`) on the Cα coordinates, and saves a `[N_ref, 3]`
float tensor of per-structure (f_H, f_E, f_C) fractions. Consumed by
`evaluate.compute_ss_metrics` via `--ss_reference_pdb_path` / `--ss_reference_afdb_path`.

Mirrors `precompute_centroids.py`'s sidecar-driven sampling so we hit the
same reference distribution the FID and centroid-novelty metrics use.

Usage:
    python precompute_ss_reference.py \
        --lmdb_path /rds/.../pdb_train/lmdb/train.lmdb \
        --output_path /rds/.../ss_reference_pdb.pt \
        --max_samples 5000

CA atom index in atom37 layout is 1 (AlphaFold convention: N=0, CA=1, C=2, O=3, ...).
"""

import argparse
import os
import pickle
import sys

import biotite.structure as struc
import lmdb
import numpy as np
import torch
from loguru import logger

CA_IDX = 1  # atom37 index for Cα


def parse_args():
    ap = argparse.ArgumentParser(description="Precompute SS reference distribution")
    ap.add_argument("--lmdb_path", required=True, type=str)
    ap.add_argument("--output_path", required=True, type=str)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Number of reference structures to sample. 5000 is plenty for a "
        "stable 3-bin reference (binomial SE on f_H ~ 0.005 at p=0.4).",
    )
    ap.add_argument(
        "--min_len",
        type=int,
        default=40,
        help="Drop structures shorter than this (P-SEA needs ≥ ~5 residues "
        "and very short chains contribute noise).",
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="Drop structures longer than this. Matches our generation length "
        "regime so the reference is comparable.",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _sidecar_paths(lmdb_path: str):
    lmdb_dir = os.path.dirname(lmdb_path)
    base = os.path.basename(lmdb_path).replace(".lmdb", "")
    lengths_path = os.path.join(lmdb_dir, f"{base}_lengths.npy")
    keys_path = os.path.join(lmdb_dir, f"{base}_keys.pkl")
    if os.path.exists(lengths_path) and os.path.exists(keys_path):
        return lengths_path, keys_path
    return None, None


def select_keys(
    lmdb_path: str, max_samples: int, min_len: int, max_len: int, seed: int
):
    """Return a length-filtered sample of LMDB keys via the sidecar index."""
    lengths_path, keys_path = _sidecar_paths(lmdb_path)
    assert lengths_path is not None, f"sidecar index not found for {lmdb_path}"

    lengths = np.load(lengths_path)
    with open(keys_path, "rb") as f:
        keys = pickle.load(f)

    eligible = np.where((lengths >= min_len) & (lengths <= max_len))[0]
    logger.info(
        f"Index: {len(keys)} entries; {len(eligible)} pass length filter "
        f"[{min_len}, {max_len}]"
    )
    rng = np.random.RandomState(seed)
    if len(eligible) > max_samples:
        eligible = rng.choice(eligible, size=max_samples, replace=False)
    return [keys[i] for i in eligible]


def coords_to_ss_fractions(ca_coords: np.ndarray) -> np.ndarray:
    """Run biotite annotate_sse on CA coords; return (f_H, f_E, f_C)."""
    n = ca_coords.shape[0]
    arr = struc.AtomArray(n)
    arr.coord = ca_coords.astype(np.float32)
    arr.atom_name = np.array(["CA"] * n)
    arr.element = np.array(["C"] * n)
    arr.res_id = np.arange(1, n + 1)
    arr.res_name = np.array(["ALA"] * n)
    arr.chain_id = np.array(["A"] * n)
    arr.hetero = np.zeros(n, dtype=bool)
    sse = struc.annotate_sse(arr)
    sse = sse[sse != ""]
    if sse.size == 0:
        return None
    counts = np.array(
        [(sse == "a").sum(), (sse == "b").sum(), (sse == "c").sum()],
        dtype=float,
    )
    return counts / counts.sum()


def main():
    args = parse_args()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    selected = select_keys(
        args.lmdb_path, args.max_samples, args.min_len, args.max_len, args.seed
    )
    logger.info(f"Sampling {len(selected)} structures from {args.lmdb_path}")

    env = lmdb.open(
        args.lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        map_size=1024 * 1024 * 1024 * 80,
    )

    fracs: list[np.ndarray] = []
    n_dropped = 0
    with env.begin() as txn:
        for i, k in enumerate(selected):
            val = txn.get(k)
            if val is None:
                n_dropped += 1
                continue
            data = pickle.loads(val)
            coords = data.coords.numpy()  # [n_res, 37, 3]
            ca = coords[:, CA_IDX, :]
            f = coords_to_ss_fractions(ca)
            if f is None:
                n_dropped += 1
                continue
            fracs.append(f)
            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i + 1}/{len(selected)} (kept {len(fracs)})")
    env.close()

    arr = np.stack(fracs, axis=0)
    means = arr.mean(axis=0)
    logger.info(
        f"Reference SS distribution (N={arr.shape[0]}, dropped={n_dropped}): "
        f"H={means[0]:.3f}, E={means[1]:.3f}, C={means[2]:.3f}"
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(torch.from_numpy(arr).float(), args.output_path)
    logger.info(f"Saved [{arr.shape[0]}, 3] tensor to {args.output_path}")


if __name__ == "__main__":
    main()
