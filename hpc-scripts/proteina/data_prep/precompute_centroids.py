"""Precompute training set cluster centroids for novelty evaluation.

Loads the training LMDB, clusters structures by length-stratified greedy
TM-score clustering, and saves the centroid atom37 coordinates as a .pt file
consumable by `compute_novelty_metrics` in
`evaluation/proteina/generation/scripts/evaluate.py`.

Two-stage approach (avoids O(N^2) full clustering on hundreds of thousands of
chains):
  1. Group training structures into length bins.
  2. Within each bin, optionally subsample to `--max_per_length_group`, then
     greedy-cluster at TM-score >= --tm_threshold. Keep one representative
     per cluster.

Index files: if `train_lengths.npy` and `train_keys.pkl` exist alongside the
LMDB, lengths are read from them and only the keys we actually need are
fetched from LMDB. This skips streaming the entire (38-51 GB) LMDB.

Output schema: torch.save'd list of numpy arrays of shape [n_i, 37, 3].

Usage:
    python precompute_centroids.py \
        --lmdb_path /rds/.../pdb_train/lmdb/train.lmdb \
        --output_path /rds/.../centroids_pdb.pt \
        --max_per_length_group 200 \
        --length_bin_width 16 \
        --tm_threshold 0.5
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict

import lmdb
import numpy as np
import torch
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute training set centroids")
    parser.add_argument(
        "--lmdb_path", type=str, required=True, help="Path to train.lmdb file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output .pt file for centroid coordinates",
    )
    parser.add_argument(
        "--tm_threshold",
        type=float,
        default=0.5,
        help="TM-score threshold for clustering (default 0.5)",
    )
    parser.add_argument(
        "--max_per_length_group",
        type=int,
        default=200,
        help="Max input structures per length bin (subsample if more). "
        "Caps the number of *candidates* fed into clustering, not the number "
        "of centroids that emerge.",
    )
    parser.add_argument(
        "--length_bin_width",
        type=int,
        default=16,
        help="Width of length bins for grouping (default 16, aligns with "
        "generation lengths 128/256/512)",
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Max total entries to consider before binning (smoke test only)",
    )
    parser.add_argument(
        "--no_index",
        action="store_true",
        help="Force full LMDB scan instead of reading sidecar lengths/keys",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sidecar_paths(lmdb_path):
    """Return (lengths.npy, keys.pkl) paths if both exist, else (None, None)."""
    lmdb_dir = os.path.dirname(lmdb_path)
    base = os.path.basename(lmdb_path).replace(".lmdb", "")
    lengths_path = os.path.join(lmdb_dir, f"{base}_lengths.npy")
    keys_path = os.path.join(lmdb_dir, f"{base}_keys.pkl")
    if os.path.exists(lengths_path) and os.path.exists(keys_path):
        return lengths_path, keys_path
    return None, None


def select_keys_via_index(
    lmdb_path, length_bin_width, max_per_length_group, max_entries, seed
):
    """Build per-bin key selections from sidecar index files.

    Returns:
        dict: {bin_key (int): list[bytes]} -- LMDB keys to fetch per bin.
    """
    lengths_path, keys_path = _sidecar_paths(lmdb_path)
    assert lengths_path is not None, "sidecar index files not found"

    logger.info(f"Reading sidecar index: {lengths_path}, {keys_path}")
    lengths = np.load(lengths_path)
    with open(keys_path, "rb") as f:
        keys = pickle.load(f)
    assert len(keys) == len(
        lengths
    ), f"keys/lengths mismatch: {len(keys)} vs {len(lengths)}"
    logger.info(
        f"Index has {len(keys)} entries (lengths {lengths.min()}-{lengths.max()})"
    )

    rng = np.random.RandomState(seed)

    if max_entries is not None and max_entries < len(keys):
        sel = rng.choice(len(keys), size=max_entries, replace=False)
        keys = [keys[i] for i in sel]
        lengths = lengths[sel]
        logger.info(f"Subsampled to {len(keys)} entries (max_entries={max_entries})")

    bin_to_indices = defaultdict(list)
    for idx, n_res in enumerate(lengths):
        bin_key = (int(n_res) // length_bin_width) * length_bin_width
        bin_to_indices[bin_key].append(idx)

    selected_keys_per_bin = {}
    total = 0
    for bin_key in sorted(bin_to_indices.keys()):
        indices = bin_to_indices[bin_key]
        if len(indices) > max_per_length_group:
            indices = rng.choice(
                indices, size=max_per_length_group, replace=False
            ).tolist()
        selected_keys_per_bin[bin_key] = [keys[i] for i in indices]
        total += len(indices)

    logger.info(
        f"Selected {total} keys across {len(selected_keys_per_bin)} bins "
        f"(width={length_bin_width}, cap={max_per_length_group})"
    )
    return selected_keys_per_bin


def fetch_coords_for_keys(lmdb_path, keys_per_bin):
    """Open LMDB once and fetch the requested keys per bin.

    Returns:
        dict: {bin_key: list[(id_str, coords_np[n,37,3])]}
    """
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        map_size=1024 * 1024 * 1024 * 80,
    )

    coords_per_bin = {}
    n_done = 0
    n_total = sum(len(v) for v in keys_per_bin.values())
    with env.begin() as txn:
        for bin_key in sorted(keys_per_bin.keys()):
            entries = []
            for k in keys_per_bin[bin_key]:
                val = txn.get(k)
                if val is None:
                    logger.warning(f"Key {k!r} missing from LMDB, skipping")
                    continue
                data = pickle.loads(val)
                coords = data.coords.numpy()
                eid = data.id if hasattr(data, "id") else k.decode()
                entries.append((eid, coords))
                n_done += 1
                if n_done % 5000 == 0:
                    logger.info(f"  Fetched {n_done}/{n_total} entries")
            coords_per_bin[bin_key] = entries

    env.close()
    logger.info(f"Fetched {n_done} entries total")
    return coords_per_bin


def stream_load_and_bin(lmdb_path, length_bin_width, max_entries):
    """Fallback: stream the whole LMDB (slow), bin coords by length."""
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        map_size=1024 * 1024 * 1024 * 80,
    )
    coords_per_bin = defaultdict(list)
    with env.begin() as txn:
        n_total = txn.stat()["entries"]
        logger.info(f"LMDB has {n_total} entries (full stream)")
        cursor = txn.cursor()
        for i, (key, val) in enumerate(cursor):
            if max_entries is not None and i >= max_entries:
                break
            data = pickle.loads(val)
            coords = data.coords.numpy()
            eid = data.id if hasattr(data, "id") else key.decode()
            n_res = coords.shape[0]
            bin_key = (n_res // length_bin_width) * length_bin_width
            coords_per_bin[bin_key].append((eid, coords))
            if (i + 1) % 10000 == 0:
                logger.info(f"  Streamed {i + 1}/{n_total} entries")
    env.close()
    return dict(coords_per_bin)


def greedy_tm_cluster(coords_list, tm_threshold):
    """Greedy TM-score clustering. Returns indices of cluster centers.

    Args:
        coords_list: List of [n_i, 37, 3] numpy arrays.
        tm_threshold: Two structures are in same cluster if TM >= threshold.

    Returns:
        List of indices into coords_list that are cluster centers.
    """
    from proteinfoundation.metrics.tm_score import compute_tm_score

    if len(coords_list) == 0:
        return []

    centers = [0]
    for i in range(1, len(coords_list)):
        is_novel = True
        for c in centers:
            tm = compute_tm_score(coords_list[i], coords_list[c])
            if tm >= tm_threshold:
                is_novel = False
                break
        if is_novel:
            centers.append(i)

    return centers


def main():
    args = parse_args()

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    rng = np.random.RandomState(args.seed)

    # Path A: index-driven (fast)
    use_index = (not args.no_index) and (_sidecar_paths(args.lmdb_path)[0] is not None)
    if use_index:
        keys_per_bin = select_keys_via_index(
            args.lmdb_path,
            args.length_bin_width,
            args.max_per_length_group,
            args.max_entries,
            args.seed,
        )
        coords_per_bin = fetch_coords_for_keys(args.lmdb_path, keys_per_bin)
    else:
        # Path B: streaming fallback (slow)
        logger.info("Sidecar index not found or disabled; streaming full LMDB")
        coords_per_bin = stream_load_and_bin(
            args.lmdb_path, args.length_bin_width, args.max_entries
        )
        # Apply subsample cap post-hoc
        for bin_key, entries in list(coords_per_bin.items()):
            if len(entries) > args.max_per_length_group:
                sel = rng.choice(
                    len(entries), size=args.max_per_length_group, replace=False
                )
                coords_per_bin[bin_key] = [entries[i] for i in sel]

    # Cluster within each bin
    all_centroid_coords = []
    bin_summary = []
    for bin_key in sorted(coords_per_bin.keys()):
        entries = coords_per_bin[bin_key]
        group_coords = [c for _, c in entries]

        logger.info(
            f"Length bin {bin_key}-{bin_key + args.length_bin_width - 1}: "
            f"{len(group_coords)} structures, clustering..."
        )

        center_indices = greedy_tm_cluster(group_coords, args.tm_threshold)

        for ci in center_indices:
            all_centroid_coords.append(group_coords[ci])

        logger.info(f"  -> {len(center_indices)} centroids")
        bin_summary.append((bin_key, len(group_coords), len(center_indices)))

        # Free non-centroid coords for this bin to keep RSS bounded.
        # torch.save serializes the full list at end - peak mem ~= centroid
        # set + serialization buffer; without this, peak ~= all fetched coords.
        coords_per_bin[bin_key] = None
        del entries, group_coords

    logger.info(f"Total centroids: {len(all_centroid_coords)}")
    logger.info("Per-bin summary (bin_start, n_input, n_centroids):")
    for b, n_in, n_out in bin_summary:
        logger.info(f"  {b:4d}: {n_in:5d} -> {n_out:5d}")

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(all_centroid_coords, args.output_path)
    logger.info(f"Saved {len(all_centroid_coords)} centroids to {args.output_path}")


if __name__ == "__main__":
    main()
