"""Precompute training set cluster centroids for novelty evaluation.

Loads the training LMDB, clusters structures by sequence similarity using
greedy TM-score clustering (or just samples representatives), and saves
the centroid atom37 coordinates as a .pt file.

For large training sets, direct pairwise TM-score clustering is infeasible
(O(n^2)). Instead we use a two-stage approach:
  1. Group training structures by length (within ±5 residues).
  2. Within each length group, greedily cluster using TM-score >= threshold.
  3. Save the first structure in each cluster as the centroid.

Usage:
    python precompute_centroids.py \
        --lmdb_path /rds/.../pdb_train/lmdb/train.lmdb \
        --output_path /rds/.../centroids.pt \
        --max_per_length_group 200 \
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
        help="Max structures per length group to cluster (subsample if more)",
    )
    parser.add_argument(
        "--length_bin_width",
        type=int,
        default=10,
        help="Width of length bins for grouping (default 10)",
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Max total entries to process (for smoke testing)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_training_coords(lmdb_path, max_entries=None):
    """Load atom37 coordinates from training LMDB.

    Returns:
        List of (id_str, coords_atom37_numpy) tuples.
    """
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        map_size=1024 * 1024 * 1024 * 60,
    )
    entries = []
    with env.begin() as txn:
        n_total = txn.stat()["entries"]
        logger.info(f"LMDB has {n_total} entries")

        cursor = txn.cursor()
        for i, (key, val) in enumerate(cursor):
            if max_entries is not None and i >= max_entries:
                break
            data = pickle.loads(val)
            coords = data.coords.numpy()  # [n_res, 37, 3]
            entry_id = data.id if hasattr(data, "id") else key.decode()
            entries.append((entry_id, coords))

            if (i + 1) % 10000 == 0:
                logger.info(f"  Loaded {i + 1}/{n_total} entries")

    env.close()
    logger.info(f"Loaded {len(entries)} training structures")
    return entries


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

    # Load training data
    entries = load_training_coords(args.lmdb_path, max_entries=args.max_entries)

    # Group by length bin
    length_groups = defaultdict(list)
    for idx, (entry_id, coords) in enumerate(entries):
        n_res = coords.shape[0]
        bin_key = (n_res // args.length_bin_width) * args.length_bin_width
        length_groups[bin_key].append(idx)

    logger.info(
        f"Grouped into {len(length_groups)} length bins "
        f"(width={args.length_bin_width})"
    )

    # Cluster within each length group
    all_centroid_coords = []
    for bin_key in sorted(length_groups.keys()):
        group_indices = length_groups[bin_key]

        # Subsample if too large
        if len(group_indices) > args.max_per_length_group:
            group_indices = rng.choice(
                group_indices, size=args.max_per_length_group, replace=False
            ).tolist()

        group_coords = [entries[i][1] for i in group_indices]

        logger.info(
            f"Length bin {bin_key}-{bin_key + args.length_bin_width}: "
            f"{len(group_coords)} structures, clustering..."
        )

        center_indices = greedy_tm_cluster(group_coords, args.tm_threshold)

        for ci in center_indices:
            all_centroid_coords.append(group_coords[ci])

        logger.info(f"  -> {len(center_indices)} centroids")

    logger.info(f"Total centroids: {len(all_centroid_coords)}")

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(all_centroid_coords, args.output_path)
    logger.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
