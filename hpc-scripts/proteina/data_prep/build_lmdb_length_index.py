"""Build a length index for proteina LMDB files.

Scans each LMDB split, extracts the number of residues per entry,
and saves a numpy array + key list alongside the LMDB. This enables
fast max_length filtering without re-scanning the LMDB at every startup.

Usage:
    source .venv/bin/activate
    python hpc-scripts/proteina/data_prep/build_lmdb_length_index.py --lmdb_dir /path/to/lmdb/

Output files (per split):
    {lmdb_dir}/{split}_lengths.npy   - int32 array of residue counts
    {lmdb_dir}/{split}_keys.pkl      - ordered list of LMDB keys (bytes)
"""

import argparse
import os
import pickle
import time

import lmdb
import numpy as np


def build_index(lmdb_path: str) -> tuple:
    """Scan an LMDB file and extract lengths.

    Args:
        lmdb_path: Path to the .lmdb file.

    Returns:
        (keys, lengths): Ordered list of keys and int32 array of residue counts.
    """
    db = lmdb.open(
        lmdb_path,
        map_size=50 * (1024**3),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )

    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        lengths = np.zeros(len(keys), dtype=np.int32)

        for i, k in enumerate(keys):
            data = pickle.loads(txn.get(k))
            lengths[i] = data.coords.shape[0]

            if (i + 1) % 50000 == 0:
                print(f"  Scanned {i + 1}/{len(keys)} entries...")

    db.close()
    return keys, lengths


def main():
    parser = argparse.ArgumentParser(description="Build LMDB length index")
    parser.add_argument(
        "--lmdb_dir",
        type=str,
        default="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb",
        help="Directory containing {split}.lmdb files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="LMDB splits to index",
    )
    args = parser.parse_args()

    for split in args.splits:
        lmdb_path = os.path.join(args.lmdb_dir, f"{split}.lmdb")
        if not os.path.exists(lmdb_path):
            print(f"Skipping {split}: {lmdb_path} not found")
            continue

        print(f"Indexing {split} ({lmdb_path})...")
        t0 = time.time()
        keys, lengths = build_index(lmdb_path)
        elapsed = time.time() - t0

        # Save
        keys_path = os.path.join(args.lmdb_dir, f"{split}_keys.pkl")
        lengths_path = os.path.join(args.lmdb_dir, f"{split}_lengths.npy")

        with open(keys_path, "wb") as f:
            pickle.dump(keys, f)
        np.save(lengths_path, lengths)

        print(f"  {split}: {len(keys)} entries in {elapsed:.1f}s")
        print(
            f"  Length range: {lengths.min()}-{lengths.max()}, "
            f"mean={lengths.mean():.1f}, median={np.median(lengths):.0f}"
        )
        print(
            f"  <= 128: {(lengths <= 128).sum()} ({100 * (lengths <= 128).mean():.1f}%)"
        )
        print(
            f"  <= 256: {(lengths <= 256).sum()} ({100 * (lengths <= 256).mean():.1f}%)"
        )
        print(f"  Saved: {keys_path} ({os.path.getsize(keys_path) / 1024:.0f} KB)")
        print(
            f"  Saved: {lengths_path} ({os.path.getsize(lengths_path) / 1024:.0f} KB)"
        )
        print()


if __name__ == "__main__":
    main()
