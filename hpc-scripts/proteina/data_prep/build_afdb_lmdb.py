"""Build LMDB from AFDB Swiss-Prot tar archive (streaming, no extraction).

Streams PDB files directly from swissprot_pdb_v6.tar without ever extracting
them to disk. Two passes over the tar:

  Pass 1 (fast): read only tar headers to collect all protein IDs, compute
                 train/val/test split, save splits.json.
  Pass 2 (slow): stream tar data blocks, parse each PDB, write to LMDB.

Supports incremental builds: re-running skips proteins already in the LMDB
(tracked via the LMDB __ids__ metadata key). New files added to the tar on
re-download are automatically appended.

Usage:
    # Full build:
    python hpc-scripts/proteina/data_prep/build_afdb_lmdb.py \\
        --data-dir $DATA_PATH/afdb_swissprot \\
        --max-residues 256 \\
        --num-workers 24

    # Smoke test (first 200 proteins per split):
    python hpc-scripts/proteina/data_prep/build_afdb_lmdb.py \\
        --data-dir /path/to/smoke \\
        --max-residues 256 \\
        --num-workers 2 \\
        --smoke-limit 200
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import tarfile

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../src/proteina/proteinfoundation")
    )
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/proteina"))
)

import proteinfoundation.repa.pyg_compat  # noqa: F401

from loguru import logger
from proteinfoundation.datasets.lmdb_utils import process_tar_to_lmdb


def _scan_tar_members(tar_path: str) -> list[str]:
    """Pass 1: read tar headers only to collect all protein ID stems.

    Uses streaming mode so we never load data blocks — just headers.
    For a 27 GB tar this takes ~30-60 seconds.
    """
    logger.info(f"Scanning tar headers: {tar_path}")
    stems = []
    with tarfile.open(tar_path, "r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stem = os.path.splitext(os.path.basename(member.name))[0]
            if stem:
                stems.append(stem)
            # extractfile() NOT called — we skip the data block automatically
    logger.info(f"Found {len(stems)} PDB members in tar")
    return stems


def load_or_create_splits(
    tar_path: str,
    splits_path: str,
    ratios=(0.98, 0.019, 0.001),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Load split assignments from splits.json, or create from tar headers."""
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        assigned = set(splits["train"] + splits["val"] + splits["test"])
        logger.info(
            f"Loaded existing splits ({len(assigned)} total) from {splits_path}"
        )

        # Check for new members in the tar not yet assigned to a split.
        # Re-scan headers to find them.
        all_stems = set(_scan_tar_members(tar_path))
        new_stems = sorted(all_stems - assigned)
        if new_stems:
            logger.info(f"{len(new_stems)} new members found; assigning to splits")
            rng = random.Random(seed + len(assigned))
            random.shuffle(new_stems, rng.random)
            n = len(new_stems)
            n_train = round(n * ratios[0])
            n_val = round(n * ratios[1])
            splits["train"].extend(new_stems[:n_train])
            splits["val"].extend(new_stems[n_train : n_train + n_val])
            splits["test"].extend(new_stems[n_train + n_val :])
            with open(splits_path, "w") as f:
                json.dump(splits, f)
            logger.info(f"Updated splits saved to {splits_path}")
        return splits

    # First run: scan tar and build splits from scratch.
    all_stems = _scan_tar_members(tar_path)
    rng = random.Random(seed)
    rng.shuffle(all_stems)
    n = len(all_stems)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    splits = {
        "train": all_stems[:n_train],
        "val": all_stems[n_train : n_train + n_val],
        "test": all_stems[n_train + n_val :],
    }
    os.makedirs(os.path.dirname(os.path.abspath(splits_path)), exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f)
    logger.info(
        f"Created splits: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    logger.info(f"Splits saved to {splits_path}")
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Build LMDB from AFDB Swiss-Prot tar (streaming, no extraction)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.environ.get("DATA_PATH", ""), "afdb_swissprot"),
        help="Root data dir containing swissprot_pdb_v6.tar",
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=256,
        help="Skip structures with more than this many residues (default: 256)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel workers for PDB parsing (0 = auto from CPU count)",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="If set, only process the first N proteins per split (smoke test)",
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=50,
        help="LMDB map size in GB per split (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Commit to LMDB every N entries (default: 500)",
    )
    parser.add_argument(
        "--tar-name",
        type=str,
        default="swissprot_pdb_v6.tar",
        help="Tar filename inside --data-dir (default: swissprot_pdb_v6.tar)",
    )
    args = parser.parse_args()

    if args.num_workers == 0:
        args.num_workers = min(24, max(1, multiprocessing.cpu_count() - 1))

    data_dir = args.data_dir
    tar_path = os.path.join(data_dir, args.tar_name)
    lmdb_dir = os.path.join(data_dir, "lmdb")
    splits_path = os.path.join(data_dir, "splits.json")

    if not os.path.isfile(tar_path):
        logger.error(f"Tar not found: {tar_path}")
        logger.error("Run download_afdb_swissprot.sh first.")
        sys.exit(1)

    os.makedirs(lmdb_dir, exist_ok=True)

    # Pass 1: assign splits (fast — headers only)
    splits = load_or_create_splits(tar_path, splits_path)

    # Pass 2: stream tar for each split
    for split_name, protein_ids in splits.items():
        if args.smoke_limit is not None:
            protein_ids = protein_ids[: args.smoke_limit]

        lmdb_path = os.path.join(lmdb_dir, f"{split_name}.lmdb")
        logger.info(
            f"\n=== {split_name}: {len(protein_ids)} proteins → {lmdb_path} ==="
            + (f" [max_residues={args.max_residues}]" if args.max_residues else "")
        )

        n_written = process_tar_to_lmdb(
            tar_path=tar_path,
            output_path=lmdb_path,
            protein_ids=protein_ids,
            max_residues=args.max_residues,
            store_het=False,
            store_bfactor=True,  # pLDDT is useful to keep
            apply_coord_reorder=True,
            map_size_gb=args.map_size_gb,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        logger.info(f"{split_name}: {n_written} new samples written")

    logger.info(f"\nLMDB files in {lmdb_dir}:")
    for f in sorted(os.listdir(lmdb_dir)):
        size_mb = os.path.getsize(os.path.join(lmdb_dir, f)) / 1e6
        logger.info(f"  {f}: {size_mb:.1f} MB")

    logger.info(
        "\nNext: build length index:\n"
        "  sbatch hpc-scripts/proteina/data_prep/build_length_index.sh afdb_swissprot"
    )


if __name__ == "__main__":
    main()
