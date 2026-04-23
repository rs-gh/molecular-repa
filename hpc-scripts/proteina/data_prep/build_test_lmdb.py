"""Build a proper held-out test.lmdb for the Proteina probe suite.

Mirrors build_val_lmdb.py exactly — only difference is it draws from the
"test" split (0.1% of PDB, ~500 proteins at max_length=512) instead of "val".

Operating principles (see feedback_cache_intermediates.md):
  1. Commits every --commit_every entries (default 100).
  2. Resumes from partial test.lmdb via the `__ids__` metadata key.
  3. Defensive: exits cleanly on SIGTERM flushing the current buffer;
     never blows away test.lmdb without --force AND explicit confirmation.
  4. Smoke mode: --n_test 10 --output_path /tmp/test_smoke.lmdb exercises the
     full pipeline in ~30 s on the login node.

Usage:
    # CPU login node smoke:
    source .venv/bin/activate
    python hpc-scripts/proteina/data_prep/build_test_lmdb.py \\
        --n_test 10 --output_path /tmp/test_smoke.lmdb --num_workers 2

    # Full build on intr GPU node:
    sbatch --qos=intr --time=00:30:00 ... build_test_lmdb.sh \\
        --n_test 500 --commit_every 100 --skip_train_check --resume
"""

import argparse
import json
import multiprocessing
import os
import pickle
import signal
import sys
import time
from pathlib import Path

import lmdb
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Script lives at hpc-scripts/proteina/data_prep/ — 3 levels up to repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src/proteina/proteinfoundation"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src/proteina"))

import proteinfoundation.repa.pyg_compat  # noqa: F401, E402

from proteinfoundation.datasets.lmdb_utils import _parse_one_structure  # noqa: E402
from proteinfoundation.datasets.pdb_data import PDBDataSplitter  # noqa: E402


_IDS_META_KEY = b"__ids__"
_MAP_SIZE_GB = 5  # test.lmdb with ~500 entries sits well under 0.5 GB.


# ---- existing-LMDB helpers -------------------------------------------------


def _read_existing_ids(lmdb_path: Path) -> set:
    """Return protein IDs already in an LMDB via `__ids__` metadata (fast)."""
    if not lmdb_path.exists():
        return set()
    db = lmdb.open(
        str(lmdb_path), readonly=True, subdir=False, lock=False, readahead=False
    )
    with db.begin() as txn:
        meta = txn.get(_IDS_META_KEY)
    db.close()
    if meta is None:
        logger.warning(
            f"{lmdb_path} has no __ids__ metadata; assuming empty for resume."
        )
        return set()
    return pickle.loads(meta)


def _next_lmdb_key(lmdb_path: Path) -> int:
    """Smallest integer cursor key available for appending."""
    if not lmdb_path.exists():
        return 0
    db = lmdb.open(
        str(lmdb_path), readonly=True, subdir=False, lock=False, readahead=False
    )
    try:
        with db.begin() as txn:
            cursor = txn.cursor()
            if not cursor.last():
                return 0
            k = cursor.key()
            if k == _IDS_META_KEY:
                if not cursor.prev():
                    return 0
                k = cursor.key()
            try:
                return int(k.decode()) + 1
            except ValueError:
                return 0
    finally:
        db.close()


# ---- CSV + splitter --------------------------------------------------------


def _find_csv(data_dir: str, fraction: float, max_length: int) -> str:
    frac_str = f"f{fraction}"
    maxl_str = f"maxl{max_length}"
    matches = [
        f
        for f in os.listdir(data_dir)
        if f.endswith(".csv") and frac_str in f and maxl_str in f
    ]
    if not matches:
        raise FileNotFoundError(f"No CSV in {data_dir} matching {frac_str}+{maxl_str}")
    return os.path.join(data_dir, matches[0])


def _on_disk_pdb_set(raw_dir: str) -> set:
    codes = set()
    for f in os.listdir(raw_dir):
        stem = f.split(".")[0]
        if stem:
            codes.add(stem.lower())
    return codes


# ---- main build loop -------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir", default=os.environ.get("DATA_PATH", "") + "/pdb_train"
    )
    ap.add_argument(
        "--output_path",
        default=None,
        help="Target LMDB path. Defaults to $data_dir/lmdb/test.lmdb.",
    )
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--train_val_test",
        type=float,
        nargs=3,
        default=[0.98, 0.019, 0.001],
        help="Must match the ratios used to build train.lmdb and val.lmdb.",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--commit_every",
        type=int,
        default=100,
        help="Flush batch to LMDB every N parsed entries.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing LMDB; skip already-written proteins.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Delete existing LMDB + lock before writing (ignored if --resume).",
    )
    ap.add_argument(
        "--skip_train_check",
        action="store_true",
        help="Skip scanning train.lmdb for disjointness. Safe when splitter "
        "config (ratios/seed/CSV) matches the historical train build.",
    )
    args = ap.parse_args()

    if args.num_workers == 0:
        args.num_workers = min(16, max(1, multiprocessing.cpu_count() - 1))

    data_dir = Path(args.data_dir)
    lmdb_dir = data_dir / "lmdb"
    raw_dir = data_dir / "raw"
    out_path = Path(args.output_path) if args.output_path else (lmdb_dir / "test.lmdb")
    out_lock = out_path.with_suffix(out_path.suffix + "-lock")
    manifest_out = out_path.with_name(out_path.stem + "_build_manifest.json")

    # --- Step 1: load CSV
    csv_path = _find_csv(str(data_dir), args.fraction, args.max_length)
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  {len(df)} rows")

    # --- Step 2: deterministic split
    logger.info(
        f"Splitting with ratios {args.train_val_test}, seed=42 (inside splitter)"
    )
    splitter = PDBDataSplitter(
        data_dir=str(data_dir),
        train_val_test=args.train_val_test,
        split_type="random",
    )
    file_id = os.path.basename(csv_path).replace(".csv", "")
    dfs_splits, _ = splitter.split_data(df, file_id)
    test_df = dfs_splits["test"].copy()
    test_ids_candidates = set(f"{r.pdb}_{r.chain}" for r in test_df.itertuples())
    logger.info(f"  test candidates: {len(test_ids_candidates)}")

    # --- Step 3: optional train disjointness
    train_lmdb_path = lmdb_dir / "train.lmdb"
    if args.skip_train_check:
        logger.info(
            "Skipping train.lmdb disjointness scan (--skip_train_check). "
            "Splitter partition is disjoint by construction when ratios/seed/CSV match."
        )
    elif train_lmdb_path.exists():
        train_ids = _read_existing_ids(train_lmdb_path)
        if train_ids:
            overlap = test_ids_candidates & train_ids
            if overlap:
                logger.warning(
                    f"  test ∩ train = {len(overlap)}; removing these from test"
                )
                test_df = test_df[
                    ~test_df.apply(lambda r: f"{r.pdb}_{r.chain}" in overlap, axis=1)
                ]
            else:
                logger.info("  test ∩ train = 0 (splitter-disjoint)")

    # --- Step 4: filter to on-disk raws
    logger.info(f"Listing {raw_dir}...")
    on_disk_pdbs = _on_disk_pdb_set(str(raw_dir))
    logger.info(f"  {len(on_disk_pdbs)} distinct PDB codes on disk")
    test_df = test_df[test_df["pdb"].str.lower().isin(on_disk_pdbs)]
    logger.info(f"  test candidates with raw on disk: {len(test_df)}")

    target_n = min(args.n_test, len(test_df))
    test_df = test_df.sample(n=target_n, random_state=args.seed).reset_index(drop=True)
    logger.info(f"Final requested test set: {len(test_df)} proteins")

    # --- Step 5: output LMDB setup
    if out_path.exists():
        if args.resume:
            logger.info(f"Resuming into existing {out_path}")
        elif args.force:
            logger.info(f"--force: removing {out_path}")
            out_path.unlink()
            if out_lock.exists():
                out_lock.unlink()
        else:
            logger.error(
                f"{out_path} exists — pass --resume to append or --force to replace."
            )
            sys.exit(1)

    existing_ids = _read_existing_ids(out_path) if args.resume else set()
    next_key = _next_lmdb_key(out_path) if args.resume else 0
    if existing_ids:
        logger.info(
            f"  {len(existing_ids)} entries already in LMDB, next key = {next_key}"
        )

    # --- Step 6: queue up only the items we still need
    items = []
    for r in test_df.itertuples():
        pid = f"{r.pdb}_{r.chain}"
        if pid in existing_ids:
            continue
        items.append((r.pdb, r.chain, pid))
    logger.info(
        f"To process: {len(items)} new proteins ({len(existing_ids)} already done)"
    )

    if not items:
        logger.info("Nothing to do.")
        return

    # --- Step 7: parse + commit with tight batching
    db = lmdb.open(
        str(out_path),
        map_size=_MAP_SIZE_GB * 1024**3,
        create=True,
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )
    all_ids = set(existing_ids)

    batch = []
    batch_ids = []
    n_written = 0
    n_failed = 0
    t_start = time.time()

    def _flush():
        nonlocal batch, batch_ids, n_written, all_ids
        if not batch:
            return
        all_ids.update(batch_ids)
        with db.begin(write=True) as txn:
            for i, pickled in enumerate(batch):
                txn.put(key=str(next_key + n_written + i).encode(), value=pickled)
            txn.put(_IDS_META_KEY, pickle.dumps(all_ids))
        n_written += len(batch)
        logger.info(
            f"  committed {n_written}/{len(items)} "
            f"(total {len(all_ids)} ids; elapsed {time.time()-t_start:.0f}s)"
        )
        batch = []
        batch_ids = []

    def _handle_signal(signum, frame):
        logger.warning(f"Signal {signum} received — flushing {len(batch)} buffered")
        try:
            _flush()
        finally:
            db.close()
            sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    worker_args = [
        (pdb, chain, pid, str(raw_dir), "cif", False, True, True)
        for pdb, chain, pid in items
    ]

    try:
        if args.num_workers > 1:
            pool = multiprocessing.Pool(args.num_workers)
            results_iter = pool.imap_unordered(
                _parse_one_structure, worker_args, chunksize=8
            )
        else:
            pool = None
            results_iter = (_parse_one_structure(a) for a in worker_args)

        for protein_id, pickled in tqdm(
            results_iter, total=len(worker_args), desc="Parsing CIFs"
        ):
            if pickled is None:
                n_failed += 1
                continue
            batch.append(pickled)
            batch_ids.append(protein_id)
            if len(batch) >= args.commit_every:
                _flush()

        _flush()
    finally:
        if pool is not None:
            pool.close()
            pool.terminate()
            pool.join()
        db.close()

    elapsed = time.time() - t_start
    logger.info(
        f"Done: wrote {n_written} new entries ({n_failed} failed) in {elapsed:.0f}s. "
        f"Total in LMDB: {len(all_ids)}."
    )

    # --- Step 8: manifest
    manifest = {
        "n_requested": target_n,
        "n_written_this_run": n_written,
        "n_failed_this_run": n_failed,
        "n_total_in_lmdb": len(all_ids),
        "csv": csv_path,
        "seed": args.seed,
        "max_length": args.max_length,
        "train_val_test": list(args.train_val_test),
        "output_path": str(out_path),
        "protein_ids_this_run": [pid for _, _, pid in items[:n_written]],
    }
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"wrote {manifest_out}")


if __name__ == "__main__":
    main()
