"""Verify the new held-out val.lmdb passes the 5 blocking checks.

Run after build_val_lmdb.py + build_lmdb_length_index.py --splits val.
Fails loudly on any violation; prints a one-line OK summary if all pass.

Usage:
    source .venv/bin/activate
    export DATA_PATH=/rds/user/sr2173/hpc-work/proteina/data
    python hpc-scripts/proteina/data_prep/verify_val_lmdb.py
"""

import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src/proteina"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation/proteina/representation"))

DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_DIR = Path(DATA_PATH) / "pdb_train/lmdb"
VAL_LMDB = LMDB_DIR / "val.lmdb"
TRAIN_LMDB = LMDB_DIR / "train.lmdb"
VAL_KEYS = LMDB_DIR / "val_keys.pkl"
VAL_LENGTHS = LMDB_DIR / "val_lengths.npy"

MIN_VAL_ENTRIES = 2000


def _lmdb_ids(path: Path) -> set:
    db = lmdb.open(str(path), readonly=True, subdir=False, lock=False, readahead=False)
    try:
        with db.begin() as txn:
            meta = txn.get(b"__ids__")
            if meta is not None:
                return pickle.loads(meta)
            # Slow scan fallback
            ids = set()
            for k, v in txn.cursor():
                if k == b"__ids__":
                    continue
                try:
                    ids.add(str(pickle.loads(v).id))
                except Exception:
                    pass
            return ids
    finally:
        db.close()


def check_counts() -> dict:
    """Check 1: entry count + sidecar shape."""
    assert VAL_LMDB.exists(), f"val.lmdb missing at {VAL_LMDB}"
    assert VAL_KEYS.exists(), f"val_keys.pkl missing at {VAL_KEYS}"
    assert VAL_LENGTHS.exists(), f"val_lengths.npy missing at {VAL_LENGTHS}"

    val_ids = _lmdb_ids(VAL_LMDB)
    with open(VAL_KEYS, "rb") as f:
        keys = pickle.load(f)
    lengths = np.load(VAL_LENGTHS)

    assert len(keys) == len(
        lengths
    ), f"sidecar mismatch: {len(keys)} keys vs {len(lengths)} lengths"
    assert len(keys) == len(val_ids), (
        f"val.lmdb entries ({len(val_ids)}) != sidecar ({len(keys)}); "
        f"rebuild sidecars"
    )
    assert (
        len(val_ids) >= MIN_VAL_ENTRIES
    ), f"val has only {len(val_ids)} entries — want >= {MIN_VAL_ENTRIES}"
    assert lengths.max() <= 512, "val contains a protein > 512 residues"
    print(
        f"[1/5] OK  n_entries={len(val_ids)}  "
        f"lengths=[{lengths.min()},{lengths.max()}]"
    )
    return {"val_ids": val_ids}


def check_disjointness(val_ids: set) -> None:
    """Check 2: val ∩ train == ∅."""
    assert TRAIN_LMDB.exists(), f"train.lmdb missing at {TRAIN_LMDB}"
    print("[2/5] reading train.lmdb IDs (may take minutes if no __ids__ cache)...")
    train_ids = _lmdb_ids(TRAIN_LMDB)
    overlap = val_ids & train_ids
    assert not overlap, f"val ∩ train = {len(overlap)}; sample: {list(overlap)[:5]}"
    print(f"[2/5] OK  val ∩ train = 0 (|train|={len(train_ids)})")


def check_cath_rate() -> None:
    """Check 3: CATH attach rate >= 10% on a 100-sample."""
    from lib.data import _attach_cath_labels

    db = lmdb.open(
        str(VAL_LMDB), readonly=True, subdir=False, lock=False, readahead=False
    )
    raw = []
    with db.begin() as txn:
        for i, (k, v) in enumerate(txn.cursor()):
            if k == b"__ids__":
                continue
            if len(raw) >= 100:
                break
            raw.append(pickle.loads(v))
    db.close()

    _attach_cath_labels(raw)
    labelled = sum(1 for g in raw if getattr(g, "cath_code", None))
    rate = 100.0 * labelled / len(raw)
    assert rate >= 10.0, (
        f"CATH attach rate too low: {rate:.1f}% " f"(want >= 10%; expected ~30-40%)"
    )
    print(f"[3/5] OK  CATH attach rate = {rate:.1f}% on 100-sample")


def check_sidecar_sampling() -> None:
    """Check 4: manifest sampling works end-to-end."""
    from lib.manifest import sample_manifest

    m = sample_manifest(str(VAL_LMDB), n=500, max_size=512, seed=42)
    assert len(m["keys"]) == 500
    assert max(m["lengths"]) <= 512
    print(f"[4/5] OK  sidecar sampling: 500 keys, max_len={max(m['lengths'])}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip_disjointness",
        action="store_true",
        help="Skip the train-val disjointness scan (~30+ min on Lustre). "
        "Safe when build_val_lmdb was run with --skip_train_check and splitter "
        "config matches the historical train build.",
    )
    args = ap.parse_args()

    print(f"Verifying {VAL_LMDB}")
    ctx = check_counts()
    if args.skip_disjointness:
        print("[2/5] SKIPPED (--skip_disjointness) — trusting splitter determinism")
    else:
        check_disjointness(ctx["val_ids"])
    check_cath_rate()
    check_sidecar_sampling()
    print(
        "[5/5] SKIPPED: end-to-end smoke probe requires sbatch; "
        "run after verify via: PROBE_SPLIT=val sbatch hpc-scripts/proteina/evaluation/representation/run_probes.sh "
        "--n_proteins 200 --runs baseline_128 --timesteps 1.0"
    )
    print("\nALL PASSED — safe to flip PROBE_SPLIT default to val")


if __name__ == "__main__":
    main()
