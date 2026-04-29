"""Verify the new held-out test.lmdb passes the blocking checks.

Mirrors verify_val_lmdb.py - only differences:
  - Operates on test.lmdb / test_keys.pkl / test_lengths.npy
  - MIN_TEST_ENTRIES is 200 (test split is ~0.1% of PDB vs val's ~1.9%)
  - Sidecar sampling check requests 200 proteins (not 500)

Run after build_test_lmdb.py + build_lmdb_length_index.py --splits test.
Fails loudly on any violation; prints a one-line OK summary if all pass.

Usage:
    source .venv/bin/activate
    export DATA_PATH=/rds/user/sr2173/hpc-work/proteina/data
    python hpc-scripts/proteina/data_prep/verify_test_lmdb.py
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
TEST_LMDB = LMDB_DIR / "test.lmdb"
TRAIN_LMDB = LMDB_DIR / "train.lmdb"
TEST_KEYS = LMDB_DIR / "test_keys.pkl"
TEST_LENGTHS = LMDB_DIR / "test_lengths.npy"

MIN_TEST_ENTRIES = 200


def _lmdb_ids(path: Path) -> set:
    db = lmdb.open(str(path), readonly=True, subdir=False, lock=False, readahead=False)
    try:
        with db.begin() as txn:
            meta = txn.get(b"__ids__")
            if meta is not None:
                return pickle.loads(meta)
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
    assert TEST_LMDB.exists(), f"test.lmdb missing at {TEST_LMDB}"
    assert TEST_KEYS.exists(), f"test_keys.pkl missing at {TEST_KEYS}"
    assert TEST_LENGTHS.exists(), f"test_lengths.npy missing at {TEST_LENGTHS}"

    test_ids = _lmdb_ids(TEST_LMDB)
    with open(TEST_KEYS, "rb") as f:
        keys = pickle.load(f)
    lengths = np.load(TEST_LENGTHS)

    assert len(keys) == len(
        lengths
    ), f"sidecar mismatch: {len(keys)} keys vs {len(lengths)} lengths"
    assert len(keys) == len(test_ids), (
        f"test.lmdb entries ({len(test_ids)}) != sidecar ({len(keys)}); "
        f"rebuild sidecars"
    )
    assert (
        len(test_ids) >= MIN_TEST_ENTRIES
    ), f"test has only {len(test_ids)} entries - want >= {MIN_TEST_ENTRIES}"
    assert lengths.max() <= 512, "test contains a protein > 512 residues"
    print(
        f"[1/4] OK  n_entries={len(test_ids)}  "
        f"lengths=[{lengths.min()},{lengths.max()}]"
    )
    return {"test_ids": test_ids}


def check_disjointness(test_ids: set) -> None:
    """Check 2: test intersect train == empty."""
    assert TRAIN_LMDB.exists(), f"train.lmdb missing at {TRAIN_LMDB}"
    print("[2/4] reading train.lmdb IDs (may take minutes if no __ids__ cache)...")
    train_ids = _lmdb_ids(TRAIN_LMDB)
    overlap = test_ids & train_ids
    assert (
        not overlap
    ), f"test intersect train = {len(overlap)}; sample: {list(overlap)[:5]}"
    print(f"[2/4] OK  test intersect train = 0 (|train|={len(train_ids)})")


def check_cath_rate() -> None:
    """Check 3: CATH attach rate >= 10% on a 100-sample."""
    from lib.data import _attach_cath_labels

    db = lmdb.open(
        str(TEST_LMDB), readonly=True, subdir=False, lock=False, readahead=False
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
    print(f"[3/4] OK  CATH attach rate = {rate:.1f}% on 100-sample")


def check_sidecar_sampling() -> None:
    """Check 4: manifest sampling works end-to-end."""
    from lib.manifest import sample_manifest

    m = sample_manifest(str(TEST_LMDB), n=200, max_size=512, seed=42)
    assert len(m["keys"]) == 200
    assert max(m["lengths"]) <= 512
    print(f"[4/4] OK  sidecar sampling: 200 keys, max_len={max(m['lengths'])}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip_disjointness",
        action="store_true",
        help="Skip the train-test disjointness scan (~30+ min on Lustre). "
        "Safe when build_test_lmdb was run with --skip_train_check and splitter "
        "config matches the historical train build.",
    )
    args = ap.parse_args()

    print(f"Verifying {TEST_LMDB}")
    ctx = check_counts()
    if args.skip_disjointness:
        print("[2/4] SKIPPED (--skip_disjointness) - trusting splitter determinism")
    else:
        check_disjointness(ctx["test_ids"])
    check_cath_rate()
    check_sidecar_sampling()
    print(
        "\nALL PASSED - test.lmdb is ready. "
        "To run probes against it: PROBE_SPLIT=test sbatch hpc-scripts/proteina/evaluation/representation/run_probes.sh "
        "--n_proteins 200 --runs baseline_128 --timesteps 1.0"
    )


if __name__ == "__main__":
    main()
