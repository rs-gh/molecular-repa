"""Backfill _res_ss_jsd_*_2d into existing sweep_results.jsonl rows.

For each row in `results/{paper,lite}/*/sweep_results.jsonl` that already has
`_res_ss_jsd_pdb` set, locate the matching eval-output dir (across /home and
/rds), load the cached per-structure fractions, and compute the 4 new 2D-binned
JSD values. Updates rows in place and rewrites the JSONL.

This is a one-time bridge: future evals will populate these fields directly via
ss_metrics.compute_ss_metrics. Once all your reported runs have the columns
this script can be deleted.

Usage:
    python backfill_ss_2d_to_jsonl.py            # all paper + lite JSONLs
    python backfill_ss_2d_to_jsonl.py --dry-run  # preview without writing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.ss_metrics import compute_ss_jsd_2d  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
RESULTS_GLOBS = [
    REPO_ROOT / "evaluation/proteina/generation/results/paper",
    REPO_ROOT / "evaluation/proteina/generation/results/lite",
]
EVAL_ROOTS = [
    REPO_ROOT / "eval_output",
    Path("/rds/user/sr2173/hpc-work/proteina/eval_output"),
]
REF_PDB_PATH = "/rds/user/sr2173/hpc-work/proteina/data/ss_reference_pdb.pt"
REF_AFDB_PATH = "/rds/user/sr2173/hpc-work/proteina/data/ss_reference_afdb.pt"


def _load_ref(path: str) -> np.ndarray:
    arr = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr, dtype=float)


def _scan_cache_index() -> dict[tuple[str, int], Path]:
    """Build {(run, step): ss_cache_path} from all eval_output roots.

    Eval dir basenames look like:
      <config-with-underscores>_sweep_<run>_step_<step>
    """
    out: dict[tuple[str, int], Path] = {}
    for root in EVAL_ROOTS:
        if not root.exists():
            continue
        for cache in root.glob("*/ss_cache/ss_fractions.npz"):
            d = cache.parent.parent.name
            # Split on "_sweep_" — left = config, right = "<run>_step_<step>"
            if "_sweep_" not in d:
                continue
            tail = d.split("_sweep_", 1)[1]
            if "_step_" not in tail:
                continue
            run, step_str = tail.rsplit("_step_", 1)
            # Trim ".stale.<ts>" suffix variants
            step_str = step_str.split(".")[0]
            if not step_str.isdigit():
                continue
            key = (run, int(step_str))
            if key in out:
                continue  # first wins (prefer /home over /rds)
            out[key] = cache
    return out


def _load_designable_basenames(eval_dir: Path) -> set[str]:
    idx = eval_dir / "designability_index.csv"
    if not idx.exists():
        return set()
    out: set[str] = set()
    with idx.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("designable", "").strip().lower() == "true":
                out.add(os.path.basename(row["pdb_path"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ref_pdb = _load_ref(REF_PDB_PATH)
    ref_afdb = _load_ref(REF_AFDB_PATH)
    print(f"Loaded refs: PDB N={len(ref_pdb)}, AFDB N={len(ref_afdb)}")

    cache_index = _scan_cache_index()
    print(f"Found {len(cache_index)} (run, step) → cache mappings")

    jsonl_files: list[Path] = []
    for root in RESULTS_GLOBS:
        if root.exists():
            jsonl_files.extend(root.glob("*/sweep_results.jsonl"))
    print(f"Scanning {len(jsonl_files)} sweep_results.jsonl files\n")

    total_rows = 0
    total_updated = 0
    total_already = 0
    total_no_cache = 0
    total_no_ss = 0

    for jsonl in jsonl_files:
        rows = []
        updated_in_file = 0
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            total_rows += 1

            if "_res_ss_jsd_pdb" not in r:
                total_no_ss += 1
                continue
            if "_res_ss_jsd_pdb_2d" in r:
                total_already += 1
                continue

            run = r.get("run")
            step = r.get("step")
            if run is None or step is None:
                continue
            cache = cache_index.get((str(run), int(step)))
            if cache is None:
                total_no_cache += 1
                continue

            d = np.load(cache, allow_pickle=True)
            fracs = d["fracs"]
            paths = d["paths"]

            r["_res_ss_jsd_pdb_2d"] = compute_ss_jsd_2d(fracs, ref_pdb)
            r["_res_ss_jsd_afdb_2d"] = compute_ss_jsd_2d(fracs, ref_afdb)

            # Designable subset, if available.
            eval_dir = cache.parent.parent
            des_set = _load_designable_basenames(eval_dir)
            if des_set:
                mask = np.array(
                    [os.path.basename(p) in des_set for p in paths], dtype=bool
                )
                fracs_des = fracs[mask]
                if fracs_des.size > 0:
                    r["_res_ss_jsd_pdb_designable_2d"] = compute_ss_jsd_2d(
                        fracs_des, ref_pdb
                    )
                    r["_res_ss_jsd_afdb_designable_2d"] = compute_ss_jsd_2d(
                        fracs_des, ref_afdb
                    )

            updated_in_file += 1
            total_updated += 1

        if updated_in_file and not args.dry_run:
            with jsonl.open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        marker = "(dry-run)" if args.dry_run else "written"
        print(f"  {jsonl.relative_to(REPO_ROOT)}: +{updated_in_file} rows  [{marker}]")

    print("\nSummary:")
    print(f"  total rows scanned:    {total_rows}")
    print(f"  updated (added 2D):    {total_updated}")
    print(f"  already had 2D:        {total_already}")
    print(f"  no SS at all:          {total_no_ss}")
    print(f"  SS but no cache match: {total_no_cache}")


if __name__ == "__main__":
    main()
