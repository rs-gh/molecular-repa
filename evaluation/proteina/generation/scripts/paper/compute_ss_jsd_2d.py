"""Recompute SS JSD with a 2D-binned metric that captures point-cloud shape.

The current `compute_ss_jsd` (ss_metrics.py) collapses each side to a single
(H̄, Ē, C̄) centroid before measuring JSD — so it cannot see variance,
bimodality, or tail behavior. This script reads the cached per-structure
SS fractions (`ss_cache/ss_fractions.npz`) and the per-PDB designability
flags (`designability_index.csv`), then recomputes JSD against the PDB and
AFDB reference distributions with two binning schemes:

  - 3-bin (centroid):  matches the current metric for sanity-check.
  - K×K-bin:           histograms each side over the (f_H, f_E) plane,
                       captures shape rather than just mean.

Output: a TSV next to the n=128 / n=256 paper figures, one row per cached
eval dir, with the old and new JSD values side-by-side so the user can see
whether rankings shift.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

REPO_ROOT = Path(__file__).resolve().parents[5]
EVAL_OUTPUTS = [
    REPO_ROOT / "eval_output",
    Path("/rds/user/sr2173/hpc-work/proteina/eval_output"),
]
FIG_ROOT = REPO_ROOT / "evaluation/proteina/generation/figures/paper"

REF_PDB_PATH = "/rds/user/sr2173/hpc-work/proteina/data/ss_reference_pdb.pt"
REF_AFDB_PATH = "/rds/user/sr2173/hpc-work/proteina/data/ss_reference_afdb.pt"


def _load_ref(path: str) -> np.ndarray:
    arr = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr, dtype=float)


def jsd_3bin(p_fracs: np.ndarray, q_fracs: np.ndarray) -> float:
    """Current metric: JSD between mean (H,E,C) vectors."""
    if p_fracs.size == 0 or q_fracs.size == 0:
        return float("nan")
    p = p_fracs.mean(axis=0)
    q = q_fracs.mean(axis=0)
    d = jensenshannon(p, q, base=2)
    return float(d**2)


def jsd_2d(p_fracs: np.ndarray, q_fracs: np.ndarray, n_bins: int) -> float:
    """JSD between 2D histograms on the (f_H, f_E) plane.

    Captures variance / shape of the point cloud, not just centroid.
    Bins are uniform over [0, 1]^2; the upper-triangle cells (f_H+f_E > 1)
    are valid grid cells that simply receive zero mass on both sides.
    """
    if p_fracs.size == 0 or q_fracs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    hp, _, _ = np.histogram2d(p_fracs[:, 0], p_fracs[:, 1], bins=(edges, edges))
    hq, _, _ = np.histogram2d(q_fracs[:, 0], q_fracs[:, 1], bins=(edges, edges))
    p = hp.flatten() / hp.sum()
    q = hq.flatten() / hq.sum()
    d = jensenshannon(p, q, base=2)
    return float(d**2)


def load_designable_set(eval_dir: Path) -> set[str]:
    """Return basenames of designable PDBs in this eval dir.

    Returns empty set if designability_index.csv is missing or empty.
    """
    idx = eval_dir / "designability_index.csv"
    if not idx.exists():
        return set()
    out = set()
    with idx.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("designable", "").strip().lower() == "true":
                out.add(os.path.basename(row["pdb_path"]))
    return out


_RUN_RE = re.compile(r"_sweep_(.+?)_step_\d+$")


def short_run_name(eval_dir_name: str) -> str:
    """Pull the underlying run name from the eval dir name.

    eval dir names look like:
      inference_paper_inference_fid_60m_paper_sweep_<run>_step_<step>
    """
    m = _RUN_RE.search(eval_dir_name)
    return m.group(1) if m else eval_dir_name


def detect_n(eval_dir_name: str) -> str:
    if "_n128_" in eval_dir_name:
        return "128"
    if "60m_paper_sweep" in eval_dir_name:
        return "256"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument(
        "--out",
        type=Path,
        default=FIG_ROOT / "ss_jsd_2d_comparison.tsv",
    )
    args = ap.parse_args()

    ref_pdb = _load_ref(REF_PDB_PATH)
    ref_afdb = _load_ref(REF_AFDB_PATH)
    print(f"Loaded references: PDB N={len(ref_pdb)}, AFDB N={len(ref_afdb)}")

    seen: set[str] = set()
    cache_files: list[Path] = []
    for root in EVAL_OUTPUTS:
        if not root.exists():
            continue
        for cache in sorted(root.glob("*/ss_cache/ss_fractions.npz")):
            eval_name = cache.parent.parent.name
            if eval_name in seen:
                continue  # prefer the first location (local /home wins)
            seen.add(eval_name)
            cache_files.append(cache)
    print(f"Found {len(cache_files)} ss_cache files across {len(EVAL_OUTPUTS)} roots")

    rows = []
    for cache in cache_files:
        eval_dir = cache.parent.parent
        run = short_run_name(eval_dir.name)
        n_label = detect_n(eval_dir.name)

        d = np.load(cache, allow_pickle=True)
        gen = d["fracs"]
        gen_paths = d["paths"]
        if gen.size == 0:
            continue

        des_set = load_designable_set(eval_dir)
        des_mask = np.array([os.path.basename(p) in des_set for p in gen_paths])
        gen_des = gen[des_mask] if des_mask.any() else np.zeros((0, 3))

        row = {
            "n": n_label,
            "run": run,
            "N_gen": len(gen),
            "N_des": int(des_mask.sum()),
            "ss_jsd_pdb_3bin": jsd_3bin(gen, ref_pdb),
            "ss_jsd_pdb_2d": jsd_2d(gen, ref_pdb, args.n_bins),
            "ss_jsd_afdb_3bin": jsd_3bin(gen, ref_afdb),
            "ss_jsd_afdb_2d": jsd_2d(gen, ref_afdb, args.n_bins),
            "ss_jsd_pdb_des_3bin": jsd_3bin(gen_des, ref_pdb),
            "ss_jsd_pdb_des_2d": jsd_2d(gen_des, ref_pdb, args.n_bins),
            "ss_jsd_afdb_des_3bin": jsd_3bin(gen_des, ref_afdb),
            "ss_jsd_afdb_des_2d": jsd_2d(gen_des, ref_afdb, args.n_bins),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["n"], r["run"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "n",
        "run",
        "N_gen",
        "N_des",
        "ss_jsd_pdb_3bin",
        "ss_jsd_pdb_2d",
        "ss_jsd_afdb_3bin",
        "ss_jsd_afdb_2d",
        "ss_jsd_pdb_des_3bin",
        "ss_jsd_pdb_des_2d",
        "ss_jsd_afdb_des_3bin",
        "ss_jsd_afdb_des_2d",
    ]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            for k in fields:
                if isinstance(r[k], float):
                    r[k] = f"{r[k]:.5f}" if not np.isnan(r[k]) else "nan"
            w.writerow(r)

    print(f"\nWrote {args.out}  ({len(rows)} rows, n_bins={args.n_bins})")
    print("\nQuick look (sorted by Δ = pdb_2d - pdb_3bin):")
    n256 = [r for r in rows if r["n"] == "256"]
    for r in n256:
        try:
            three = (
                float(r["ss_jsd_pdb_3bin"])
                if r["ss_jsd_pdb_3bin"] != "nan"
                else float("nan")
            )
            twod = (
                float(r["ss_jsd_pdb_2d"])
                if r["ss_jsd_pdb_2d"] != "nan"
                else float("nan")
            )
            r["_delta"] = twod - three
        except ValueError:
            r["_delta"] = float("nan")
    n256.sort(key=lambda r: -r.get("_delta", 0))
    print(f"  {'run':<60}  {'N_gen':>5}  {'pdb_3bin':>9}  {'pdb_2d':>9}  {'Δ':>7}")
    for r in n256:
        print(
            f"  {r['run']:<60}  {r['N_gen']:>5}  "
            f"{r['ss_jsd_pdb_3bin']:>9}  {r['ss_jsd_pdb_2d']:>9}  "
            f"{r.get('_delta', float('nan')):>+7.4f}"
        )


if __name__ == "__main__":
    main()
