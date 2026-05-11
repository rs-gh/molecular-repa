"""Bootstrap 95% CIs on SS JSD (3-bin + 2D) and Designability per eval run.

Reads cached per-structure SS fractions and designability_index.csv files.
For each run:
  - Bootstrap-resamples N_gen structures with replacement to build a CI
    on SS JSD (3-bin) and SS JSD (2D, K=5).
  - Bernoulli SE on Designability (no resampling needed; closed form).
  - Same SS JSD CI restricted to the designable subset.

Writes a TSV with point estimates + 95% CIs alongside.

Notes:
  - fS_T and FID CIs aren't computed here — they need the underlying
    per-structure GearNet features / CATH-class predictions which aren't
    cached. Add that separately if needed.
  - 2000 bootstrap iterations gives MC error ~1/sqrt(2000) ≈ 2% on the
    quantile estimates. Plenty for headline numbers.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.ss_metrics import compute_ss_jsd, compute_ss_jsd_2d  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
FIG_ROOT = REPO_ROOT / "evaluation/proteina/generation/figures/paper"
EVAL_ROOTS = [
    REPO_ROOT / "eval_output",
    Path("/rds/user/sr2173/hpc-work/proteina/eval_output"),
]
REF_PDB_PATH = "/rds/user/sr2173/hpc-work/proteina/data/ss_reference_pdb.pt"


def load_ref(path: str) -> np.ndarray:
    arr = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr, dtype=float)


def load_designable_set(eval_dir: Path) -> set[str]:
    idx = eval_dir / "designability_index.csv"
    if not idx.exists():
        return set()
    out = set()
    with idx.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("designable", "").strip().lower() == "true":
                out.add(os.path.basename(row["pdb_path"]))
    return out


def load_evaluated_count(eval_dir: Path) -> int:
    """How many PDBs were actually evaluated for designability."""
    idx = eval_dir / "designability_index.csv"
    if not idx.exists():
        return 0
    n = 0
    with idx.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("evaluated", "").strip().lower() == "true":
                n += 1
    return n


def bootstrap_jsd_ci(
    fracs: np.ndarray,
    ref: np.ndarray,
    jsd_fn,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) at 95%."""
    point = jsd_fn(fracs, ref)
    if fracs.shape[0] < 2:
        return point, float("nan"), float("nan")
    N = fracs.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        v = jsd_fn(fracs[idx], ref)
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return point, lo, hi


def des_bernoulli_ci(k: int, n: int) -> tuple[float, float, float]:
    """Wilson score 95% CI for Bernoulli proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def short_run_name(name: str) -> str:
    if "_sweep_" not in name:
        return name
    tail = name.split("_sweep_", 1)[1]
    if "_step_" not in tail:
        return name
    return tail.rsplit("_step_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_bins_2d", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=FIG_ROOT / "ss_des_bootstrap_ci.tsv",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ref_pdb = load_ref(REF_PDB_PATH)

    # Build {name: best_cache_path} preferring the root that ALSO has
    # designability_index.csv. /home and /rds can both hold an eval dir,
    # but /home's were partially pruned (PDBs and des-index gone) while
    # the ss_cache survived — so we must pair the cache with the matching
    # designability index, not just the first cache we see.
    by_name: dict[str, Path] = {}
    for root in EVAL_ROOTS:
        if not root.exists():
            continue
        for c in sorted(root.glob("*/ss_cache/ss_fractions.npz")):
            name = c.parent.parent.name
            has_idx = (c.parent.parent / "designability_index.csv").exists()
            if name not in by_name:
                by_name[name] = c
            elif (
                has_idx
                and not (
                    by_name[name].parent.parent / "designability_index.csv"
                ).exists()
            ):
                by_name[name] = c  # upgrade to a cache whose dir has the index
    caches = sorted(by_name.values())
    print(f"Found {len(caches)} ss_cache files")

    rows = []
    for cache in caches:
        eval_dir = cache.parent.parent
        run = short_run_name(eval_dir.name)
        if "_n128_" in eval_dir.name:
            n_label = "128"
        elif "60m_paper" in eval_dir.name:
            n_label = "256"
        else:
            n_label = "?"

        d = np.load(cache, allow_pickle=True)
        fracs = d["fracs"]
        paths = d["paths"]
        if fracs.size == 0:
            continue

        des_set = load_designable_set(eval_dir)
        des_mask = np.array([os.path.basename(p) in des_set for p in paths])
        des_n_evaluated = load_evaluated_count(eval_dir)

        def jsd2d_k(g, r):
            return compute_ss_jsd_2d(g, r, n_bins=args.n_bins_2d)

        pdb_3, pdb_3_lo, pdb_3_hi = bootstrap_jsd_ci(
            fracs, ref_pdb, compute_ss_jsd, args.n_boot, rng
        )
        pdb_2, pdb_2_lo, pdb_2_hi = bootstrap_jsd_ci(
            fracs, ref_pdb, jsd2d_k, args.n_boot, rng
        )

        fracs_des = fracs[des_mask] if des_mask.any() else np.zeros((0, 3))
        pdb_3_d, pdb_3_d_lo, pdb_3_d_hi = bootstrap_jsd_ci(
            fracs_des, ref_pdb, compute_ss_jsd, args.n_boot, rng
        )
        pdb_2_d, pdb_2_d_lo, pdb_2_d_hi = bootstrap_jsd_ci(
            fracs_des, ref_pdb, jsd2d_k, args.n_boot, rng
        )

        des_count = int(des_mask.sum())
        des_p, des_lo, des_hi = des_bernoulli_ci(
            des_count, des_n_evaluated or fracs.shape[0]
        )

        rows.append(
            {
                "n": n_label,
                "run": run,
                "N_gen": fracs.shape[0],
                "N_des_eval": des_n_evaluated,
                "N_des": des_count,
                "des_p": des_p,
                "des_ci_lo": des_lo,
                "des_ci_hi": des_hi,
                "des_ci_halfwidth": (des_hi - des_lo) / 2,
                "pdb_3bin": pdb_3,
                "pdb_3bin_ci_lo": pdb_3_lo,
                "pdb_3bin_ci_hi": pdb_3_hi,
                "pdb_2d": pdb_2,
                "pdb_2d_ci_lo": pdb_2_lo,
                "pdb_2d_ci_hi": pdb_2_hi,
                "pdb_3bin_des": pdb_3_d,
                "pdb_3bin_des_ci_lo": pdb_3_d_lo,
                "pdb_3bin_des_ci_hi": pdb_3_d_hi,
                "pdb_2d_des": pdb_2_d,
                "pdb_2d_des_ci_lo": pdb_2_d_lo,
                "pdb_2d_des_ci_hi": pdb_2_d_hi,
            }
        )

    rows.sort(key=lambda r: (r["n"], r["run"]))
    fields = list(rows[0].keys())
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            for k in fields:
                if isinstance(r[k], float):
                    r[k] = "nan" if math.isnan(r[k]) else f"{r[k]:.5f}"
            w.writerow(r)

    print(f"Wrote {args.out}  ({len(rows)} rows, n_boot={args.n_boot})")


if __name__ == "__main__":
    main()
