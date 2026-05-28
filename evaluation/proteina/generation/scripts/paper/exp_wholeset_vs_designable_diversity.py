"""Experiment A: whole-set vs designable-subset structural diversity.

Tests whether REPA's diversity divergence (whole-set fS-A ↑ while designable
#Clust ↓) is a SUBSET effect or a METRIC effect. We use pwTM (mean pairwise
TM-score), which is n-independent (an average), so whole-set and designable
values are directly comparable — unlike #clusters which scales with sample count.

For each (model, step), compute:
  - pwTM over ALL generated samples (per-length, length-pooled)
  - pwTM over the DESIGNABLE subset only
Lower pwTM = more structurally diverse.

If REPA whole-set pwTM ≈ baseline whole-set pwTM (both diverse), but REPA
designable pwTM > baseline designable pwTM (REPA designable concentrated):
  → the concentration is SUBSET-specific (designability filter selects REPA's
    concentrated high-confidence modes). The fS-A increase is real whole-set
    diversity (more fold classes), and only the designable modes concentrate.

If REPA whole-set pwTM > baseline whole-set pwTM too:
  → REPA concentrates everywhere; fS-A's rise is a class-coverage artifact
    decoupled from structural diversity.

Whole-set is subsampled to `--cap` per length for tractability (pwTM is a mean,
so a 100-sample estimate is stable). Designable uses the full designable pool.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[5] / "src/proteina"))
from proteinfoundation.metrics.tm_score import compute_diversity  # noqa: E402

ROOT = _HERE.parents[5]
EVAL_OUT = ROOT / "eval_output"
SAMPLER_TAG = "sde_n0.45"


# All n256 variants × representative steps spanning the T-D crossover
# (~before / ~crossover / ~after / latest). Missing dirs are skipped.
def _expand(fam, steps):
    return [
        (f"{label}_{s//1000}K", f"{fam}_step{s//1000}k_step_{s}")
        for label, s in [(label, st) for st in steps]
    ]


CASES = []
for label, fam, steps in [
    # --- PDB ---
    ("PDB_baseline", "baseline_256_bs24_2gpu", [400000, 700000, 1000000, 1600000]),
    ("PDB_L4_GN", "repa_l4_256_per_residue_bs24_2gpu", [400000, 700000, 1000000]),
    (
        "PDB_L9_GN",
        "repa_l9_256_per_residue_bs24_2gpu",
        [400000, 700000, 1000000, 1200000],
    ),
    ("PDB_L4_rand", "repa_l4_256_per_residue_random_bs24_2gpu", [400000, 700000]),
    ("PDB_L4_MPNN", "repa_mpnn_l4_256_per_residue", [400000, 700000, 1000000, 1600000]),
    ("PDB_L9_MPNN", "repa_mpnn_l9_256_per_residue", [400000, 700000, 1000000, 1300000]),
    # --- AFDB ---
    ("AFDB_baseline", "baseline_afdb_256", [400000, 700000, 1000000, 1600000]),
    ("AFDB_L4_GN", "repa_l4_afdb_256", [400000, 700000, 1000000, 1300000]),
    ("AFDB_L9_GN", "repa_l9_afdb_256", [700000, 900000, 1000000]),
    ("AFDB_L9_MPNN", "repa_mpnn_l9_afdb_256", [400000, 700000, 1000000, 1300000]),
]:
    for s in steps:
        CASES.append((f"{label}_{s//1000}K", f"{fam}_step{s//1000}k_step_{s}"))


def load_atom37(pdb_path):
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                coords.append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
    arr = np.zeros((len(coords), 37, 3), dtype=np.float32)
    arr[:, 1] = np.array(coords, dtype=np.float32)
    return arr


def find_dirs(pat):
    return sorted(
        glob(
            str(
                EVAL_OUT
                / f"inference_paper_inference_fid_60m_paper_sweep_{pat}__{SAMPLER_TAG}__rep*"
            )
        )
    )


def pwtm_pooled(pdb_by_len):
    """Weighted-mean pwTM across length bins."""
    num, den = 0.0, 0
    nclust, ntot = 0, 0
    for L, pdbs in pdb_by_len.items():
        if len(pdbs) < 2:
            nclust += len(pdbs)
            ntot += len(pdbs)
            continue
        coords = []
        for p in pdbs:
            try:
                coords.append(load_atom37(p))
            except Exception:
                pass
        if len(coords) < 2:
            continue
        d = compute_diversity(coords)
        if not np.isnan(d["mean_pairwise_tm"]):
            npairs = len(coords) * (len(coords) - 1) // 2
            num += d["mean_pairwise_tm"] * npairs
            den += npairs
        nclust += d["n_clusters"]
        ntot += len(coords)
    return (num / den if den else float("nan")), nclust, ntot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cap", type=int, default=100, help="max whole-set samples per length"
    )
    args = ap.parse_args()
    rng = random.Random(42)

    print(
        f"{'case':<18} | {'whole pwTM':>10} {'(n)':>6} | {'des pwTM':>9} {'(n)':>6} | {'Δ(des-whole)':>12}"
    )
    print("-" * 75)
    results = {}
    for label, pat in CASES:
        whole_by_len = defaultdict(list)
        des_by_len = defaultdict(list)
        for rep_dir in find_dirs(pat):
            di_csv = Path(rep_dir) / "designability_index.csv"
            samples = Path(rep_dir) / "samples_fid"
            if not di_csv.exists() or not samples.is_dir():
                continue
            di = pd.read_csv(di_csv)
            # length column may be present
            for _, r in di.iterrows():
                rel = r["pdb_path"]
                p = Path(rep_dir) / rel
                if not p.exists():
                    continue
                L = (
                    int(r["length"])
                    if "length" in di.columns and not pd.isna(r["length"])
                    else 0
                )
                whole_by_len[L].append(p)
                if r["designable"]:
                    des_by_len[L].append(p)
        # subsample whole-set per length
        whole_capped = {}
        for L, ps in whole_by_len.items():
            ps2 = ps[:]
            rng.shuffle(ps2)
            whole_capped[L] = ps2[: args.cap]
        if not whole_capped:
            print(f"{label:<18} | <no data>")
            continue
        w_pwtm, w_clust, w_n = pwtm_pooled(whole_capped)
        d_pwtm, d_clust, d_n = pwtm_pooled(des_by_len)
        delta = (
            (d_pwtm - w_pwtm)
            if (not np.isnan(d_pwtm) and not np.isnan(w_pwtm))
            else float("nan")
        )
        print(
            f"{label:<18} | {w_pwtm:>10.3f} {w_n:>6} | {d_pwtm:>9.3f} {d_n:>6} | {delta:>+12.3f}"
        )
        results[label] = dict(
            whole_pwtm=w_pwtm,
            whole_n=w_n,
            des_pwtm=d_pwtm,
            des_n=d_n,
            whole_clust=w_clust,
            des_clust=d_clust,
        )

    out = (
        ROOT
        / "evaluation/proteina/generation/results/variance/wholeset_vs_designable_diversity.json"
    )
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print(
        "\nReading: if des pwTM >> whole pwTM for REPA but not baseline → designable-subset concentration is REPA-specific."
    )


if __name__ == "__main__":
    main()
