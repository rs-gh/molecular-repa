"""Experiment 1 from proteina_narratives.md: β-stratified diversity within designable.

Question: is REPA's lower #clusters explained by its higher β content (β-rich folds
are topologically constrained → fewer architectures naturally), or does REPA reduce
diversity *within* a fixed β-content slice (i.e., it's something beyond SS composition)?

Method:
  For each (model, step, sampler=γ=0.45), across all available reps:
    1. Walk eval_output dirs, load ss_fractions.npz + designability_index.csv
    2. Filter to designable; record (rep, length, per_sample_β_frac, pdb_path)
    3. Bin by per-sample β-fraction: low (<10%), mid (10–25%), high (≥25%)
    4. For each (length, β-bin) tuple, compute intra-set TM-score diversity:
       - #clusters (TM>=0.5 single-link), mean pairwise TM
    5. Aggregate across lengths within each β-bin (mean over bins, weighted by n)

Outputs a comparison table baseline vs REPA per β-bin per training step.

If REPA's pwTM within (length, β-bin) ≈ baseline's pwTM within (length, β-bin):
  → low-T-D is a composition effect, "sheets→fewer folds" holds at sample level
If REPA's pwTM within (length, β-bin) < baseline's:
  → REPA concentrates folds beyond what β composition explains
"""

from __future__ import annotations

import argparse
import json
import os
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

# Cases to analyze. Each (model_key, run_prefix, ckpt_step_label_in_dir).
# The eval_output dir naming we built earlier:
#   inference_paper_inference_fid_60m_paper_sweep_<run>_step_<step>__<sampler_tag>__rep<N>
CASES = [
    # --- PDB ---
    ("pdb_baseline_700K", "baseline_256_bs24_2gpu_step700k_step_700000"),
    ("pdb_baseline_1.0M", "baseline_256_bs24_2gpu_step1000k_step_1000000"),
    ("pdb_baseline_1.5M", "baseline_256_bs24_2gpu_step1500k_step_1500000"),
    ("pdb_baseline_1.6M", "baseline_256_bs24_2gpu_step1600k_step_1600000"),
    ("pdb_REPA_L9_GN_700K", "repa_l9_256_per_residue_bs24_2gpu_step700k_step_700000"),
    ("pdb_REPA_L9_GN_900K", "repa_l9_256_per_residue_bs24_2gpu_step900k_step_900000"),
    (
        "pdb_REPA_L9_GN_1000K",
        "repa_l9_256_per_residue_bs24_2gpu_step1000k_step_1000000",
    ),
    ("pdb_REPA_L4_GN_700K", "repa_l4_256_per_residue_bs24_2gpu_step700k_step_700000"),
    ("pdb_REPA_L9_MPNN_700K", "repa_mpnn_l9_256_per_residue_step700k_step_700000"),
    ("pdb_REPA_L9_MPNN_1000K", "repa_mpnn_l9_256_per_residue_step1000k_step_1000000"),
    # MPNN-L4 PDB (added 2026-05-27 per handoff TODO): another data point on
    # encoder × depth interaction. β-rich pwTM expected ≈ MPNN-L9 (~0.67)
    # if encoder dominates, ≈ baseline (~0.13) if depth dominates.
    ("pdb_REPA_L4_MPNN_400K", "repa_mpnn_l4_256_per_residue_step400k_step_400000"),
    ("pdb_REPA_L4_MPNN_700K", "repa_mpnn_l4_256_per_residue_step700k_step_700000"),
    ("pdb_REPA_L4_MPNN_1000K", "repa_mpnn_l4_256_per_residue_step1000k_step_1000000"),
    # random-encoder control (added 2026-06-09 for full-variant-grid concentration
    # table 6.8). Lets us say the β-concentration is REPA-wide vs variant-specific.
    # 700k sde_n0.45 generations exist in eval_output (ss_cache/ss_fractions.npz +
    # designability_index.csv present); post-processing only, no GPU.
    (
        "pdb_REPA_L4_random_700K",
        "repa_l4_256_per_residue_random_bs24_2gpu_step700k_step_700000",
    ),
    # --- AFDB ---
    ("afdb_baseline_700K", "baseline_afdb_256_step700k_step_700000"),
    ("afdb_baseline_1.0M", "baseline_afdb_256_step1000k_step_1000000"),
    ("afdb_baseline_1.6M", "baseline_afdb_256_step1600k_step_1600000"),
    ("afdb_REPA_L4_GN_700K", "repa_l4_afdb_256_step700k_step_700000"),
    ("afdb_REPA_L4_GN_1.0M", "repa_l4_afdb_256_step1000k_step_1000000"),
    ("afdb_REPA_L9_GN_700K", "repa_l9_afdb_256_step700k_step_700000"),
    ("afdb_REPA_L9_GN_900K", "repa_l9_afdb_256_step900k_step_900000"),
    ("afdb_REPA_L9_MPNN_700K", "repa_mpnn_l9_afdb_256_step700k_step_700000"),
    ("afdb_REPA_L9_MPNN_1.0M", "repa_mpnn_l9_afdb_256_step1000k_step_1000000"),
]

SAMPLER_TAG = "sde_n0.45"

# --- Auto-discover all-variant coverage at the report's standard steps --------
# Overrides the hand-curated CASES above so the all-model grid stays in sync with
# what is actually on disk in eval_output. Re-run after new evals land (e.g. job
# 30352629's 1.3M structures) to pick them up automatically. A (variant, step)
# is included only if a default-sampler rep dir with both ss_fractions.npz and
# designability_index.csv exists. CPU-only post-processing.
_VARIANTS = [
    ("pdb_baseline", "baseline_256_bs24_2gpu"),
    ("pdb_REPA_L4_random", "repa_l4_256_per_residue_random_bs24_2gpu"),
    ("pdb_REPA_L4_GN", "repa_l4_256_per_residue_bs24_2gpu"),
    ("pdb_REPA_L9_GN", "repa_l9_256_per_residue_bs24_2gpu"),
    ("pdb_REPA_L4_MPNN", "repa_mpnn_l4_256_per_residue"),
    ("pdb_REPA_L9_MPNN", "repa_mpnn_l9_256_per_residue"),
    ("afdb_baseline", "baseline_afdb_256"),
    ("afdb_REPA_L4_random", "repa_l4_afdb_256_random"),
    ("afdb_REPA_L4_GN", "repa_l4_afdb_256"),
    ("afdb_REPA_L9_GN", "repa_l9_afdb_256"),
    ("afdb_REPA_L4_MPNN", "repa_mpnn_l4_afdb_256"),
    ("afdb_REPA_L9_MPNN", "repa_mpnn_l9_afdb_256"),
]
_STEPS = [(400000, "400K"), (700000, "700K"), (1000000, "1.0M"), (1300000, "1.3M")]


def _has_inputs(run_base, step):
    pat = str(
        EVAL_OUT
        / (
            f"inference_paper_inference_fid_60m_paper_sweep_{run_base}"
            f"_step{step // 1000}k_step_{step}__{SAMPLER_TAG}__rep*"
        )
    )
    return any(
        (Path(rd) / "ss_cache/ss_fractions.npz").exists()
        and (Path(rd) / "designability_index.csv").exists()
        for rd in glob(pat)
    )


CASES = [
    (f"{label}_{slabel}", f"{run_base}_step{step // 1000}k_step_{step}")
    for label, run_base in _VARIANTS
    for step, slabel in _STEPS
    if _has_inputs(run_base, step)
]


def load_atom37(pdb_path: Path) -> np.ndarray:
    """Load atom37 coords from a generated PDB. Uses a minimal Cα-only loader
    since compute_diversity's tm-score function only needs CA (per code review)."""
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
    n = len(coords)
    arr = np.zeros((n, 37, 3), dtype=np.float32)
    # atom37 index 1 = CA in the atom37 convention; the tm_score code uses [:, 1]
    arr[:, 1] = np.array(coords, dtype=np.float32)
    return arr


def find_eval_dirs(run_step_pattern: str, sampler_tag: str):
    """Find all rep dirs matching a (run, step) prefix."""
    glob_pat = str(
        EVAL_OUT
        / f"inference_paper_inference_fid_60m_paper_sweep_{run_step_pattern}__{sampler_tag}__rep*"
    )
    return sorted(glob(glob_pat))


def collect_designable(case_label: str, run_step_pattern: str):
    """Return rows: (rep_idx, length, beta_frac, abs_pdb_path)."""
    rows = []
    for rep_dir in find_eval_dirs(run_step_pattern, SAMPLER_TAG):
        rep_idx = int(rep_dir.rsplit("__rep", 1)[1])
        ss_npz = Path(rep_dir) / "ss_cache/ss_fractions.npz"
        di_csv = Path(rep_dir) / "designability_index.csv"
        if not ss_npz.exists() or not di_csv.exists():
            continue
        npz = np.load(ss_npz, allow_pickle=True)
        # fracs columns are (H, E, C)
        fracs = npz["fracs"]
        paths = [str(p) for p in npz["paths"]]
        # Normalise paths to absolute (file may have been written with leading "./")
        path_to_beta = {
            os.path.abspath(p): float(fracs[i, 1]) for i, p in enumerate(paths)
        }
        di = pd.read_csv(di_csv)
        for _, r in di.iterrows():
            if not r["designable"]:
                continue
            rel = r["pdb_path"]
            abs_p = os.path.abspath(os.path.join(rep_dir, rel))
            if abs_p not in path_to_beta:
                continue
            rows.append((rep_idx, int(r["length"]), path_to_beta[abs_p], abs_p))
    return rows


BETA_BINS = [
    ("β<10", (0.00, 0.10)),
    ("10-25", (0.10, 0.25)),
    ("β≥25", (0.25, 1.01)),
]


def bin_by_beta(beta_frac: float):
    for name, (lo, hi) in BETA_BINS:
        if lo <= beta_frac < hi:
            return name
    return BETA_BINS[-1][0]


def diversity_for_subset(pdb_paths):
    """compute_diversity expects same-length intra-set; we pool by length here."""
    by_len = defaultdict(list)
    for p in pdb_paths:
        try:
            arr = load_atom37(Path(p))
            by_len[arr.shape[0]].append(arr)
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
    if not by_len:
        return {
            "n_clusters_total": 0,
            "pwtm_weighted": float("nan"),
            "n_samples": 0,
            "n_lengths": 0,
        }
    total_clusters = 0
    pwtm_sum = 0.0
    pwtm_weight = 0
    total_n = 0
    for L, coords in by_len.items():
        if len(coords) < 2:
            # singleton: contributes 1 cluster, pwtm undefined
            total_clusters += len(coords)
            total_n += len(coords)
            continue
        d = compute_diversity(coords)
        total_clusters += d["n_clusters"]
        if not np.isnan(d["mean_pairwise_tm"]):
            n_pairs = len(coords) * (len(coords) - 1) // 2
            pwtm_sum += d["mean_pairwise_tm"] * n_pairs
            pwtm_weight += n_pairs
        total_n += len(coords)
    return {
        "n_clusters_total": total_clusters,
        "pwtm_weighted": pwtm_sum / pwtm_weight if pwtm_weight else float("nan"),
        "n_samples": total_n,
        "n_lengths": len(by_len),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit_per_case",
        type=int,
        default=None,
        help="cap n samples per case for speed",
    )
    args = ap.parse_args()

    print(
        f"{'case':<20} {'β-bin':<8} | {'n_samp':>7} {'n_len':>5} {'#clust':>6} {'pwTM':>6}"
    )
    print("-" * 70)
    summary = []
    for case_label, run_step in CASES:
        rows = collect_designable(case_label, run_step)
        if not rows:
            print(f"{case_label:<20} <no eval_output dirs found>")
            continue
        for bin_name, _ in BETA_BINS:
            paths = [r[3] for r in rows if bin_by_beta(r[2]) == bin_name]
            if args.limit_per_case is not None:
                paths = paths[: args.limit_per_case]
            if not paths:
                print(
                    f"{case_label:<20} {bin_name:<8} | {'-':>7} {'-':>5} {'-':>6} {'-':>6}"
                )
                continue
            d = diversity_for_subset(paths)
            print(
                f"{case_label:<20} {bin_name:<8} | {d['n_samples']:>7} {d['n_lengths']:>5} {d['n_clusters_total']:>6} {d['pwtm_weighted']:>6.3f}"
            )
            summary.append(
                {
                    "case": case_label,
                    "bin": bin_name,
                    **d,
                }
            )
        print()

    # Write json
    out_path = (
        ROOT
        / "evaluation/proteina/generation/results/variance/beta_stratified_diversity.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
