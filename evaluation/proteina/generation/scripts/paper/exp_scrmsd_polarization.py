"""scRMSD polarization analysis (formerly "bimodality").

Origin TODO predicted REPA → bimodal scRMSD (peaks at <2Å AND >5Å). That
prediction was NOT borne out: the histograms are unimodal for both baseline
and REPA (one peak near 0–2Å + a monotone-decaying tail; no second mode). The
bimodality coefficient does not separate baseline from REPA — a fat right tail
inflates it even for a unimodal distribution.

What the data DOES show is polarization: REPA depletes the marginal 2–4Å bin
relative to baseline. Early in training the depleted mass goes to <2Å
(designable); late in training it splits, with a growing share landing in the
>4Å broken tail. So REPA's higher scRMSD-mean at matched/higher Des% is a
fatter far tail, not a second mode.

Outputs:
  - figures/.../scrmsd_polarization.png  (linear + log-y histograms per pair)
  - results/variance/scrmsd_polarization_stats.json (per-bin mass + fractions;
    bimodality_coef retained only to document that it does NOT separate the two)
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[5]
EVAL_OUT = REPO_ROOT / "eval_output"
SAMPLER_TAG = "sde_n0.45"

# Pairs at matched steps. (label, baseline_prefix, repa_prefix).
PAIRS = [
    (
        "PDB 700K",
        "baseline_256_bs24_2gpu_step700k_step_700000",
        "repa_l9_256_per_residue_bs24_2gpu_step700k_step_700000",
    ),
    (
        "PDB 1000K",
        "baseline_256_bs24_2gpu_step1000k_step_1000000",
        "repa_l9_256_per_residue_bs24_2gpu_step1000k_step_1000000",
    ),
    (
        "AFDB 700K",
        "baseline_afdb_256_step700k_step_700000",
        "repa_l4_afdb_256_step700k_step_700000",
    ),
    (
        "AFDB 1000K (REPA L4)",
        "baseline_afdb_256_step1000k_step_1000000",
        "repa_l4_afdb_256_step1000k_step_1000000",
    ),
]


def collect_scrmsd(run_step_prefix: str):
    """Aggregate per-sample (scRMSD, designable) across all reps."""
    rows = []
    pat = str(
        EVAL_OUT
        / f"inference_paper_inference_fid_60m_paper_sweep_{run_step_prefix}__{SAMPLER_TAG}__rep*"
    )
    for rep_dir in sorted(glob(pat)):
        di_csv = Path(rep_dir) / "designability_index.csv"
        if not di_csv.exists():
            continue
        d = pd.read_csv(di_csv)
        d = d[d["evaluated"] == True].copy()  # noqa: E712
        rows.append(d[["scRMSD", "designable", "length"]])
    if not rows:
        return pd.DataFrame(columns=["scRMSD", "designable", "length"])
    return pd.concat(rows, ignore_index=True)


def bimodality_coef(x: np.ndarray) -> float:
    """SAS bimodality coefficient. >5/9 ≈ 0.555 suggests bimodality."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return float("nan")
    mu = x.mean()
    sigma = x.std(ddof=1)
    if sigma == 0:
        return float("nan")
    z = (x - mu) / sigma
    g1 = (z**3).mean()  # skewness
    g2 = (z**4).mean() - 3.0  # excess kurtosis
    # SAS formula
    return (g1**2 + 1.0) / (g2 + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))


def bin_counts(x: np.ndarray, edges=(0, 2, 4, 8, np.inf)) -> dict:
    h, _ = np.histogram(x, bins=edges)
    keys = [f"[{edges[i]},{edges[i+1]})" for i in range(len(edges) - 1)]
    return dict(zip(keys, [int(v) for v in h]))


def main():
    out_fig_dir = (
        REPO_ROOT / "evaluation/proteina/generation/figures/paper/n256_sampler_ablation"
    )
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_json = (
        REPO_ROOT
        / "evaluation/proteina/generation/results/variance/scrmsd_polarization_stats.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    stats = {}
    fig, axes = plt.subplots(
        len(PAIRS), 2, figsize=(11, 3.0 * len(PAIRS)), sharex="row"
    )
    if len(PAIRS) == 1:
        axes = axes[None, :]

    for i, (label, base_prefix, repa_prefix) in enumerate(PAIRS):
        base = collect_scrmsd(base_prefix)
        repa = collect_scrmsd(repa_prefix)
        if base.empty or repa.empty:
            print(f"{label:<25} skip (base n={len(base)}, repa n={len(repa)})")
            stats[label] = {"missing": True}
            continue
        bx = base["scRMSD"].dropna().values
        rx = repa["scRMSD"].dropna().values

        stats[label] = {
            "baseline": {
                "n": int(len(bx)),
                "mean": float(bx.mean()),
                "median": float(np.median(bx)),
                "frac_below_2": float((bx < 2).mean()),
                "frac_marginal_2_4": float(((bx >= 2) & (bx < 4)).mean()),
                "frac_above_4": float((bx > 4).mean()),
                "bimodality_coef": float(bimodality_coef(bx)),
                "bins": bin_counts(bx),
            },
            "repa": {
                "n": int(len(rx)),
                "mean": float(rx.mean()),
                "median": float(np.median(rx)),
                "frac_below_2": float((rx < 2).mean()),
                "frac_marginal_2_4": float(((rx >= 2) & (rx < 4)).mean()),
                "frac_above_4": float((rx > 4).mean()),
                "bimodality_coef": float(bimodality_coef(rx)),
                "bins": bin_counts(rx),
            },
        }

        # Histograms: full evaluated subset (left); designable filter is implicit
        # via the <2Å mass — but we also show clipped log-y to make tails visible.
        edges = np.linspace(0, 20, 41)
        ax_l, ax_r = axes[i]
        ax_l.hist(
            bx,
            bins=edges,
            alpha=0.6,
            color="C0",
            label=f"baseline (n={len(bx)})",
            density=True,
        )
        ax_l.hist(
            rx,
            bins=edges,
            alpha=0.6,
            color="C1",
            label=f"REPA (n={len(rx)})",
            density=True,
        )
        ax_l.axvline(
            2.0, color="k", linestyle=":", linewidth=0.8, label="Des threshold (2Å)"
        )
        ax_l.set_title(f"{label}  (linear)")
        ax_l.set_xlabel("scRMSD (Å)")
        ax_l.set_ylabel("density")
        ax_l.legend(fontsize=8)

        ax_r.hist(
            bx, bins=edges, alpha=0.6, color="C0", label="baseline", density=False
        )
        ax_r.hist(rx, bins=edges, alpha=0.6, color="C1", label="REPA", density=False)
        ax_r.axvline(2.0, color="k", linestyle=":", linewidth=0.8)
        ax_r.set_yscale("log")
        ax_r.set_title(f"{label}  (log-y)")
        ax_r.set_xlabel("scRMSD (Å)")
        ax_r.set_ylabel("count (log)")

        b, r = stats[label]["baseline"], stats[label]["repa"]
        print(
            f"{label:<24}  <2Å {b['frac_below_2']:.2f}→{r['frac_below_2']:.2f}  "
            f"| 2-4Å(marginal) {b['frac_marginal_2_4']:.2f}→{r['frac_marginal_2_4']:.2f}  "
            f"| >4Å {b['frac_above_4']:.2f}→{r['frac_above_4']:.2f}  "
            f"(BC {b['bimodality_coef']:.2f} vs {r['bimodality_coef']:.2f} — does NOT separate)"
        )

    fig.suptitle(
        "Per-sample scRMSD — polarization (unimodal; REPA depletes the 2–4Å marginal bin)",
        y=1.005,
    )
    fig.tight_layout()
    fig_path = out_fig_dir / "scrmsd_polarization.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"\nWrote {fig_path}")

    out_json.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
