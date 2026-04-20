# ruff: noqa: F841
"""Joint probe × FID plots — Fig 3c analogue + probe-redundancy scatter.

Merges:
  - `playground/proteina/probes/sweep_results.csv`   (probe quality metrics)
  - `evaluation/proteina/results/pdb/fid/lite_convergence_all.csv`   (FID / fJSD / fS)

on `(run, samples_seen)` (nsamples = step × batch_size; that's the fair x-axis
given baseline bs=6 vs REPA bs=4 at 512 residues).

Outputs to `playground/proteina/probes/figures/`:

  fig_pareto_fid_vs_probe.png        — Fig 3c analogue
    One point per (run, samples_seen). X = probe quality (P@L/5 at aligned
    layer), Y = PDB FID (lower = better). Colour by run, connected in
    training order. "Better FID and representation" = top-left.

  fig_pareto_fjsd_vs_probe.png       — same for fJSD_T
  fig_pareto_fS_vs_probe.png         — same for fS_T

  fig_probe_redundancy.png           — P@L/5 vs CATH accuracy scatter
    Are the two probes measuring the same thing? Strong diagonal → redundant;
    scatter → orthogonal (keep both).

Usage:
  python playground/proteina/probes/pareto.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROBE_CSV = HERE / "sweep_results.csv"
FID_CSV = HERE / "../../../evaluation/proteina/results/pdb/fid/lite_convergence_all.csv"
FIG_DIR = HERE / "figures"

# Map FID CSV run names ↔ probe CSV run names (same underlying _v2 runs).
PROBE_TO_FID = {
    "baseline": "Baseline (60M)",
    "repa_l0": "REPA (L0)",
    "repa_l4": "REPA (L4)",
    "repa_l9": "REPA (L9)",
}
FID_TO_PROBE = {v: k for k, v in PROBE_TO_FID.items()}

RUN_COLORS = {
    "baseline": "#1f77b4",
    "repa_l0": "#ff7f0e",
    "repa_l4": "#d62728",
    "repa_l9": "#2ca02c",
}
RUN_ALIGNED_LAYER = {"baseline": None, "repa_l0": 0, "repa_l4": 4, "repa_l9": 9}
BATCH_SIZES = {"baseline": 6, "repa_l0": 4, "repa_l4": 4, "repa_l9": 4}


# --------------------------------------------------------------------------- #
# Data loading + join
# --------------------------------------------------------------------------- #


def _load_probe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    for c in ["p_at_L", "p_at_L_2", "p_at_L_5", "cath_acc", "cath_f1", "step", "layer"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # samples_seen on the same convention as lite_convergence_all.csv
    df["samples_seen"] = df["step"] * df["run"].map(BATCH_SIZES).fillna(1).astype(int)
    return df


def _load_fid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_probe"] = df["run"].map(FID_TO_PROBE)
    df = df.dropna(subset=["run_probe"]).copy()
    return df


def _aligned_layer_rows(probe: pd.DataFrame) -> pd.DataFrame:
    """For each (run, step), pick the row at the run's aligned layer.
    For baseline (no alignment) pick layer 4 (matches training_repa default)."""
    picked = []
    for run, grp in probe.groupby("run"):
        if run == "gearnet" or grp["layer"].max() < 0:
            continue
        al = RUN_ALIGNED_LAYER.get(run)
        if al is None:
            al = 4  # baseline probe layer convention
        rows = grp[grp["layer"] == al]
        picked.append(rows)
    if not picked:
        return pd.DataFrame()
    return pd.concat(picked, ignore_index=True)


def _join(probe: pd.DataFrame, fid: pd.DataFrame) -> pd.DataFrame:
    """Merge probe × FID on (run_probe/run, samples_seen).  Uses nearest-step
    matching in samples_seen within 10% tolerance since FID and probe schedules
    may not agree exactly (e.g. 740k vs 750k on baseline vs REPA)."""
    probe_al = _aligned_layer_rows(probe)
    if probe_al.empty or fid.empty:
        return pd.DataFrame()
    # For each probe row, find closest FID row in same run.
    joined = []
    for _, p in probe_al.iterrows():
        cand = fid[fid["run_probe"] == p["run"]].copy()
        if cand.empty:
            continue
        cand["delta"] = (cand["samples_seen"] - p["samples_seen"]).abs()
        cand = cand.sort_values("delta").head(1)
        if cand.iloc[0]["delta"] > max(1000, 0.1 * p["samples_seen"]):
            continue  # nothing close enough
        r = p.to_dict()
        for col in [
            "_res_PDB_FID",
            "_res_PDB_fJSD_T",
            "_res_fS_T",
            "_res_AFDB_FID",
            "_res_AFDB_fJSD_T",
        ]:
            r[col] = cand.iloc[0].get(col, np.nan)
        r["fid_samples_seen"] = int(cand.iloc[0]["samples_seen"])
        joined.append(r)
    return pd.DataFrame(joined)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def _plot_pareto(
    joined: pd.DataFrame,
    y_col: str,
    y_label: str,
    x_col: str,
    x_label: str,
    title: str,
    out_path: Path,
    y_lower_is_better: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for run, grp in joined.groupby("run"):
        grp = grp.sort_values("samples_seen")
        grp = grp.dropna(subset=[x_col, y_col])
        if grp.empty:
            continue
        c = RUN_COLORS.get(run, "gray")
        ax.plot(grp[x_col], grp[y_col], "-", color=c, linewidth=1.3, alpha=0.6)
        sc = ax.scatter(
            grp[x_col],
            grp[y_col],
            c=np.log10(grp["samples_seen"].clip(lower=1)),
            cmap="viridis" if run == "baseline" else "plasma",
            s=55,
            edgecolors=c,
            linewidths=1.4,
            zorder=5,
            label=run,
        )
        # Label the final (largest-step) point
        last = grp.iloc[-1]
        ax.annotate(
            f"{int(last['samples_seen']/1e6)}M"
            if last["samples_seen"] >= 1e6
            else f"{int(last['samples_seen']/1e3)}K",
            (last[x_col], last[y_col]),
            fontsize=7,
            color=c,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_lower_is_better:
        ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_redundancy(probe: pd.DataFrame, out_path: Path) -> None:
    """Scatter P@L/5 vs CATH acc across all (run, step, layer) to see if they're
    redundant or orthogonal."""
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    sub = probe.dropna(subset=["p_at_L_5", "cath_acc"])
    if sub.empty:
        print(f"(no joint data for redundancy plot — skipping {out_path.name})")
        return
    for run, grp in sub.groupby("run"):
        c = RUN_COLORS.get(run, "gray")
        ax.scatter(
            grp["p_at_L_5"], grp["cath_acc"], s=25, alpha=0.55, color=c, label=run
        )
    # Correlation
    r = float(sub[["p_at_L_5", "cath_acc"]].corr().iloc[0, 1])
    ax.set_xlabel("Contact P@L/5")
    ax.set_ylabel("CATH accuracy")
    ax.set_title(f"Probe redundancy: Pearson r = {r:+.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}  (n={len(sub)}, correlation r={r:.3f})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_csv", type=str, default=str(PROBE_CSV))
    ap.add_argument("--fid_csv", type=str, default=str(FID_CSV))
    ap.add_argument("--outdir", type=str, default=str(FIG_DIR))
    args = ap.parse_args()

    global FIG_DIR
    FIG_DIR = Path(args.outdir)

    probe = _load_probe(Path(args.probe_csv))
    print(f"Probe rows: {len(probe)}  (runs: {sorted(probe['run'].unique().tolist())})")

    fid = _load_fid(Path(args.fid_csv))
    print(
        f"FID rows:   {len(fid)}  (runs: {sorted(fid['run_probe'].dropna().unique().tolist())})"
    )

    joined = _join(probe, fid)
    print(f"Joined (run, samples_seen within 10%): {len(joined)}")

    # Probe-redundancy plot works from probe data alone.
    _plot_redundancy(probe[probe["layer"] >= 0], FIG_DIR / "fig_probe_redundancy.png")

    if joined.empty:
        print("No joined probe×FID rows — skip Pareto plots until sweep completes.")
        return

    for y_col, y_label, fname in [
        ("_res_PDB_FID", "PDB FID (↓)", "fig_pareto_fid_vs_probe.png"),
        ("_res_PDB_fJSD_T", "PDB fJSD (Topology)", "fig_pareto_fjsd_vs_probe.png"),
        ("_res_fS_T", "Fold score (Topology)", "fig_pareto_fS_vs_probe.png"),
    ]:
        _plot_pareto(
            joined,
            y_col=y_col,
            y_label=y_label,
            x_col="p_at_L_5",
            x_label="Probe quality — contact P@L/5 (aligned layer)",
            title=f"Pareto — {y_label} vs P@L/5",
            out_path=FIG_DIR / fname,
            y_lower_is_better=(fname != "fig_pareto_fS_vs_probe.png"),
        )


if __name__ == "__main__":
    main()
