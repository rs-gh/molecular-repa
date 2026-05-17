"""Correlation between representation quality and generation quality.

For each n=128 convergence dataset (PDB, AFDB), joins generation metrics
(_res_PDB_FID, _res_designability_rate) with representation probes
(CATH-T top1 accuracy, inverse-folding top1, dihedral MAE) on (run, step).

Representation metrics are reduced across layers per checkpoint by taking the
best layer (max for accuracies, min for dihedral MAE). All rep rows are at
t=1.0 in this sweep, so no t-reduction is needed.

Outputs:
  - figures/paper/n128_convergence/gen_vs_rep_correlation.png  (3 rep x 2 gen
    scatter grid per dataset; panel titles show Spearman rho).
  - results/paper/n128_convergence_gen_vs_rep_correlation.{csv,md} with
    Pearson/Spearman correlations, partial correlations controlling for step,
    and a step >= 200k robustness variant.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[5]
GEN_RESULTS = REPO_ROOT / "evaluation/proteina/generation/results/paper"
REP_RESULTS = REPO_ROOT / "evaluation/proteina/representation/results/paper"
FIG_OUT = REPO_ROOT / "evaluation/proteina/joint/figures/paper/n128_convergence"
TAB_OUT = REPO_ROOT / "evaluation/proteina/joint/results/paper"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": {
        "gen_jsonl": GEN_RESULTS / "n128_convergence_pdb" / "sweep_results.jsonl",
        "rep_csv": REP_RESULTS
        / "n128_convergence_cath_if_dih_pdb"
        / "pretrained_sweep_results.csv",
    },
    "AFDB": {
        "gen_jsonl": GEN_RESULTS / "n128_convergence_afdb" / "sweep_results.jsonl",
        "rep_csv": REP_RESULTS
        / "n128_convergence_cath_if_dih_afdb"
        / "pretrained_sweep_results.csv",
    },
}

# (prefix, label, color)
RUN_FAMILIES = {
    "PDB": [
        ("baseline_128_bs80", "Baseline", "tab:blue"),
        ("repa_l4_128_bs80", "REPA L4 GearNet", "tab:red"),
        ("repa_l9_128_bs80", "REPA L9 GearNet", "tab:orange"),
        ("repa_mpnn_l4_128_bs80", "REPA L4 MPNN", "tab:green"),
        ("repa_mpnn_l9_128_bs80_2gpu", "REPA L9 MPNN", "tab:purple"),
    ],
    "AFDB": [
        ("baseline_afdb_128_bs80", "Baseline", "tab:blue"),
        ("repa_l4_afdb_128_bs80", "REPA L4 GearNet", "tab:red"),
        ("repa_mpnn_l4_afdb_128_bs80", "REPA L4 MPNN", "tab:green"),
        ("repa_mpnn_l9_afdb_128_bs80_2gpu", "REPA L9 MPNN", "tab:purple"),
    ],
}

REP_METRICS = [
    # (column, label, reduction: "max" or "min")
    ("cath_C_top1", "CATH-C top1 acc", "max"),
    ("cath_A_top1", "CATH-A top1 acc", "max"),
    ("cath_T_top1", "CATH-T top1 acc", "max"),
    ("if_top1_acc", "IF top1 acc", "max"),
    ("dih_mae_total_deg", "Dihedral MAE (deg)", "min"),
]

GEN_METRICS = [
    ("_res_PDB_FID", "FID vs PDB"),
    ("_res_AFDB_FID", "FID vs AFDB"),
    ("_res_designability_rate", "Designability rate"),
]


def fnum(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def load_gen(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    if not path.exists():
        return pd.DataFrame(columns=["run", "step"] + [m for m, _ in GEN_METRICS])
    for line in path.open():
        r = json.loads(line)
        err = r.get("error")
        if err and err != "NONE":
            continue
        run = r.get("run")
        step = r.get("step")
        if run is None or step is None:
            continue
        row = {"run": run, "step": int(step)}
        for col, _ in GEN_METRICS:
            row[col] = fnum(r.get(col))
        rows.append(row)
    return pd.DataFrame(rows)


def load_rep(path: Path) -> pd.DataFrame:
    """One row per (run, step): best layer for each metric."""
    cath_C: Dict[Tuple[str, int], float] = defaultdict(lambda: -np.inf)
    cath_A: Dict[Tuple[str, int], float] = defaultdict(lambda: -np.inf)
    cath_T: Dict[Tuple[str, int], float] = defaultdict(lambda: -np.inf)
    iff: Dict[Tuple[str, int], float] = defaultdict(lambda: -np.inf)
    dih: Dict[Tuple[str, int], float] = defaultdict(lambda: np.inf)
    if not path.exists():
        return pd.DataFrame(columns=["run", "step"] + [m for m, _, _ in REP_METRICS])
    with path.open() as fh:
        for r in csv.DictReader(fh):
            run = r.get("run")
            s = fnum(r.get("step"))
            if run is None or s is None:
                continue
            key = (run, int(s))
            kind = r.get("probe_kind")
            if kind == "cath":
                level = r.get("cath_level")
                v = fnum(r.get("cath_accuracy"))
                if v is None:
                    continue
                if level == "C" and v > cath_C[key]:
                    cath_C[key] = v
                elif level == "A" and v > cath_A[key]:
                    cath_A[key] = v
                elif level == "T" and v > cath_T[key]:
                    cath_T[key] = v
            elif kind == "inverse_folding":
                v = fnum(r.get("if_top1_acc"))
                if v is not None and v > iff[key]:
                    iff[key] = v
            elif kind == "dihedral":
                v = fnum(r.get("dih_mae_total_deg"))
                if v is not None and v < dih[key]:
                    dih[key] = v
    keys = set(cath_C) | set(cath_A) | set(cath_T) | set(iff) | set(dih)
    rows = []
    for run, step in keys:
        rows.append(
            {
                "run": run,
                "step": step,
                "cath_C_top1": cath_C.get((run, step))
                if np.isfinite(cath_C.get((run, step), -np.inf))
                else None,
                "cath_A_top1": cath_A.get((run, step))
                if np.isfinite(cath_A.get((run, step), -np.inf))
                else None,
                "cath_T_top1": cath_T.get((run, step))
                if np.isfinite(cath_T.get((run, step), -np.inf))
                else None,
                "if_top1_acc": iff.get((run, step))
                if np.isfinite(iff.get((run, step), -np.inf))
                else None,
                "dih_mae_total_deg": dih.get((run, step))
                if np.isfinite(dih.get((run, step), np.inf))
                else None,
            }
        )
    return pd.DataFrame(rows)


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Spearman partial correlation of x and y controlling for z (rank-based)."""
    if len(x) < 4:
        return float("nan"), float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rz = pd.Series(z).rank().to_numpy()

    # residuals after linear regression on rz
    def resid(a: np.ndarray) -> np.ndarray:
        A = np.column_stack([np.ones_like(rz), rz])
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta

    ex = resid(rx)
    ey = resid(ry)
    r, p = pearsonr(ex, ey)
    return float(r), float(p)


def compute_correlations(
    df: pd.DataFrame, dataset: str, step_filter: Optional[int]
) -> List[dict]:
    if step_filter is not None:
        df = df[df["step"] >= step_filter]
    out = []
    for rep_col, _, reduction in REP_METRICS:
        for gen_col, _ in GEN_METRICS:
            sub = df[["run", "step", rep_col, gen_col]].dropna()
            if len(sub) < 4:
                out.append(
                    {
                        "dataset": dataset,
                        "step_filter": step_filter or 0,
                        "rep_metric": rep_col,
                        "gen_metric": gen_col,
                        "n": len(sub),
                        "pearson_r": None,
                        "pearson_p": None,
                        "spearman_r": None,
                        "spearman_p": None,
                        "partial_spearman_r_ctrl_step": None,
                        "partial_spearman_p_ctrl_step": None,
                    }
                )
                continue
            pr, pp = pearsonr(sub[rep_col], sub[gen_col])
            sr, sp = spearmanr(sub[rep_col], sub[gen_col])
            par_r, par_p = partial_corr(
                sub[rep_col].to_numpy(), sub[gen_col].to_numpy(), sub["step"].to_numpy()
            )
            out.append(
                {
                    "dataset": dataset,
                    "step_filter": step_filter or 0,
                    "rep_metric": rep_col,
                    "gen_metric": gen_col,
                    "n": int(len(sub)),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                    "partial_spearman_r_ctrl_step": par_r,
                    "partial_spearman_p_ctrl_step": par_p,
                }
            )
    return out


def plot_scatter_grid(
    df: pd.DataFrame, dataset: str, families: List[Tuple[str, str, str]]
):
    fig, axes = plt.subplots(
        len(REP_METRICS),
        len(GEN_METRICS),
        figsize=(5.0 * len(GEN_METRICS), 3.6 * len(REP_METRICS)),
    )
    if len(REP_METRICS) == 1:
        axes = np.array([axes])
    for i, (rep_col, rep_label, _) in enumerate(REP_METRICS):
        for j, (gen_col, gen_label) in enumerate(GEN_METRICS):
            ax = axes[i, j]
            sub_all = df[["run", "step", rep_col, gen_col]].dropna()
            for prefix, label, color in families:
                sub = sub_all[sub_all["run"].str.startswith(prefix)]
                if sub.empty:
                    continue
                sizes = 20 + 0.00006 * sub["step"].to_numpy()
                ax.scatter(
                    sub[rep_col],
                    sub[gen_col],
                    s=sizes,
                    color=color,
                    marker="o",
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.85,
                    label=label if (i == 0 and j == 0) else None,
                )
            if len(sub_all) >= 4:
                sr, sp = spearmanr(sub_all[rep_col], sub_all[gen_col])
                pr, _ = pearsonr(sub_all[rep_col], sub_all[gen_col])
                ax.set_title(
                    f"{rep_label} vs {gen_label}\nSpearman ρ={sr:.2f} (p={sp:.1e}), Pearson r={pr:.2f}",
                    fontsize=9,
                )
            else:
                ax.set_title(f"{rep_label} vs {gen_label}", fontsize=9)
            ax.set_xlabel(rep_label)
            ax.set_ylabel(gen_label)
            ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{dataset} (n=128): generation vs representation — all checkpoints\nPoint size ∝ training step",
        fontsize=12,
    )
    fig.legend(
        loc="lower center", ncol=len(families), bbox_to_anchor=(0.5, -0.02), fontsize=9
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = FIG_OUT / f"gen_vs_rep_correlation_{dataset.lower()}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    all_corr: List[dict] = []
    for ds_name, paths in DATASETS.items():
        gen_df = load_gen(paths["gen_jsonl"])
        rep_df = load_rep(paths["rep_csv"])
        joined = pd.merge(gen_df, rep_df, on=["run", "step"], how="inner")
        print(
            f"[{ds_name}] gen rows={len(gen_df)}, rep rows={len(rep_df)}, joined={len(joined)}"
        )
        if joined.empty:
            continue
        plot_scatter_grid(joined, ds_name, RUN_FAMILIES[ds_name])
        for sf in (None, 200_000):
            all_corr.extend(compute_correlations(joined, ds_name, sf))

    corr_df = pd.DataFrame(all_corr)
    csv_path = TAB_OUT / "n128_convergence_gen_vs_rep_correlation.csv"
    md_path = TAB_OUT / "n128_convergence_gen_vs_rep_correlation.md"
    corr_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Markdown: one block per (dataset, step_filter)
    lines: List[str] = ["# Gen vs Rep correlations (n=128 convergence)\n"]
    lines.append(
        "Rep metrics are reduced across layers per checkpoint (max for accuracies, min for MAE). All rep rows are at t=1.0.\n"
    )
    lines.append(
        "`partial_spearman_*_ctrl_step` is the Spearman partial correlation controlling for training step.\n"
    )
    for (ds, sf), g in corr_df.groupby(["dataset", "step_filter"]):
        tag = "all checkpoints" if sf == 0 else f"step >= {sf}"
        lines.append(f"\n## {ds} — {tag}\n")
        cols = [
            "rep_metric",
            "gen_metric",
            "n",
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
            "partial_spearman_r_ctrl_step",
            "partial_spearman_p_ctrl_step",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in g.iterrows():

            def fmt(v):
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return "—"
                if isinstance(v, float):
                    return f"{v:.3f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
                return str(v)

            lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
