"""Render the baseline metric × step × γ tables used in § 1 / § 7.{1,2} of
docs/research/proteina_sampler_regime_audit_2026-05-28.md.

Run from repo root:
    source .venv/bin/activate
    python evaluation/proteina/generation/scripts/paper/audit_2026-05-28/baseline_tradeoff.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from load import load_all
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

GAMMA_ORDER = ["ODE", "0.0", "0.35", "0.45", "0.5", "1.0"]

METRICS = [
    ("Des", "_res_designability_rate", 3),
    ("scRMSD", "_res_scRMSD_mean", 2),
    ("pLDDT", "_res_plddt_mean", 3),
    ("β=ss_E", "_res_ss_frac_E", 3),
    ("#Clust", "_res_diversity_clusters_mean", 1),
    ("pwTM", "_res_diversity_pairwise_tm_mean", 3),
]


def render(df, ds, col, prec=3):
    sub = df[df["dataset"] == ds]
    if col not in sub.columns:
        return None
    g = sub.groupby(["step_K", "gamma"])[col].mean().unstack("gamma")
    cols = [c for c in GAMMA_ORDER if c in g.columns]
    if not cols:
        return None
    return g[cols].round(prec)


def main():
    df = load_all()
    df = df[df["encoder"] == "baseline"].copy()
    df["step_K"] = (df["step"] / 1000).astype(int)

    for ds in ["PDB", "AFDB"]:
        ds_metrics = METRICS + [
            ("FID", f"_res_{ds}_FID", 1),
            ("fJSD-A", f"_res_{ds}_fJSD_A", 2),
            ("fJSD-T", f"_res_{ds}_fJSD_T", 2),
            ("ssJSD2D", f"_res_ss_jsd_{ds.lower()}_2d", 3),
        ]
        print(f"\n========= {ds} baseline · metric × step × γ =========")
        for label, col, prec in ds_metrics:
            tbl = render(df, ds, col, prec)
            if tbl is None or tbl.empty:
                continue
            print(f"\n[{label}]  ({col})")
            print(tbl.to_string())


if __name__ == "__main__":
    main()
