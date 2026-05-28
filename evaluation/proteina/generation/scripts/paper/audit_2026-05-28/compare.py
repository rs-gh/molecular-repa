"""For each (encoder, layer, dataset), compute Δ = REPA − baseline at every
(step, γ) cell, then render 10 Δ tables (Des / scRMSD / pLDDT / FID / fJSD-A /
fJSD-T / β=ss_E / ssJSD-2D / #Clust / pwTM) indexed by step × γ.

This is the data source for §§ 2–3 / § 7.3 of
docs/research/proteina_sampler_regime_audit_2026-05-28.md.

Run from repo root (writes ~930 lines):
    source .venv/bin/activate
    python evaluation/proteina/generation/scripts/paper/audit_2026-05-28/compare.py \
        > evaluation/proteina/generation/scripts/paper/audit_2026-05-28/compare_out.txt
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from load import load_all
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 80)

GAMMA_ORDER = ["ODE", "0.0", "0.35", "0.45", "0.5", "1.0"]

METRICS_BASIC = [
    "_res_designability_rate",
    "_res_scRMSD_mean",
    "_res_plddt_mean",
    "_res_ss_frac_E",
    "_res_diversity_clusters_mean",
    "_res_diversity_pairwise_tm_mean",
]

DISPLAY = [
    ("Des ↑", "_res_designability_rate", 3),
    ("scRMSD ↓", "_res_scRMSD_mean", 2),
    ("pLDDT ↑", "_res_plddt_mean", 3),
    ("FID ↓", "_res_FID", 1),
    ("fJSD-A ↓", "_res_fJSD_A", 2),
    ("fJSD-T ↓", "_res_fJSD_T", 2),
    ("β=ss_E ↑", "_res_ss_frac_E", 3),
    ("ssJSD2D ↓", "_res_ss_jsd_2d", 3),
    ("#Clust ↑", "_res_diversity_clusters_mean", 1),
    ("pwTM ↓", "_res_diversity_pairwise_tm_mean", 3),
]


def add_homedb(df):
    """Resolve the PDB-vs-AFDB homedb metric columns into single dataset-agnostic ones."""
    df = df.copy()
    df["_res_FID"] = df.apply(lambda r: r.get(f"_res_{r['dataset']}_FID"), axis=1)
    df["_res_fJSD_A"] = df.apply(lambda r: r.get(f"_res_{r['dataset']}_fJSD_A"), axis=1)
    df["_res_fJSD_C"] = df.apply(lambda r: r.get(f"_res_{r['dataset']}_fJSD_C"), axis=1)
    df["_res_fJSD_T"] = df.apply(lambda r: r.get(f"_res_{r['dataset']}_fJSD_T"), axis=1)
    df["_res_ss_jsd_2d"] = df.apply(
        lambda r: r.get(f"_res_ss_jsd_{r['dataset'].lower()}_2d"), axis=1
    )
    return df


def main():
    df = load_all()
    df["step_K"] = (df["step"] / 1000).astype(int)
    df = add_homedb(df)

    agg_cols = METRICS_BASIC + [
        "_res_FID",
        "_res_fJSD_A",
        "_res_fJSD_C",
        "_res_fJSD_T",
        "_res_ss_jsd_2d",
    ]
    group_cols = ["dataset", "encoder", "layer", "step_K", "gamma"]
    agg = df.groupby(group_cols)[agg_cols].mean(numeric_only=True).reset_index()

    base = agg[agg.encoder == "baseline"].drop(columns=["encoder", "layer"])
    base = base.rename(columns={c: f"base__{c}" for c in agg_cols})

    repa = agg[agg.encoder != "baseline"]
    merged = repa.merge(base, on=["dataset", "step_K", "gamma"], how="left")
    for c in agg_cols:
        merged[f"d__{c}"] = merged[c] - merged[f"base__{c}"]

    for ds in ["PDB", "AFDB"]:
        for (enc, layer), g in merged[merged.dataset == ds].groupby(
            ["encoder", "layer"]
        ):
            if g.empty:
                continue
            n_cells = g.dropna(subset=[f"d__{m[1]}" for m in DISPLAY], how="all").shape[
                0
            ]
            print(
                f"\n========= {ds} · REPA {enc}-{layer} − baseline · {n_cells} (step×γ) cells ========="
            )
            for label, col, prec in DISPLAY:
                dcol = f"d__{col}"
                tbl = g.pivot_table(index="step_K", columns="gamma", values=dcol)
                cols = [c for c in GAMMA_ORDER if c in tbl.columns]
                if not cols or tbl[cols].isna().all().all():
                    continue
                print(f"\n  [Δ {label}]")
                print(tbl[cols].round(prec).to_string())


if __name__ == "__main__":
    main()
