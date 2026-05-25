"""Mode-collapse diagnostic plot for n=128 and n=256.

Three-panel figure per n:

  Top-left  — Designability (%) vs fS_T scatter.
              Size  = sqrt(absolute clusters).
              Color = SS JSD on the designable subset (lower = closer to
              natural PDB SS distribution after the designability filter).
  Top-right — SS %H_overall (x) vs SS %H_designable (y) scatter.
              y=x diagonal drawn; vertical/horizontal dotted lines at
              the natural-PDB %H reference. Distance above the diagonal
              = how much the designability filter (ProteinMPNN→ESMFold)
              selects for helix in this model.
  Bottom    — Paired SS stacked bars per run, overall (faded) vs designable
              (solid), sorted by Δ%H = %H_des − %H_overall (most-biased on
              top). One bar pair = one run.

Reads the per-n TSVs produced by ``jsonl_to_tsv.py`` (schema: ``profile``,
``run``, ``step``, ``ckpt_path``, ``_res_*`` metric columns).

Legend lineage rule: every annotation/bar label is rendered via
``pretty_run_label`` so the checkpoint step is visible on the figure.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from evaluation.proteina.lib.plot_labels import (
    block_label_plan,
    compose_legend_label,
)


def _humanize(v, _pos=None):
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:g}M"
    if av >= 1e3:
        return f"{v / 1e3:g}K"
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


ROOT = Path(__file__).resolve().parents[2]  # .../proteina/generation
FIG_ROOT = ROOT / "figures/paper"

NATURAL_PDB_H = 0.35

DES_COL = "_res_designability_rate"
FS_T_COL = "_res_fS_T"
CLUSTERS_COL = "_res_diversity_clusters_total"
DES_N_COL = "_res_designability_n"
SS_H_COL = "_res_ss_frac_H"
SS_E_COL = "_res_ss_frac_E"
SS_JSD_PDB_COL = "_res_ss_jsd_pdb"
SS_H_DES_COL = "_res_ss_frac_H_designable"
SS_E_DES_COL = "_res_ss_frac_E_designable"
SS_JSD_DES_PDB_COL = "_res_ss_jsd_pdb_designable"

TARGETS = {
    128: (
        FIG_ROOT / "n128_paper/n128_paper_tables.tsv",
        FIG_ROOT / "n128_paper/n128_mode_collapse.png",
    ),
    256: (
        FIG_ROOT / "n256_paper/n256_paper_tables.tsv",
        FIG_ROOT / "n256_paper/n256_mode_collapse.png",
    ),
}


def parse_tsv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def to_float(x):
    if x is None:
        return None
    x = x.strip()
    if x in ("", "—", "-"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def collect(rows: list[dict]) -> list[dict]:
    # Compute the metadata fields that vary across this run set so the legend
    # only carries deltas (model/encoder/dataset annotated per-run when they
    # differ between rows). Falls back to pretty_run_label for runs without a
    # RUN_META entry.
    all_runs = [row["run"] for row in rows]
    _, varying = block_label_plan(all_runs)
    pts = []
    for row in rows:
        des = to_float(row.get(DES_COL, ""))
        fs_t = to_float(row.get(FS_T_COL, ""))
        if des is None or fs_t is None:
            continue
        step_str = row.get("step", "").strip()
        try:
            step = int(float(step_str)) if step_str else None
        except ValueError:
            step = None
        clusters = to_float(row.get(CLUSTERS_COL, ""))
        des_n = to_float(row.get(DES_N_COL, ""))
        ss_h = to_float(row.get(SS_H_COL, ""))
        ss_e = to_float(row.get(SS_E_COL, ""))
        ss_jsd = to_float(row.get(SS_JSD_PDB_COL, ""))
        ss_h_des = to_float(row.get(SS_H_DES_COL, ""))
        ss_e_des = to_float(row.get(SS_E_DES_COL, ""))
        ss_jsd_des = to_float(row.get(SS_JSD_DES_PDB_COL, ""))
        pts.append(
            {
                "run": row["run"],
                "step": step,
                "label": compose_legend_label(
                    row["run"],
                    step=step,
                    varying_fields=varying,
                    fallback_display=row["run"],
                ),
                "des_pct": des * 100.0,
                "fs_t": fs_t,
                "clusters": clusters,
                "des_count": des * des_n if des_n is not None else None,
                "ss_h": ss_h,
                "ss_e": ss_e,
                "ss_jsd": ss_jsd,
                "ss_h_des": ss_h_des,
                "ss_e_des": ss_e_des,
                "ss_jsd_des": ss_jsd_des,
            }
        )
    return pts


def size_from_clusters(c):
    if c is None or c <= 0:
        return 30.0
    return 30.0 + 18.0 * math.sqrt(c)


def plot_one(n: int, tsv: Path, out: Path):
    rows = parse_tsv(tsv)
    pts = collect(rows)
    if not pts:
        raise RuntimeError(f"no usable rows in {tsv}")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.0], hspace=0.32, wspace=0.30)
    ax_sc1 = fig.add_subplot(gs[0, 0])
    ax_sc2 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    # ---- Top-left: Des vs fS_T, color = SS JSD des ----
    jsd_des_vals = [p["ss_jsd_des"] for p in pts if p["ss_jsd_des"] is not None]
    norm1 = Normalize(vmin=0.0, vmax=max(jsd_des_vals)) if jsd_des_vals else None
    cmap1 = plt.get_cmap("viridis_r")

    for p in pts:
        size = size_from_clusters(p["clusters"])
        if norm1 is not None and p["ss_jsd_des"] is not None:
            color = cmap1(norm1(p["ss_jsd_des"]))
        else:
            color = "lightgray"
        ax_sc1.scatter(
            p["des_pct"],
            p["fs_t"],
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.88,
            zorder=3,
        )
        ax_sc1.annotate(
            p["label"],
            (p["des_pct"], p["fs_t"]),
            fontsize=6.5,
            xytext=(4, 3),
            textcoords="offset points",
            alpha=0.85,
            zorder=4,
        )

    ax_sc1.set_xlabel("Designability (%) ↑")
    ax_sc1.set_ylabel("fS_T ↑")
    ax_sc1.set_title(
        "Des vs fS_T  —  color = SS JSD on designable subset (low/yellow = closer to natural)"
    )
    ax_sc1.grid(True, alpha=0.3)
    ax_sc1.xaxis.set_major_formatter(FuncFormatter(_humanize))
    ax_sc1.yaxis.set_major_formatter(FuncFormatter(_humanize))

    if norm1 is not None:
        sm1 = ScalarMappable(norm=norm1, cmap=cmap1)
        sm1.set_array([])
        cb1 = fig.colorbar(sm1, ax=ax_sc1, fraction=0.04, pad=0.02)
        cb1.set_label("SS JSD on designable subset ↓")

    leg_handles = []
    for v in [10, 50, 100, 200]:
        leg_handles.append(
            ax_sc1.scatter(
                [],
                [],
                s=size_from_clusters(v),
                color="lightgray",
                edgecolor="black",
                linewidth=0.6,
                label=f"clusters={v}",
            )
        )
    ax_sc1.legend(
        handles=leg_handles,
        title="absolute clusters",
        loc="lower left",
        fontsize=7,
        title_fontsize=7,
        frameon=True,
        labelspacing=1.0,
    )

    ax_sc1.text(
        0.98,
        0.03,
        "high Des, low fS_T",
        transform=ax_sc1.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="right",
        style="italic",
        color="darkred",
    )
    ax_sc1.text(
        0.98,
        0.97,
        "high Des, high fS_T\n(target zone)",
        transform=ax_sc1.transAxes,
        fontsize=7.5,
        va="top",
        ha="right",
        style="italic",
        color="darkgreen",
    )

    # ---- Top-right: %H_overall vs %H_des ----
    drawn = 0
    for p in pts:
        if p["ss_h"] is None or p["ss_h_des"] is None:
            continue
        delta = p["ss_h_des"] - p["ss_h"]
        color = "firebrick" if delta > 0 else "steelblue"
        size = size_from_clusters(p["clusters"])
        ax_sc2.scatter(
            p["ss_h"],
            p["ss_h_des"],
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            zorder=3,
        )
        ax_sc2.annotate(
            p["label"],
            (p["ss_h"], p["ss_h_des"]),
            fontsize=6.5,
            xytext=(4, 3),
            textcoords="offset points",
            alpha=0.85,
            zorder=4,
        )
        drawn += 1

    if drawn:
        ax_sc2.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, zorder=2)
        ax_sc2.text(
            0.55,
            0.50,
            "y = x  (filter doesn't shift %H)",
            transform=ax_sc2.transData,
            fontsize=7.5,
            rotation=45,
            ha="left",
            va="bottom",
            style="italic",
            color="dimgray",
        )

    ax_sc2.axhline(NATURAL_PDB_H, color="gray", linestyle=":", linewidth=0.6)
    ax_sc2.axvline(NATURAL_PDB_H, color="gray", linestyle=":", linewidth=0.6)
    ax_sc2.text(
        NATURAL_PDB_H + 0.005,
        0.02,
        f"natural %H ≈ {NATURAL_PDB_H:.2f}",
        fontsize=7,
        va="bottom",
        color="gray",
    )

    ax_sc2.set_xlabel("SS %H (overall sample)")
    ax_sc2.set_ylabel("SS %H (designable subset)")
    ax_sc2.set_title(
        "Helix-bias of designability filter  —  red: filter selects helix; blue: filter selects sheet"
    )
    ax_sc2.grid(True, alpha=0.3)
    ax_sc2.xaxis.set_major_formatter(FuncFormatter(_humanize))
    ax_sc2.yaxis.set_major_formatter(FuncFormatter(_humanize))
    ax_sc2.set_xlim(0, 1)
    ax_sc2.set_ylim(0, 1)

    # ---- Bottom: paired SS bars ----
    bp = [p for p in pts if p["ss_h"] is not None and p["ss_e"] is not None]

    def delta_h_key(p):
        if p["ss_h_des"] is None:
            return -1.0
        return p["ss_h_des"] - p["ss_h"]

    bp.sort(key=delta_h_key)

    bar_height = 0.40
    pair_gap = 0.04
    row_gap = 0.55
    pair_total = 2 * bar_height + pair_gap

    y_overall: list[float] = []
    y_des: list[float] = []
    labels: list[str] = []
    for i, p in enumerate(bp):
        base = i * (pair_total + row_gap)
        y_overall.append(base)
        y_des.append(base + bar_height + pair_gap)
        labels.append(p["label"])

    h_o = [p["ss_h"] for p in bp]
    e_o = [p["ss_e"] for p in bp]
    c_o = [max(0.0, 1.0 - h - e) for h, e in zip(h_o, e_o)]

    ax_bar.barh(
        y_overall,
        h_o,
        height=bar_height,
        color="#d62728",
        alpha=0.40,
        edgecolor="black",
        linewidth=0.4,
    )
    ax_bar.barh(
        y_overall,
        e_o,
        left=h_o,
        height=bar_height,
        color="#1f77b4",
        alpha=0.40,
        edgecolor="black",
        linewidth=0.4,
    )
    ax_bar.barh(
        y_overall,
        c_o,
        left=[h + e for h, e in zip(h_o, e_o)],
        height=bar_height,
        color="#bcbcbc",
        alpha=0.40,
        edgecolor="black",
        linewidth=0.4,
    )

    for i, p in enumerate(bp):
        if p["ss_h_des"] is None or p["ss_e_des"] is None:
            ax_bar.text(
                0.02,
                y_des[i],
                "(no designable SS data)",
                fontsize=6.5,
                va="center",
                color="dimgray",
                style="italic",
            )
            continue
        h = p["ss_h_des"]
        e = p["ss_e_des"]
        c = max(0.0, 1.0 - h - e)
        ax_bar.barh(
            y_des[i],
            h,
            height=bar_height,
            color="#d62728",
            alpha=0.95,
            edgecolor="black",
            linewidth=0.4,
        )
        ax_bar.barh(
            y_des[i],
            e,
            left=h,
            height=bar_height,
            color="#1f77b4",
            alpha=0.95,
            edgecolor="black",
            linewidth=0.4,
        )
        ax_bar.barh(
            y_des[i],
            c,
            left=h + e,
            height=bar_height,
            color="#bcbcbc",
            alpha=0.95,
            edgecolor="black",
            linewidth=0.4,
        )

    y_label = [(yo + yd) / 2 for yo, yd in zip(y_overall, y_des)]
    ax_bar.set_yticks(y_label)
    ax_bar.set_yticklabels(labels, fontsize=6.5)

    for yo, yd in zip(y_overall, y_des):
        ax_bar.text(
            -0.012, yo, "ovr", fontsize=5.5, ha="right", va="center", color="dimgray"
        )
        ax_bar.text(
            -0.012,
            yd,
            "des",
            fontsize=5.5,
            ha="right",
            va="center",
            color="black",
            weight="bold",
        )

    ax_bar.set_xlim(-0.04, 1.0)
    ax_bar.set_xlabel("Secondary-structure fraction")
    ax_bar.xaxis.set_major_formatter(FuncFormatter(_humanize))
    ax_bar.yaxis.set_major_formatter(FuncFormatter(_humanize))
    ax_bar.set_title(
        "SS distribution per run: overall (faded) vs designable (solid). "
        "Sorted ascending by Δ%H = des − overall  (top = most helix-biased by filter)"
    )
    ax_bar.axvline(NATURAL_PDB_H, color="black", linestyle="--", linewidth=0.9)
    if y_des:
        ax_bar.text(
            NATURAL_PDB_H + 0.005,
            max(y_des) + bar_height,
            f"natural %H ≈ {NATURAL_PDB_H:.2f}",
            fontsize=8,
            va="top",
            color="black",
        )

    legend_elements = [
        Patch(facecolor="#d62728", edgecolor="black", label="α-helix"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="β-sheet"),
        Patch(facecolor="#bcbcbc", edgecolor="black", label="loop/coil"),
        Patch(
            facecolor="white",
            edgecolor="black",
            label="ovr = overall (faded), des = designable (solid)",
        ),
    ]
    ax_bar.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=2)

    fig.suptitle(
        f"n={n} — Mode-collapse diagnostic (with designable-subset SS)",
        fontsize=14,
        y=0.998,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({len(pts)} runs, {len(bp)} with overall SS data)")


def main():
    for n, (tsv, out) in TARGETS.items():
        plot_one(n, tsv, out)


if __name__ == "__main__":
    main()
