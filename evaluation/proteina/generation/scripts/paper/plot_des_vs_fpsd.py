"""Scatter Designability(%) vs FPSD (PDB / AFDB) for n=128 and n=256.

Reads the per-n TSVs produced by ``jsonl_to_tsv.py`` (schema: ``profile``,
``run``, ``step``, ``ckpt_path``, ``_res_*`` metric columns), overlays paper
Table 1 baselines, and writes a two-panel figure per n.

Legend lineage rule: every point label includes the checkpoint step (e.g.
``baseline_128_bs24 @ 200k``) so the figure is traceable to a specific
checkpoint. Step is read from the TSV's ``step`` column, not parsed from the
run id (the ``_steplast`` suffix is ambiguous across suites — see
``plot_labels.pretty_run_label``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.proteina.lib.plot_labels import (
    block_label_plan,
    compose_legend_label,
)

ROOT = Path(__file__).resolve().parents[2]  # .../proteina/generation
FIG_ROOT = ROOT / "figures/paper"

# Paper Table 1, unconditional generation block, n=256 lengths 50-275.
# (model, designability_%, FPSD_PDB, FPSD_AFDB)
PAPER_BASELINES = [
    ("FrameDiff", 65.4, 194.2, 258.1),
    ("FoldFlow (base)", 96.6, 601.5, 566.2),
    ("FoldFlow (stoc.)", 97.0, 543.6, 520.4),
    ("FoldFlow (OT)", 97.2, 431.4, 414.1),
    ("FrameFlow", 88.6, 129.9, 159.9),
    ("ESM3", 22.0, 933.9, 855.4),
    ("Chroma", 74.8, 189.0, 184.1),
    ("RFDiffusion", 94.4, 253.7, 252.4),
    ("Proteus", 94.2, 225.7, 226.2),
    ("Genie2", 95.2, 350.0, 313.8),
    ("Proteína M_FS γ=0.35", 98.2, 411.2, 392.1),
    ("Proteína M_FS γ=0.45", 96.4, 388.0, 368.2),
    ("Proteína M_FS γ=0.5", 91.4, 380.1, 359.8),
    ("Proteína M_FS^no-tri γ=0.45", 93.8, 322.2, 306.2),
    ("Proteína M_21M γ=0.3", 99.0, 280.7, 319.9),
    ("Proteína M_21M γ=0.6", 84.6, 280.7, 301.8),
    ("Proteína M_LoRA γ=0.5", 96.6, 274.1, 336.0),
]

# Map of n → (tsv_path, output_path).
TARGETS = {
    128: (
        FIG_ROOT / "n128_paper/n128_paper_tables.tsv",
        FIG_ROOT / "n128_paper/n128_des_vs_fpsd.png",
    ),
    256: (
        FIG_ROOT / "n256_paper/n256_paper_tables.tsv",
        FIG_ROOT / "n256_paper/n256_des_vs_fpsd.png",
    ),
}

DES_COL = "_res_designability_rate"
PDB_FID_COL = "_res_PDB_FID"
AFDB_FID_COL = "_res_AFDB_FID"


def parse_tsv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def to_float(x: str):
    if x is None:
        return None
    x = x.strip()
    if x in ("", "—", "-"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def short_profile(label: str) -> str:
    """``"n128_paper_bs_lr" -> "bs_lr"`` for shorter legend entries."""
    for prefix in ("n128_paper_", "n256_paper_", "n512_paper_"):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def collect(rows: list[dict]) -> list[dict]:
    all_runs = [row["run"] for row in rows]
    _, varying = block_label_plan(all_runs)
    pts = []
    for row in rows:
        des_frac = to_float(row.get(DES_COL, ""))
        if des_frac is None:
            continue
        pdb = to_float(row.get(PDB_FID_COL, ""))
        afdb = to_float(row.get(AFDB_FID_COL, ""))
        if pdb is None and afdb is None:
            continue
        step_str = row.get("step", "").strip()
        try:
            step = int(float(step_str)) if step_str else None
        except ValueError:
            step = None
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
                "block": short_profile(row.get("profile", "(unlabelled)")),
                "des_pct": des_frac * 100.0,
                "pdb": pdb,
                "afdb": afdb,
            }
        )
    return pts


def plot_one(n: int, tsv: Path, out: Path):
    rows = parse_tsv(tsv)
    pts = collect(rows)
    if not pts:
        raise RuntimeError(f"no usable rows in {tsv}")

    blocks = sorted({p["block"] for p in pts})
    cmap = plt.get_cmap("tab10")
    colors = {b: cmap(i % 10) for i, b in enumerate(blocks)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    panels = [
        (axes[0], "pdb", 2, "FPSD vs PDB ↓"),
        (axes[1], "afdb", 3, "FPSD vs AFDB ↓"),
    ]
    for ax, key, paper_idx, title in panels:
        for b in blocks:
            xs = [p[key] for p in pts if p["block"] == b and p[key] is not None]
            ys = [p["des_pct"] for p in pts if p["block"] == b and p[key] is not None]
            if not xs:
                continue
            ax.scatter(
                xs,
                ys,
                s=80,
                color=colors[b],
                label=b,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
            )
        for p in pts:
            if p[key] is None:
                continue
            ax.annotate(
                p["label"],
                (p[key], p["des_pct"]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
                alpha=0.8,
                zorder=4,
            )

        bx = [b[paper_idx] for b in PAPER_BASELINES]
        by = [b[1] for b in PAPER_BASELINES]
        ax.scatter(
            bx,
            by,
            s=70,
            marker="x",
            color="dimgray",
            linewidths=1.5,
            label="Paper Table 1 (n=256)",
            zorder=2,
        )
        for b in PAPER_BASELINES:
            ax.annotate(
                b[0],
                (b[paper_idx], b[1]),
                fontsize=6.5,
                xytext=(4, -8),
                textcoords="offset points",
                color="dimgray",
                zorder=2,
            )

        ax.set_xlabel(title + " ↓")
        ax.invert_xaxis()  # right = better (FPSD/fJSD natural direction is lower)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Designability (%) ↑")
    fig.suptitle(f"n={n} — Designability vs FPSD (with paper Table 1 baselines)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({len(pts)} runs + {len(PAPER_BASELINES)} paper baselines)")


def main():
    for n, (tsv, out) in TARGETS.items():
        plot_one(n, tsv, out)


if __name__ == "__main__":
    main()
