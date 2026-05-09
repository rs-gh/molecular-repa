"""Scatter Designability(%) vs FPSD (PDB / AFDB) for n=128 and n=256.

Reads the per-n TSVs produced by md_tables_to_tsv.py, overlays paper Table 1
baselines (FrameDiff, FoldFlow*, FrameFlow, ESM3, Chroma, RFDiffusion, Proteus,
Genie2, Proteína M_FS / M_21M / M_LoRA at the listed γ values), and writes a
two-panel figure per n.

Notes:
- Our TSV labels the FPSD column "PDB FID" / "AFDB FID"; the paper calls it
  FPSD. The reference-range row in the n=256 table (129.9–933.9 PDB,
  159.9–855.4 AFDB) matches FrameFlow's FPSD column from Table 1, confirming
  these are the same number under different names.
- Our `Des` is a fraction in [0, 1]; the paper reports %. We multiply ours
  by 100 so paper baselines and our runs share the y-axis.
- Designability column is `Des ↑` in n=256 and `Designability (↑)` in n=128.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

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

# Map of n → (tsv_path, output_path, designability_column).
TARGETS = {
    128: (
        FIG_ROOT / "n128_paper/n128_paper_tables.tsv",
        FIG_ROOT / "n128_paper/n128_des_vs_fpsd.png",
        "Designability (↑)",
    ),
    256: (
        FIG_ROOT / "n256_paper/n256_paper_tables.tsv",
        FIG_ROOT / "n256_paper/n256_des_vs_fpsd.png",
        "Des ↑",
    ),
}

# Column names for FPSD differ slightly across the two TSVs.
PDB_FID_KEYS = ("PDB FID ↓", "PDB FID (↓)")
AFDB_FID_KEYS = ("AFDB FID ↓", "AFDB FID (↓)")


def parse_tsv(path: Path):
    with path.open() as fh:
        reader = list(csv.reader(fh, delimiter="\t"))
    header = reader[0]
    ncol = len(header)
    rows: list[dict] = []
    block = "(unlabelled)"
    for r in reader[1:]:
        if not r or all(c == "" for c in r):
            continue
        nonempty = [c for c in r if c != ""]
        if len(nonempty) == 1:
            block = nonempty[0]
            continue
        r = (r + [""] * ncol)[:ncol]
        d = dict(zip(header, r))
        d["_block"] = block
        rows.append(d)
    return rows


def to_float(x: str):
    x = x.strip()
    if x in ("", "—", "-"):
        return None
    x = x.replace(",", "")
    for ch in "†‡*":
        x = x.replace(ch, "")
    try:
        return float(x)
    except ValueError:
        return None


def first_present(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in row:
            return row[k]
    return ""


def short_block(label: str) -> str:
    """Trim block titles to the leading phrase for legend readability."""
    head = label.split("—")[0].strip()
    return head or label[:40]


def collect(rows, des_col: str):
    pts = []
    for row in rows:
        des_frac = to_float(row.get(des_col, ""))
        if des_frac is None:
            continue
        pdb = to_float(first_present(row, PDB_FID_KEYS))
        afdb = to_float(first_present(row, AFDB_FID_KEYS))
        if pdb is None and afdb is None:
            continue
        pts.append(
            {
                "label": row["Run"],
                "block": short_block(row["_block"]),
                "des_pct": des_frac * 100.0,
                "pdb": pdb,
                "afdb": afdb,
            }
        )
    return pts


def plot_one(n: int, tsv: Path, out: Path, des_col: str):
    rows = parse_tsv(tsv)
    pts = collect(rows, des_col)
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
        # Our runs.
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

        # Paper Table 1 baselines (n=256 settings; gray ×, labelled).
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

        ax.set_xlabel(title)
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
    for n, (tsv, out, des_col) in TARGETS.items():
        plot_one(n, tsv, out, des_col)


if __name__ == "__main__":
    main()
