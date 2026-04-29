"""Per-n layerwise plots for the proteina probe sweep - single 3x3 grid.

Emits two figures (P@L/5 linear, CATH-acc). Each figure is a 3x3 grid:
    rows = model size (n=128, 256, 512_sm)
    cols = probe timestep t in {0.5, 0.75, 1.0}
Each panel shows probe score vs transformer layer, one curve per run family.
Frozen-encoder baselines (gearnet, seq_onehot, random_gauss, distance_only)
are drawn as horizontal reference lines.

Row titles include "samples seen" per run (step x batch_size), analogous to
`fig_grid_sample_matched.png` on the generation side.

Usage:
  python evaluation/proteina/representation/scripts/plot_per_n.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIG_DIR = HERE.parent / "figures"

SIZES = ["n128_val", "n256_val", "n512_val"]
TS = [0.50, 0.75, 1.00]

FAMILY_COLORS = {
    "baseline": "#1f77b4",
    "repa_l0": "#ff7f0e",
    "repa_l4": "#d62728",
    "repa_l9": "#2ca02c",
    "esm_repa_l0": "#ffbb78",
    "esm_repa_l4": "#ff9896",
    "esm_repa_l9": "#98df8a",
    "untrained_proteina": "#e377c2",
}
REF_COLORS = {
    "gearnet": "black",
    "seq_onehot": "#8c564b",
    "random_gauss": "#7f7f7f",
    "distance_only": "#17becf",
}
REF_SENTINELS = {
    -1: "gearnet",
    -2: "random_gauss",
    -3: "seq_onehot",
    -101: "distance_only",
}
UNTRAINED_MIN, UNTRAINED_MAX = -19, -10


# Samples-seen computation per (size, family).
# n=128 gearnet-REPA ran bs=24 until step ~220k, then bs=80 to 400k.
# n=128 esm-REPA ran bs=80 throughout. n=128 baseline ran bs=24 throughout.
# n=256: bs=12 across. n=512_sm: baseline bs=6, REPA bs=4.
def _samples_seen(size_dir: str, family: str, step: int) -> int | None:
    if size_dir == "n128_val":
        if family == "baseline":
            return step * 24
        if family.startswith("esm_repa"):
            return step * 80
        if family.startswith("repa"):  # gearnet REPA: bs switch at 220k
            switch = 220_000
            if step <= switch:
                return step * 24
            return switch * 24 + (step - switch) * 80
    if size_dir == "n256_val":
        return step * 12
    if size_dir == "n512_val":
        if family == "baseline":
            return step * 6
        if family.startswith("repa"):
            return step * 4
    return None


SIZE_LABEL = {"n128_val": "n = 128", "n256_val": "n = 256", "n512_val": "n = 512"}


def _family(run: str) -> str:
    for suffix in ("_128", "_256", "_512_sm", "_512"):
        if run.endswith(suffix):
            return run[: -len(suffix)]
    return run


def _fmt_samples(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{int(n)}"


def _load(size_dir: str) -> pd.DataFrame:
    csv = RESULTS / size_dir / "sweep_results.csv"
    df = pd.read_csv(csv)
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    for c in ["layer", "step", "t", "p_at_L_5_linear", "p_at_L_5_mlp", "cath_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["layer", "t"])
    df["family"] = df["run"].map(_family)
    return df


def _ref_values(df: pd.DataFrame, metric: str) -> dict:
    out = {}
    for sentinel, name in REF_SENTINELS.items():
        sub = df[df["layer"] == sentinel]
        if not sub.empty:
            v = float(sub[metric].mean())
            if v == v:
                out[name] = v
    return out


def _samples_note(df: pd.DataFrame, size_dir: str) -> str:
    """Condensed 'baseline XM / REPA YM samples' note for the row y-label."""

    def _summary(prefix: str) -> str | None:
        fams = [
            f
            for f in FAMILY_COLORS
            if f != "untrained_proteina" and f.startswith(prefix)
        ]
        ns_vals = []
        for fam in fams:
            sub = df[df["family"] == fam]
            if sub.empty:
                continue
            step = int(sub["step"].max())
            ns = _samples_seen(size_dir, fam, step)
            if ns is None:
                continue
            ns_vals.append(ns)
        if not ns_vals:
            return None
        lo, hi = min(ns_vals), max(ns_vals)
        if lo == hi:
            return _fmt_samples(hi)
        return f"{_fmt_samples(lo)}-{_fmt_samples(hi)}"

    parts = []
    b = _summary("baseline")
    r = _summary("repa")
    if b:
        parts.append(f"baseline {b}")
    if r:
        parts.append(f"REPA {r}")
    return " / ".join(parts) + " samples" if parts else ""


def _panel(ax, df: pd.DataFrame, size_dir: str, t: float, metric: str) -> None:
    sub = df[(df["layer"] >= 0) & (df["t"].round(2) == t)].copy()
    for fam, color in FAMILY_COLORS.items():
        if fam == "untrained_proteina":
            continue
        cur = sub[sub["family"] == fam].sort_values("layer")
        if cur.empty:
            continue
        ax.plot(
            cur["layer"],
            cur[metric],
            "-o",
            color=color,
            label=fam,
            linewidth=1.6,
            markersize=3.5,
        )

    untr = df[
        (df["run"] == "untrained_proteina")
        & (df["layer"] >= UNTRAINED_MIN)
        & (df["layer"] <= UNTRAINED_MAX)
        & (df["t"].round(2) == t)
    ].copy()
    if not untr.empty:
        untr["trunk_layer"] = -untr["layer"] - 10
        untr = untr.sort_values("trunk_layer")
        ax.plot(
            untr["trunk_layer"],
            untr[metric],
            "--s",
            color=FAMILY_COLORS["untrained_proteina"],
            label="untrained",
            linewidth=1.2,
            markersize=3,
            alpha=0.8,
        )

    for name, v in _ref_values(df, metric).items():
        ax.axhline(
            v,
            linestyle="--",
            linewidth=0.9,
            color=REF_COLORS[name],
            label=name,
            alpha=0.7,
        )

    ax.grid(True, alpha=0.3)


def _plot_grid(metric: str, ylabel: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=True)
    loaded = {s: _load(s) for s in SIZES}

    for row, size_dir in enumerate(SIZES):
        df = loaded[size_dir]
        for col, t in enumerate(TS):
            ax = axes[row, col]
            if df.empty:
                ax.set_visible(False)
                continue
            _panel(ax, df, size_dir, t, metric)
            if row == 0:
                ax.set_title(f"t = {t:.2f}", fontsize=11, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Transformer layer")
        if not df.empty:
            note = _samples_note(df, size_dir)
            axes[row, 0].set_ylabel(
                f"{SIZE_LABEL[size_dir]}\n{note}\n{ylabel}",
                fontsize=10,
                fontweight="bold",
            )

    # De-duplicated legend
    handles, labels = [], []
    seen = set()
    for ax_row in axes:
        for ax in ax_row:
            if not ax.get_visible():
                continue
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                if lbl in seen:
                    continue
                seen.add(lbl)
                handles.append(h)
                labels.append(lbl)
    fig.legend(
        handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.5)
    )

    fig.suptitle(
        f"{ylabel} - layerwise per timestep, per model size",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.9, 0.98])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    _plot_grid("p_at_L_5_linear", "P@L/5 (linear)", FIG_DIR / "fig_grid_P_at_L5.png")
    _plot_grid("cath_acc", "CATH accuracy", FIG_DIR / "fig_grid_cath_acc.png")


if __name__ == "__main__":
    main()
