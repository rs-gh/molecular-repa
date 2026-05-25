"""Shared helpers for multi-seed (rep_idx) convergence plots.

Companion to the single-seed `plot_convergence_{fid,des}{,_n128}.py`
scripts. Those dedup on (run, step) and silently keep one rep — these
preserve all reps from `sweep_results.clean.jsonl` and emit mean lines
with min/max envelope bands across reps. Legacy single-seed rows
(rep_idx=None) are kept as singleton "reps" so they appear as bare
markers without a shaded band.

PDB has full 3-rep coverage (rep_idx 0/1/2 → seeds 42/1042/2042) on
both n=128 and n=256. AFDB n=128 has 3-rep coverage only on the ext-
sweep ckpts (2026-05-23): baseline 800/900/1100/1200k and mpnn-L4
800/900/1100k. AFDB n=256 has 3-rep coverage on baseline, GearNet L4,
and MPNN L9 across the full convergence sweep (added 2026-05-24);
GearNet L9 and MPNN L4 on AFDB n=256 remain seed 42 only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


# Rate metrics stored in JSONL/TSV as 0–1 fractions; multiplied by 100 only at
# plot/display time so units read as percent without rewriting source data.
RATE_KEYS = {
    "_res_designability_rate",
    "_res_novelty_foldseek_pdb_rate",
    "_res_novelty_foldseek_afdb_swissprot_rate",
}


def _rate_to_pct(values, key):
    return [v * 100.0 for v in values] if key in RATE_KEYS else values


def humanize(v, _pos=None):
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:g}M"
    if av >= 1e3:
        return f"{v / 1e3:g}K"
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


def style_axes(ax, log_y: bool = False) -> None:
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.3, which="both" if log_y else "major")
    # Fixed ticks at the canonical sweep step grid — using LogLocator subs
    # would also place 1.3/1.6 ticks inside each lower decade (e.g. 130K/160K).
    ax.xaxis.set_major_locator(
        FixedLocator(
            [100_000, 200_000, 400_000, 700_000, 1_000_000, 1_300_000, 1_600_000]
        )
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(humanize))
    ax.yaxis.set_major_formatter(FuncFormatter(humanize))
    for lbl in ax.get_xticklabels(which="both"):
        lbl.set_rotation(0)
        lbl.set_ha("center")
        lbl.set_fontsize(7)


def load_jsonl_with_reps(path: Path) -> List[Dict]:
    """Load a sweep_results.clean.jsonl. Includes single-seed legacy rows
    (rep_idx=None) — extract_bands treats them as singleton reps so they
    plot as bare markers without a shaded band."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def _metric_fn(metric) -> Callable[[Dict], object]:
    if isinstance(metric, tuple):
        _, fn = metric
        return fn
    return lambda r, _m=metric: r.get(_m)


def extract_bands(
    rows: List[Dict], run_prefix: str, metric
) -> Tuple[List[int], List[float], List[float], List[float], List[int]]:
    """Per training step, aggregate reps with mean / min / max.

    Returns (steps, means, mins, maxs, n_reps_per_step) sorted by step.
    Steps with zero usable reps are skipped.
    """
    fn = _metric_fn(metric)
    by_step: Dict[int, List[float]] = {}
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        # ensure it's a step-suffixed label of the family (not a sibling like
        # `<prefix>_per_sample`); mirrors the single-seed extractor.
        if "_step" not in run.split(run_prefix)[-1] and run != run_prefix:
            continue
        if r.get("error", "NONE") != "NONE":
            continue
        v = fn(r)
        s = r.get("step")
        if v is None or s is None:
            continue
        by_step.setdefault(int(s), []).append(float(v))

    steps = sorted(by_step)
    means, mins, maxs, ns = [], [], [], []
    for s in steps:
        vals = by_step[s]
        means.append(sum(vals) / len(vals))
        mins.append(min(vals))
        maxs.append(max(vals))
        ns.append(len(vals))
    return steps, means, mins, maxs, ns


def plot_band(ax, xs, means, mins, maxs, color, ls, marker, label):
    """Mean line + min/max envelope.

    Visual policy:
      * Multi-rep, ≥3 points: faded line + min/max fill (default).
      * Single-rep, ≥3 points: opaque line, no fill (line carries the trend).
      * <3 points (any rep count): markers-only with a label, no line —
        connecting two isolated checkpoints with a line implies a trend that
        doesn't exist. Markers are enlarged so they read clearly.
    """
    if not xs:
        return
    has_variance = any(mn != mx for mn, mx in zip(mins, maxs))
    too_few_for_trend = len(xs) < 3

    if too_few_for_trend:
        # Honest sparse rendering: markers only, no connecting line.
        ax.plot(
            xs,
            means,
            marker=marker,
            markersize=9,
            markeredgewidth=1.0,
            markeredgecolor="white",
            linestyle="none",
            color=color,
            label=label,
        )
        if has_variance:
            ax.fill_between(xs, mins, maxs, color=color, alpha=0.18, linewidth=0)
        return

    # Enough points to suggest a trend.
    line_alpha = 0.45 if has_variance else 0.85
    line_width = 1.4 if has_variance else 1.7
    ax.plot(
        xs,
        means,
        linewidth=line_width,
        color=color,
        linestyle=ls,
        alpha=line_alpha,
        marker="none",
    )
    ax.fill_between(xs, mins, maxs, color=color, alpha=0.18, linewidth=0)
    ax.plot(
        xs,
        means,
        marker=marker,
        markersize=6,
        markeredgewidth=0.8,
        markeredgecolor="white",
        linestyle="none",
        color=color,
        label=label,
    )


# Convenience re-export so the per-plot scripts don't need pyplot directly.
__all__ = [
    "plt",
    "humanize",
    "style_axes",
    "load_jsonl_with_reps",
    "extract_bands",
    "plot_band",
    "RATE_KEYS",
    "_rate_to_pct",
]
