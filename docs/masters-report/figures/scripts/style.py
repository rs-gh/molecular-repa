"""Common styling for master's-report figures.

Conventions (locked 2026-05-28):
  Color:    baseline = blue,  L9 = green,  L4 = red,  random = grey
  Marker:   GearNet  = circle (o),  MPNN = triangle (^),  baseline = square (s),
            random = x
  Bands:    shaded +/- 1 SD across seeds where >=2 seeds exist
"""

import re


def use_report_style():
    """Match figure text to the report body font (Computer Modern / Latin Modern).

    The report loads no font package, so LaTeX typesets it in Computer Modern
    Roman. matplotlib ships the matching ``cmr10`` TTF and a ``cm`` mathtext
    fontset, so we can match the look without shelling out to LaTeX (fast, no
    external dependency, can never fail a figure regen).

    Caveats handled here:
      * cmr10 has no proper minus glyph -> ``axes.unicode_minus=False``.
      * cmr10 has no arrow/unicode-symbol glyphs; write arrows as mathtext
        (e.g. ``$\\downarrow$``) in labels, not literal "down" Unicode.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
        }
    )


# Locked palette
COLORS = {
    "baseline": "#1f77b4",  # blue
    "L9": "#2ca02c",  # green
    "L4": "#d62728",  # red
    "random": "#7f7f7f",  # grey
    "ceiling": "#bbbbbb",  # light grey (NVIDIA-60M reference, etc.)
}

MARKERS = {
    "baseline": "s",
    "GearNet": "o",
    "MPNN": "^",
    "random": "x",
}


def classify_family(family):
    """Return (color, marker, label, zorder) from a run/family name."""
    f = family.lower()
    if "baseline" in f:
        return COLORS["baseline"], MARKERS["baseline"], "Baseline", 10
    if "random" in f:
        return COLORS["random"], MARKERS["random"], "REPA L4-random", 5
    is_l9 = bool(re.search(r"(^|_)l9(_|$)", f))
    is_mpnn = "mpnn" in f
    layer = "L9" if is_l9 else "L4"
    enc = "MPNN" if is_mpnn else "GearNet"
    return COLORS[layer], MARKERS[enc], f"REPA {layer}-{enc}", 8


def setup_axes(ax, title="", xlabel="", ylabel="", grid=True):
    if title:
        ax.set_title(title, fontsize=11, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if grid:
        ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=9)


def _humanize_step_k(x, _):
    """Tick formatter for an x-axis expressed in *thousands* of steps.

    100 -> "100k", 1000 -> "1M", 1600 -> "1.6M", 2400 -> "2.4M".
    """
    if x >= 1000:
        return f"{x / 1000:g}M"
    return f"{x:g}k"


def log_step_axis(ax, label="Training step (log scale)"):
    """Apply log scale on x-axis for training-step plots, with humanised ticks.

    x-data is in thousands of steps; ticks span the full sweep range (data
    runs to 2.4M steps) and are humanised to 100k / 1M / 2.4M form. Ticks
    beyond a given figure's data are auto-clipped by the data-driven xlim.
    """
    import matplotlib.ticker as mticker

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_humanize_step_k))
    ax.xaxis.set_major_locator(
        mticker.FixedLocator([100, 200, 400, 700, 1000, 1500, 2400])
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(label, fontsize=10)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.0)


def plot_trajectory(ax, steps_k, means, stds, color, marker, label, zorder=5):
    """Plot a single trajectory with shaded band."""
    line = ax.plot(
        steps_k,
        means,
        "-",
        marker=marker,
        color=color,
        label=label,
        linewidth=1.8,
        markersize=5,
        zorder=zorder,
        alpha=0.95,
    )
    if stds is not None and any(s is not None and s > 0 for s in stds):
        lo = [m - (s or 0) for m, s in zip(means, stds)]
        hi = [m + (s or 0) for m, s in zip(means, stds)]
        ax.fill_between(steps_k, lo, hi, color=color, alpha=0.18, zorder=zorder - 1)
    return line


def legend(ax, loc="best", **kw):
    leg = ax.legend(loc=loc, fontsize=9, frameon=True, framealpha=0.85, **kw)
    leg.get_frame().set_linewidth(0.5)
    return leg
