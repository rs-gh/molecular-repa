"""Shared plot/table helpers for representation-eval plotting scripts.

Sources for the constants below:
- `FAMILY_COLORS` / `REF_COLORS`: matched across plot_convergence.py and
  plot_per_n.py so curves keep the same colour from one figure to the next.
- `BASELINE_SENTINELS` / `REF_SENTINELS`: the negative-`layer` codes that
  `scripts/lite/run_sweep.py` writes to JSONL for frozen-encoder reference
  rows (gearnet, distance-only, etc.). These are *the* canonical mapping —
  any new plot script should import from here rather than redefining.
- `RUN_ALIGNED_LAYER`: convention used by plot_convergence.py to pick a
  "comparison layer" for step-progression curves (each REPA variant probes
  at the layer it was aligned to during training).
- `UNTRAINED_LAYER_{MIN,MAX}`: sentinel-layer range for the
  `untrained_proteina` rep source — probe row for trunk layer L is written
  with `layer = -(L + 10)`, so values land in [-19, -10] for a 10-layer trunk.

Importing modules should reach for these via the package alias::

    from utils.plot_helpers import FAMILY_COLORS, BASELINE_SENTINELS

Adding new figure code: extend this file rather than duplicating constants
into your script. Cross-script style consistency is the point.
"""

from __future__ import annotations


# Per-run family colours shared across all layer-wise plots.
FAMILY_COLORS: dict[str, str] = {
    # 10L / 12L proteina trunk runs
    "baseline": "#1f77b4",  # blue
    "repa_l0": "#ff7f0e",  # orange
    "repa_l4": "#d62728",  # red
    "repa_l9": "#2ca02c",  # green
    # ESM-target variants (lighter shades, same hue family as their non-ESM peer)
    "esm_repa_l0": "#ffbb78",
    "esm_repa_l4": "#ff9896",
    "esm_repa_l9": "#98df8a",
    # Untrained-proteina (random initialised trunk)
    "untrained_proteina": "#e377c2",
    # NGC reference ceiling
    "pretrained_dfs_60m": "#9467bd",
}

# Reference-source colours (analytic + frozen-encoder baselines).
REF_COLORS: dict[str, str] = {
    "gearnet": "black",
    "seq_onehot": "#8c564b",
    "random_gauss": "#7f7f7f",
    "random_rank": "#bcbd22",
    "distance_only": "#17becf",
}

# Negative-layer sentinels that the in-place-sweep harness uses to encode
# frozen-encoder / analytic baseline rows.
BASELINE_SENTINELS: dict[int, str] = {
    -1: "gearnet",
    -2: "random_gauss",
    -3: "seq_onehot",
    -100: "random_rank",
    -101: "distance_only",
}
# Alias maintained for backwards-compat (plot_per_n.py uses this name).
REF_SENTINELS = BASELINE_SENTINELS

# Per-run alignment layer used by REPA training; plot_convergence.py reads this
# to draw "comparison layer" markers on layer-wise figures.
RUN_ALIGNED_LAYER: dict[str, int | None] = {
    "baseline": None,
    "repa_l0": 0,
    "repa_l4": 4,
    "repa_l9": 9,
}

# Untrained-proteina sentinel range: probe row for trunk layer L has
# layer = -(L + 10). Inverse: trunk_layer = -L - 10 for L in [-19, -10].
UNTRAINED_LAYER_MIN: int = -19
UNTRAINED_LAYER_MAX: int = -10
