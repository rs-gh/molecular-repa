"""Shared plot/table helpers for representation-eval plotting scripts.

Top-level `utils/` (sibling of `lib/`) houses *figure-layer* helpers — colour
palettes, sentinel layer codes, CSV loaders shared across plot scripts under
`scripts/{lite,convergence,paper}/`.

`lib/` remains the home for backbone-and-probe infrastructure (extract,
manifest, probes). Do not mix the two: `lib/` should never import from
`utils/`, and `utils/` should never import from `lib/`.
"""

from utils.plot_helpers import (
    BASELINE_SENTINELS,
    FAMILY_COLORS,
    REF_COLORS,
    REF_SENTINELS,
    RUN_ALIGNED_LAYER,
    UNTRAINED_LAYER_MAX,
    UNTRAINED_LAYER_MIN,
)

__all__ = [
    "BASELINE_SENTINELS",
    "FAMILY_COLORS",
    "REF_COLORS",
    "REF_SENTINELS",
    "RUN_ALIGNED_LAYER",
    "UNTRAINED_LAYER_MAX",
    "UNTRAINED_LAYER_MIN",
]
