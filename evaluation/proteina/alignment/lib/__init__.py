"""Alignment-analysis library — CKNNA computation + helpers.

Kept deliberately small: this package is read-only against the proteina
critical path. All shared utilities (batch loading, feature extraction)
are imported from the existing ``evaluation.proteina.representation`` package
or ``evaluation.proteina.lib`` rather than reimplemented here.
"""

from lib.cknna import cknna, cknna_bootstrap

__all__ = ["cknna", "cknna_bootstrap"]
