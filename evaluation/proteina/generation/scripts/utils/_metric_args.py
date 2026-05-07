"""Shared argparse definitions for the proteina generation eval suite.

Both ``evaluate.py`` (per-checkpoint) and ``run_sweep.py`` (orchestrator) need
the same 16 metric-knob flags. Keeping the definitions here keeps them in
sync; previously they were declared twice and threaded between scripts via
``sys.argv`` reconstruction.

All flags here default to ``None`` so callers can distinguish "user did not
specify" from a real value. ``evaluate.main()`` resolves None to its
standalone defaults at runtime; ``run_sweep.py`` resolves None against the
sweep-profile defaults from ``sweep_config.yaml``. The canonical default
table lives in ``EVALUATE_DEFAULTS`` below.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict


# Prefix for metric columns surfaced to the sweep CSV/JSONL. Single source of
# truth so evaluate.py (writer) and run_sweep.py (consumer) agree.
METRIC_PREFIX = "_res_"


EVALUATE_DEFAULTS: Dict[str, Any] = {
    "seed": None,  # cfg.seed if None
    # TODO: cleanup. designability_subset_per_length is dual-semantic — see
    # the help text on the CLI flag below. Once all profiles set
    # designability_lengths, drop the global-cap fallback and require it.
    "designability_subset_per_length": 50,
    "designability_lengths": None,  # comma-sep ints; None = all lengths
    "cath_subset": 100,
    "cath_head_path": None,
    "centroid_path": None,
    "centroid_filter_designable": True,  # paper App. F protocol
    "foldseek_target_dbs": None,
    "foldseek_alignment_type": 2,
    "foldseek_max_seqs": 1000,
    "foldseek_sensitivity": 9.5,
    "foldseek_threads": 0,
    "foldseek_filter_designable": True,
    "skip_fid": False,
    "fast_inference": True,
    "metrics": None,
}


def add_metric_args(parser: argparse.ArgumentParser) -> None:
    """Register the metric-knob flags shared by evaluate.py and run_sweep.py.

    All defaults are None so the calling script can detect "unset" and apply
    its own resolution (profile defaults for run_sweep, EVALUATE_DEFAULTS for
    evaluate standalone).
    """
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--designability_subset_per_length",
        type=int,
        default=None,
        help="Per-length cap on designability PDBs. When --designability_lengths "
        "is unset, this falls back to a GLOBAL cap on the whole pool — a "
        "dual-semantic kept for legacy profiles. TODO: remove the fallback "
        "once every profile sets --designability_lengths explicitly. 0 = skip.",
    )
    parser.add_argument(
        "--designability_lengths",
        type=str,
        default=None,
        help="Comma-separated lengths (e.g. '50,100,150,200,250') restricting "
        "the designability and diversity metrics to those length bins. Subsamples "
        "--designability_subset_per_length PDBs *per length*. Unset = use all "
        "generated lengths (legacy behaviour, see TODO above).",
    )
    parser.add_argument(
        "--cath_subset",
        type=int,
        default=None,
        help="PDBs to score with the CATH head (0=skip).",
    )
    parser.add_argument(
        "--cath_head_path",
        type=str,
        default=None,
        help="Path to .pkl from build_cath_classifier.py. Required for CATH metrics.",
    )
    parser.add_argument(
        "--centroid_path",
        type=str,
        default=None,
        help="Path to training-set centroid .pt from precompute_centroids.py.",
    )
    parser.add_argument(
        "--centroid_filter_designable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Filter centroid-novelty queries to the designable subset "
        "(paper App. F protocol). Default ON.",
    )
    parser.add_argument(
        "--foldseek_target_dbs",
        type=str,
        default=None,
        help="Comma-separated 'label=path' pairs for Foldseek novelty target DBs.",
    )
    parser.add_argument("--foldseek_alignment_type", type=int, default=None)
    parser.add_argument("--foldseek_max_seqs", type=int, default=None)
    parser.add_argument("--foldseek_sensitivity", type=float, default=None)
    parser.add_argument("--foldseek_threads", type=int, default=None)
    parser.add_argument(
        "--foldseek_filter_designable",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        default=None,
        help="Skip GearNet FID/fJSD/fS metrics.",
    )
    parser.add_argument(
        "--fast_inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="torch.compile + bf16 autocast for generation. Default ON.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated subset of metrics to run. Names: fid, "
        "designability, diversity, novelty_centroid, novelty_foldseek, cath. "
        "Unset = run everything per per-knob settings.",
    )


def resolve_evaluate_defaults(args: argparse.Namespace) -> None:
    """Fill None metric-knob fields on ``args`` with EVALUATE_DEFAULTS.

    Mutates in place. Used by ``evaluate.main()`` so standalone invocations
    of evaluate.py see the same defaults they used to have when each flag
    had its own argparse default.
    """
    for k, v in EVALUATE_DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
