"""Pretrained Proteina (NVIDIA NGC v1.3 DFS 60M) metrics for plot overlays.

The pretrained checkpoint is NOT step-comparable to our n256_convergence runs
(it was trained ~1.3M steps with a different schedule, data mix, and possibly
n_layers — see project_proteina_60m_layer_mismatch.md). Use these numbers as
an *eyeball* external baseline, not as a like-for-like comparison.

Numbers are loaded from the paper-sweep result files where the pretrained
model was evaluated under the same eval manifest (eval_v1):
  - Generation:   evaluation/proteina/generation/results/paper/n256_paper_pretrained/
  - CATH probes:  evaluation/proteina/representation/results/paper/n256_paper_cath{,_afdb}/
  - IF + dih:     evaluation/proteina/representation/results/paper/n256_paper_struct{,_afdb}/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_PRETRAINED = (
    REPO_ROOT
    / "evaluation/proteina/generation/results/paper/n256_paper_pretrained/sweep_results.jsonl"
)
REP_PAPER = REPO_ROOT / "evaluation/proteina/representation/results/paper"

PRETRAINED_RUN = "pretrained_dfs_60m"
PRETRAINED_LABEL = "Proteina pretrained (60M, ~1.3M steps)"
PRETRAINED_COLOR = "gold"
PRETRAINED_MARKER = "*"


def _best(
    csv_path: Path, probe: str, col: str, level: Optional[str], direction: str
) -> Optional[float]:
    """Best-layer reduction of the pretrained row's `col` value."""
    if not csv_path.exists():
        return None
    best: Optional[float] = None
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("run") != PRETRAINED_RUN:
                continue
            if r.get("probe_kind") != probe:
                continue
            if level is not None and r.get("cath_level") != level:
                continue
            try:
                v = float(r[col])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                best is None
                or (direction == "max" and v > best)
                or (direction == "min" and v < best)
            ):
                best = v
    return best


def load_gen() -> Dict[str, Optional[float]]:
    """Pretrained generation metrics from the n256_paper_pretrained sweep."""
    if not GEN_PRETRAINED.exists():
        return {
            "_res_PDB_FID": None,
            "_res_AFDB_FID": None,
            "_res_designability_rate": None,
        }
    for line in GEN_PRETRAINED.open():
        r = json.loads(line)
        if r.get("error") and r["error"] != "NONE":
            continue
        return {
            "_res_PDB_FID": r.get("_res_PDB_FID"),
            "_res_AFDB_FID": r.get("_res_AFDB_FID"),
            "_res_designability_rate": r.get("_res_designability_rate"),
        }
    return {
        "_res_PDB_FID": None,
        "_res_AFDB_FID": None,
        "_res_designability_rate": None,
    }


def load_rep(eval_set: str = "PDB") -> Dict[str, Optional[float]]:
    """Pretrained representation probes, eval set in {"PDB", "AFDB"}.

    Returns best-layer reduction for each metric (max for accuracies, min for
    dihedral MAE). Returns None for any metric that wasn't evaluated.
    """
    if eval_set == "PDB":
        cath_csv = REP_PAPER / "n256_paper_cath/pretrained_sweep_results.csv"
        struct_csv = REP_PAPER / "n256_paper_struct/pretrained_sweep_results.csv"
    elif eval_set == "AFDB":
        cath_csv = REP_PAPER / "n256_paper_cath_afdb/pretrained_sweep_results.csv"
        struct_csv = REP_PAPER / "n256_paper_struct_afdb/pretrained_sweep_results.csv"
    else:
        raise ValueError(f"eval_set must be 'PDB' or 'AFDB', got {eval_set!r}")
    return {
        "cath_C_top1": _best(cath_csv, "cath", "cath_accuracy", "C", "max"),
        "cath_A_top1": _best(cath_csv, "cath", "cath_accuracy", "A", "max"),
        "cath_T_top1": _best(cath_csv, "cath", "cath_accuracy", "T", "max"),
        "if_top1_acc": _best(struct_csv, "inverse_folding", "if_top1_acc", None, "max"),
        "dih_mae_total_deg": _best(
            struct_csv, "dihedral", "dih_mae_total_deg", None, "min"
        ),
    }
