"""Pretrained Proteina (NVIDIA NGC v1.3 DFS 60M) metrics for plot overlays.

The pretrained checkpoint is NOT step-comparable to our convergence runs
(it was trained ~1.3M steps with a different schedule, data mix, and possibly
n_layers — see project_proteina_60m_layer_mismatch.md). Use these numbers as
an *eyeball* external baseline, not as a like-for-like comparison.

Numbers are loaded from the paper-sweep result files where the pretrained
model was evaluated under the same eval manifest (eval_v1). load_gen() and
load_rep() take an `n` argument ("n256" or "n128") so callers can match the
overlay to the sample size of the plot they're decorating.

Layout (per `n`):
  - Generation:   evaluation/proteina/generation/results/paper/{n}_paper_pretrained/
  - CATH probes:  evaluation/proteina/representation/results/paper/{n}_paper_cath{,_afdb}/
  - IF + dih:     evaluation/proteina/representation/results/paper/{n}_paper_struct{,_afdb}/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_PAPER = REPO_ROOT / "evaluation/proteina/generation/results/paper"
REP_PAPER = REPO_ROOT / "evaluation/proteina/representation/results/paper"

# Per-`n` paths.
_GEN_DIRS = {
    "n256": GEN_PAPER / "n256_paper_pretrained",
    "n128": GEN_PAPER / "n128_paper_pretrained",
}
_REP_DIRS = {
    "n256": {
        "PDB": (REP_PAPER / "n256_paper_cath", REP_PAPER / "n256_paper_struct"),
        "AFDB": (
            REP_PAPER / "n256_paper_cath_afdb",
            REP_PAPER / "n256_paper_struct_afdb",
        ),
    },
    "n128": {
        "PDB": (REP_PAPER / "n128_paper_cath", REP_PAPER / "n128_paper_struct"),
        "AFDB": (
            REP_PAPER / "n128_paper_cath_afdb",
            REP_PAPER / "n128_paper_struct_afdb",
        ),
    },
}

PRETRAINED_RUN_PREFIX = "pretrained_dfs_60m"
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
            if not r.get("run", "").startswith(PRETRAINED_RUN_PREFIX):
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


def load_gen(n: str = "n256") -> Dict[str, Optional[float]]:
    """Pretrained generation metrics from the matching paper-pretrained sweep.

    Returns *all* numeric `_res_*` fields from the row, plus a derived `H/E`
    helix-to-sheet ratio. Callers can therefore use this as a dict lookup
    against any metric column name they plot. Returns an empty dict if the
    sweep file is missing.
    """
    sweep = _GEN_DIRS.get(n)
    if sweep is None:
        return {}
    path = sweep / "sweep_results.jsonl"
    if not path.exists():
        return {}
    row = None
    for line in path.open():
        r = json.loads(line)
        if r.get("error") and r["error"] != "NONE":
            continue
        row = r
        break
    if row is None:
        return {}
    out: Dict[str, Optional[float]] = {}
    for k, v in row.items():
        if not k.startswith("_res_"):
            continue
        if isinstance(v, bool):  # avoid numpy bool truthy
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    # Derived H/E ratio (designable subset), matches the metric definition in
    # plot_convergence_speedup.py.
    h = row.get("_res_ss_frac_H_designable")
    e = row.get("_res_ss_frac_E_designable")
    if isinstance(h, (int, float)) and isinstance(e, (int, float)) and e > 0:
        out["H/E"] = float(h) / float(e)
    return out


def load_rep(eval_set: str = "PDB", n: str = "n256") -> Dict[str, Optional[float]]:
    """Pretrained representation probes, eval set in {"PDB", "AFDB"}.

    Returns best-layer reduction for each metric (max for accuracies, min for
    dihedral MAE). Returns None for any metric that wasn't evaluated.
    """
    dirs_for_n = _REP_DIRS.get(n)
    if dirs_for_n is None:
        return {}
    paths = dirs_for_n.get(eval_set)
    if paths is None:
        raise ValueError(f"eval_set must be 'PDB' or 'AFDB', got {eval_set!r}")
    cath_dir, struct_dir = paths
    cath_csv = cath_dir / "pretrained_sweep_results.csv"
    struct_csv = struct_dir / "pretrained_sweep_results.csv"
    return {
        "cath_C_top1": _best(cath_csv, "cath", "cath_accuracy", "C", "max"),
        "cath_A_top1": _best(cath_csv, "cath", "cath_accuracy", "A", "max"),
        "cath_T_top1": _best(cath_csv, "cath", "cath_accuracy", "T", "max"),
        "if_top1_acc": _best(struct_csv, "inverse_folding", "if_top1_acc", None, "max"),
        "if_macro_f1": _best(struct_csv, "inverse_folding", "if_macro_f1", None, "max"),
        "dih_mae_total_deg": _best(
            struct_csv, "dihedral", "dih_mae_total_deg", None, "min"
        ),
    }
