"""Markdown ablation table for the n=128 paper-protocol sweeps.

Companion to plot_n128_paper.py: same data, same ablation grouping, but
rendered as a single combined table — the kind you'd drop into a paper
appendix. One row per checkpoint, columns are run / step / training bs /
designability-N / each metric. Best-per-metric within each ablation block
is bolded, matching the gold-edge highlighting on the figure.

Section headers (e.g. "Layer ablation") render as a heading-style row
spanning the first column.

Usage:
    python evaluation/proteina/generation/scripts/tabulate_n128_paper.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent.parent
GENERATION_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))  # for utils._results_io
sys.path.insert(0, str(HERE))  # for plot_n128_paper (sibling)

from utils._results_io import load_sweep_rows  # noqa: E402
from plot_n128_paper import (  # noqa: E402
    ABLATIONS,
    METRICS,
    MIN_DESIGNABILITY_N,
    N_NOTES,
    _scrub_corrupt_designability,
)

RESULTS_ROOT = GENERATION_ROOT / "results" / "paper"
OUT = GENERATION_ROOT / "figures" / "paper" / "n128_paper" / "n128_paper_tables.md"

# Training batch size per run name. Encodes the actual effective bs used
# for the run; not derivable cleanly from the checkpoint, so listed here
# (mirrors what the plot legend communicates implicitly).
# Per-ablation row reordering (overrides ABLATIONS order in the table only —
# plot ordering is left untouched). For bs_lr we pair BL/L4 at matching
# (bs, step) so eye-level comparison is one row apart.
TABLE_ROW_ORDER = {
    "bs_lr": [
        "baseline_128_bs24_step200k",
        "repa_l4_128_bs24_step200k",
        "baseline_128_bs24_step400k",
        "repa_l4_128_bs24_step400k",
        "baseline_128_bs80_step200k",
        "repa_l4_128_bs80_step200k",
        "baseline_128_bs80_lr3x_step200k",
        "repa_l4_128_bs80_lr3x_steplast",
    ],
}


RUN_BS = {
    "baseline_128_bs24_step200k": 24,
    "baseline_128_bs24_step400k": 24,
    "baseline_128_bs80_step200k": 80,
    "baseline_128_bs80_lr3x_step200k": 80,
    "repa_l0_128_bs80_steplast": 80,
    "repa_l4_128_bs24_step200k": 24,
    "repa_l4_128_bs24_step400k": 24,
    "repa_l4_128_bs80_step200k": 80,
    "repa_l4_128_bs80_lr3x_steplast": 80,
    "repa_l9_128_bs80_steplast": 80,
    "repa_l4_128_random_step200k": 80,
    "repa_l4_128_pw_structure_step100k": 80,
    "repa_l4_128_pw_torsional_step100k": 80,
    "repa_mpnn_l4_128_bs80_step200k": 80,
    "repa_esm_l4_128_step200k": 80,
}


def _val(row: dict | None, mkey: str) -> float | None:
    if row is None:
        return None
    v = row.get(mkey)
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _step_label(step: int | None) -> str:
    if step is None:
        return "—"
    return f"{step // 1000}K"


def _load_block(akey: str) -> list[tuple[str, str, dict]]:
    """Return [(run_name, display_label, row)] for an ablation, in config order."""
    cfg = ABLATIONS[akey]
    rows = {
        r["run"]: _scrub_corrupt_designability(r)
        for r in load_sweep_rows(RESULTS_ROOT / cfg["dir"] / "sweep_results.jsonl")
    }
    label_by_run = {rn: lbl for rn, lbl, _c in cfg["runs"]}
    order = TABLE_ROW_ORDER.get(akey) or [rn for rn, _l, _c in cfg["runs"]]
    return [(rn, label_by_run[rn], rows.get(rn, {})) for rn in order]


def _best_indices(block: list[tuple[str, str, dict]]) -> dict[str, int]:
    """For each metric, the index in `block` that holds the best value."""
    best: dict[str, int] = {}
    for mkey, (_mlabel, lower_better) in METRICS.items():
        finite = [(i, _val(r, mkey)) for i, (_rn, _lbl, r) in enumerate(block)]
        finite = [(i, v) for i, v in finite if v is not None]
        if not finite:
            continue
        best[mkey] = (
            min(finite, key=lambda t: t[1])[0]
            if lower_better
            else max(finite, key=lambda t: t[1])[0]
        )
    return best


def render() -> str:
    metric_cols = list(METRICS.items())  # [(mkey, (label, lower_better)), ...]
    headers = ["Run", "Step", "bs", "des N"] + [
        f"{lbl} ({'↓' if low else '↑'})" for _mk, (lbl, low) in metric_cols
    ]
    n_cols = len(headers)

    lines: list[str] = []
    lines.append("# n=128 paper-protocol sweep — ablation table")
    lines.append("")
    lines.append(
        "Companion to `n128_paper_sweep.png`. Pool: 500 PDBs at "
        "L∈{50,75,100,125} × 125 for FID/fJSD/fS_T (N=500); designability "
        "on 50/L × 4 lengths (N=200); diversity on the designable subset. "
        f"Rows whose `des N` < {MIN_DESIGNABILITY_N} have downstream metrics "
        "(designability, scRMSD, diversity) suppressed — they hit the "
        "PDB-index-shift bug and the FID family is the only safe column."
    )
    lines.append("")
    lines.append(
        "**N per metric:** "
        + ", ".join(f"{lbl}={N_NOTES[mk]}" for mk, (lbl, _) in metric_cols)
        + "."
    )
    lines.append("")
    lines.append("Best per metric within each ablation block is **bolded**.")
    lines.append("")

    # Header row + separator. Numeric columns right-aligned for readability.
    lines.append("| " + " | ".join(headers) + " |")
    aligns = [":---"] + ["---:"] * (n_cols - 1)
    lines.append("| " + " | ".join(aligns) + " |")

    for akey, acfg in ABLATIONS.items():
        block = _load_block(akey)
        best = _best_indices(block)

        # Section header row — collapse onto first column.
        section_label = acfg["label"].replace("\n", " — ")
        section_cells = [f"**{section_label}**"] + [""] * (n_cols - 1)
        lines.append("| " + " | ".join(section_cells) + " |")

        for i, (_run_name, label, row) in enumerate(block):
            step = row.get("step")
            step_int = int(step) if step is not None else None
            bs = RUN_BS.get(_run_name)
            des_n = row.get("_res_designability_n")
            des_n_str = (
                "—"
                if des_n is None or (isinstance(des_n, float) and math.isnan(des_n))
                else f"{int(des_n)}"
            )

            cells = [
                label,
                _step_label(step_int),
                f"{bs}" if bs is not None else "—",
                des_n_str,
            ]
            for mkey, _meta in metric_cols:
                v = _val(row, mkey)
                s = _fmt(v)
                if v is not None and best.get(mkey) == i:
                    s = f"**{s}**"
                cells.append(s)
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
