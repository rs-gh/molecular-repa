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
PAPER_DIR = HERE.parent
SCRIPTS_DIR = HERE.parent.parent
GENERATION_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))  # for utils._results_io
sys.path.insert(0, str(PAPER_DIR))  # for md_tables_to_tsv
sys.path.insert(0, str(HERE))  # for plot_n128_paper (sibling)

from md_tables_to_tsv import md_to_tsv  # noqa: E402
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


# Min/max across all rows of Proteina paper Table 1 (unconditional backbone
# generation; FrameDiff/FoldFlow×3/FrameFlow/ESM3/Chroma/RFDiffusion/Proteus/
# Genie2 plus M_FS γ∈{0.35,0.45,0.5}, M_FS-no-tri γ=0.45, M_21M γ∈{0.3,0.6},
# M_LoRA γ=0.5). Caveat: paper protocol is n=256 over lengths 50-275, ours
# is n=128 over lengths 50-125, so absolute values are NOT directly
# comparable — these are sanity-band reference numbers, not targets. The
# scRMSD column has no paper-table counterpart and is left blank.
# Designability is converted from percent (paper 22.0-99.0) to rate (0-1).
# Diversity-clusters maps to the count-in-parens (paper 64-323), not the
# 0-1 cluster-fraction column the paper bolds.
PAPER_REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "_res_PDB_FID": (129.9, 933.9),
    "_res_AFDB_FID": (159.9, 855.4),
    "_res_fS_T": (9.72, 30.11),
    # Paper Table 1 only reports one fJSD column per dataset (T = Topology).
    # No reference range for fJSD A/C — leave empty so they render as "—".
    "_res_PDB_fJSD_T": (0.68, 3.69),
    "_res_AFDB_fJSD_T": (0.91, 3.10),
    "_res_designability_rate": (0.220, 0.990),
    "_res_diversity_clusters_total": (64.0, 323.0),
    "_res_diversity_pairwise_tm_mean": (0.35, 0.45),
}


RUN_BS = {
    "baseline_128_bs24_step200k": 24,
    "baseline_128_bs24_step400k": 24,
    "baseline_128_bs80_step200k": 80,
    "baseline_128_bs80_lr3x_step200k": 80,
    "repa_l0_128_bs80_steplast": 80,
    "repa_l0_128_bs80_step200k": 80,
    "repa_l4_128_bs24_step200k": 24,
    "repa_l4_128_bs24_step400k": 24,
    "repa_l4_128_bs80_step200k": 80,
    "repa_l4_128_bs80_lr3x_steplast": 80,
    "repa_l4_128_bs80_wd1e2_step200k": 80,
    "repa_l9_128_bs80_steplast": 80,
    "repa_l9_128_bs80_step200k": 80,
    "repa_l4_128_random_step200k": 80,
    "repa_l4_128_pw_structure_step100k": 80,
    "repa_l4_128_pw_torsional_step100k": 80,
    "repa_mpnn_l4_128_bs80_step200k": 80,
    "repa_esm_l4_128_step200k": 80,
    "pretrained_dfs_60m_n128_paper": None,  # external NVIDIA NGC ckpt
    "repa_l4_128_bs80_lambda2_steplast": 80,
    # per_residue layer-ablation block: mixed bs=24→80 at step 220k.
    # Display "24→80" to flag the bump; samples computed via override below.
    "repa_l0_128_per_residue_step400k": "24→80",
    "repa_l4_128_per_residue_step400k": "24→80",
    "repa_l9_128_per_residue_step400k": "24→80",
}


# Samples-seen override for runs where step × bs is not directly meaningful
# (mixed-bs runs). Per-residue layer ablation: 220k × 24 + 180k × 80 = 19.68M
# at step=400k. Source: comment in `evaluation/proteina/lib/checkpoints.py`
# (n=128 sample-matched arithmetic).
SAMPLES_OVERRIDE: dict[str, int] = {
    "repa_l0_128_per_residue_step400k": 19_680_000,
    "repa_l4_128_per_residue_step400k": 19_680_000,
    "repa_l9_128_per_residue_step400k": 19_680_000,
}


# n=128 batch-size audit (2026-05-07). Method: read `nsamples_processed /
# global_step` from the lightning ckpt at multiple step snapshots; non-
# monotonic counters across resumes flag a bs change. ✅ = clean (one bs
# throughout), 🔁 = mid-run bump, ⚠️ = ambiguous (only `last.ckpt`, no
# periodic snapshots to verify), 🚫 = wrong/mixed throughout. † next to bs
# marks rows whose underlying run was actually a different bs for some
# prefix or throughout. The pretrained NGC reference is N/A.
BS_AUDIT: dict[str, tuple[str, str]] = {
    "baseline_128_bs24_step200k": ("✅", "clean bs=24 across 10 ckpts (100k–1M)."),
    "baseline_128_bs24_step400k": ("✅", "clean bs=24 across 10 ckpts (100k–1M)."),
    "baseline_128_bs80_step200k": ("✅", "clean bs=80 (post-bump era)."),
    "baseline_128_bs80_lr3x_step200k": ("✅", "clean bs=80; started 04-27."),
    "repa_l0_128_bs80_steplast": (
        "⚠️",
        "only `last.ckpt`, no periodic ckpts to verify. Re-run candidate.",
    ),
    "repa_l4_128_bs24_step200k": ("✅", "clean bs=24 at 100k+200k."),
    "repa_l4_128_bs24_step400k": (
        "✅",
        "clean bs=24; counter reset on resume but per-step delta 300k→400k = 2.4M = bs=24.",
    ),
    "repa_l4_128_bs80_step200k": (
        "✅",
        "clean bs=80 (verified on wandb run `proteina_60m_repa_l4_128_per_residue_bs80`). The earlier `nsamples_processed / global_step` discrepancy was a Lightning resume-counter artefact, not a real bs change.",
    ),
    "repa_l4_128_bs80_lr3x_steplast": ("✅", "clean bs=80; started 04-26."),
    "repa_l9_128_bs80_steplast": (
        "⚠️",
        "only `last.ckpt`, no periodic ckpts to verify. Re-run candidate.",
    ),
    "repa_l4_128_random_step200k": ("✅", "clean bs=80; started 04-23."),
    "repa_l4_128_pw_structure_step100k": (
        "🚫",
        "**rerun candidate**: yaml claims bs=80 but bs_eff=24 throughout (config-vs-runtime divergence).",
    ),
    "repa_l4_128_pw_torsional_step100k": (
        "🚫",
        "**rerun candidate**: bs_eff=21.2 — neither clean 24 nor 80 (mixed/ramped phases).",
    ),
    "repa_mpnn_l4_128_bs80_step200k": ("✅", "clean bs=80; started 05-05."),
    "repa_esm_l4_128_step200k": ("✅", "clean bs=80; started 04-18 at bump cutoff."),
    "pretrained_dfs_60m_n128_paper": (
        "—",
        "external NVIDIA NGC v1.3 (12L, ep=177, global_step=1.3M); N/A.",
    ),
    "repa_l0_128_per_residue_step400k": (
        "🔁",
        "mixed bs=24→80 at step 220k. Same schedule as L4/L9 per_residue → cross-layer comparison fair.",
    ),
    "repa_l4_128_per_residue_step400k": (
        "🔁",
        "mixed bs=24→80 at step 220k. Same schedule as L0/L9 per_residue → cross-layer comparison fair.",
    ),
    "repa_l9_128_per_residue_step400k": (
        "🔁",
        "mixed bs=24→80 at step 220k. Same schedule as L0/L4 per_residue → cross-layer comparison fair.",
    ),
    "repa_l4_128_bs80_lambda2_steplast": (
        "✅",
        "clean bs=80 (config-declared); wandb verification recommended per audit method.",
    ),
    "repa_l0_128_bs80_step200k": (
        "✅",
        "clean bs=80 (config-declared); evaluated 2026-05-09 at step=200K EMA snapshot.",
    ),
    "repa_l9_128_bs80_step200k": (
        "✅",
        "clean bs=80 (config-declared); evaluated 2026-05-09 at step=200K EMA snapshot.",
    ),
    "repa_l4_128_bs80_wd1e2_step200k": (
        "✅",
        "clean bs=80, wd=1e-2 (config-declared); evaluated 2026-05-09 at step=200K EMA snapshot.",
    ),
}

# bs values whose verdict is 🔁 or 🚫 get a † suffix in the bs column.
_BS_TAINT_VERDICTS = {"🔁", "🚫"}


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


def _samples_label(step: int | None, bs: int | str | None, run_name: str = "") -> str:
    """Samples seen formatted as e.g. '16.0M' / '480K'.

    For runs with a fixed integer bs: samples = step × bs. For mixed-bs runs
    the integer arithmetic is wrong, so consult SAMPLES_OVERRIDE first.
    Returns '—' when neither override nor (step, int_bs) is available.
    """
    n = SAMPLES_OVERRIDE.get(run_name)
    if n is None:
        if step is None or not isinstance(bs, int):
            return "—"
        n = step * bs
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{n // 1000}K"


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
    headers = (
        ["Run", "Step", "bs", "samples", "des N"]
        + [f"{lbl} ({'↓' if low else '↑'})" for _mk, (lbl, low) in metric_cols]
        + ["Notes"]
    )
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
    lines.append(
        "**Notes column legend.** ✅ = clean bs throughout (verified via "
        "`nsamples_processed / global_step` at multiple step snapshots). "
        "🔁 = mid-run bs change; rerun-from-scratch recommended. "
        "⚠️ = ambiguous (only `last.ckpt`, no periodic snapshots to verify). "
        "🚫 = wrong/mixed bs throughout. † next to bs marks rows whose "
        "underlying run was actually a different bs for some prefix or "
        "throughout. Audit dated 2026-05-07."
    )
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
            audit_emoji, audit_note = BS_AUDIT.get(_run_name, ("", ""))
            taint = audit_emoji in _BS_TAINT_VERDICTS
            bs_str = "—" if bs is None else f"{bs}†" if taint else f"{bs}"
            des_n = row.get("_res_designability_n")
            des_n_str = (
                "—"
                if des_n is None or (isinstance(des_n, float) and math.isnan(des_n))
                else f"{int(des_n)}"
            )
            notes_cell = f"{audit_emoji} {audit_note}".strip() if audit_emoji else ""

            cells = [
                label,
                _step_label(step_int),
                bs_str,
                _samples_label(step_int, bs, _run_name),
                des_n_str,
            ]
            for mkey, _meta in metric_cols:
                v = _val(row, mkey)
                s = _fmt(v)
                if v is not None and best.get(mkey) == i:
                    s = f"**{s}**"
                cells.append(s)
            cells.append(notes_cell)
            lines.append("| " + " | ".join(cells) + " |")

    # Paper-reference range row appended after all ablation blocks. Same
    # column layout — Run/Step/bs/des-N collapsed to the label.
    ref_section = (
        "**Reference ranges — Proteina paper Table 1** "
        "(n=256 lengths 50-275; sanity band only, NOT directly comparable)"
    )
    lines.append("| " + " | ".join([f"_{ref_section}_"] + [""] * (n_cols - 1)) + " |")
    ref_cells = ["paper min–max", "—", "—", "—", "—"]
    for mkey, _meta in metric_cols:
        rng = PAPER_REFERENCE_RANGES.get(mkey)
        ref_cells.append("—" if rng is None else f"{_fmt(rng[0])}–{_fmt(rng[1])}")
    ref_cells.append("—")  # Notes column
    lines.append("| " + " | ".join(ref_cells) + " |")

    lines.append("")
    lines.append(
        "*Reference-range source:* Proteina paper Table 1 (unconditional "
        "backbone generation), 16 method rows. Designability converted from "
        "% to rate (0-1). Diversity-clusters maps to the cluster *count* in "
        "parens (paper col shows count beside a 0-1 cluster-fraction we "
        "don't track). scRMSD has no paper-table counterpart. **The paper "
        "uses n=256 over lengths 50-275 with γ-tuned guidance; we use n=128 "
        "over lengths 50-125 unconditional, so the bands are sanity checks, "
        "not targets.**"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"Wrote {OUT}")
    tsv_path = OUT.with_suffix(".tsv")
    n = md_to_tsv(OUT, tsv_path)
    print(f"Wrote {tsv_path} ({n} table{'s' if n != 1 else ''})")


if __name__ == "__main__":
    main()
