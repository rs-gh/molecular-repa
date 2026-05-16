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

import yaml

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
SCRIPTS_DIR = HERE.parent.parent
GENERATION_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))  # for utils._results_io
sys.path.insert(0, str(PAPER_DIR))  # for md_tables_to_tsv
sys.path.insert(0, str(HERE))  # for plot_n128_paper (sibling)

from utils._results_io import load_sweep_rows  # noqa: E402
from plot_n128_paper import (  # noqa: E402
    METRICS,
    MIN_DESIGNABILITY_N,
    N_NOTES,
    _scrub_corrupt_designability,
)

RESULTS_ROOT = GENERATION_ROOT / "results" / "paper"
OUT = GENERATION_ROOT / "figures" / "paper" / "n128_paper" / "n128_paper_tables.md"
ABLATION_BLOCKS_YAML = RESULTS_ROOT / "ablation_blocks.yaml"

# Training batch size per run name. Encodes the actual effective bs used
# for the run; not derivable cleanly from the checkpoint, so listed here
# (mirrors what the plot legend communicates implicitly).
# Display label per (ablation_id, run). Falls back to `run` if missing.
# Ablation IDs come from `ablation_blocks.yaml`. The same run can have
# different labels in different ablation contexts (e.g. "Baseline" in the
# layer block, "BL bs80 200k" in the bs block).
RUN_LABELS: dict[tuple[str, str], str] = {
    # --- External reference
    ("n128_external_reference", "pretrained_dfs_60m_n128_paper"): "Pretrained DFS-60M",
    # --- n=128 PDB bs ablation
    ("n128_pdb_bs", "baseline_128_bs24_step200k"): "BL bs24 200k",
    ("n128_pdb_bs", "baseline_128_bs24_step400k"): "BL bs24 400k",
    ("n128_pdb_bs", "baseline_128_bs80_step200k"): "BL bs80 200k",
    ("n128_pdb_bs", "repa_l4_128_bs24_step200k"): "L4 bs24 200k",
    ("n128_pdb_bs", "repa_l4_128_bs24_step400k"): "L4 bs24 400k",
    ("n128_pdb_bs", "repa_l4_128_bs80_step200k"): "L4 bs80 200k",
    # --- n=128 PDB L4 REPA wd + λ + lr ablation
    ("n128_pdb_l4_wd_lambda_lr", "baseline_128_bs80_step200k"): "BL bs80",
    ("n128_pdb_l4_wd_lambda_lr", "repa_l4_128_bs80_step200k"): "λ=0.5, wd=0, lr=1×",
    ("n128_pdb_l4_wd_lambda_lr", "repa_l4_128_bs80_lambda025_step200k"): "λ=0.25",
    ("n128_pdb_l4_wd_lambda_lr", "repa_l4_128_bs80_lambda1_step200k"): "λ=1.0",
    ("n128_pdb_l4_wd_lambda_lr", "repa_l4_128_bs80_lambda2_steplast"): "λ=2.0",
    ("n128_pdb_l4_wd_lambda_lr", "repa_l4_128_bs80_wd1e2_step200k"): "wd=1e-2",
    ("n128_pdb_l4_wd_lambda_lr", "baseline_128_bs80_lr3x_step200k"): "BL lr3×",
    ("n128_pdb_l4_wd_lambda_lr", "repa_l4_128_bs80_lr3x_steplast"): "L4 lr3×",
    # --- n=128 PDB L4 REPA encoder ablation
    ("n128_pdb_l4_encoder", "baseline_128_bs80_step200k"): "Baseline",
    ("n128_pdb_l4_encoder", "repa_l4_128_bs80_step200k"): "CA-GearNet",
    ("n128_pdb_l4_encoder", "repa_l4_128_random_step200k"): "GearNet random",
    ("n128_pdb_l4_encoder", "repa_mpnn_l4_128_bs80_step200k"): "ProteinMPNN",
    ("n128_pdb_l4_encoder", "repa_esm_l4_128_step200k"): "ESM2",
    ("n128_pdb_l4_encoder", "repa_l4_128_pw_structure_step100k"): "PW-Structure",
    ("n128_pdb_l4_encoder", "repa_l4_128_pw_torsional_step100k"): "PW-Torsional",
    # --- n=128 PDB L9 REPA encoder ablation
    ("n128_pdb_l9_encoder", "baseline_128_bs80_step200k"): "Baseline",
    ("n128_pdb_l9_encoder", "repa_l9_128_bs80_step200k"): "CA-GearNet",
    # --- n=128 PDB REPA bs80 layer ablation
    ("n128_pdb_bs80_layer", "baseline_128_bs80_step200k"): "Baseline",
    ("n128_pdb_bs80_layer", "repa_l0_128_bs80_step200k"): "REPA L0",
    ("n128_pdb_bs80_layer", "repa_l4_128_bs80_step200k"): "REPA L4",
    ("n128_pdb_bs80_layer", "repa_l9_128_bs80_step200k"): "REPA L9",
    # --- n=128 PDB per_residue layer @ 400K (mixed bs)
    (
        "n128_pdb_per_residue_layer_400k",
        "baseline_128_bs24_step400k",
    ): "Baseline (bs24)",
    ("n128_pdb_per_residue_layer_400k", "repa_l0_128_per_residue_step400k"): "REPA L0",
    ("n128_pdb_per_residue_layer_400k", "repa_l4_128_per_residue_step400k"): "REPA L4",
    ("n128_pdb_per_residue_layer_400k", "repa_l9_128_per_residue_step400k"): "REPA L9",
    # --- n=128 AFDB encoder ablation
    ("n128_afdb_encoder", "baseline_afdb_128_bs80_step200k"): "Baseline AFDB 200k",
    ("n128_afdb_encoder", "baseline_afdb_128_bs80_step400k"): "Baseline AFDB 400k",
    ("n128_afdb_encoder", "baseline_afdb_128_bs80_step600k"): "Baseline AFDB 600k",
    ("n128_afdb_encoder", "baseline_afdb_128_bs80_step800k"): "Baseline AFDB 800k",
    ("n128_afdb_encoder", "baseline_afdb_128_bs80_step1000k"): "Baseline AFDB 1000k",
    ("n128_afdb_encoder", "baseline_afdb_128_bs80_step1200k"): "Baseline AFDB 1200k",
    ("n128_afdb_encoder", "repa_l4_afdb_128_bs80_step200k"): "CA-GearNet 200k",
    ("n128_afdb_encoder", "repa_l4_afdb_128_bs80_step600k"): "CA-GearNet 600k",
    ("n128_afdb_encoder", "repa_mpnn_l4_afdb_128_bs80_step200k"): "ProteinMPNN 200k",
    ("n128_afdb_encoder", "repa_mpnn_l4_afdb_128_bs80_step400k"): "ProteinMPNN 400k",
    ("n128_afdb_encoder", "repa_mpnn_l4_afdb_128_bs80_step600k"): "ProteinMPNN 600k",
    ("n128_afdb_encoder", "repa_mpnn_l4_afdb_128_bs80_step800k"): "ProteinMPNN 800k",
    ("n128_afdb_encoder", "repa_mpnn_l4_afdb_128_bs80_step1000k"): "ProteinMPNN 1000k",
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
    # AFDB n=128 (2-GPU × per-GPU bs=40, eff bs=80; doc label "bs80_2gpu")
    "baseline_afdb_128_bs80_step200k": 80,
    "baseline_afdb_128_bs80_step400k": 80,
    "baseline_afdb_128_bs80_step600k": 80,
    "baseline_afdb_128_bs80_step800k": 80,
    "baseline_afdb_128_bs80_step1000k": 80,
    "baseline_afdb_128_bs80_step1200k": 80,
    "repa_l4_afdb_128_bs80_step200k": 80,
    "repa_l4_afdb_128_bs80_step600k": 80,
    "repa_mpnn_l4_afdb_128_bs80_step200k": 80,
    "repa_mpnn_l4_afdb_128_bs80_step400k": 80,
    "repa_mpnn_l4_afdb_128_bs80_step600k": 80,
    "repa_mpnn_l4_afdb_128_bs80_step800k": 80,
    "repa_mpnn_l4_afdb_128_bs80_step1000k": 80,
    # Extra bs/λ rows mapped into the wd+λ+lr block
    "repa_l4_128_bs80_lambda025_step200k": 80,
    "repa_l4_128_bs80_lambda1_step200k": 80,
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
    # λ extension rows (λ=0.25 / λ=1.0)
    "repa_l4_128_bs80_lambda025_step200k": (
        "✅",
        "clean bs=80, λ=0.25 (config-declared); evaluated 2026-05-09.",
    ),
    "repa_l4_128_bs80_lambda1_step200k": (
        "✅",
        "clean bs=80, λ=1.0 (config-declared); evaluated 2026-05-09.",
    ),
    # AFDB n=128 (per-GPU bs=40 × 2 GPU = eff bs=80; verified 2026-05-16
    # from data_config.json:batch_size + exp_config.json:ngpus_per_node_).
    "baseline_afdb_128_bs80_step200k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); AFDB-SwissProt; val_check_interval=10000.",
    ),
    "baseline_afdb_128_bs80_step400k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); AFDB-SwissProt; continuation.",
    ),
    "baseline_afdb_128_bs80_step600k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); AFDB-SwissProt; continuation.",
    ),
    "baseline_afdb_128_bs80_step800k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); AFDB-SwissProt; continuation.",
    ),
    "baseline_afdb_128_bs80_step1000k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); AFDB-SwissProt; continuation.",
    ),
    "baseline_afdb_128_bs80_step1200k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); AFDB-SwissProt; continuation.",
    ),
    "repa_l4_afdb_128_bs80_step200k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); REPA L4 CA-GearNet, per_residue, λ=0.5, AFDB.",
    ),
    "repa_l4_afdb_128_bs80_step600k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); REPA L4 CA-GearNet, per_residue, λ=0.5, AFDB; continuation.",
    ),
    "repa_mpnn_l4_afdb_128_bs80_step200k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); REPA L4 ProteinMPNN, per_residue, λ=0.5, AFDB.",
    ),
    "repa_mpnn_l4_afdb_128_bs80_step400k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); REPA L4 ProteinMPNN, per_residue, λ=0.5, AFDB; continuation.",
    ),
    "repa_mpnn_l4_afdb_128_bs80_step600k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); REPA L4 ProteinMPNN, per_residue, λ=0.5, AFDB; continuation.",
    ),
    "repa_mpnn_l4_afdb_128_bs80_step800k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); REPA L4 ProteinMPNN, per_residue, λ=0.5, AFDB; continuation.",
    ),
    "repa_mpnn_l4_afdb_128_bs80_step1000k": (
        "✅",
        "clean per-GPU bs=40 × 2 GPU (eff bs=80); REPA L4 ProteinMPNN, per_residue, λ=0.5, AFDB; continuation.",
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


def _load_blocks_n128() -> list[dict]:
    """Load yaml ablation blocks for set=n128, in file order."""
    blocks = yaml.safe_load(ABLATION_BLOCKS_YAML.read_text())
    return [b for b in blocks if b.get("set") == "n128"]


def _index_jsonls() -> dict[tuple[str, str, str], dict]:
    """Build (profile, run, str(step)) → row index across all n128 profiles."""
    idx: dict[tuple[str, str, str], dict] = {}
    for jp in sorted(RESULTS_ROOT.glob("n128_paper_*/sweep_results.jsonl")):
        profile = jp.parent.name
        for r in load_sweep_rows(jp):
            r = _scrub_corrupt_designability(r)
            r["profile"] = profile
            idx[(profile, str(r.get("run", "")), str(r.get("step", "")))] = r
    return idx


def _load_block(block: dict, idx: dict) -> list[tuple[str, str, dict]]:
    """Return [(run_name, display_label, row)] for one yaml block."""
    out: list[tuple[str, str, dict]] = []
    for spec in block["rows"]:
        key = (spec["profile"], str(spec["run"]), str(spec["step"]))
        row = idx.get(key, {})
        run_name = str(spec["run"])
        label = RUN_LABELS.get((block["id"], run_name), run_name)
        out.append((run_name, label, row))
    return out


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

    idx = _index_jsonls()
    for ablation_block in _load_blocks_n128():
        block = _load_block(ablation_block, idx)
        best = _best_indices(block)

        # Section header row — collapse onto first column.
        section_label = ablation_block["title"]
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
    # TSV is the responsibility of jsonl_to_tsv.py (driven by the same yaml).
    # Do not emit md_to_tsv here — would clobber the canonical TSV format.


if __name__ == "__main__":
    main()
