"""Markdown ablation table for the n=256 paper-protocol sweeps.

Sister to tabulate_n128_paper.py: same yaml-driven block structure
(`results/paper/ablation_blocks.yaml`), trimmed metric column set the n=256
md historically uses. One row per (block, run, step) tuple from the yaml.
Best-per-metric within each ablation block is bolded.

Run after any sweep result update OR after editing ablation_blocks.yaml:

    python evaluation/proteina/generation/scripts/paper/n256_paper/tabulate_n256_paper.py
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
sys.path.insert(0, str(SCRIPTS_DIR))

from utils._results_io import load_sweep_rows  # noqa: E402

RESULTS_ROOT = GENERATION_ROOT / "results" / "paper"
OUT = GENERATION_ROOT / "figures" / "paper" / "n256_paper" / "n256_paper_tables.md"
ABLATION_BLOCKS_YAML = RESULTS_ROOT / "ablation_blocks.yaml"

# (metric key, display label, lower_is_better)
METRICS: dict[str, tuple[str, bool]] = {
    "_res_PDB_FID": ("PDB FID", True),
    "_res_PDB_fJSD_C": ("PDB fJSD C", True),
    "_res_PDB_fJSD_A": ("PDB fJSD A", True),
    "_res_PDB_fJSD_T": ("PDB fJSD T", True),
    "_res_AFDB_FID": ("AFDB FID", True),
    "_res_AFDB_fJSD_C": ("AFDB fJSD C", True),
    "_res_AFDB_fJSD_A": ("AFDB fJSD A", True),
    "_res_AFDB_fJSD_T": ("AFDB fJSD T", True),
    "_res_fS_T": ("fS T", False),
    "_res_designability_rate": ("Des", False),
    "_res_scRMSD_mean": ("scRMSD", True),
    "_res_diversity_clusters_total": ("Div clust total", False),
    "_res_diversity_pairwise_tm_mean": ("Div pairwise TM", True),
}

N_NOTES: dict[str, str] = {
    "_res_PDB_FID": "N=1125",
    "_res_PDB_fJSD_C": "N=1125",
    "_res_PDB_fJSD_A": "N=1125",
    "_res_PDB_fJSD_T": "N=1125",
    "_res_AFDB_FID": "N=1125",
    "_res_AFDB_fJSD_C": "N=1125",
    "_res_AFDB_fJSD_A": "N=1125",
    "_res_AFDB_fJSD_T": "N=1125",
    "_res_fS_T": "N=1125",
    "_res_designability_rate": "N=250",
    "_res_scRMSD_mean": "N=250",
    "_res_diversity_clusters_total": "designable",
    "_res_diversity_pairwise_tm_mean": "designable",
}

PAPER_REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "_res_PDB_FID": (129.9, 933.9),
    "_res_AFDB_FID": (159.9, 855.4),
    "_res_PDB_fJSD_T": (0.68, 3.69),
    "_res_AFDB_fJSD_T": (0.91, 3.10),
    "_res_fS_T": (9.72, 30.11),
    "_res_designability_rate": (0.220, 0.990),
    "_res_diversity_clusters_total": (64.0, 323.0),
    "_res_diversity_pairwise_tm_mean": (0.35, 0.45),
}

# Display label per (ablation_id, run). Falls back to run name.
RUN_LABELS: dict[tuple[str, str], str] = {
    (
        "n256_external_reference",
        "pretrained_dfs_60m_n256_paper",
    ): "Pretrained DFS-60M (NGC)",
    # Encoder + layer + bs (PDB)
    ("n256_pdb_encoder_layer_bs", "baseline_256_ep21"): "Baseline ep21",
    ("n256_pdb_encoder_layer_bs", "repa_l4_256_ep22"): "CA-GearNet L4 ep22",
    ("n256_pdb_encoder_layer_bs", "repa_l4_256_random_ep17"): "GearNet random L4 ep17",
    ("n256_pdb_encoder_layer_bs", "repa_l9_256_ep25"): "CA-GearNet L9 ep25",
    (
        "n256_pdb_encoder_layer_bs",
        "repa_mpnn_l4_256_per_residue_step300k",
    ): "ProteinMPNN L4 ep26",
    (
        "n256_pdb_encoder_layer_bs",
        "repa_esm_l9_t30_256_steplast",
    ): "ESM2 L9-t30 steplast",
    # wd + λ + bs
    ("n256_pdb_l4_wd_lambda_bs", "baseline_256_ep21"): "Baseline ep21",
    ("n256_pdb_l4_wd_lambda_bs", "repa_l4_256_ep22"): "λ=0.5 ep22",
    (
        "n256_pdb_l4_wd_lambda_bs",
        "repa_l4_256_per_residue_lambda1_step200k",
    ): "λ=1.0 step200k",
    (
        "n256_pdb_l4_wd_lambda_bs",
        "repa_l4_256_per_residue_lambda1_step300k",
    ): "λ=1.0 ep26 step300k",
    (
        "n256_pdb_l4_wd_lambda_bs",
        "repa_l4_256_per_residue_lambda2_step200k",
    ): "λ=2.0 ep17 step200k",
    (
        "n256_pdb_l4_wd_lambda_bs",
        "repa_l4_256_per_residue_lambda2_step300k",
    ): "λ=2.0 step300k",
    ("n256_pdb_l4_wd_lambda_bs", "repa_l4_256_random_ep17"): "GearNet random L4 ep17",
    (
        "n256_pdb_l4_wd_lambda_bs",
        "repa_l4_256_ep13_step300k",
    ): "λ=0.5 ep13 step300k (anchor)",
    # L4 step extension
    ("n256_pdb_l4_step_extension", "repa_l4_256_ep13_step300k"): "L4 ep13 step300k",
    ("n256_pdb_l4_step_extension", "repa_l4_256_ep22"): "L4 ep22 step400k",
    ("n256_pdb_l4_step_extension", "repa_l4_256_ep31_step500k"): "L4 ep31 step500k",
    # AFDB encoder + layer + bs
    (
        "n256_afdb_encoder_layer_bs",
        "baseline_afdb_256_ep20",
    ): "Baseline AFDB ep20 step200k",
    (
        "n256_afdb_encoder_layer_bs",
        "baseline_afdb_256_step400k",
    ): "Baseline AFDB step400k",
    (
        "n256_afdb_encoder_layer_bs",
        "baseline_afdb_256_step700k",
    ): "Baseline AFDB step700k",
    (
        "n256_afdb_encoder_layer_bs",
        "baseline_afdb_256_step900k",
    ): "Baseline AFDB step900k",
    (
        "n256_afdb_encoder_layer_bs",
        "repa_l4_afdb_256_ep20",
    ): "CA-GearNet AFDB ep20 step200k",
    (
        "n256_afdb_encoder_layer_bs",
        "repa_l4_afdb_256_step400k",
    ): "CA-GearNet AFDB step400k",
    (
        "n256_afdb_encoder_layer_bs",
        "repa_l4_afdb_256_step700k",
    ): "CA-GearNet AFDB step700k",
    (
        "n256_afdb_encoder_layer_bs",
        "repa_l9_afdb_256_step200k",
    ): "CA-GearNet L9 AFDB step200k",
    (
        "n256_afdb_encoder_layer_bs",
        "repa_l9_afdb_256_step400k",
    ): "CA-GearNet L9 AFDB step400k",
    (
        "n256_afdb_encoder_layer_bs",
        "repa_mpnn_l4_afdb_256_step400k",
    ): "ProteinMPNN L4 AFDB step400k",
    (
        "n256_afdb_encoder_layer_bs",
        "repa_mpnn_l9_afdb_256_step400k",
    ): "ProteinMPNN L9 AFDB step400k",
    (
        "n256_afdb_encoder_layer_bs",
        "repa_mpnn_l9_afdb_256_step700k",
    ): "ProteinMPNN L9 AFDB step700k",
    # Averaging
    ("n256_pdb_averaging", "baseline_256_ep21"): "Baseline ep21 (anchor)",
    ("n256_pdb_averaging", "repa_l0_256_ep26"): "L0 per_residue ep26",
    ("n256_pdb_averaging", "repa_l0_256_per_sample_steplast"): "L0 per_sample steplast",
    ("n256_pdb_averaging", "repa_l4_256_ep22"): "L4 per_residue ep22",
    ("n256_pdb_averaging", "repa_l4_256_per_sample_step400k"): "L4 per_sample step400k",
    ("n256_pdb_averaging", "repa_l9_256_ep25"): "L9 per_residue ep25",
    ("n256_pdb_averaging", "repa_l9_256_per_sample_steplast"): "L9 per_sample steplast",
}

# Nominal training bs per run. None = external pretrained.
# n=256 runs have a 12→24 bump around 2026-04-18 for many older runs; the bs
# column shows the nominal target (24) with a † from the audit dict.
RUN_BS: dict[str, int | str | None] = {
    "pretrained_dfs_60m_n256_paper": None,
    # PDB
    "baseline_256_ep21": 24,
    "repa_l0_256_ep26": 24,
    "repa_l4_256_ep22": 24,
    "repa_l9_256_ep25": 24,
    "repa_l4_256_random_ep17": 24,
    "repa_l4_256_ep13_step300k": 24,
    "repa_l4_256_ep31_step500k": 24,
    "repa_l4_256_per_residue_lambda1_step200k": 24,
    "repa_l4_256_per_residue_lambda1_step300k": 24,
    "repa_l4_256_per_residue_lambda2_step200k": 24,
    "repa_l4_256_per_residue_lambda2_step300k": 24,
    "repa_mpnn_l4_256_per_residue_step300k": 24,
    "repa_esm_l9_t30_256_steplast": 12,  # OOMs at 24
    # Averaging — per_sample variants
    "repa_l0_256_per_sample_steplast": 24,
    "repa_l4_256_per_sample_step400k": 24,
    "repa_l9_256_per_sample_steplast": 24,
    # AFDB
    "baseline_afdb_256_ep20": 24,
    "baseline_afdb_256_step400k": 24,
    "baseline_afdb_256_step700k": 24,
    "baseline_afdb_256_step900k": 24,
    "repa_l4_afdb_256_ep20": 24,
    "repa_l4_afdb_256_step400k": 24,
    "repa_l4_afdb_256_step700k": 24,
    "repa_l9_afdb_256_step200k": 24,
    "repa_l9_afdb_256_step400k": 24,
    "repa_mpnn_l4_afdb_256_step400k": 24,
    "repa_mpnn_l9_afdb_256_step400k": 24,
    "repa_mpnn_l9_afdb_256_step700k": 24,
}

# Samples-seen override for runs whose `step × bs` arithmetic doesn't match
# the ckpt's `nsamples_processed` field (mixed-bs runs or known bumps).
SAMPLES_OVERRIDE: dict[str, int] = {
    # 12→24 bumped runs (n=256): pre-bump fraction shrinks effective samples
    # below `step × 24`. Numbers taken from n256_bump_steps.md / Notes in the
    # previous md revision (audit dated 2026-05-07).
    "baseline_256_ep21": 5_740_000,
    "repa_l0_256_ep26": 7_080_000,
    "repa_l4_256_ep22": 6_370_000,
    "repa_l9_256_ep25": 7_250_000,
    "repa_l4_256_random_ep17": 1_440_000,
    "repa_l4_256_ep13_step300k": 3_970_000,
    "repa_l4_256_ep31_step500k": 8_770_000,
    "repa_l4_256_per_sample_step400k": 7_250_000,
    "repa_l0_256_per_sample_steplast": 7_520_000,
    "repa_l9_256_per_sample_steplast": 7_560_000,
    # AFDB sample-budget caveat from prior audit
    "repa_l4_afdb_256_ep20": 2_530_000,
}

# Audit emoji + note per run. Mirrors the n128 BS_AUDIT structure.
# 🔁 = bs=12→24 mid-run bump (rerun-from-scratch candidate). ✅ = clean.
# ⚠️ = nominal bs vs avg-bs divergence. † appended in bs column for 🔁/🚫.
BS_AUDIT: dict[str, tuple[str, str]] = {
    "pretrained_dfs_60m_n256_paper": (
        "—",
        "external NVIDIA NGC v1.3 (12L, ep=177, global_step=1.3M); N/A.",
    ),
    # PDB encoder + layer + bs
    "baseline_256_ep21": (
        "🔁",
        "rerun candidate: wandb confirms bs=12 → bs=24 at step ~322K (full run reached step 615K). Anchors 3 ablation blocks (layer/encoder/dataset).",
    ),
    "repa_l0_256_ep26": (
        "🔁",
        "rerun candidate: bs=12 → bs=24 at step ~210K (started 04-17, bumped 04-18). Two snaps preserved.",
    ),
    "repa_l4_256_ep22": (
        "🔁",
        "rerun candidate: bs=12 → bs=24 at step ~269K. Anchor for encoder/dataset/λ/averaging blocks — taints all four.",
    ),
    "repa_l9_256_ep25": ("🔁", "rerun candidate: bs=12 → bs=24 at step ~196K."),
    "repa_l4_256_random_ep17": (
        "⚠️",
        "avg bs=7.20 over the run (1.44M smp / 200K steps), NOT clean bs=24 as the start-date had implied. Likely length-bucketed sampling effect; sample budget 3.3× smaller than the bs=24 baseline AFDB row.",
    ),
    "repa_mpnn_l4_256_per_residue_step300k": (
        "✅",
        "clean bs=24 (started 05-06, post-bump era); strongly outperforms CA-GearNet ep22 on FID/Des/scRMSD/clusters.",
    ),
    "repa_esm_l9_t30_256_steplast": (
        "⚠️",
        "bs=12 throughout (≠ rest of block at bs=24) — ESM-650M OOMs at bs=24. L9-t30 ≠ L4 default — second axis of incomparability. Directional comparison only.",
    ),
    # wd + λ + bs
    "repa_l4_256_per_residue_lambda1_step200k": (
        "✅",
        "clean bs=24 (started 05-07, post-bump era); evaluated 2026-05-09.",
    ),
    "repa_l4_256_per_residue_lambda1_step300k": (
        "✅",
        "clean bs=24 (started 05-07, post-bump era); λ=1.0 ep26 — worse than λ=0.5 on most metrics despite +0.83M smp budget.",
    ),
    "repa_l4_256_per_residue_lambda2_step200k": (
        "✅",
        "clean bs=24 (started 05-07, post-bump era); λ=2.0 ep17 — best fS_T (43.83) but collapse on Designability (0.032).",
    ),
    "repa_l4_256_per_residue_lambda2_step300k": (
        "✅",
        "clean bs=24 (started 05-07, post-bump era); λ=2.0 ep26 step300k.",
    ),
    "repa_l4_256_ep13_step300k": (
        "🔁",
        "same run as λ=0.5 ep22 (bumped 12→24 at ~269K), earlier ckpt. Step-matched to λ=1.0 ep26@300K; NOT sample-matched (λ=1.0 has 1.8× more samples).",
    ),
    # L4 step extension
    "repa_l4_256_ep31_step500k": (
        "🔁",
        "clean bs=24 from step ~269K onward (same run as ep22, +100K steps). Designability monotonically improves (ep13→ep22→ep31); FID/fS_T peak at ep22.",
    ),
    # Averaging (per_sample variants)
    "repa_l0_256_per_sample_steplast": (
        "🔁",
        "rerun candidate: bs=12 → bs=24 at step ~143K. Per_residue ep26 better on FID/fJSD/fS_T but worse on Des; resolved live last-EMA at 381500.",
    ),
    "repa_l4_256_per_sample_step400k": (
        "🔁",
        "explicit ep25/400K snapshot pin (sample-matched to per_residue ep22, ~6.69M smp); bs=12→24 mid-training. Notably better PDB FID (276.7) than per_residue ep22 but loses on fS_T and Des.",
    ),
    "repa_l9_256_per_sample_steplast": (
        "🔁",
        "rerun candidate: bs=12 → bs=24 at step ~145K. Per_residue ep25 better on Des/scRMSD/clusters; resolved live last-EMA at 385000.",
    ),
    # AFDB
    "baseline_afdb_256_ep20": ("✅", "clean bs=24 throughout (started 04-23)."),
    "baseline_afdb_256_step400k": ("✅", "clean bs=24; continuation of ep20."),
    "baseline_afdb_256_step700k": ("✅", "clean bs=24; continuation of ep20."),
    "baseline_afdb_256_step900k": ("✅", "clean bs=24; continuation of ep20."),
    "repa_l4_afdb_256_ep20": (
        "⚠️",
        "avg bs=12.66 (2.53M / 200K from ckpt nsamples_processed), NOT clean bs=24. ~half the sample budget of Baseline AFDB at the same step — headline FID/Des wins are despite a 1.9× sample-budget disadvantage.",
    ),
    "repa_l4_afdb_256_step400k": (
        "⚠️",
        "AFDB trajectory continuation; avg bs may still diverge from nominal — verify on wandb.",
    ),
    "repa_l4_afdb_256_step700k": (
        "⚠️",
        "AFDB trajectory continuation; avg bs may still diverge from nominal — verify on wandb.",
    ),
    "repa_mpnn_l4_afdb_256_step400k": (
        "✅",
        "REPA L4 ProteinMPNN AFDB; clean bs=24 (post-bump era).",
    ),
    "repa_mpnn_l9_afdb_256_step400k": (
        "✅",
        "REPA L9 ProteinMPNN AFDB; clean bs=24 (post-bump era).",
    ),
    "repa_mpnn_l9_afdb_256_step700k": (
        "✅",
        "REPA L9 ProteinMPNN AFDB; trajectory continuation.",
    ),
    "repa_l9_afdb_256_step200k": (
        "✅",
        "REPA L9 CA-GearNet AFDB; clean per-GPU bs=12 × 2 GPU (eff bs=24); first L9-GN AFDB sweep.",
    ),
    "repa_l9_afdb_256_step400k": (
        "✅",
        "REPA L9 CA-GearNet AFDB; clean per-GPU bs=12 × 2 GPU (eff bs=24); trajectory continuation.",
    ),
}

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
    n = SAMPLES_OVERRIDE.get(run_name)
    if n is None:
        if step is None or not isinstance(bs, int):
            return "—"
        n = step * bs
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    return f"{n // 1000}K"


def _load_blocks_n256() -> list[dict]:
    blocks = yaml.safe_load(ABLATION_BLOCKS_YAML.read_text())
    return [b for b in blocks if b.get("set") == "n256"]


def _index_jsonls() -> dict[tuple[str, str, str], dict]:
    idx: dict[tuple[str, str, str], dict] = {}
    for jp in sorted(RESULTS_ROOT.glob("n256_paper_*/sweep_results.jsonl")):
        profile = jp.parent.name
        for r in load_sweep_rows(jp):
            r["profile"] = profile
            idx[(profile, str(r.get("run", "")), str(r.get("step", "")))] = r
    return idx


def _load_block(block: dict, idx: dict) -> list[tuple[str, str, dict]]:
    out: list[tuple[str, str, dict]] = []
    for spec in block["rows"]:
        key = (spec["profile"], str(spec["run"]), str(spec["step"]))
        run_name = str(spec["run"])
        label = RUN_LABELS.get((block["id"], run_name), run_name)
        out.append((run_name, label, idx.get(key, {})))
    return out


def _best_indices(block: list[tuple[str, str, dict]]) -> dict[str, int]:
    best: dict[str, int] = {}
    for mkey, (_lbl, lower_better) in METRICS.items():
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
    metric_cols = list(METRICS.items())
    headers = (
        ["Run", "Step", "bs", "samples", "des N"]
        + [f"{lbl} ({'↓' if low else '↑'})" for _mk, (lbl, low) in metric_cols]
        + ["Notes"]
    )
    n_cols = len(headers)

    lines: list[str] = []
    lines.append("# n=256 paper-protocol sweep — ablation table")
    lines.append("")
    lines.append(
        "Companion to `n256_paper_sweep.png`. Pool: 1125 PDBs at "
        "L∈{50,75,100,125,150,175,200,225,250} × 125 for FID/fJSD/fS_T (N=1125); "
        "designability on 50/L × 5 paper lengths {50,100,150,200,250} (N=250); "
        "diversity/novelty on the designable subset."
    )
    lines.append("")
    lines.append(
        "**N per metric:** "
        + ", ".join(f"{lbl}={N_NOTES[mk]}" for mk, (lbl, _) in metric_cols)
        + "."
    )
    lines.append("")
    lines.append(
        "Block grouping is driven by `evaluation/proteina/generation/results/paper/ablation_blocks.yaml` "
        "and mirrors the section headers in [`docs/research/proteina_ablation_checkpoints.md`]"
        "(../../../../../docs/research/proteina_ablation_checkpoints.md). "
        "Runs that anchor multiple ablations are repeated, one row per block."
    )
    lines.append("")
    lines.append("Best per metric within each ablation block is **bolded**.")
    lines.append("")
    lines.append(
        "**Notes column legend.** ✅ = clean bs throughout. "
        "🔁 = mid-run bs change (typically bs=12→24 around 2026-04-18 SDPA-contiguous fix); rerun-from-scratch recommended. "
        "⚠️ = nominal-vs-avg-bs divergence. 🚫 = wrong bs throughout. "
        "† next to bs marks rows whose underlying run was actually a different bs for some prefix or throughout."
    )
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    aligns = [":---"] + ["---:"] * (n_cols - 2) + [":---"]
    lines.append("| " + " | ".join(aligns) + " |")

    idx = _index_jsonls()
    for ablation_block in _load_blocks_n256():
        block = _load_block(ablation_block, idx)
        best = _best_indices(block)
        section_label = ablation_block["title"]
        section_cells = [f"**{section_label}**"] + [""] * (n_cols - 1)
        lines.append("| " + " | ".join(section_cells) + " |")

        for i, (run_name, label, row) in enumerate(block):
            step = row.get("step")
            step_int = int(step) if step is not None else None
            bs = RUN_BS.get(run_name)
            audit_emoji, audit_note = BS_AUDIT.get(run_name, ("", ""))
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
                _samples_label(step_int, bs, run_name),
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

    # Paper-reference range row
    ref_section = (
        "**Reference ranges — Proteina paper Table 1** "
        "(n=256 lengths 50-275 with γ-tuned guidance; sanity band only, NOT directly comparable)"
    )
    lines.append("| " + " | ".join([f"_{ref_section}_"] + [""] * (n_cols - 1)) + " |")
    ref_cells = ["paper min–max", "—", "—", "—", "—"]
    for mkey, _meta in metric_cols:
        rng = PAPER_REFERENCE_RANGES.get(mkey)
        ref_cells.append("—" if rng is None else f"{_fmt(rng[0])}–{_fmt(rng[1])}")
    ref_cells.append("—")
    lines.append("| " + " | ".join(ref_cells) + " |")

    lines.append("")
    lines.append(
        "*Reference-range source:* Proteina paper Table 1 (unconditional backbone "
        "generation), 17 method rows. Designability converted from % to rate (0-1). "
        "Diversity-clusters maps to the cluster *count* the paper reports. "
        "fJSD C/A levels and scRMSD have no Table 1 counterparts — left blank. "
        "**The paper uses n=256 over lengths 50–275 with γ-tuned guidance; we use "
        "n=256 over lengths 50–250 unconditional, so the band is a sanity check, "
        "not a target.**"
    )
    lines.append("")
    lines.append(
        "**Bump-step reference.** Per-run bs=12→24 bump steps and the analysis of "
        "how earlier estimates over-counted samples by 0.6–1.5M are tabulated in "
        "[`n256_bump_steps.md`](n256_bump_steps.md)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
