"""Compile data for all 4 narrative claims across every dataset×scale×encoder×layer config we have.

For each claim's metric set, produces a per-regime markdown table where:
  - Rows: training steps (union across all variants in that regime)
  - Columns: each REPA variant, showing Δ vs same-step baseline
  - Convention: positive Δ = REPA *better* than baseline (sign-corrected per metric direction)

Output: docs/research/proteina_claims_compilation.md

Regimes handled:
  - {n128, n256} × {PDB, AFDB} at γ=0.45 (paper-default sampler, multi-rep where available)
  - Sampler-regime robustness: n256 sampler ablation jsonls (PDB and AFDB) at all 5 γ values

Each REPA variant column shows the absolute Δ. Where the baseline doesn't have data at that
exact step, we look for the nearest baseline step within ±100K and flag with (~) — cleanly
handles e.g. n128_afdb where baseline is sparse.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]

CONVERGENCE_JSONLS = {
    "n256_pdb": ROOT
    / "evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl",
    "n256_afdb": ROOT
    / "evaluation/proteina/generation/results/paper/n256_convergence_afdb/sweep_results.clean.jsonl",
    "n128_pdb": ROOT
    / "evaluation/proteina/generation/results/paper/n128_convergence_pdb/sweep_results.clean.jsonl",
    "n128_afdb": ROOT
    / "evaluation/proteina/generation/results/paper/n128_convergence_afdb/sweep_results.clean.jsonl",
}
SAMPLER_JSONLS = {
    "n256_pdb_sampler": ROOT
    / "evaluation/proteina/generation/results/variance/n256_sampler_ablation/sweep_results.clean.jsonl",
    "n256_afdb_sampler": ROOT
    / "evaluation/proteina/generation/results/variance/n256_afdb_sampler_ablation/sweep_results.clean.jsonl",
}

# Variants by regime. (key, baseline-prefix, list of (variant_label, run_prefix)).
# Variant labels chosen to be compact for table headers.
VARIANT_GROUPS = {
    "n256_pdb": {
        "baseline": "baseline_256_bs24_2gpu",
        "variants": [
            ("L4-GN", "repa_l4_256_per_residue_bs24_2gpu"),
            ("L9-GN", "repa_l9_256_per_residue_bs24_2gpu"),
            ("L4-rand", "repa_l4_256_per_residue_random_bs24_2gpu"),
            ("L4-MPNN", "repa_mpnn_l4_256_per_residue"),
            ("L9-MPNN", "repa_mpnn_l9_256_per_residue"),
        ],
    },
    "n256_afdb": {
        "baseline": "baseline_afdb_256",
        "variants": [
            ("L4-GN", "repa_l4_afdb_256"),
            ("L9-GN", "repa_l9_afdb_256"),
            ("L9-MPNN", "repa_mpnn_l9_afdb_256"),
        ],
    },
    "n128_pdb": {
        "baseline": "baseline_128_bs80",
        "variants": [
            ("L4-GN", "repa_l4_128_bs80"),
            ("L9-GN", "repa_l9_128_bs80"),
            ("L4-rand", "repa_l4_128_random"),
            ("L4-MPNN", "repa_mpnn_l4_128_bs80"),
            ("L9-MPNN", "repa_mpnn_l9_128_bs80_2gpu"),
        ],
    },
    "n128_afdb": {
        "baseline": "baseline_afdb_128_bs80",
        "variants": [
            ("L4-GN", "repa_l4_afdb_128_bs80"),
            ("L4-MPNN", "repa_mpnn_l4_afdb_128_bs80"),
            ("L9-MPNN", "repa_mpnn_l9_afdb_128_bs80_2gpu"),
        ],
    },
}

# Claim definitions: list of (display_name, metric_key, scale, lower_is_better)
CLAIMS = {
    "Claim 1 — REPA accelerates whole-distribution learning (T-W / S-W)": [
        ("FID-1.1K (PDB)", "_res_PDB_FID", 1.0, True),
        ("FID-1.1K (AFDB)", "_res_AFDB_FID", 1.0, True),
        ("fJSD-A", "_res_PDB_fJSD_A", 1.0, True),
        ("fJSD-T", "_res_PDB_fJSD_T", 1.0, True),
        ("fJSD-C", "_res_PDB_fJSD_C", 1.0, True),
        ("fS-A ↑", "_res_fS_A", 1.0, False),
        ("fS-C ↑", "_res_fS_C", 1.0, False),
    ],
    "Claim 2 — REPA preserves SS balance (β-content / ssJSD)": [
        ("β % ↑", "_res_ss_frac_E_designable", 100.0, False),
        ("α %", "_res_ss_frac_H_designable", 100.0, None),  # neutral
        ("ssJSD-2D", "_res_ss_jsd_pdb_designable_2d", 1.0, True),
        ("ssJSD-2D-AFDB", "_res_ss_jsd_afdb_designable_2d", 1.0, True),
        ("fJSD-C", "_res_PDB_fJSD_C", 1.0, True),
    ],
    "Claim 3 — REPA reaches good designability faster": [
        ("Des% ↑", "_res_designability_rate", 100.0, False),
        ("scRMSD", "_res_scRMSD_mean", 1.0, True),
        ("pLDDT ↑", "_res_plddt_mean", 1.0, False),
        ("TM-self ↑", "_res_tm_score_self_mean", 1.0, False),
    ],
    "Claim 4 — REPA improves novelty (where measurable)": [
        ("Nov-PDB% ↑", "_res_novelty_foldseek_pdb_rate", 100.0, False),
        ("Nov-AFDB% ↑", "_res_novelty_foldseek_afdb_swissprot_rate", 100.0, False),
        ("max-TM-PDB", "_res_novelty_foldseek_pdb_max_tm_mean", 1.0, True),
        ("max-TM-AFDB", "_res_novelty_foldseek_afdb_swissprot_max_tm_mean", 1.0, True),
    ],
}


def load_rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.open()]


def grab_meta(rows, run_prefix, step, sampler_filter=None):
    """Return mean of each metric across reps for (run_prefix, step, optional sampler)."""
    acc = defaultdict(list)
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        if run != run_prefix and "_step" not in run.split(run_prefix)[-1]:
            continue
        if r.get("step") != step:
            continue
        if sampler_filter is not None:
            sm, sn = r.get("sampling_mode"), r.get("sc_scale_noise")
            # Legacy rows (None, None) were generated with the default sampler
            # (γ=0.45). Accept them when the target filter is the default.
            is_default_target = sampler_filter == ("sc", 0.45)
            is_legacy = sm is None and sn is None
            if not (is_legacy and is_default_target):
                if sm != sampler_filter[0] or sn != sampler_filter[1]:
                    continue
        for k, v in r.items():
            if isinstance(v, (int, float)) and k.startswith("_res_"):
                acc[k].append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def find_baseline_at_step(
    rows, baseline_prefix, step, sampler_filter=None, window=200000
):
    """Return (nearest_baseline_step, metrics_dict, is_exact) for the closest baseline step within window."""
    # Collect all baseline steps with data at this sampler
    candidates = set()
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(baseline_prefix):
            continue
        if run != baseline_prefix and "_step" not in run.split(baseline_prefix)[-1]:
            continue
        if sampler_filter is not None:
            sm, sn = r.get("sampling_mode"), r.get("sc_scale_noise")
            # Legacy rows (None, None) were generated with the default sampler
            # (γ=0.45). Accept them when the target filter is the default.
            is_default_target = sampler_filter == ("sc", 0.45)
            is_legacy = sm is None and sn is None
            if not (is_legacy and is_default_target):
                if sm != sampler_filter[0] or sn != sampler_filter[1]:
                    continue
        s = r.get("step")
        if s is None:
            continue
        candidates.add(int(s))
    if not candidates:
        return None, {}, False
    if step in candidates:
        return step, grab_meta(rows, baseline_prefix, step, sampler_filter), True
    nearest = min(candidates, key=lambda s: abs(s - step))
    if abs(nearest - step) > window:
        return None, {}, False
    return nearest, grab_meta(rows, baseline_prefix, nearest, sampler_filter), False


def fmt_delta(val, base_val, lower_is_better, scale=1.0):
    if val is None or base_val is None:
        return ""
    raw = (val - base_val) * scale
    if lower_is_better is True:
        # negative raw delta = REPA better = positive sign
        signed = -raw
    elif lower_is_better is False:
        signed = raw
    else:
        # neutral metric: just show raw delta
        signed = raw
    arrow = "✓" if signed > 0 else ("✗" if signed < 0 else "=")
    # format
    a = abs(signed)
    if a >= 100:
        s = f"{signed:+.0f}"
    elif a >= 10:
        s = f"{signed:+.1f}"
    else:
        s = f"{signed:+.2f}"
    return f"{s}{arrow}"


def render_claim_for_regime(claim_metrics, regime_key, rows, sampler_filter):
    """Build markdown table for one claim × one regime × one sampler.
    Rows = training steps (union over variants), Cols = variants (Δ from baseline)."""
    cfg = VARIANT_GROUPS[regime_key]
    baseline_prefix = cfg["baseline"]
    variants = cfg["variants"]

    # Collect union of steps where ANY variant has data
    variant_steps = defaultdict(set)
    for vlabel, vprefix in variants:
        for r in rows:
            run = r.get("run", "")
            if not run.startswith(vprefix):
                continue
            if run != vprefix and "_step" not in run.split(vprefix)[-1]:
                continue
            if sampler_filter is not None:
                sm, sn = r.get("sampling_mode"), r.get("sc_scale_noise")
                is_default_target = sampler_filter == ("sc", 0.45)
                is_legacy = sm is None and sn is None
                if not (is_legacy and is_default_target):
                    if sm != sampler_filter[0] or sn != sampler_filter[1]:
                        continue
            s = r.get("step")
            if s is None:
                continue
            variant_steps[vlabel].add(int(s))
    all_steps = sorted({s for steps in variant_steps.values() for s in steps})
    if not all_steps:
        return None

    lines = []
    # Header: metric, step, then one column per variant
    len(variants)

    # Build separate sub-table per metric (each metric is its own row block, cols=variants)
    for mlabel, mkey, mscale, lower_is_better in claim_metrics:
        lines.append(f"\n**{mlabel}** — {regime_key.replace('_', ' ').upper()}\n")
        header = ["step", "baseline"] + [v[0] for v in variants]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for step in all_steps:
            # Baseline value
            b_step, b_metrics, b_exact = find_baseline_at_step(
                rows, baseline_prefix, step, sampler_filter
            )
            b_val = b_metrics.get(mkey)
            # Format baseline cell
            if b_val is None:
                b_cell = "—"
            else:
                bv = b_val * mscale
                if abs(bv) >= 100:
                    bs = f"{bv:.0f}"
                elif abs(bv) >= 10:
                    bs = f"{bv:.1f}"
                else:
                    bs = f"{bv:.2f}"
                b_cell = bs if b_exact else f"{bs}~"
            row = [f"{step//1000}K", b_cell]
            # Each variant column: Δ from baseline at nearest step
            for vlabel, vprefix in variants:
                if step not in variant_steps.get(vlabel, set()):
                    row.append("")
                    continue
                v_metrics = grab_meta(rows, vprefix, step, sampler_filter)
                v_val = v_metrics.get(mkey)
                if v_val is None or b_val is None:
                    row.append("")
                    continue
                row.append(fmt_delta(v_val, b_val, lower_is_better, mscale))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    out = [
        "# Proteína narrative claims — full data compilation",
        "",
        "Auto-generated from `build_claims_compilation.py`. For each claim, shows the metric trajectory across training steps with **Δ vs baseline at the SAME step** for each REPA variant.",
        "",
        "**Reading**: Each cell is `signedΔ` followed by ✓ (REPA better than baseline at that step) or ✗ (REPA worse) or = (tied). Signs are corrected per metric direction. `~` next to a baseline value means the baseline at the *nearest* step within ±200K was used (no exact same-step baseline data).",
        "",
        "Variants: L4/L9 = REPA at layer 4/9. GN = GearNet encoder. MPNN = MPNN encoder. rand = random-init GearNet (falsifier control).",
        "",
        "All entries at γ=0.45 (paper-default SDE noise scale) unless flagged otherwise.",
        "",
        "---",
        "",
    ]

    sampler_default = ("sc", 0.45)
    # Main table set: per claim, per (n_scale, dataset) at γ=0.45
    for claim_name, metrics in CLAIMS.items():
        out.append(f"## {claim_name}")
        out.append("")
        for regime in ["n256_pdb", "n256_afdb", "n128_pdb", "n128_afdb"]:
            jsonl_path = CONVERGENCE_JSONLS[regime]
            rows = load_rows(jsonl_path)
            if not rows:
                out.append(f"_no data for {regime}_")
                continue
            block = render_claim_for_regime(metrics, regime, rows, sampler_default)
            if block:
                out.append(f"### {regime.replace('_', ' ').upper()} (γ=0.45)")
                out.append(block)
            else:
                out.append(f"_no rows for {regime} at γ=0.45_")
        out.append("\n---\n")

    # Sampler-regime robustness (n256 only — that's where multi-sampler data exists)
    out.append("## Sampler-regime robustness (n256, all 5 γ values)\n")
    out.append("Available REPA variants for sampler ablations:")
    out.append(
        "- **n256_pdb_sampler**: baseline_256_bs24_2gpu × repa_l9_256_per_residue_bs24_2gpu"
    )
    out.append("- **n256_afdb_sampler**: baseline_afdb_256 × repa_l4_afdb_256")
    out.append("")

    # Override the variant config for sampler tables — only one REPA per regime
    sampler_cfg = {
        "n256_pdb_sampler": {
            "baseline": "baseline_256_bs24_2gpu",
            "variants": [("L9-GN", "repa_l9_256_per_residue_bs24_2gpu")],
        },
        "n256_afdb_sampler": {
            "baseline": "baseline_afdb_256",
            "variants": [("L4-GN", "repa_l4_afdb_256")],
        },
    }
    SAMPLERS = [
        ("ODE", ("vf", None)),
        ("γ=0.0", ("sc", 0.0)),
        ("γ=0.35", ("sc", 0.35)),
        ("γ=0.45", ("sc", 0.45)),
        ("γ=0.5", ("sc", 0.5)),
        ("γ=1.0", ("sc", 1.0)),
    ]

    # Pair each sampler-regime with the convergence jsonl that contains γ=0.45 data
    SAMPLER_REGIME_TO_CONVERGENCE = {
        "n256_pdb_sampler": "n256_pdb",
        "n256_afdb_sampler": "n256_afdb",
    }

    for claim_name, metrics in CLAIMS.items():
        out.append(f"### {claim_name}\n")
        for regime, srows_key in [
            ("n256_pdb_sampler", "n256_pdb_sampler"),
            ("n256_afdb_sampler", "n256_afdb_sampler"),
        ]:
            jsonl_path = SAMPLER_JSONLS[srows_key]
            rows = load_rows(jsonl_path)
            # Merge γ=0.45 data from the corresponding convergence jsonl
            conv_path = CONVERGENCE_JSONLS[SAMPLER_REGIME_TO_CONVERGENCE[srows_key]]
            rows = rows + load_rows(conv_path)
            if not rows:
                out.append(f"_no data for {regime}_")
                continue
            cfg = sampler_cfg[regime]
            baseline_prefix = cfg["baseline"]
            variants = cfg["variants"]

            out.append(f"#### {regime.replace('_', ' ').upper()}")
            for mlabel, mkey, mscale, lower_is_better in metrics:
                # Get union of steps
                all_steps = set()
                for r in rows:
                    run = r.get("run", "")
                    for vlabel, vprefix in variants + [("baseline", baseline_prefix)]:
                        if run.startswith(vprefix):
                            s = r.get("step")
                            if s:
                                all_steps.add(int(s))
                            break
                all_steps = sorted(all_steps)
                if not all_steps:
                    continue
                out.append(f"\n**{mlabel}**\n")
                header = ["step"] + [s[0] for s in SAMPLERS]
                out.append("| " + " | ".join(header) + " |")
                out.append("|" + "|".join(["---"] * len(header)) + "|")
                for step in all_steps:
                    row = [f"{step//1000}K"]
                    for slabel, sfilt in SAMPLERS:
                        # Get baseline + variant at this step+sampler
                        b_step, b_metrics, b_exact = find_baseline_at_step(
                            rows, baseline_prefix, step, sfilt
                        )
                        v_prefix = variants[0][1]  # only one variant per sampler table
                        v_metrics = grab_meta(rows, v_prefix, step, sfilt)
                        v_val = v_metrics.get(mkey)
                        b_val = b_metrics.get(mkey)
                        if v_val is None or b_val is None:
                            row.append("")
                        else:
                            row.append(fmt_delta(v_val, b_val, lower_is_better, mscale))
                    out.append("| " + " | ".join(row) + " |")
                out.append("")
        out.append("\n---\n")

    out_path = ROOT / "docs/research/proteina_claims_compilation.md"
    out_path.write_text("\n".join(out))
    print(f"Wrote {out_path}  ({len(out)} lines)")


if __name__ == "__main__":
    main()
