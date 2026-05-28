"""Claim 5 deep-dive: T-D crossover analysis.

1. Compile β%, #clusters, pwTM step-matched (Δ vs baseline) for all configs.
2. Identify the T-D crossover step per regime (where REPA #clusters Δ flips + → -).
3. For ALL claim metrics, split win-fraction into BEFORE vs AFTER the crossover.

Hypothesis under test: "REPA improves T-D until a point, then does worse."
If true, we expect: before-crossover win fractions high, after-crossover low — and
we can check whether OTHER metrics also degrade after that point (i.e., is the cliff
a T-D-specific phenomenon or a general 'REPA advantage expires' inflection?).

Output: docs/research/proteina_td_crossover.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CONV = {
    "n256_pdb": ROOT
    / "evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl",
    "n256_afdb": ROOT
    / "evaluation/proteina/generation/results/paper/n256_convergence_afdb/sweep_results.clean.jsonl",
    "n128_pdb": ROOT
    / "evaluation/proteina/generation/results/paper/n128_convergence_pdb/sweep_results.clean.jsonl",
    "n128_afdb": ROOT
    / "evaluation/proteina/generation/results/paper/n128_convergence_afdb/sweep_results.clean.jsonl",
}
GROUPS = {
    "n256_pdb": (
        "baseline_256_bs24_2gpu",
        [
            ("L4-GN", "repa_l4_256_per_residue_bs24_2gpu"),
            ("L9-GN", "repa_l9_256_per_residue_bs24_2gpu"),
            ("L4-rand", "repa_l4_256_per_residue_random_bs24_2gpu"),
            ("L4-MPNN", "repa_mpnn_l4_256_per_residue"),
            ("L9-MPNN", "repa_mpnn_l9_256_per_residue"),
        ],
    ),
    "n256_afdb": (
        "baseline_afdb_256",
        [
            ("L4-GN", "repa_l4_afdb_256"),
            ("L9-GN", "repa_l9_afdb_256"),
            ("L9-MPNN", "repa_mpnn_l9_afdb_256"),
        ],
    ),
    "n128_pdb": (
        "baseline_128_bs80",
        [
            ("L4-GN", "repa_l4_128_bs80"),
            ("L9-GN", "repa_l9_128_bs80"),
            ("L4-rand", "repa_l4_128_random"),
            ("L4-MPNN", "repa_mpnn_l4_128_bs80"),
            ("L9-MPNN", "repa_mpnn_l9_128_bs80_2gpu"),
        ],
    ),
    "n128_afdb": (
        "baseline_afdb_128_bs80",
        [
            ("L4-GN", "repa_l4_afdb_128_bs80"),
            ("L4-MPNN", "repa_mpnn_l4_afdb_128_bs80"),
            ("L9-MPNN", "repa_mpnn_l9_afdb_128_bs80_2gpu"),
        ],
    ),
}

# All metrics across the 4 claims, for the before/after split.
ALL_METRICS = [
    # (label, key, lower_is_better)
    ("FID-PDB", "_res_PDB_FID", True),
    ("FID-AFDB", "_res_AFDB_FID", True),
    ("fJSD-A", "_res_PDB_fJSD_A", True),
    ("fJSD-C", "_res_PDB_fJSD_C", True),
    ("fS-A", "_res_fS_A", False),
    ("Des%", "_res_designability_rate", False),
    ("scRMSD", "_res_scRMSD_mean", True),
    ("pLDDT", "_res_plddt_mean", False),
    ("ssJSD2D", "_res_ss_jsd_pdb_designable_2d", True),
    ("β%", "_res_ss_frac_E_designable", False),
    ("#Clust", "_res_diversity_clusters_total", False),
    ("pwTM", "_res_diversity_pairwise_tm_mean", True),
    ("Nov-PDB", "_res_novelty_foldseek_pdb_rate", False),
]

TD_CLUSTER_KEY = "_res_diversity_clusters_total"


def is_default(r):
    sm, sn = r.get("sampling_mode"), r.get("sc_scale_noise")
    return (sm == "sc" and sn == 0.45) or (sm is None and sn is None)


def load(p):
    return [json.loads(ln) for ln in p.open()] if p.exists() else []


def grab(rows, pre, step):
    acc = defaultdict(list)
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(pre):
            continue
        if run != pre and "_step" not in run.split(pre)[-1]:
            continue
        if not is_default(r):
            continue
        if r.get("step") != step:
            continue
        for k, v in r.items():
            if isinstance(v, (int, float)) and k.startswith("_res_"):
                acc[k].append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def steps_of(rows, pre):
    out = set()
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(pre):
            continue
        if run != pre and "_step" not in run.split(pre)[-1]:
            continue
        if not is_default(r):
            continue
        s = r.get("step")
        if s:
            out.add(s)
    return sorted(out)


def find_crossover(rows, bp, vp):
    """Find the step where #clusters Δ (REPA - baseline) flips from + to - (last
    positive step and first negative step). Returns (last_pos, first_neg) or None."""
    common = sorted(set(steps_of(rows, bp)) & set(steps_of(rows, vp)))
    deltas = []
    for s in common:
        b = grab(rows, bp, s)
        v = grab(rows, vp, s)
        cb = b.get(TD_CLUSTER_KEY)
        cv = v.get(TD_CLUSTER_KEY)
        if cb is None or cv is None:
            continue
        deltas.append((s, cv - cb))
    # Find first step where delta goes negative and stays mostly negative
    last_pos, first_neg = None, None
    for s, d in deltas:
        if d > 0:
            last_pos = s
        elif d < 0 and first_neg is None and last_pos is not None:
            first_neg = s
    return deltas, last_pos, first_neg


def fmt_delta(v, b, lower, scale=1.0):
    if v is None or b is None:
        return ""
    raw = (v - b) * scale
    signed = -raw if lower else raw
    a = abs(signed)
    s = (
        f"{signed:+.0f}"
        if a >= 100
        else (f"{signed:+.1f}" if a >= 10 else f"{signed:+.2f}")
    )
    return s + ("✓" if signed > 0 else ("✗" if signed < 0 else "="))


def main():
    out = [
        "# Claim 5 — T-D crossover analysis",
        "",
        "Compiles β% / #clusters / pwTM step-matched across all configs, identifies the T-D crossover step (where REPA #clusters Δ flips + → −), and splits all-metric win fractions into before/after that step.",
        "",
        "Δ = REPA − baseline, sign-corrected (✓ = REPA better). Crossover from #clusters trajectory.",
        "",
        "---",
        "",
    ]

    # Section 1: T-D metric trajectories
    out.append("## 1. T-D metric trajectories (Δ vs baseline at same step)\n")
    crossover_by_regime_variant = {}
    for regime, (bp, variants) in GROUPS.items():
        rows = load(CONV[regime])
        if not rows:
            continue
        out.append(f"### {regime.upper()}\n")
        for tdlabel, tdkey, tdlower in [
            ("#Clusters ↑", TD_CLUSTER_KEY, False),
            ("pwTM ↓", "_res_diversity_pairwise_tm_mean", True),
            ("β% ↑", "_res_ss_frac_E_designable", False),
        ]:
            out.append(f"**{tdlabel}**\n")
            # union of steps
            allsteps = sorted(
                {
                    s
                    for _, vp in variants
                    for s in (set(steps_of(rows, bp)) & set(steps_of(rows, vp)))
                }
            )
            header = ["variant"] + [f"{s//1000}K" for s in allsteps]
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "|".join(["---"] * len(header)) + "|")
            for vlabel, vp in variants:
                vsteps = set(steps_of(rows, bp)) & set(steps_of(rows, vp))
                row = [vlabel]
                scale = 100.0 if tdkey == "_res_ss_frac_E_designable" else 1.0
                for s in allsteps:
                    if s not in vsteps:
                        row.append("")
                        continue
                    b = grab(rows, bp, s)
                    v = grab(rows, vp, s)
                    row.append(fmt_delta(v.get(tdkey), b.get(tdkey), tdlower, scale))
                out.append("| " + " | ".join(row) + " |")
            out.append("")

        # crossover per variant
        out.append(
            "**T-D crossover (from #clusters)** — last step REPA wins / first step REPA loses:\n"
        )
        for vlabel, vp in variants:
            deltas, last_pos, first_neg = find_crossover(rows, bp, vp)
            crossover_by_regime_variant[(regime, vlabel)] = (last_pos, first_neg)
            cross_str = ""
            if last_pos is not None and first_neg is not None:
                cross_str = f"between {last_pos//1000}K and {first_neg//1000}K"
            elif last_pos is not None and first_neg is None:
                cross_str = f"never crosses (stays ✓ through {last_pos//1000}K)"
            elif first_neg is not None:
                cross_str = (
                    f"crosses before first common step ({first_neg//1000}K already ✗)"
                )
            else:
                cross_str = "insufficient data"
            out.append(f"- {vlabel}: {cross_str}")
        out.append("")

    # Section 2: regime-level crossover decision
    out.append("\n---\n## 2. Regime-level crossover step (for before/after split)\n")
    # Use GearNet variants to define the regime crossover (main subject)
    REGIME_CROSSOVER = {
        "n256_pdb": 850000,  # L4/L9-GN cross between 700K and 1000K
        "n256_afdb": 150000,  # L4/L9-GN cross between 100K and 200K
        "n128_pdb": None,  # most variants stay ✓ through 600-700K — no clean crossover in range
        "n128_afdb": 50000,  # REPA T-D negative from first ckpt (100K already ✗)
    }
    out.append("| Regime | Crossover step | Basis |")
    out.append("|---|---|---|")
    out.append(
        "| n256 PDB | **~850K** | L4-GN & L9-GN #clusters Δ flip between 700K (+) and 1000K (−) |"
    )
    out.append(
        "| n256 AFDB | **~150K** | L4-GN & L9-GN flip between 100K (+) and 200K (−); MPNN never flips |"
    )
    out.append(
        "| n128 PDB | none in range | most variants stay T-D-positive through 600-700K (baseline hasn't overtaken yet) |"
    )
    out.append("| n128 AFDB | <100K | REPA T-D-negative from first ckpt |")
    out.append("")
    out.append(
        "**The crossover step is NOT universal** — it tracks when the *baseline's* #clusters growth overtakes REPA's plateau. PDB baseline keeps growing → crossover ~850K (n256) / not-yet (n128). AFDB baseline is high from early → crossover ~150K (n256) / immediate (n128).\n"
    )

    # Section 3: before/after win-fraction for ALL metrics (n256_pdb and n256_afdb where crossover is clear)
    out.append("\n---\n## 3. Before/after-crossover win fractions for ALL metrics\n")
    out.append(
        "For regimes with a clear crossover. `before` = steps ≤ crossover, `after` = steps > crossover. Each cell = (#REPA-wins / #comparisons).\n"
    )

    for regime in ["n256_pdb", "n256_afdb"]:
        xover = REGIME_CROSSOVER[regime]
        rows = load(CONV[regime])
        bp, variants = GROUPS[regime]
        out.append(f"### {regime.upper()} (crossover ~{xover//1000}K)\n")
        out.append(
            "| Variant | window | " + " | ".join(m[0] for m in ALL_METRICS) + " |"
        )
        out.append("|" + "|".join(["---"] * (len(ALL_METRICS) + 2)) + "|")
        for vlabel, vp in variants:
            common = sorted(set(steps_of(rows, bp)) & set(steps_of(rows, vp)))
            for window_name, window_steps in [
                ("before", [s for s in common if s <= xover]),
                ("after", [s for s in common if s > xover]),
            ]:
                cells = []
                for mlabel, mkey, mlower in ALL_METRICS:
                    wins, n = 0, 0
                    for s in window_steps:
                        b = grab(rows, bp, s)
                        v = grab(rows, vp, s)
                        bv = b.get(mkey)
                        vv = v.get(mkey)
                        if bv is None or vv is None:
                            continue
                        n += 1
                        d = vv - bv
                        if mlower:
                            d = -d
                        if d > 0:
                            wins += 1
                    cells.append(f"{wins}/{n}" if n else "—")
                out.append(f"| {vlabel} | {window_name} | " + " | ".join(cells) + " |")
            out.append("| | | " + " | ".join([""] * len(ALL_METRICS)) + " |")
        out.append("")

    out_path = ROOT / "docs/research/proteina_td_crossover.md"
    out_path.write_text("\n".join(out))
    print(f"Wrote {out_path}  ({len(out)} lines)")


if __name__ == "__main__":
    main()
