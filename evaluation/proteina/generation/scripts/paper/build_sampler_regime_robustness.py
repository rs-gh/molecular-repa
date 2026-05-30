"""Sampler-regime robustness of the convergence claims.

The narrative claims (proteina_narratives.md) are compiled at gamma=0.45 across
the training trajectory. This script asks: do those trends hold across the
sampler-noise regime (ODE, gamma in {0, 0.35, 0.45, 0.5, 1.0})?

For every baseline/REPA pair that has multi-gamma coverage it builds:
  1. a per-gamma win-rate table over the common step-matched steps, and
  2. a step-matched gamma x step delta grid per metric, and
  3. the single-step (700K) gamma row (the table the doc originally had).

Data sources (chosen for freshness/completeness):
  - sampler-ablation gammas (ode/0.0/0.35/0.5/1.0): the *.clean.jsonl in
    results/variance/n256{,_afdb}_sampler_ablation/ . The raw PDB jsonl is
    missing 0.35/0.5; the clean has all five. 1 rep/cell.
  - gamma=0.45: the convergence-sweep raw jsonl (results/paper/n256_convergence_*)
    deduped by mean over reps. Read raw (not clean) because the convergence raw
    is mutated more often than its clean snapshot. 3 reps/cell.

A pair is emitted only if it has >=2 non-0.45 sampler tags with step-matched
data, so MPNN pairs appear automatically once the afdb sampler-ablation-ext
jobs (and any future PDB MPNN ablation) land.

Run: python evaluation/proteina/generation/scripts/paper/build_sampler_regime_robustness.py
Prints a markdown block to stdout (splice into proteina_narratives.md).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
GEN = ROOT / "evaluation/proteina/generation/results"

GAMMAS = ["ode", "sde_n0.0", "sde_n0.35", "sde_n0.45", "sde_n0.5", "sde_n1.0"]
GLABEL = {
    "ode": "ODE",
    "sde_n0.0": "γ=0",
    "sde_n0.35": "γ=.35",
    "sde_n0.45": "**γ=.45**",
    "sde_n0.5": "γ=.5",
    "sde_n1.0": "γ=1",
}

# metric -> (jsonl key template, higher_is_better). {ds} -> PDB/AFDB.
METRICS = [
    ("Des", "_res_designability_rate", True),
    ("FID", "_res_{ds}_FID", False),
    ("fJSD-A", "_res_{ds}_fJSD_A", False),
    ("fJSD-C", "_res_{ds}_fJSD_C", False),
    ("ssJSD2D", "_res_ss_jsd_{ds_lc}_2d", False),
    ("β", "_res_ss_frac_E", True),
    ("scRMSD", "_res_scRMSD_mean", False),
    ("pLDDT", "_res_plddt_mean", True),
    ("#Clust", "_res_diversity_clusters_mean", True),
    ("pwTM", "_res_diversity_pairwise_tm_mean", False),
]

# dataset -> (ablation clean, convergence raw, baseline, [(label, repa_run), ...])
CONFIGS = {
    "PDB": (
        GEN / "variance/n256_sampler_ablation/sweep_results.clean.jsonl",
        GEN / "paper/n256_convergence_pdb/sweep_results.jsonl",
        "baseline_256_bs24_2gpu",
        [
            ("L9-GN", "repa_l9_256_per_residue_bs24_2gpu"),
            ("L4-GN", "repa_l4_256_per_residue_bs24_2gpu"),
            ("L4-MPNN", "repa_mpnn_l4_256_per_residue"),
            ("L9-MPNN", "repa_mpnn_l9_256_per_residue"),
        ],
    ),
    "AFDB": (
        GEN / "variance/n256_afdb_sampler_ablation/sweep_results.clean.jsonl",
        GEN / "paper/n256_convergence_afdb/sweep_results.jsonl",
        "baseline_afdb_256",
        [
            ("L4-GN", "repa_l4_afdb_256"),
            ("L9-GN", "repa_l9_afdb_256"),
            ("L9-MPNN", "repa_mpnn_l9_afdb_256"),
        ],
    ),
}


def load(path):
    out = []
    if not path.exists():
        return out
    for line in path.open():
        s = line.strip()
        if not s or s == "}":
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            pass
    return out


def fam(r):
    return r["run"].rsplit("_step", 1)[0]


def build_data(dataset):
    abl_path, conv_path, base, repas = CONFIGS[dataset]
    keys = {m: tmpl.format(ds=dataset, ds_lc=dataset.lower()) for m, tmpl, _ in METRICS}
    runs = {base} | {r for _, r in repas}
    agg = defaultdict(
        lambda: defaultdict(list)
    )  # (fam, step, gamma) -> metric -> [vals]

    for r in load(abl_path):
        if fam(r) not in runs or r.get("sampler_tag") not in GAMMAS:
            continue
        for m, _, _ in METRICS:
            v = r.get(keys[m])
            if v is not None:
                agg[(fam(r), r["step"], r["sampler_tag"])][m].append(v)
    for r in load(conv_path):
        if r.get("sampler_tag") != "sde_n0.45" or fam(r) not in runs:
            continue
        for m, _, _ in METRICS:
            v = r.get(keys[m])
            if v is not None:
                agg[(fam(r), r["step"], "sde_n0.45")][m].append(v)

    data = {
        k: {m: statistics.mean(vs) for m, vs in md.items()} for k, md in agg.items()
    }
    return data, base, repas


def gammas_present(data, base, repa):
    g = set()
    for f, s, gm in data:
        if f in (base, repa):
            g.add(gm)
    return [gm for gm in GAMMAS if gm in g]


def matched_steps(data, base, repa, gamma):
    bs = {s for (f, s, gm) in data if f == base and gm == gamma}
    rs = {s for (f, s, gm) in data if f == repa and gm == gamma}
    return bs & rs


def delta(data, base, repa, step, gamma, metric):
    bk, rk = (base, step, gamma), (repa, step, gamma)
    if bk in data and rk in data and metric in data[bk] and metric in data[rk]:
        return data[rk][metric] - data[bk][metric]
    return None


HIGHER = {m: hb for m, _, hb in METRICS}
DISPLAY = {"FID": "FID-1.1K"}


def is_win(metric, d):
    return (d > 0) if HIGHER[metric] else (d < 0)


def fmt_mark(metric, d):
    if d is None:
        return ""
    return "✓" if is_win(metric, d) else "✗"


def common_steps(data, base, repa, gs):
    """Steps present (step-matched) at *every* gamma in gs (the ablation gammas
    define the common set; gamma=0.45 typically has more steps)."""
    abl_gs = [g for g in gs if g != "sde_n0.45"]
    if not abl_gs:
        return set()
    sets = [matched_steps(data, base, repa, g) for g in abl_gs]
    common = set.intersection(*sets) if sets else set()
    return common


def emit():
    lines = []
    P = lines.append
    for dataset in ["PDB", "AFDB"]:
        data, base, repas = build_data(dataset)
        for label, repa in repas:
            gs = gammas_present(data, base, repa)
            non045 = [g for g in gs if g != "sde_n0.45"]
            if len(non045) < 2:
                P(
                    f"\n#### {dataset} — {label}: no multi-γ ablation data yet "
                    f"(have {[GLABEL[g].strip('*') for g in gs] or 'none'}). Skipped.\n"
                )
                continue
            common = common_steps(data, base, repa, gs)
            if not common:
                P(
                    f"\n#### {dataset} — {label}: no step-matched ablation steps. Skipped.\n"
                )
                continue
            common_sorted = sorted(common)
            P(f"\n#### {dataset} — REPA {label} vs baseline")
            P(
                f"\nWin-rate (REPA beats baseline) over step-matched steps "
                f"{[s // 1000 for s in common_sorted]}K · ablation γ are single-rep, γ=.45 is 3-rep mean.\n"
            )
            hdr = "| metric | " + " | ".join(GLABEL[g] for g in gs) + " |"
            P(hdr)
            P("|" + "---|" * (len(gs) + 1))
            for m, _, hb in METRICS:
                cells = []
                for g in gs:
                    # win-rate uses the same common step set for every gamma so columns compare
                    w = tot = 0
                    for s in common:
                        d = delta(data, base, repa, s, g, m)
                        if d is not None:
                            tot += 1
                            w += int(is_win(m, d))
                    cells.append(f"{w}/{tot}" if tot else "–")
                arrow = "↑" if hb else "↓"
                P(f"| {DISPLAY.get(m, m)} {arrow} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def emit_700k():
    """Single-step (700K) gamma row per dataset — the corrected version of the
    table the doc originally carried (its ssJSD2D/β columns were wrong)."""
    lines = []
    P = lines.append
    STEP = 700000
    show = [
        ("Des", "_res", True),
        ("FID", "", False),
        ("fJSD-A", "", False),
        ("fJSD-C", "", False),
        ("ssJSD2D", "", False),
        ("β", "", True),
        ("#Clust", "", True),
    ]
    for dataset in ["PDB", "AFDB"]:
        data, base, repas = build_data(dataset)
        label, repa = repas[0]  # the GearNet pair with full coverage
        gs = [g for g in GAMMAS if delta(data, base, repa, STEP, g, "FID") is not None]
        if not gs:
            continue
        P(
            f"\n**{dataset} 700K step-matched (baseline vs REPA {label}), Δ = REPA−base:**\n"
        )
        P("| γ | " + " | ".join(DISPLAY.get(m, m) for m, _, _ in show) + " |")
        P("|" + "---|" * (len(show) + 1))
        for g in gs:
            cells = []
            for m, _, _ in show:
                d = delta(data, base, repa, STEP, g, m)
                if d is None:
                    cells.append("–")
                else:
                    cells.append(f"{fmt_mark(m, d)}{d:+.2f}")
            P(f"| {GLABEL[g]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(emit())
    print("\n\n=====  700K single-step tables  =====")
    print(emit_700k())
