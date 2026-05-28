"""TODO: cross-check the n256 700K T-D cliff against n=128.

Pulls per-(run, step) #clusters and Des% from the multi-seed n128 convergence
results at γ=0.45 only, aggregates across reps, and writes a JSON + a small
markdown summary. Looks for a sharp #clusters drop in the REPA-GearNet runs
(predicted cliff position: ~300K for n=128 vs ~700K for n=256 — same
proportional point of training).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[5]
SAMPLER = "sde_n0.45"

JSONL_PDB = (
    REPO_ROOT
    / "evaluation/proteina/generation/results/paper/n128_convergence_pdb/sweep_results.clean.jsonl"
)
JSONL_AFDB = (
    REPO_ROOT
    / "evaluation/proteina/generation/results/paper/n128_convergence_afdb/sweep_results.clean.jsonl"
)

OUT_JSON = (
    REPO_ROOT / "evaluation/proteina/generation/results/variance/n128_cliff_check.json"
)
OUT_MD = (
    REPO_ROOT / "evaluation/proteina/generation/results/variance/n128_cliff_check.md"
)


def aggregate(jsonl_path: Path):
    nclust = defaultdict(lambda: defaultdict(list))
    desr = defaultdict(lambda: defaultdict(list))
    for line in open(jsonl_path):
        d = json.loads(line)
        if d.get("sampler_tag") and d["sampler_tag"] != SAMPLER:
            continue
        run = d["run"]
        # Normalize trailing _stepXXXk into the run group: e.g.
        # baseline_128_bs80_step100k -> baseline_128_bs80
        run_base = run.rsplit("_step", 1)[0]
        step = int(d["step"])
        c = d.get("_res_diversity_clusters_total")
        r = d.get("_res_designability_rate")
        if c is not None:
            nclust[run_base][step].append(c)
        if r is not None:
            desr[run_base][step].append(r)
    return nclust, desr


def cliff_diagnose(steps_to_clust: dict[int, list[float]]) -> dict:
    """Find the largest single-step #clusters drop and characterize it."""
    by_step = sorted(steps_to_clust.items())
    if len(by_step) < 3:
        return {"verdict": "insufficient_data", "n_points": len(by_step)}
    means = [(s, mean(v)) for s, v in by_step]
    # Peak step
    peak_idx, (peak_step, peak_val) = max(enumerate(means), key=lambda kv: kv[1][1])
    if peak_idx == len(means) - 1:
        return {
            "verdict": "monotone_or_late_peak",
            "peak_step": peak_step,
            "peak_clust": peak_val,
            "n_points": len(means),
        }
    # post-peak min
    post = means[peak_idx + 1 :]
    post_min_step, post_min_val = min(post, key=lambda kv: kv[1])
    drop_abs = peak_val - post_min_val
    drop_rel = drop_abs / peak_val if peak_val > 0 else float("nan")
    # Does it recover? (max of post-min steps)
    recovery = max([v for s, v in post if s > post_min_step], default=post_min_val)
    return {
        "n_points": len(means),
        "trajectory": [(s, round(v, 1)) for s, v in means],
        "peak_step": peak_step,
        "peak_clust": round(peak_val, 1),
        "trough_step": post_min_step,
        "trough_clust": round(post_min_val, 1),
        "drop_abs": round(drop_abs, 1),
        "drop_rel": round(drop_rel, 2),
        "post_trough_recovery": round(recovery, 1),
        "verdict": "cliff"
        if drop_rel >= 0.40 and (recovery - post_min_val) < drop_abs / 2
        else "soft_decline"
        if drop_rel >= 0.20
        else "stable",
    }


def main():
    out = {}
    md_lines = ["# n=128 cliff check (γ=0.45, multi-seed aggregated)\n"]
    for dataset, jsonl in [("pdb", JSONL_PDB), ("afdb", JSONL_AFDB)]:
        nc, des = aggregate(jsonl)
        out[dataset] = {}
        md_lines.append(f"\n## {dataset.upper()}\n")
        md_lines.append(
            "| run group | trajectory (step → #clust mean) | verdict | drop |"
        )
        md_lines.append("|---|---|---|---|")
        for run_base in sorted(nc):
            diag = cliff_diagnose(nc[run_base])
            out[dataset][run_base] = diag
            traj = diag.get("trajectory", [])
            traj_str = ", ".join(f"{s//1000}k→{v}" for s, v in traj)
            drop = f"{diag.get('drop_abs','?')}/{diag.get('drop_rel','?')}"
            md_lines.append(f"| {run_base} | {traj_str} | {diag['verdict']} | {drop} |")

    OUT_JSON.write_text(json.dumps(out, indent=2))
    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")

    # Print a quick summary of cliffs
    print("\n=== Cliffs detected ===")
    for ds in out:
        for run, diag in out[ds].items():
            if diag.get("verdict") == "cliff":
                print(
                    f"  {ds:5} {run:<35}  peak={diag['peak_step']//1000}k ({diag['peak_clust']}) → trough={diag['trough_step']//1000}k ({diag['trough_clust']})  drop={diag['drop_rel']*100:.0f}%"
                )


if __name__ == "__main__":
    main()
