"""Generation-vs-representation envelope — REPA paper Fig 3c analog.

Scatter:
    x = best CATH-T probe accuracy across layers at (run, step)
    y = designability rate at the same (run, step)
Each point is one checkpoint; lines connect same-run points in step order to
visualise per-run trajectories. REPA variants should sit above-right of the
baseline curve (better generation AND better representations).

Output: ``evaluation/proteina/joint/figures/paper/n256_convergence/gen_vs_rep_envelope.{png,pdf}``
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
GEN_RESULTS = REPO_ROOT / "evaluation/proteina/generation/results/paper"
REP_RESULTS = REPO_ROOT / "evaluation/proteina/representation/results/paper"
FIG_OUT = REPO_ROOT / "evaluation/proteina/joint/figures/paper/n256_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": {
        "gen_jsonl": GEN_RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl",
        "rep_csv": REP_RESULTS
        / "n256_convergence_cath_if_dih_pdb"
        / "pretrained_sweep_results.csv",
    },
    "AFDB": {
        "gen_jsonl": GEN_RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl",
        "rep_csv": REP_RESULTS
        / "n256_convergence_cath_if_dih_afdb"
        / "pretrained_sweep_results.csv",
    },
}

RUN_FAMILIES = {
    "PDB": [
        ("baseline_256_bs24_2gpu", "Baseline (PDB)", "tab:blue", "o"),
        ("repa_l4_256_per_residue_bs24_2gpu", "REPA L4 GearNet (PDB)", "tab:red", "s"),
        (
            "repa_l9_256_per_residue_bs24_2gpu",
            "REPA L9 GearNet (PDB)",
            "tab:orange",
            "s",
        ),
        ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN (PDB)", "tab:red", "^"),
    ],
    "AFDB": [
        ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "o"),
        ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "s"),
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB)", "tab:orange", "s"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "^"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:orange", "^"),
    ],
}


def fnum(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_gen(path: Path) -> Dict[Tuple[str, int], float]:
    """Return {(run, step): designability_rate} for rows without errors."""
    out: Dict[Tuple[str, int], float] = {}
    if not path.exists():
        return out
    for line in path.open():
        r = json.loads(line)
        if r.get("error", "NONE") != "NONE":
            continue
        run = r.get("run")
        step = r.get("step")
        des = r.get("_res_designability_rate")
        if run is None or step is None or des is None:
            continue
        out[(run, int(step))] = float(des)
    return out


def load_rep_best(
    path: Path,
    probe_kind: str = "cath",
    cath_level: str = "T",
    metric_col: str = "cath_accuracy",
) -> Dict[Tuple[str, int], float]:
    """Return {(run, step): max-over-layers metric}."""
    out: Dict[Tuple[str, int], float] = defaultdict(lambda: float("-inf"))
    if not path.exists():
        return {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("probe_kind") != probe_kind:
                continue
            if cath_level is not None and r.get("cath_level") != cath_level:
                continue
            run = r.get("run")
            s = fnum(r.get("step"))
            v = fnum(r.get(metric_col))
            if run is None or s is None or v is None:
                continue
            k = (run, int(s))
            if v > out[k]:
                out[k] = v
    return dict(out)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    for ax, (ds_name, paths) in zip(axes, DATASETS.items()):
        gen_map = load_gen(paths["gen_jsonl"])
        rep_map = load_rep_best(paths["rep_csv"])
        for prefix, label, color, marker in RUN_FAMILIES[ds_name]:
            pts = []
            for (run, step), des in gen_map.items():
                if not run.startswith(prefix):
                    continue
                acc = rep_map.get((run, step))
                if acc is None or acc == float("-inf"):
                    continue
                pts.append((step, acc, des))
            if not pts:
                continue
            pts.sort()  # by step
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            sizes = [20 + 0.00006 * p[0] for p in pts]  # marker grows with step
            ax.plot(xs, ys, linewidth=1.0, color=color, alpha=0.6)
            ax.scatter(
                xs,
                ys,
                s=sizes,
                color=color,
                marker=marker,
                edgecolor="black",
                linewidth=0.4,
                label=label,
                zorder=3,
            )
        ax.set_xlabel("CATH-T probe accuracy (best layer)")
        ax.set_ylabel("Designability rate")
        ax.set_title(f"{ds_name}: generation vs representation envelope")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "n=256 — generation vs representation envelope (point size ∝ training step)\n"
        "Top-right = better on both axes; lines connect same-run checkpoints in step order.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = FIG_OUT / "gen_vs_rep_envelope.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
