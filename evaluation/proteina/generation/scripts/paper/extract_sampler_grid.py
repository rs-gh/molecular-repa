"""Extract the seed-averaged sampler grid for the Ch6 full-variant tables (6.6/6.7).

Reads results/variance/n256_sampler_ablation/sweep_results.clean.jsonl and prints,
for every PDB n=256 variant x sampler_tag, the mean FPSD (= _res_PDB_FID) and
designability (= _res_designability_rate), seed-averaged. This is the
layout-independent data behind:
  - Table 6.6 (table_sampler.tex): FPSD + Des across {ode, sde 0/0.35/0.45/0.5/1.0}
  - Table 6.7 (table_ode.tex): the ODE column only.

Run from the repo root:
    python evaluation/proteina/generation/scripts/paper/extract_sampler_grid.py
    python .../extract_sampler_grid.py --step 700000   # restrict to one step

Variants (run-name prefix -> label), PDB headline is MPNN-L9:
    baseline_256_bs24_2gpu              -> Baseline
    repa_l4_256_per_residue (no rnd/mpnn)-> GN-L4
    repa_l9_256_per_residue (no rnd/mpnn)-> GN-L9
    repa_mpnn_l4_256                    -> MPNN-L4
    repa_mpnn_l9_256                    -> MPNN-L9   (headline)
    repa_l4_256_per_residue_random      -> random-L4

Use after the SL3 sampler sweeps land + clean_variance_jsonl.py is re-run.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

# The grid spans TWO files: the sampler-ablation file holds {ode, sde 0/0.35/
# 0.5/1.0} (single-seed), and the default gamma=0.45 column (3-seed) lives in the
# convergence file. We merge both.
CLEAN_GLOBS = [
    "evaluation/proteina/generation/results/variance/"
    "n256_sampler_ablation/sweep_results.clean.jsonl",
    "evaluation/proteina/generation/results/paper/"
    "n256_convergence_pdb/sweep_results.clean.jsonl",
    "evaluation/proteina/generation/results/variance/"
    "n256_convergence_pdb*/sweep_results.clean.jsonl",
]

TAGS = ["ode", "sde_n0.0", "sde_n0.35", "sde_n0.45", "sde_n0.5", "sde_n1.0"]
TAG_LABEL = {
    "ode": "ODE",
    "sde_n0.0": "g=0",
    "sde_n0.35": "g=0.35",
    "sde_n0.45": "g=0.45",
    "sde_n0.5": "g=0.5",
    "sde_n1.0": "g=1.0",
}
VAR_ORDER = ["Baseline", "GN-L4", "GN-L9", "MPNN-L4", "MPNN-L9", "random-L4"]


def variant(run: str) -> str:
    r = run.lower()
    if "baseline" in r:
        return "Baseline"
    if "random" in r:
        return "random-L4"
    enc = "MPNN" if "mpnn" in r else "GN"
    lay = "L9" if "l9" in r else ("L4" if "l4" in r else "L?")
    return f"{enc}-{lay}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=None, help="restrict to one step")
    args = ap.parse_args()

    files = []
    for g in CLEAN_GLOBS:
        files.extend(glob.glob(g))
    files = sorted(set(files))
    if not files:
        raise SystemExit("no clean jsonl found -- run clean_variance_jsonl.py first")

    # cell[(variant, step, tag)] = {seed: (fpsd, des)}
    cell = defaultdict(dict)
    for f in files:
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            run = d.get("run", "")
            m = re.search(r"step(\d+)k", run)
            if not m:
                continue
            step = int(m.group(1)) * 1000
            if args.step and step != args.step:
                continue
            tag = str(d.get("sampler_tag") or "sde_n0.45")
            fpsd = d.get("_res_PDB_FID")
            des = d.get("_res_designability_rate")
            if des is None:
                continue
            cell[(variant(run), step, tag)][d.get("seed")] = (fpsd, des)

    def avg(byseed, i):
        xs = [v[i] for v in byseed.values() if v[i] is not None]
        return sum(xs) / len(xs) if xs else None

    steps = sorted({k[1] for k in cell})
    for step in steps:
        print(f"\n===== step {step // 1000}k =====")
        print(f"{'variant':10s} | " + " ".join(f"{TAG_LABEL[t]:>7}" for t in TAGS))
        for metric, idx, fmt in [("FPSD", 0, "{:>7.0f}"), ("Des", 1, "{:>7.2f}")]:
            print(f"  -- {metric} --")
            for var in VAR_ORDER:
                cells = []
                for t in TAGS:
                    bs = cell.get((var, step, t))
                    if not bs:
                        cells.append(f"{'--':>7}")
                    else:
                        v = avg(bs, idx)
                        cells.append(fmt.format(v) if v is not None else f"{'--':>7}")
                ns = max((len(cell.get((var, step, t), {})) for t in TAGS), default=0)
                print(f"{var:10s} | " + " ".join(cells) + f"   (n≤{ns})")


if __name__ == "__main__":
    main()
