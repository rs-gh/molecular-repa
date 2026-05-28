"""Load both sampler-ablation clean jsonls + the γ=0.45 convergence-sweep rows
into one normalized pandas frame. Used by baseline_tradeoff.py and compare.py.

This is a one-off audit utility — not part of the paper pipeline.
"""

import json
import pandas as pd
from pathlib import Path

ROOT = Path(
    "/home/sr2173/git/molecular-repa/evaluation/proteina/generation/results/variance"
)
FILES = {
    "PDB": ROOT / "n256_sampler_ablation" / "sweep_results.clean.jsonl",
    "AFDB": ROOT / "n256_afdb_sampler_ablation" / "sweep_results.clean.jsonl",
}

CONV = {
    "PDB": Path(
        "/home/sr2173/git/molecular-repa/evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl"
    ),
    "AFDB": Path(
        "/home/sr2173/git/molecular-repa/evaluation/proteina/generation/results/paper/n256_convergence_afdb/sweep_results.clean.jsonl"
    ),
}

GAMMA_FROM_TAG = {
    "ode": "ODE",
    "sde_n0.0": "0.0",
    "sde_n0.35": "0.35",
    "sde_n0.45": "0.45",
    "sde_n0.5": "0.5",
    "sde_n1.0": "1.0",
}


def parse_run(name: str):
    """Return (encoder, layer, dataset) parsed from a run name."""
    ds = "AFDB" if "afdb" in name else "PDB"
    if "baseline" in name:
        return ("baseline", "-", ds)
    enc = "GN"
    if "mpnn" in name:
        enc = "MPNN"
    if "random" in name:
        enc = "GN-random"
    layer = "L4" if "_l4_" in name else ("L9" if "_l9_" in name else "?")
    return (enc, layer, ds)


def load_all():
    frames = []
    for ds, f in FILES.items():
        rows = [json.loads(line) for line in open(f)]
        df = pd.DataFrame(rows)
        df["__src"] = "ablation"
        frames.append(df)
    for ds, f in CONV.items():
        if not f.exists():
            print(f"missing {f}")
            continue
        rows = [json.loads(line) for line in open(f)]
        for r in rows:
            r.setdefault("sampler_tag", "sde_n0.45")
        df = pd.DataFrame(rows)
        df["__src"] = "conv"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["gamma"] = df["sampler_tag"].map(GAMMA_FROM_TAG)
    parsed = df["run"].apply(parse_run)
    df["encoder"] = [p[0] for p in parsed]
    df["layer"] = [p[1] for p in parsed]
    df["dataset"] = [p[2] for p in parsed]
    return df


if __name__ == "__main__":
    df = load_all()
    print(
        df.groupby(["dataset", "encoder", "layer", "gamma"])
        .size()
        .unstack(fill_value=0)
    )
