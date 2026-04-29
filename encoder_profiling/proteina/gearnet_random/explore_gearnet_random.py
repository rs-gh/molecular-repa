"""Random-init CA-GearNet baseline.

Runs the standard probe over N seeds (default 3) and aggregates each metric to
mean/std. Provides the architecture-only floor against which the trained
CA-GearNet is judged - if a metric isn't notably better than this, the
pretraining isn't contributing.

Each seed run writes its own results.json under
    results/seed{seed}_<timestamp>/
and a final mean_std.json sits at
    results/<timestamp>/mean_std.json
which is what collate.py reads.

Run:
  source .venv/bin/activate
  python encoder_profiling/proteina/gearnet_random/explore_gearnet_random.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))
sys.path.insert(0, str(REPO_ROOT / "encoder_profiling" / "proteina"))

# Reuse the trained driver's setup + layerwise logic
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gearnet"))
from explore_gearnet import LMDB_PATH, layerwise_fn, setup_encoder  # noqa: E402

from _probes import (  # noqa: E402
    EncoderProbe,
    load_proteins,
    make_embed_fn,
    run_pipeline,
)


def _flatten(d, prefix=""):
    """Flatten a nested results dict to {dotted.key: scalar}. Skips lists/dicts of strings."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[key] = float(v)
    return flat


def aggregate(per_seed_results):
    """Mean/std across seeds for every flattened scalar metric."""
    flats = [_flatten(r) for r in per_seed_results]
    keys = set().union(*[f.keys() for f in flats])
    agg = {}
    for k in sorted(keys):
        vals = [f[k] for f in flats if k in f]
        if len(vals) >= 1:
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=200)
    ap.add_argument(
        "--random-seed", type=int, default=0, help="Protein selection seed."
    )
    ap.add_argument(
        "--init-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="GearNet random-init seeds to run (each = one full pipeline pass).",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--output-root", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    proteins = load_proteins(LMDB_PATH, args.n_proteins, seed=args.random_seed)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = args.output_root or os.path.join(os.path.dirname(__file__), "results", stamp)
    os.makedirs(root, exist_ok=True)

    per_seed = []
    for s in args.init_seeds:
        print("\n" + "*" * 70)
        print(f"* Init seed {s}")
        print("*" * 70)
        encoder = setup_encoder(device, random_init=True, random_seed=s)
        seed_dir = os.path.join(root, f"seed{s}")
        probe = EncoderProbe(
            name=f"ca-gearnet-random-seed{s}",
            encoder=encoder,
            embed_fn=make_embed_fn(encoder, device),
            is_3d_aware=True,
            accepts_residue_type=False,
            context_mode="structural",
            layerwise_fn=layerwise_fn,
            output_dir=seed_dir,
        )
        per_seed.append(run_pipeline(probe, proteins, device))
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    agg = aggregate(per_seed)
    out_path = os.path.join(root, "mean_std.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "name": "ca-gearnet-random",
                "init_seeds": args.init_seeds,
                "n_seeds": len(args.init_seeds),
                "n_proteins": args.n_proteins,
                "aggregate": agg,
            },
            f,
            indent=2,
        )
    print(f"\nAggregated mean/std across {len(args.init_seeds)} seeds -> {out_path}")


if __name__ == "__main__":
    main()
