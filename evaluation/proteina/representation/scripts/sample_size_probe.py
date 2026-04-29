# ruff: noqa: E402
"""Phase 1 — Sample-size learning curve for the pretrained contact probe.

Picks a single representative checkpoint + layer and runs the pretrained-split
probe at several train-set sizes, evaluating each against the same fixed val
subset. Plot P@L/5 vs N_train, pick the elbow as the canonical N_train for
the Phase 2 sweep.

This is a one-off: run it once, decide N_train, commit it to
``sweep_config.yaml`` pretrained_probe profile, then run
``pretrain_probe_sweep.py`` across all checkpoints with that value.

Default target: baseline @ step=400000, layer=4, t=1.0. Layer-4 mid-trunk
output is a reasonable compromise — not the L0 peak (where signal is already
very strong and the curve might saturate faster) and not the deep layers
(where numbers are low and noise dominates).

Usage:
  python evaluation/proteina/representation/scripts/sample_size_probe.py \
      --n_train_sweep 500,1000,2000,5000,10000 \
      --n_eval 500
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import List

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from lib.checkpoints import LMDB_PATH, RUN_SCHEDULES, find_checkpoint_path, resolve_step
from lib.data import _default_device
from lib.extract import extract_model_hidden_states_multilayer
from lib.manifest import build_or_load_manifest, load_proteina_batch_from_manifest
from lib.probes.contact_pretrained import run_pretrained_contact_probe


def _default_train_lmdb() -> str:
    """Same directory as val.lmdb, swapping the stem."""
    import os

    val = os.environ.get("PROBES_LMDB_PATH", LMDB_PATH)
    p = Path(val)
    # val.lmdb -> train.lmdb (same directory)
    return str(p.parent / "train.lmdb")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="baseline")
    ap.add_argument("--step", type=int, default=400000)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument(
        "--n_train_sweep",
        type=str,
        default="500,1000,2000,5000,10000",
        help="Comma-separated list of N_train values to try.",
    )
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument("--probe_epochs", type=int, default=15)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--head_type", type=str, default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--train_lmdb_path",
        type=str,
        default=None,
        help="Path to train.lmdb. Default: sibling of PROBES_LMDB_PATH val.lmdb.",
    )
    ap.add_argument(
        "--eval_lmdb_path",
        type=str,
        default=None,
        help="Path to val.lmdb (eval set). Default: PROBES_LMDB_PATH env var.",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(HERE.parent / "results" / "pretrained_probe"),
    )
    args = ap.parse_args()

    train_lmdb = args.train_lmdb_path or _default_train_lmdb()
    eval_lmdb = args.eval_lmdb_path or LMDB_PATH
    n_trains: List[int] = [int(x) for x in args.n_train_sweep.split(",") if x.strip()]
    max_n_train = max(n_trains)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "sample_size_curve.csv"
    json_path = outdir / "sample_size_curve.json"

    print(f"Train LMDB: {train_lmdb}")
    print(f"Eval  LMDB: {eval_lmdb}")
    print(f"N_train sweep: {n_trains}")
    print(f"N_eval: {args.n_eval}")
    print(f"Target: run={args.run} step={args.step} layer={args.layer}")

    device = _default_device()

    # --- Manifests: build the largest train manifest once; slice for smaller Ns ---
    # Using the same seed for all sizes means N=500 is a strict subset of N=10000,
    # so any change in P@L/5 is genuinely from adding more data rather than
    # swapping which proteins are in the set.
    print(f"\n[manifest] Building eval manifest: n={args.n_eval}, seed={args.seed}")
    eval_manifest = build_or_load_manifest(
        outdir=outdir,
        version=f"eval_n{args.n_eval}_seed{args.seed}",
        lmdb_path=eval_lmdb,
        n=args.n_eval,
        max_size=args.max_size,
        seed=args.seed,
    )
    print(f"[manifest] Building train manifest: n={max_n_train}, seed={args.seed}")
    train_manifest_full = build_or_load_manifest(
        outdir=outdir,
        version=f"train_n{max_n_train}_seed{args.seed}",
        lmdb_path=train_lmdb,
        n=max_n_train,
        max_size=args.max_size,
        seed=args.seed,
    )
    # Override the cached lmdb_path with the CURRENT run's path. The manifest
    # on disk may have been written by a previous job whose /dev/shm dir no
    # longer exists. Keys are stable across runs (same source LMDB), so this
    # just retargets the open() at a valid file.
    eval_manifest["lmdb_path"] = eval_lmdb
    train_manifest_full["lmdb_path"] = train_lmdb

    # Load eval batch once.
    print(f"\n[load] Eval batch from {eval_lmdb}")
    eval_batch, eval_raw = load_proteina_batch_from_manifest(
        eval_manifest, device=device
    )
    print(f"  Loaded {len(eval_raw)} eval proteins")

    # Load full train batch (largest N), will slice it down for smaller Ns.
    print(f"\n[load] Train batch from {train_lmdb} (N={max_n_train})")
    train_batch_full, train_raw_full = load_proteina_batch_from_manifest(
        train_manifest_full, device=device
    )
    print(f"  Loaded {len(train_raw_full)} train proteins")

    # --- Load checkpoint & extract features ONCE (expensive — backbone forward) ---
    run_dir, is_repa, _, _ = RUN_SCHEDULES[args.run]
    ckpt = find_checkpoint_path(run_dir, args.step, prefer_ema=True)
    if ckpt is None:
        raise RuntimeError(f"No EMA checkpoint for {args.run} @ step={args.step}")
    step = resolve_step(ckpt, args.step)
    print(f"\n[ckpt] {args.run} @ step={step} : {ckpt.name}")

    from lib.checkpoints import load_checkpoint_by_path

    model = load_checkpoint_by_path(str(ckpt), is_repa=is_repa, device=device)

    print(f"[extract] Running backbone forward on {max_n_train} train proteins...")
    t0 = time.time()
    reps_train_full = extract_model_hidden_states_multilayer(
        model, train_batch_full, [args.layer], t_value=1.0
    )[args.layer]  # [max_n_train, N, D]
    print(f"  done in {time.time() - t0:.1f}s (shape={tuple(reps_train_full.shape)})")

    print(f"[extract] Running backbone forward on {args.n_eval} eval proteins...")
    t0 = time.time()
    reps_eval = extract_model_hidden_states_multilayer(
        model, eval_batch, [args.layer], t_value=1.0
    )[args.layer]  # [n_eval, N, D]
    print(f"  done in {time.time() - t0:.1f}s (shape={tuple(reps_eval.shape)})")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Sweep N_train: slice the features, retrain probe, reuse eval features ---
    rows = []
    for n in n_trains:
        # Slice: same order as the full manifest, so N=500 ⊂ N=1000 ⊂ ....
        reps_train = reps_train_full[:n]
        # Slice train_batch tensors correspondingly.
        train_batch_n = {
            k: (v[:n] if torch.is_tensor(v) and v.dim() > 0 else v)
            for k, v in train_batch_full.items()
        }
        print(f"\n[probe] N_train={n}")
        t0 = time.time()
        result = run_pretrained_contact_probe(
            reps_train=reps_train,
            train_batch=train_batch_n,
            reps_eval=reps_eval,
            eval_batch=eval_batch,
            head_type=args.head_type,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            seed=args.seed,
        )
        t_probe = time.time() - t0

        row = {
            "n_train": n,
            "n_eval": args.n_eval,
            "n_proteins_test": result["n_proteins_test"],
            "head_type": args.head_type,
            "p_at_L_5": result["p_at_L_5"],
            "p_at_L_2": result["p_at_L_2"],
            "p_at_L": result["p_at_L"],
            "probe_train_s": float(t_probe),
            "run": args.run,
            "step": int(step),
            "layer": args.layer,
            "ckpt_path": str(ckpt),
        }
        rows.append(row)
        print(
            f"  P@L/5={row['p_at_L_5']:.3f}  P@L/2={row['p_at_L_2']:.3f}  "
            f"P@L={row['p_at_L']:.3f}  ({t_probe:.1f}s)"
        )

    # --- Persist ---
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[out] wrote {csv_path}")
    print(f"[out] wrote {json_path}")

    # --- Simple matplotlib plot ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            [r["n_train"] for r in rows],
            [r["p_at_L_5"] for r in rows],
            marker="o",
            label=f"P@L/5 ({args.head_type})",
        )
        ax.set_xscale("log")
        ax.set_xlabel("N_train (pretrained contact probe)")
        ax.set_ylabel("P@L/5 on fixed val set")
        ax.set_title(
            f"{args.run} step={step} layer={args.layer}  |  "
            f"n_eval={args.n_eval} ({result['n_proteins_test']} after L≥50 filter)"
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plot_path = outdir / "sample_size_curve.png"
        fig.savefig(plot_path, dpi=150)
        print(f"[out] wrote {plot_path}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
