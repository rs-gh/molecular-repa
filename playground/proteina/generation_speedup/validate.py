"""Correctness validation: compare geometry distributions across the 4 modes at
len=256, using same random seeds for direct sample-wise comparison.

Usage:
    python playground/proteina/generation_speedup/validate.py
    python playground/proteina/generation_speedup/validate.py --n_samples 50 --length 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _speedup_utils import prepare_model, MODES  # noqa: E402
from _geometry import (  # noqa: E402
    cross_mode_ca_rmsd,
    per_sample_summary,
)
from checkpoints import resolve_active  # noqa: E402

# Mirror bench.py
GENERATE_KWARGS = dict(
    dt=0.0025,
    self_cond=True,
    cath_code=None,
    guidance_weight=1.0,
    autoguidance_ratio=1.0,
    schedule_mode="log",
    schedule_p=2.0,
    sampling_mode="sc",
    sc_scale_noise=0.45,
    sc_scale_score=1.0,
    gt_mode="1/t",
    gt_p=1.0,
    gt_clamp_val=None,
)

BATCH_SIZE = 8

# PDB writer
sys.path.insert(0, str(HERE.parent.parent.parent / "src" / "proteina"))
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb  # noqa: E402


def generate_all_modes(
    ckpt_path: Path,
    is_repa: bool,
    modes: list[str],
    nres: int,
    n_samples: int,
    base_seed: int,
) -> dict[str, np.ndarray]:
    """Generate n_samples atom37 coords per mode, using same per-batch seed across modes.

    Returns:
        dict mode -> numpy array [n_samples, nres, 37, 3]
    """
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    total = n_batches * BATCH_SIZE  # pad up; we trim at the end
    out = {}

    for mode in modes:
        print(f"\n--- generating {total} samples @ mode={mode} ---")
        sys.stdout.flush()
        model, autocast_ctx = prepare_model(ckpt_path, is_repa, mode)

        batches = []
        for b in range(n_batches):
            seed = base_seed + b
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.no_grad(), autocast_ctx():
                x = model.generate(
                    nsamples=BATCH_SIZE,
                    n=nres,
                    dtype=torch.float32,
                    **GENERATE_KWARGS,
                )
                coords = model.samples_to_atom37(x).cpu().numpy()  # [B, N, 37, 3]
            batches.append(coords)
            print(f"    batch {b + 1}/{n_batches} done (seed={seed})")
            sys.stdout.flush()

        out[mode] = np.concatenate(batches, axis=0)[:n_samples]
        del model
        torch.cuda.empty_cache()

    return out


def compute_rows(samples_by_mode: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for mode, coords in samples_by_mode.items():
        for i in range(coords.shape[0]):
            summary = per_sample_summary(coords[i])
            summary.update(mode=mode, sample_idx=i)
            rows.append(summary)
    return pd.DataFrame(rows)


def compute_cross_mode_rmsd(
    samples_by_mode: dict[str, np.ndarray], reference: str = "eager_fp32"
) -> pd.DataFrame:
    if reference not in samples_by_mode:
        return pd.DataFrame()
    ref = samples_by_mode[reference]
    rows = []
    for mode, coords in samples_by_mode.items():
        if mode == reference:
            continue
        for i in range(min(ref.shape[0], coords.shape[0])):
            r = cross_mode_ca_rmsd(ref[i], coords[i])
            rows.append(dict(mode=mode, sample_idx=i, ca_rmsd_vs_ref=r))
    return pd.DataFrame(rows)


def plot_distributions(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("bond_mean", "mean CA-CA bond length (A)"),
        ("angle_mean", "mean CA-CA-CA virtual bond angle (deg)"),
        ("rg", "radius of gyration (A)"),
        ("end_to_end", "end-to-end distance (A)"),
        ("clash_rate", "CA-CA clash rate (frac. < 3.6 A, non-adj)"),
    ]
    for key, label in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, grp in df.groupby("mode"):
            ax.hist(grp[key], bins=20, alpha=0.5, label=mode, density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title(f"{key} across modes")
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=120)
        plt.close(fig)


def plot_rmsd(rmsd_df: pd.DataFrame, out_dir: Path):
    if rmsd_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    mode_order = sorted(rmsd_df["mode"].unique())
    data = [rmsd_df[rmsd_df["mode"] == m]["ca_rmsd_vs_ref"].values for m in mode_order]
    ax.boxplot(data, labels=mode_order)
    ax.set_ylabel("CA RMSD vs eager_fp32 (A)")
    ax.set_title("Per-sample best-aligned CA RMSD vs reference mode")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_mode_rmsd.png", dpi=120)
    plt.close(fig)


def save_samples(
    samples_by_mode: dict[str, np.ndarray], out_root: Path, per_mode: int = 3
):
    for mode, coords in samples_by_mode.items():
        mode_dir = out_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(per_mode, coords.shape[0])):
            pdb_path = mode_dir / f"sample_{i:02d}.pdb"
            write_prot_to_pdb(
                coords[i], str(pdb_path), overwrite=True, no_indexing=True
            )


def summarize(df: pd.DataFrame, rmsd_df: pd.DataFrame):
    print("\n=== Per-mode geometry summary ===")
    agg = df.groupby("mode").agg(
        rg_mean=("rg", "mean"),
        rg_std=("rg", "std"),
        bond_mean=("bond_mean", "mean"),
        bond_std=("bond_mean", "std"),
        angle_mean=("angle_mean", "mean"),
        clash_mean=("clash_rate", "mean"),
        end_to_end_mean=("end_to_end", "mean"),
    )
    print(agg.round(3).to_string())
    if not rmsd_df.empty:
        print("\n=== Cross-mode CA RMSD (median, p90) ===")
        print(
            rmsd_df.groupby("mode")["ca_rmsd_vs_ref"]
            .agg(median="median", p90=lambda s: s.quantile(0.9))
            .round(3)
            .to_string()
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--modes",
        type=str,
        default="all",
        help="'all' (default, 4 modes) or comma-separated list.",
    )
    args = ap.parse_args()

    if args.modes == "all":
        modes = list(MODES)
    else:
        modes = args.modes.split(",")
    for m in modes:
        assert m in MODES, f"unknown mode {m!r}; choose from {MODES}"

    active = resolve_active()
    ckpt = active[0]
    print(f"Ckpt: {ckpt.name} -> {ckpt.ckpt_path}")
    print(
        f"n_samples={args.n_samples} length={args.length} seed={args.seed} modes={modes}"
    )

    samples_by_mode = generate_all_modes(
        ckpt.ckpt_path, ckpt.is_repa, modes, args.length, args.n_samples, args.seed
    )

    metrics_df = compute_rows(samples_by_mode)
    rmsd_df = compute_cross_mode_rmsd(samples_by_mode, reference="eager_fp32")

    figures_dir = HERE / "figures"
    plot_distributions(metrics_df, figures_dir)
    plot_rmsd(rmsd_df, figures_dir)

    samples_root = HERE / "samples" / "validate"
    save_samples(samples_by_mode, samples_root, per_mode=3)

    metrics_df.to_csv(HERE / "validate_metrics.csv", index=False)
    if not rmsd_df.empty:
        rmsd_df.to_csv(HERE / "validate_rmsd.csv", index=False)

    summarize(metrics_df, rmsd_df)
    print(f"\nFigures saved to {figures_dir}")
    print(f"Samples (3/mode) saved to {samples_root}")


if __name__ == "__main__":
    main()
