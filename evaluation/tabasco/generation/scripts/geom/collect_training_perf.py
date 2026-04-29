"""Collect training performance statistics across all GEOM runs and generate plots.

Outputs:
    evaluation/tabasco/generation/results/geom/training_performance/training_performance.csv  - raw data
    evaluation/tabasco/generation/results/geom/training_performance/figures/training_perf_*.png - plots

Usage:
    python evaluation/tabasco/generation/scripts/tabasco/geom/collect_training_perf.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wandb

STEPS_PER_EPOCH = {
    "geom": 4461,  # 1,142,099 / 256
    "qm9": 374,  # 95,793 / 256
}

# (wandb display_name, human label, dataset, encoder phase)
RUNS = [
    ("0320-1409-tabasco-geom-mace-additive", "MACE add (CPU, f64)", "geom", "mace-cpu"),
    (
        "0320-1409-tabasco-geom-mace-tradeoff",
        "MACE trade (CPU, f64)",
        "geom",
        "mace-cpu",
    ),
    (
        "0321-0006-tabasco-geom-mace-additive-1",
        "MACE add (GPU, f32)",
        "geom",
        "mace-gpu",
    ),
    (
        "0321-0010-tabasco-geom-mace-tradeoff-1",
        "MACE trade (GPU, f32)",
        "geom",
        "mace-gpu",
    ),
    (
        "0224-1849-tabasco-geom-chemprop-additive-fused-projector",
        "CheMeleon additive",
        "geom",
        "chemeleon",
    ),
    (
        "0224-1848-tabasco-geom-chemprop-tradeoff-fused-projector",
        "CheMeleon tradeoff",
        "geom",
        "chemeleon",
    ),
    ("final-tabasco-mild-geom-part2", "Baseline (no REPA)", "geom", "baseline"),
]

# Cached MACE runs - matched by regex since name includes timestamp
CACHED_RUNS_REGEX = [
    ("mace-cached-additive", "MACE cached add", "geom", "mace-cached"),
    ("mace-cached-tradeoff", "MACE cached trade", "geom", "mace-cached"),
]

VAL_METRICS = [
    "val/coords_loss",
    "val/atomics_loss",
    "val/validity",
    "val/connectivity",
    "val/novelty",
    "val/pb_all_atoms_connected",
    "val/atom_type_distribution",
]


def collect_run_data(api, name, label, dataset, phase, is_regex=False):
    """Collect performance and validation data from a single wandb run."""
    if is_regex:
        runs = api.runs(
            "tabasco",
            filters={"display_name": {"$regex": name}},
            order="-created_at",
            per_page=2,
        )
    else:
        runs = api.runs("tabasco", filters={"display_name": name})

    for run in runs:
        hist = list(run.scan_history(keys=["trainer/global_step", "_runtime"]))
        if len(hist) < 2:
            continue

        steps = hist[-1]["trainer/global_step"] - hist[0]["trainer/global_step"]
        dt = hist[-1]["_runtime"] - hist[0]["_runtime"]
        if steps == 0:
            continue

        sps = dt / steps
        spe = STEPS_PER_EPOCH.get(dataset, 4461)

        row = {
            "run_name": run.name,
            "label": label,
            "dataset": dataset,
            "phase": phase,
            "state": run.state,
            "total_steps": hist[-1]["trainer/global_step"],
            "epoch": run.summary.get("epoch", None),
            "s_per_step": round(sps, 3),
            "steps_per_hr": round(3600 / sps),
            "hr_per_epoch": round(spe * sps / 3600, 2),
            "runtime_hr": round(dt / 3600, 2),
        }

        # GPU utilization from system metrics
        try:
            sys_metrics = run.history(stream="events", samples=500)
            gpu_util = sys_metrics.get("system.gpu.0.gpu")
            if gpu_util is not None and len(gpu_util.dropna()) > 0:
                row["gpu_util_mean"] = round(gpu_util.mean(), 1)
                row["gpu_util_max"] = round(gpu_util.max(), 1)
        except Exception:
            pass

        # Validation metrics
        for key in VAL_METRICS:
            val = run.summary.get(key)
            if isinstance(val, (int, float)):
                row[key] = round(val, 4)

        return row
    return None


def make_plots(df, out_dir):
    """Generate performance comparison plots."""
    os.makedirs(out_dir, exist_ok=True)

    # Use a consistent color scheme
    phase_colors = {
        "mace-cpu": "#d62728",
        "mace-gpu": "#ff7f0e",
        "mace-cached": "#2ca02c",
        "chemeleon": "#1f77b4",
        "baseline": "#7f7f7f",
    }

    # --- Plot 1: s/step comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        df["label"],
        df["s_per_step"],
        color=[phase_colors.get(p, "#333") for p in df["phase"]],
    )
    ax.set_xlabel("Seconds per step")
    ax.set_title("Training Speed Comparison (GEOM dataset)")
    ax.invert_yaxis()
    for bar, val in zip(bars, df["s_per_step"]):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_perf_speed.png"), dpi=150)
    plt.close()

    # --- Plot 2: GPU utilization ---
    if "gpu_util_mean" in df.columns:
        gpu_df = df.dropna(subset=["gpu_util_mean"])
        if len(gpu_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(
                gpu_df["label"],
                gpu_df["gpu_util_mean"],
                color=[phase_colors.get(p, "#333") for p in gpu_df["phase"]],
            )
            ax.set_xlabel("Mean GPU Utilization (%)")
            ax.set_title("GPU Utilization Comparison")
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "training_perf_gpu_util.png"), dpi=150)
            plt.close()

    # --- Plot 3: Speedup over CPU MACE ---
    cpu_sps = df.loc[df["phase"] == "mace-cpu", "s_per_step"].mean()
    if cpu_sps > 0:
        df_speedup = df.copy()
        df_speedup["speedup"] = cpu_sps / df_speedup["s_per_step"]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(
            df_speedup["label"],
            df_speedup["speedup"],
            color=[phase_colors.get(p, "#333") for p in df_speedup["phase"]],
        )
        ax.set_xlabel("Speedup vs CPU MACE (f64)")
        ax.set_title("Training Speedup Progression")
        ax.invert_yaxis()
        ax.set_xscale("log")
        for bar, val in zip(bars, df_speedup["speedup"]):
            ax.text(
                bar.get_width() * 1.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}x",
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_perf_speedup.png"), dpi=150)
        plt.close()

    # --- Plot 4: hr/epoch ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        df["label"],
        df["hr_per_epoch"],
        color=[phase_colors.get(p, "#333") for p in df["phase"]],
    )
    ax.set_xlabel("Hours per epoch")
    ax.set_title("Time per Epoch (GEOM, 4461 steps)")
    ax.invert_yaxis()
    for bar, val in zip(bars, df["hr_per_epoch"]):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}h",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_perf_epoch_time.png"), dpi=150)
    plt.close()


def main():
    api = wandb.Api()
    rows = []

    print("Collecting run data...")
    for name, label, dataset, phase in RUNS:
        row = collect_run_data(api, name, label, dataset, phase)
        if row:
            rows.append(row)
            print(f"  {label}: {row['s_per_step']} s/step")

    for regex, label, dataset, phase in CACHED_RUNS_REGEX:
        row = collect_run_data(api, regex, label, dataset, phase, is_regex=True)
        if row:
            rows.append(row)
            print(f"  {label}: {row['s_per_step']} s/step")

    if not rows:
        print("No data collected!")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Save CSV
    eval_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    out_dir = os.path.join(
        eval_root, "results", "tabasco", "geom", "training_performance"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "training_performance.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print summary table
    print("\n" + "=" * 90)
    cols = [
        "label",
        "s_per_step",
        "steps_per_hr",
        "hr_per_epoch",
        "gpu_util_mean",
        "epoch",
    ]
    print(df[cols].to_string(index=False))
    print("=" * 90)

    # Generate plots
    fig_dir = os.path.join(out_dir, "figures")
    make_plots(df, fig_dir)
    print(f"\nPlots saved to: {fig_dir}/")


if __name__ == "__main__":
    main()
