"""Back-calculate REPA validation loss from existing WandB logs.

For additive mode with lambda=0.5:
    total_val_loss = fm_loss + aux_loss + lambda * repa_loss
    repa_loss = (total_val_loss - fm_loss - aux_loss) / lambda

Usage:
    source .venv/bin/activate
    python playground/proteina/val_loss_breakdown/backcalc_repa_val_loss.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import wandb

OUTDIR = "playground/proteina/val_loss_breakdown/figures"
os.makedirs(OUTDIR, exist_ok=True)

LAMBDA_REPA = 0.5

RUNS = {
    "REPA L4": "proteina_60m_repa_v2",
    "REPA L0": "proteina_60m_repa_layer0_v2",
    "REPA L9": "proteina_60m_repa_layer9_v2",
}

# Baseline has no REPA loss, but we fetch its validation loss for comparison
BASELINE_RUN = {
    "Baseline": "proteina_60m_baseline_v2",
}

VAL_KEYS = [
    "validation_loss/loss_step",
    "validation_loss/trans_loss_step",
    "validation_loss/auxiliary_loss_step",
    "trainer/global_step",
    "epoch",
]

api = wandb.Api(timeout=120)


def fetch_validation_data(run_id, keys, samples=500):
    """Fetch validation metrics from WandB."""
    run = api.run(f"proteina-repa/{run_id}")
    df = run.history(samples=samples, pandas=True)
    # Keep only rows with validation loss
    df = df[df["validation_loss/loss_step"].notna()].copy()
    return df


def backcalc_repa_loss(df):
    """Compute REPA loss by subtraction: (total - fm - aux) / lambda."""
    df = df.copy()
    df["repa_loss_backcalc"] = (
        df["validation_loss/loss_step"]
        - df["validation_loss/trans_loss_step"]
        - df["validation_loss/auxiliary_loss_step"]
    ) / LAMBDA_REPA
    return df


def main():
    all_data = {}

    # Fetch baseline
    print("Fetching baseline validation data...")
    for label, rid in BASELINE_RUN.items():
        try:
            df = fetch_validation_data(rid, VAL_KEYS)
            all_data[label] = df
            print(
                f"  {label}: {len(df)} validation points, "
                f"steps {df['trainer/global_step'].min():.0f}-{df['trainer/global_step'].max():.0f}"
            )
        except Exception as e:
            print(f"  {label}: FAILED - {e}")

    # Fetch REPA runs and back-calculate
    print("\nFetching REPA validation data...")
    for label, rid in RUNS.items():
        try:
            df = fetch_validation_data(rid, VAL_KEYS)
            df = backcalc_repa_loss(df)
            all_data[label] = df
            print(
                f"  {label}: {len(df)} validation points, "
                f"steps {df['trainer/global_step'].min():.0f}-{df['trainer/global_step'].max():.0f}"
            )
            print(
                f"    Back-calculated REPA loss: "
                f"mean={df['repa_loss_backcalc'].mean():.4f}, "
                f"last={df['repa_loss_backcalc'].iloc[-1]:.4f}"
            )
        except Exception as e:
            print(f"  {label}: FAILED - {e}")

    if not all_data:
        print("No data fetched. Check WandB connectivity.")
        return

    # ── Plot 1: Total validation loss (all models) ──
    colors = {
        "Baseline": "#4C72B0",
        "REPA L4": "#DD8452",
        "REPA L0": "#55A868",
        "REPA L9": "#C44E52",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: total validation loss
    ax = axes[0, 0]
    for label, df in all_data.items():
        ax.plot(
            df["trainer/global_step"],
            df["validation_loss/loss_step"],
            label=label,
            color=colors[label],
            alpha=0.7,
            linewidth=1,
        )
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title("Total Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: FM (trans) loss only
    ax = axes[0, 1]
    for label, df in all_data.items():
        ax.plot(
            df["trainer/global_step"],
            df["validation_loss/trans_loss_step"],
            label=label,
            color=colors[label],
            alpha=0.7,
            linewidth=1,
        )
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title("FM Translation Loss (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: back-calculated REPA loss (REPA models only)
    ax = axes[1, 0]
    for label, df in all_data.items():
        if "repa_loss_backcalc" in df.columns:
            ax.plot(
                df["trainer/global_step"],
                df["repa_loss_backcalc"],
                label=label,
                color=colors[label],
                alpha=0.7,
                linewidth=1,
            )
    ax.set_xlabel("Global Step")
    ax.set_ylabel("REPA Loss (back-calculated)")
    ax.set_title(f"REPA Loss = (total - fm - aux) / {LAMBDA_REPA}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: auxiliary loss
    ax = axes[1, 1]
    for label, df in all_data.items():
        ax.plot(
            df["trainer/global_step"],
            df["validation_loss/auxiliary_loss_step"],
            label=label,
            color=colors[label],
            alpha=0.7,
            linewidth=1,
        )
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title("Auxiliary Loss (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Validation Loss Breakdown (back-calculated REPA)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/val_loss_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {OUTDIR}/val_loss_breakdown.png")

    # ── Print summary table ──
    print("\n" + "=" * 80)
    print(
        f"{'Model':<12} {'Steps':>10} {'Val Loss':>10} {'FM Loss':>10} {'Aux Loss':>10} {'REPA Loss':>11}"
    )
    print("=" * 80)
    for label, df in all_data.items():
        last = df.iloc[-1]
        step = last["trainer/global_step"]
        total = last["validation_loss/loss_step"]
        fm = last["validation_loss/trans_loss_step"]
        aux = last["validation_loss/auxiliary_loss_step"]
        repa = last.get("repa_loss_backcalc", float("nan"))
        repa_str = f"{repa:.4f}" if not np.isnan(repa) else "N/A"
        print(
            f"{label:<12} {step:>10.0f} {total:>10.4f} {fm:>10.4f} {aux:>10.4f} {repa_str:>11}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
