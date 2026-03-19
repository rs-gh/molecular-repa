"""Generate comprehensive metric plots for all GEOM training runs.

Usage:
    source .venv/bin/activate
    python playground/tabasco/model_evaluation/generate_metric_plots.py
"""

import wandb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

api = wandb.Api()

RUNS = {
    "yy363ps7": {"label": "Baseline (mild)", "color": "#2c3e50", "ls": "-", "lw": 2.5},
    "x3c4vid0": {
        "label": "Additive, fused proj",
        "color": "#e74c3c",
        "ls": "-",
        "lw": 1.8,
    },
    "0fbrr8vx": {
        "label": "Additive, same proj",
        "color": "#e67e22",
        "ls": "--",
        "lw": 1.8,
    },
    "7u3l0zpy": {
        "label": "Tradeoff, fused proj",
        "color": "#3498db",
        "ls": "-",
        "lw": 1.8,
    },
    "cqjant8r": {
        "label": "Tradeoff, same proj",
        "color": "#2ecc71",
        "ls": "--",
        "lw": 1.8,
    },
}

TRAIN_METRICS = [
    "train/loss_epoch",
    "train/coords_loss_epoch",
    "train/atomics_loss_epoch",
    "train/repa_loss_epoch",
    "train/grad_norm_step",
    "train/coords_loss_bin_0_epoch",
    "train/coords_loss_bin_1_epoch",
    "train/coords_loss_bin_2_epoch",
    "train/coords_loss_bin_3_epoch",
    "train/coords_loss_bin_4_epoch",
]

VAL_METRICS = [
    "val/loss",
    "val/coords_loss",
    "val/atomics_loss",
    "val/repa_loss",
    "val/validity",
    "val/connectivity",
    "val/uniqueness",
    "val/novelty",
    "val/qed",
    "val/lipinski",
    "val/mol_logp",
    "val/atom_type_distribution",
    "val/pb_intersection",
    "val/pb_bond_angles",
    "val/pb_bond_lengths",
    "val/pb_internal_steric_clash",
    "val/pb_aromatic_ring_flatness",
    "val/pb_double_bond_flatness",
]

OUT_DIR = "/home/sr2173/git/molecular-repa"


def fetch_all():
    all_data = {}
    for rid, info in RUNS.items():
        run = api.run(f"sr2173-university-of-cambridge/tabasco/{rid}")
        df = run.history(
            keys=["epoch", "_step"] + TRAIN_METRICS + VAL_METRICS, samples=2000
        )
        all_data[rid] = df
        print(f"  {info['label']}: {len(df)} rows")
    return all_data


def plot_training(all_data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Metrics Over Epochs", fontsize=16, fontweight="bold")

    specs = [
        ("train/loss_epoch", "Total Loss"),
        ("train/coords_loss_epoch", "Coords Loss"),
        ("train/atomics_loss_epoch", "Atomics Loss"),
        ("train/repa_loss_epoch", "REPA Loss (cosine sim)"),
        ("train/grad_norm_step", "Gradient Norm"),
    ]

    for idx, (metric, title) in enumerate(specs):
        ax = axes[idx // 3, idx % 3]
        for rid, info in RUNS.items():
            df = all_data[rid]
            valid = df.dropna(subset=[metric])
            if len(valid) > 0:
                if "_epoch" in metric:
                    grouped = valid.groupby("epoch")[metric].mean()
                    ax.plot(
                        grouped.index,
                        grouped.values,
                        label=info["label"],
                        color=info["color"],
                        linestyle=info["ls"],
                        linewidth=info["lw"],
                        alpha=0.85,
                    )
                else:
                    step = max(1, len(valid) // 200)
                    ax.plot(
                        valid["_step"].values[::step],
                        valid[metric].values[::step],
                        label=info["label"],
                        color=info["color"],
                        linestyle=info["ls"],
                        linewidth=info["lw"],
                        alpha=0.6,
                    )
                    ax.set_xlabel("Step")
        if "_epoch" in metric:
            ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    # Coords loss by time bin (baseline)
    ax = axes[1, 2]
    bin_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    bin_labels = [
        "t in [0,0.2)",
        "t in [0.2,0.4)",
        "t in [0.4,0.6)",
        "t in [0.6,0.8)",
        "t in [0.8,1.0]",
    ]
    df = all_data["yy363ps7"]
    for bi in range(5):
        m = f"train/coords_loss_bin_{bi}_epoch"
        valid = df.dropna(subset=[m])
        if len(valid) > 0:
            grouped = valid.groupby("epoch")[m].mean()
            ax.plot(
                grouped.index,
                grouped.values,
                label=bin_labels[bi],
                color=bin_colors[bi],
                linewidth=1.5,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Coords Loss")
    ax.set_title("Coords Loss by Time Bin (Baseline)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = f"{OUT_DIR}/plots_training_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_validation(all_data):
    specs = [
        ("val/validity", "Validity", 0.7),
        ("val/pb_intersection", "PoseBusters (all checks)", 0.2),
        ("val/pb_internal_steric_clash", "No Steric Clash", 0.4),
        ("val/pb_bond_angles", "Bond Angles OK", 0.4),
        ("val/pb_bond_lengths", "Bond Lengths OK", 0.4),
        ("val/pb_aromatic_ring_flatness", "Aromatic Ring Flatness", 0.4),
        ("val/qed", "QED Score", 0.4),
        ("val/novelty", "Novelty", 0.4),
        ("val/mol_logp", "LogP", None),
        ("val/uniqueness", "Uniqueness", 0.8),
        ("val/atom_type_distribution", "Atom Type Dist.", 0.8),
        ("val/connectivity", "Connectivity", 0.8),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        "Validation Metrics Over Epochs\n(n=100 samples per evaluation)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for idx, (metric, title, ybot) in enumerate(specs):
        ax = axes[idx // 3, idx % 3]
        for rid, info in RUNS.items():
            df = all_data[rid]
            valid = df.dropna(subset=[metric])
            if len(valid) > 0:
                ax.plot(
                    valid["epoch"],
                    valid[metric],
                    label=info["label"],
                    color=info["color"],
                    linestyle=info["ls"],
                    linewidth=info["lw"],
                    marker="o",
                    markersize=3,
                    alpha=0.85,
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if ybot is not None:
            ax.set_ylim(bottom=ybot)
        ax.legend(fontsize=7, loc="best")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = f"{OUT_DIR}/plots_validation_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_losses(all_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Loss Curves: Validation", fontsize=14, fontweight="bold")

    for ax, (metric, title) in zip(
        axes,
        [
            ("val/coords_loss", "Val Coords Loss"),
            ("val/atomics_loss", "Val Atomics Loss"),
            ("val/repa_loss", "Val REPA Loss"),
        ],
    ):
        for rid, info in RUNS.items():
            df = all_data[rid]
            valid = df.dropna(subset=[metric])
            if len(valid) > 0:
                ax.plot(
                    valid["epoch"],
                    valid[metric],
                    label=info["label"],
                    color=info["color"],
                    linestyle=info["ls"],
                    linewidth=info["lw"],
                    marker=".",
                    markersize=4,
                    alpha=0.85,
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{OUT_DIR}/plots_loss_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    print("Fetching run histories from WandB...")
    data = fetch_all()
    print("\nGenerating plots...")
    plot_training(data)
    plot_validation(data)
    plot_losses(data)
    print("\nDone! All plots saved.")
