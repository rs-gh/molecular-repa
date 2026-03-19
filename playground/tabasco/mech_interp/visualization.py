"""Visualization functions for trajectory analysis.

Produces heatmaps and line plots analogous to proteina's crystallization figures.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from playground.tabasco.mech_interp.trajectory_analyzer import TrajectoryMetrics


def _setup_style():
    """Set consistent plot style."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def plot_heatmap(
    data: np.ndarray,
    timesteps: np.ndarray,
    title: str,
    ylabel: str = "Layer",
    cbar_label: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    output_path: Optional[str] = None,
):
    """Generic [layers x timesteps] heatmap.

    Args:
        data: [T, L] array.
        timesteps: [T] timestep values for x-axis labels.
        title: Plot title.
        ylabel: Y-axis label.
        cbar_label: Colorbar label.
        cmap: Matplotlib colormap.
        vmin, vmax: Colorbar range.
        output_path: If set, save figure here.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        data.T,  # Transpose so layers are on y-axis
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[timesteps[0], timesteps[-1], -0.5, data.shape[1] - 0.5],
    )

    ax.set_xlabel("Generation time $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_structure_lens_heatmap(
    metrics: TrajectoryMetrics,
    use_gt: bool = False,
    output_path: Optional[str] = None,
):
    """Structure lens: RMSD to final (or GT) across layers and timesteps."""
    if use_gt and metrics.coord_rmsd_to_gt is not None:
        data = metrics.coord_rmsd_to_gt
        title = "Structure Lens: RMSD to Ground Truth"
    else:
        data = metrics.coord_rmsd_to_final
        title = "Structure Lens: RMSD to Final-Layer Output"

    return plot_heatmap(
        data,
        metrics.timesteps,
        title=title,
        cbar_label="RMSD",
        cmap="viridis_r",
        output_path=output_path,
    )


def plot_entropy_heatmap(
    metrics: TrajectoryMetrics,
    output_path: Optional[str] = None,
):
    """Attention entropy (mean over heads) across layers and timesteps."""
    # Average over heads
    data = np.nanmean(metrics.attn_entropy, axis=-1)  # [T, L]

    return plot_heatmap(
        data,
        metrics.timesteps,
        title="Attention Entropy (mean over heads, normalized)",
        cbar_label="$\\hat{H}$",
        cmap="RdYlBu",
        vmin=0,
        vmax=1,
        output_path=output_path,
    )


def plot_bond_precision_heatmap(
    metrics: TrajectoryMetrics,
    output_path: Optional[str] = None,
):
    """Bond precision (mean over heads) across layers and timesteps."""
    if metrics.bond_precision_vals is None:
        return None

    data = np.nanmean(metrics.bond_precision_vals, axis=-1)  # [T, L]

    return plot_heatmap(
        data,
        metrics.timesteps,
        title="Bond Precision@N (mean over heads)",
        cbar_label="Precision",
        cmap="YlOrRd",
        output_path=output_path,
    )


def plot_repa_landscape(
    metrics: TrajectoryMetrics,
    output_path: Optional[str] = None,
):
    """REPA alignment landscape: cosine similarity to ChemProp."""
    if metrics.chemprop_cosine_sim is None:
        return None

    return plot_heatmap(
        metrics.chemprop_cosine_sim,
        metrics.timesteps,
        title="REPA Alignment Landscape (cosine sim to ChemProp)",
        cbar_label="Cosine similarity",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        output_path=output_path,
    )


def plot_metric_lines(
    metrics: TrajectoryMetrics,
    layers: Optional[List[int]] = None,
    output_path: Optional[str] = None,
):
    """Line plots of key metrics for selected layers over time.

    Produces a 2x2 grid: RMSD, atom accuracy, entropy, REPA sim.
    """
    _setup_style()
    L = metrics.num_layers
    if layers is None:
        layers = [0, L // 4, L // 2, 3 * L // 4, L - 1]
    layers = [li for li in layers if li < L]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = metrics.timesteps

    # RMSD to final
    ax = axes[0, 0]
    for li in layers:
        ax.plot(t, metrics.coord_rmsd_to_final[:, li], label=f"L{li}")
    ax.set_ylabel("RMSD to final")
    ax.set_title("Structure Lens")
    ax.legend(fontsize=8)

    # Atom accuracy
    ax = axes[0, 1]
    for li in layers:
        ax.plot(t, metrics.atom_accuracy[:, li], label=f"L{li}")
    ax.set_ylabel("Atom type accuracy")
    ax.set_title("Atom Prediction from Intermediate Layer")
    ax.legend(fontsize=8)

    # Attention entropy (mean over heads)
    ax = axes[1, 0]
    entropy_mean = np.nanmean(metrics.attn_entropy, axis=-1)  # [T, L]
    for li in layers:
        ax.plot(t, entropy_mean[:, li], label=f"L{li}")
    ax.set_ylabel("$\\hat{H}$ (normalized)")
    ax.set_xlabel("Generation time $t$")
    ax.set_title("Attention Entropy")
    ax.legend(fontsize=8)

    # REPA sim or distance correlation
    ax = axes[1, 1]
    if metrics.chemprop_cosine_sim is not None:
        for li in layers:
            ax.plot(t, metrics.chemprop_cosine_sim[:, li], label=f"L{li}")
        ax.set_ylabel("Cosine sim")
        ax.set_title("REPA Alignment")
    elif metrics.dist_correlation is not None:
        dc_mean = np.nanmean(metrics.dist_correlation, axis=-1)
        for li in layers:
            ax.plot(t, dc_mean[:, li], label=f"L{li}")
        ax.set_ylabel("$\\rho$")
        ax.set_title("Distance Correlation")
    else:
        ax.text(
            0.5,
            0.5,
            "No REPA/distance data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Generation time $t$")
    ax.legend(fontsize=8)

    fig.suptitle("Trajectory Metrics by Layer", fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_ablation_summary(
    ablation_results: List[Dict],
    output_path: Optional[str] = None,
):
    """Bar chart comparing ablation conditions.

    Args:
        ablation_results: List of dicts from TrajectoryAnalyzer.analyze_ablation,
            each with "label" and "molecules" keys. Molecules should have had
            validity/quality metrics computed externally and stored under
            "validity", "connectivity", "rg" keys.
        output_path: Where to save.
    """
    _setup_style()

    labels = [r.get("label", f"condition_{i}") for i, r in enumerate(ablation_results)]

    # Extract metrics from results (these should be added by the runner script)
    validity = [r.get("validity", float("nan")) for r in ablation_results]
    connectivity = [r.get("connectivity", float("nan")) for r in ablation_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    width = 0.6

    # Validity
    ax = axes[0]
    ax.bar(x, validity, width, color="steelblue")
    ax.set_ylabel("Validity (%)")
    ax.set_title("Molecular Validity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Connectivity
    ax = axes[1]
    ax.bar(x, connectivity, width, color="darkorange")
    ax.set_ylabel("Connectivity (%)")
    ax.set_title("Molecular Connectivity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Input Ablation Results", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_all(
    metrics: TrajectoryMetrics,
    output_dir: str = "analysis_output/figures",
):
    """Generate all standard plots from trajectory metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_structure_lens_heatmap(
        metrics, output_path=str(out / "structure_lens_rmsd.png")
    )
    plot_entropy_heatmap(metrics, output_path=str(out / "attention_entropy.png"))
    plot_bond_precision_heatmap(metrics, output_path=str(out / "bond_precision.png"))
    plot_repa_landscape(metrics, output_path=str(out / "repa_landscape.png"))
    plot_metric_lines(metrics, output_path=str(out / "metric_lines.png"))

    if metrics.coord_rmsd_to_gt is not None:
        plot_structure_lens_heatmap(
            metrics,
            use_gt=True,
            output_path=str(out / "structure_lens_rmsd_gt.png"),
        )

    print(f"Saved figures to {out}/")
