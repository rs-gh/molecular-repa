#!/usr/bin/env python
"""Entry-point for trajectory analysis on a trained tabasco checkpoint.

Usage:
    source .venv/bin/activate
    export PROJECT_ROOT=$(pwd)/src/tabasco
    python playground/mech_interp/run_analysis.py \
        --checkpoint evaluation_checkpoints/baseline.ckpt \
        --num-molecules 50 \
        --num-steps 100 \
        --output-dir playground/mech_interp/results/baseline

To also run ablation experiments:
    python playground/mech_interp/run_analysis.py \
        --checkpoint evaluation_checkpoints/baseline.ckpt \
        --num-molecules 50 \
        --run-ablations \
        --output-dir playground/mech_interp/results/ablations
"""

import argparse
from pathlib import Path

import torch
import numpy as np

import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # repo root for playground.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "tabasco" / "src"))


def load_model_from_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Load a stripped checkpoint and reconstruct the FlowMatchingModel.

    Returns:
        (model, data_stats) tuple.
    """
    from tabasco.models.components.transformer_module import TransformerModule
    from tabasco.models.flow_model import FlowMatchingModel
    from tabasco.flow.interpolate import CenteredMetricInterpolant, DiscreteInterpolant

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    data_stats = ckpt.get("data_stats")
    ckpt.get("hyper_parameters")  # available but not needed for reconstruction

    # Strip state_dict keys to bare module names.
    # Handles both "model.net." and "model.net._orig_mod." (torch.compile).
    net_state = {}
    for k, v in state_dict.items():
        if k.startswith("model.net._orig_mod."):
            net_state[k[len("model.net._orig_mod.") :]] = v
        elif k.startswith("model.net."):
            net_state[k[len("model.net.") :]] = v

    # Infer model dimensions from stripped keys
    linear_embed_w = net_state["linear_embed.weight"]
    hidden_dim = linear_embed_w.shape[0]
    spatial_dim = linear_embed_w.shape[1]

    atom_embed_w = net_state["atom_type_embed.weight"]
    atom_dim = atom_embed_w.shape[0]

    # Count transformer layers
    layer_indices = set()
    for k in net_state:
        if "transformer.layers." in k:
            parts = k.split("transformer.layers.")
            if len(parts) > 1:
                idx = parts[1].split(".")[0]
                if idx.isdigit():
                    layer_indices.add(int(idx))
    num_layers = max(layer_indices) + 1 if layer_indices else 16

    # Detect cross-attention
    has_cross_attn = any("coord_cross_attention" in k for k in net_state)

    # Detect num_heads from attention in_proj_weight: shape [3*hidden_dim, hidden_dim]
    num_heads = 8  # default

    print(
        f"Reconstructing model: hidden_dim={hidden_dim}, spatial_dim={spatial_dim}, "
        f"atom_dim={atom_dim}, num_layers={num_layers}, num_heads={num_heads}, "
        f"cross_attention={has_cross_attn}"
    )

    net = TransformerModule(
        spatial_dim=spatial_dim,
        atom_dim=atom_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        activation="SiLU",
        implementation="reimplemented",
        cross_attention=has_cross_attn,
    )

    # Load weights (already stripped above)
    net.load_state_dict(net_state, strict=True)

    # Build interpolants
    coords_interpolant = CenteredMetricInterpolant(
        key="coords",
        centered=True,
    )
    atomics_interpolant = DiscreteInterpolant(key="atomics")

    model = FlowMatchingModel(
        net=net,
        coords_interpolant=coords_interpolant,
        atomics_interpolant=atomics_interpolant,
        sample_schedule="linear",
    )

    if data_stats is not None:
        model.set_data_stats(data_stats)

    return model, data_stats


def _build_bond_matrices(x_final):
    """Build bond adjacency matrices from generated molecules via RDKit.

    Returns:
        (bond_matrices, valid_count, total_count) or (None, 0, total_count) on failure.
    """
    from tabasco.chem.convert import MoleculeConverter

    converter = MoleculeConverter()
    try:
        mols = converter.from_batch(x_final.cpu())
    except Exception as e:
        print(f"  Warning: MoleculeConverter.from_batch failed: {e}")
        return None, 0, x_final["coords"].shape[0]

    B = len(mols)
    N = x_final["coords"].shape[1]
    valid_count = sum(1 for m in mols if m is not None)
    print(f"  Converted molecules: {valid_count}/{B} valid")

    if valid_count == 0:
        return None, 0, B

    bond_matrices = torch.zeros(B, N, N)
    for bi, mol in enumerate(mols):
        if mol is None:
            continue
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < N and j < N:
                bond_matrices[bi, i, j] = 1.0
                bond_matrices[bi, j, i] = 1.0

    avg_bonds = bond_matrices.sum(dim=(1, 2)).mean().item() / 2
    print(f"  Bond matrices: avg {avg_bonds:.1f} bonds/mol")

    return bond_matrices, valid_count, B


def _print_summary(metrics, valid_mols: int = 0, total_mols: int = 0):
    """Print a summary table of key findings."""
    print("\n" + "=" * 65)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("=" * 65)

    T, L = metrics.coord_rmsd_to_final.shape

    # Structure lens
    print("\nStructure Lens (RMSD to final output):")
    print(f"  {'':20s}  {'Mean':>10s}  {'Last layer':>10s}")
    for label, idx in [("Early (t~0)", 0), ("Mid (t~0.5)", T // 2), ("Late (t~1)", -1)]:
        mean_val = np.nanmean(metrics.coord_rmsd_to_final[idx, :])
        last_val = metrics.coord_rmsd_to_final[idx, -1]
        print(f"  {label:20s}  {mean_val:10.4f}  {last_val:10.4f}")

    # Atom accuracy
    print("\nAtom Type Accuracy:")
    print(f"  {'':20s}  {'Mean':>10s}  {'Last layer':>10s}")
    for label, idx in [("Early (t~0)", 0), ("Mid (t~0.5)", T // 2), ("Late (t~1)", -1)]:
        mean_val = np.nanmean(metrics.atom_accuracy[idx, :])
        last_val = metrics.atom_accuracy[idx, -1]
        print(f"  {label:20s}  {mean_val:10.4f}  {last_val:10.4f}")

    # Entropy
    entropy_mean = np.nanmean(metrics.attn_entropy, axis=-1)  # [T, L]
    print("\nAttention Entropy (normalized, mean over heads):")
    print(f"  {'':20s}  {'Mean':>10s}  {'Last layer':>10s}")
    for label, idx in [("Early (t~0)", 0), ("Mid (t~0.5)", T // 2), ("Late (t~1)", -1)]:
        mean_val = np.nanmean(entropy_mean[idx, :])
        last_val = entropy_mean[idx, -1]
        print(f"  {label:20s}  {mean_val:10.4f}  {last_val:10.4f}")

    # Distance correlation
    if metrics.dist_correlation is not None:
        dc_mean = np.nanmean(metrics.dist_correlation, axis=-1)
        print("\nDistance Correlation (attn vs pairwise distance, mean over heads):")
        print(f"  {'':20s}  {'Mean':>10s}  {'Last layer':>10s}")
        for label, idx in [
            ("Early (t~0)", 0),
            ("Mid (t~0.5)", T // 2),
            ("Late (t~1)", -1),
        ]:
            mean_val = np.nanmean(dc_mean[idx, :])
            last_val = dc_mean[idx, -1]
            print(f"  {label:20s}  {mean_val:10.4f}  {last_val:10.4f}")

    # Bond precision
    if metrics.bond_precision_vals is not None:
        bp_mean = np.nanmean(metrics.bond_precision_vals, axis=-1)
        print(
            f"\nBond Precision@N (mean over heads, {valid_mols}/{total_mols} valid mols):"
        )
        print(f"  {'':20s}  {'Mean':>10s}  {'Last layer':>10s}")
        for label, idx in [
            ("Early (t~0)", 0),
            ("Mid (t~0.5)", T // 2),
            ("Late (t~1)", -1),
        ]:
            mean_val = np.nanmean(bp_mean[idx, :])
            last_val = bp_mean[idx, -1]
            print(f"  {label:20s}  {mean_val:10.4f}  {last_val:10.4f}")

    print("\n" + "=" * 65)


def run_trajectory_analysis(args):
    """Run the full trajectory analysis."""
    from playground.mech_interp.trajectory_analyzer import TrajectoryAnalyzer
    from playground.mech_interp.visualization import plot_all

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    model, data_stats = load_model_from_checkpoint(args.checkpoint, device)
    analyzer = TrajectoryAnalyzer(model, device=device)

    # Phase 1: Generate molecules with analysis capture
    print(
        f"\nPhase 1: Generating {args.num_molecules} molecules, "
        f"{args.num_steps} steps, capture every {args.capture_every_n}..."
    )
    with torch.no_grad():
        x_final, captures = model.sample_with_analysis(
            batch_size=args.num_molecules,
            num_steps=args.num_steps,
            capture_every_n=args.capture_every_n,
        )
    print(f"  Generated {len(captures)} capture points")

    # Phase 2: Compute distance matrices from final coords
    print("\nPhase 2: Computing distance matrices from final coordinates...")
    final_coords = x_final["coords"].cpu()
    dist_matrices = torch.cdist(final_coords, final_coords)  # [B, N, N]
    print(f"  Distance matrices: {dist_matrices.shape}")

    # Phase 3: Compute bond adjacency matrices
    print("\nPhase 3: Computing bond matrices from generated molecules...")
    bond_matrices, valid_count, total_count = _build_bond_matrices(x_final)

    # Phase 4: Run metric computation on pre-generated data
    print("\nPhase 4: Computing trajectory metrics...")
    metrics = analyzer.analyze_generation(
        num_molecules=args.num_molecules,
        num_steps=args.num_steps,
        capture_every_n=args.capture_every_n,
        bond_matrices=bond_matrices,
        dist_matrices=dist_matrices,
        precomputed_x_final=x_final,
        precomputed_captures=captures,
    )

    # Phase 5: Summary, save, plot
    _print_summary(metrics, valid_count, total_count)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "trajectory_metrics.npz"
    metrics.save(str(metrics_path))
    print(f"\nSaved metrics to {metrics_path}")

    plot_all(metrics, output_dir=str(out_dir / "figures"))

    return metrics


def run_ablation_experiments(args):
    """Run the ablation sweep."""
    from playground.mech_interp.trajectory_analyzer import (
        TrajectoryAnalyzer,
        get_standard_ablation_configs,
    )
    from playground.mech_interp.visualization import plot_ablation_summary
    from tabasco.chem.convert import MoleculeConverter
    from rdkit import Chem

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, data_stats = load_model_from_checkpoint(args.checkpoint, device)
    analyzer = TrajectoryAnalyzer(model, device=device)
    converter = MoleculeConverter()

    configs = get_standard_ablation_configs()
    results = []

    for config in configs:
        print(f"\nRunning ablation: {config.label}")
        result = analyzer.analyze_ablation(
            config,
            num_molecules=args.num_molecules,
            num_steps=args.num_steps,
        )

        # Evaluate molecular quality
        try:
            mols = converter.from_batch(result["molecules"])
            valid = sum(1 for m in mols if m is not None)
            result["validity"] = valid / len(mols) * 100

            # Compute connectivity (single connected component)
            connected = 0
            for mol in mols:
                if mol is None:
                    continue
                frags = Chem.GetMolFrags(mol)
                if len(frags) == 1:
                    connected += 1
            result["connectivity"] = connected / len(mols) * 100

            print(
                f"  Validity: {result['validity']:.1f}%, "
                f"Connectivity: {result['connectivity']:.1f}%"
            )
        except Exception as e:
            print(f"  Error evaluating molecules: {e}")
            result["validity"] = 0.0
            result["connectivity"] = 0.0

        results.append(result)

    # Save and plot
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out_dir / "ablation_results.npz"),
        **{
            "labels": [r["label"] for r in results],
            "validity": [r["validity"] for r in results],
            "connectivity": [r["connectivity"] for r in results],
        },
    )

    plot_ablation_summary(
        results, output_path=str(out_dir / "figures" / "ablation_summary.png")
    )

    # Print ablation summary table
    print("\n" + "=" * 65)
    print("ABLATION RESULTS")
    print("=" * 65)
    print(f"  {'Condition':40s}  {'Valid%':>8s}  {'Connect%':>8s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['label']:40s}  {r['validity']:8.1f}  {r['connectivity']:8.1f}")
    print("=" * 65)

    return results


def main():
    parser = argparse.ArgumentParser(description="Tabasco trajectory analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to stripped checkpoint (.ckpt)",
    )
    parser.add_argument("--num-molecules", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--capture-every-n", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=str, default="playground/mech_interp/results"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Also run input ablation experiments",
    )

    args = parser.parse_args()

    run_trajectory_analysis(args)

    if args.run_ablations:
        run_ablation_experiments(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
