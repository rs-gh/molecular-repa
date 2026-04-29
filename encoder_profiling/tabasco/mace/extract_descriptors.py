"""
Minimal script to extract MACE descriptors from .xyz files and save to disk.
Supports batched processing for large datasets.
"""

import argparse
import os
from pathlib import Path

# E3NN doesn't work without this env variable set
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import ase.io as aio
import numpy as np
import torch
from e3nn import o3
from mace import data
from mace.calculators import mace_off
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric
from tqdm import tqdm


def extract_descriptors_batched(
    in_path: str,
    out_path: str,
    batch_size: int = 64,
    model_name: str = "medium",
    invariants_only: bool = True,
    num_layers: int = -1,
):
    """
    Extract MACE descriptors from atoms in .xyz/.extxyz file and save to disk.

    Args:
        in_path: Path to input .xyz/.extxyz file
        out_path: Path to output .npz file
        batch_size: Number of structures to process in parallel
        model_name: MACE model to use (small, medium, large)
        invariants_only: If True, only extract invariant features
        num_layers: Number of interaction layers to extract (-1 for all)
    """
    # Load atoms from file
    print(f"Loading atoms from {in_path}...")
    atoms_list = aio.read(in_path, index=":")
    print(f"Loaded {len(atoms_list)} structures")

    # Initialize device and calculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    calc = mace_off(model_name, device=device)

    # Get model parameters for descriptor extraction
    model = calc.models[0]
    num_interactions = int(model.num_interactions)
    if num_layers == -1:
        num_layers = num_interactions

    # This calculates how many features to keep based on requested layers
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
    per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
    per_layer_features[-1] = num_invariant_features
    to_keep = int(np.sum(per_layer_features[:num_layers]))

    # Prepare data loader
    print("Creating data loader...")
    keyspec = data.KeySpecification(
        info_keys=calc.info_keys, arrays_keys=calc.arrays_keys
    )

    configs = [
        data.config_from_atoms(x, key_specification=keyspec, head_name=calc.head)
        for x in atoms_list
    ]

    dataset = [
        data.AtomicData.from_config(
            config,
            z_table=calc.z_table,
            cutoff=calc.r_max,
            heads=calc.available_heads,
        )
        for config in configs
    ]

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Extract descriptors batch by batch
    all_descriptors = []
    all_indices = []  # To track which atoms object each descriptor belongs to

    print(f"Extracting descriptors (batch_size={batch_size})...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            batch = batch.to(device)

            # Forward pass to get node features
            output = model(batch.to_dict(), compute_force=False)
            node_feats = output["node_feats"]

            # Extract invariant features if requested
            if invariants_only:
                descriptors = extract_invariant(
                    node_feats,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )
            else:
                descriptors = node_feats

            # Keep only requested layers
            descriptors = descriptors[:, :to_keep].cpu().numpy()

            # Track which structure each atom belongs to
            batch_indices = batch["batch"].cpu().numpy()
            global_indices = batch_indices + batch_idx * batch_size

            all_descriptors.append(descriptors)
            all_indices.append(global_indices)

    # Concatenate all results
    all_descriptors = np.concatenate(all_descriptors, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    print(f"Extracted descriptors shape: {all_descriptors.shape}")
    print(f"(total_atoms, features) where features = {to_keep}")

    # Save to disk
    print(f"Saving to {out_path}...")
    np.savez_compressed(
        out_path,
        descriptors=all_descriptors,
        structure_indices=all_indices,
        num_structures=len(atoms_list),
        num_layers=num_layers,
        invariants_only=invariants_only,
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MACE descriptors from .xyz/.extxyz files"
    )
    parser.add_argument(
        "--in-file", type=str, required=True, help="Input .xyz or .extxyz file"
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Output .npz file (default: input_file_descriptors.npz)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for parallel processing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="MACE model size",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=-1,
        help="Number of layers to extract (-1 for all)",
    )
    parser.add_argument(
        "--full-descriptors",
        action="store_true",
        help="Extract full descriptors (not just invariants)",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.out_file is None:
        in_path = Path(args.in_file)
        args.out_file = str(in_path.parent / f"{in_path.stem}_descriptors.npz")

    extract_descriptors_batched(
        in_path=args.in_file,
        out_path=args.out_file,
        batch_size=args.batch_size,
        model_name=args.model,
        invariants_only=not args.full_descriptors,
        num_layers=args.num_layers,
    )
