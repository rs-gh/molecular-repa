"""
Evaluate REPA cosine similarity stratified by timestep.

Key question: Does the MACE negative gap (-0.30) disappear at high t?

The identity baseline achieves 0.861 on clean molecules. REPA val reports 0.56
averaged across ALL timesteps. But at t~0 the denoiser sees pure noise, so low
cos_sim is expected. If cos_sim at t=0.9 exceeds the identity baseline, the
aggregate metric was misleading and REPA does learn useful structure.

Usage (requires GPU and a trained REPA checkpoint):
  srun --partition=ampere --gres=gpu:1 --time=00:30:00 --pty bash
  source .venv/bin/activate
  export PROJECT_ROOT=$(pwd)/src/tabasco
  python -u playground/projector/timestep_stratified_eval.py \\
      --checkpoint /path/to/checkpoint.ckpt \\
      --dataset geom  # or qm9

The script:
  1. Loads a trained REPA checkpoint (FlowMatchingModel with encoder + projector)
  2. Loads validation data
  3. For each fixed t in {0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}:
     - Creates FlowPaths at that fixed t
     - Runs model forward to get hidden states
     - Computes cosine similarity between projected hidden states and encoder targets
  4. Reports per-timestep cos_sim and compares to identity baseline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]

TIMESTEPS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

# Identity baselines from projector analysis (to be updated after encoder_analysis.py rerun)
IDENTITY_BASELINES = {
    "CheMeleon": 0.455,  # hidden_dim=256, will be updated
    "MACE": 0.861,
    "GearNet": 0.426,
}


def load_model(checkpoint_path, device="cuda"):
    """Load a trained REPA model from checkpoint.

    Returns the FlowMatchingModel (not the Lightning wrapper).
    """
    project_root = os.environ.get("PROJECT_ROOT", str(REPO_ROOT / "src" / "tabasco"))
    sys.path.insert(0, os.path.join(project_root, "src"))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint if available
    if "hyper_parameters" in checkpoint:
        print(
            f"  Checkpoint has hyper_parameters keys: {list(checkpoint['hyper_parameters'].keys())}"
        )

    # We need to instantiate the model from the checkpoint's config.
    # Lightning checkpoints store the full state_dict under "state_dict".
    # The model is accessible as lightning_module.model after loading.
    from tabasco.models.lightning_tabasco import LightningTabasco

    # Try loading via Lightning's load_from_checkpoint
    # This requires the original config, so we use a simpler approach:
    # instantiate from Hydra config stored alongside the checkpoint.
    ckpt_dir = Path(checkpoint_path).parent

    # Look for Hydra config in the checkpoint directory or parent
    config_path = None
    for candidate in [
        ckpt_dir / ".hydra" / "config.yaml",
        ckpt_dir.parent / ".hydra" / "config.yaml",
        ckpt_dir.parent.parent / ".hydra" / "config.yaml",
    ]:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is not None:
        print(f"  Found Hydra config at {config_path}")
        from omegaconf import OmegaConf
        import hydra

        cfg = OmegaConf.load(config_path)

        # Instantiate model from config
        model = hydra.utils.instantiate(cfg.model)
        optimizer_fn = hydra.utils.instantiate(cfg.optimizer)

        lightning_module = LightningTabasco(model=model, optimizer=optimizer_fn)

        # Load state dict
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = lightning_module.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        # Restore data stats
        if "data_stats" in checkpoint:
            lightning_module.model.data_stats = checkpoint["data_stats"]

        model = lightning_module.model
    else:
        print("  No Hydra config found - attempting direct state_dict load")
        print("  This requires the model architecture to match the checkpoint.")
        raise FileNotFoundError(
            f"Could not find .hydra/config.yaml near {checkpoint_path}. "
            "Please provide a checkpoint directory with the Hydra config."
        )

    model = model.to(device).eval()

    # Verify REPA is configured
    if not hasattr(model, "repa_loss") or model.repa_loss is None:
        raise ValueError("Checkpoint does not have REPA loss configured")

    encoder_name = type(model.repa_loss.encoder).__name__
    print(f"  Model loaded: encoder={encoder_name}")
    print(f"  Projector: {model.repa_loss.projector}")

    return model, encoder_name


def load_dataloader(dataset="geom", batch_size=64, n_batches=None):
    """Load validation data."""
    project_root = os.environ.get("PROJECT_ROOT", str(REPO_ROOT / "src" / "tabasco"))
    sys.path.insert(0, os.path.join(project_root, "src"))

    from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
    from torch.utils.data import DataLoader

    if dataset == "geom":
        data_dir = os.path.join(project_root, "data", "processed_geom_train.pt")
        lmdb_dir = os.path.join(project_root, "data", "lmdb_geom")
    elif dataset == "qm9":
        data_dir = os.path.join(project_root, "data", "processed_qm9_train.pt")
        lmdb_dir = os.path.join(project_root, "data", "lmdb_qm9")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    ds = UnconditionalLMDBDataset(
        data_dir=data_dir,
        split="train",
        add_random_rotation=False,
        add_random_permutation=False,
        reorder_to_smiles_order=False,
        remove_hydrogens=True,
        lmdb_dir=lmdb_dir,
    )

    # Use a fixed subset for reproducibility
    rng = np.random.RandomState(42)
    n_total = min(1000, len(ds))
    indices = rng.choice(len(ds), size=n_total, replace=False)

    from torch.utils.data import Subset

    subset = Subset(ds, indices.tolist())

    # Use collate_fn from the dataset
    from tensordict.nn import TensorDictCollator

    collator = TensorDictCollator()

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    if n_batches is not None:
        # Limit number of batches
        class LimitedLoader:
            def __init__(self, loader, n):
                self.loader = loader
                self.n = n

            def __iter__(self):
                for i, batch in enumerate(self.loader):
                    if i >= self.n:
                        break
                    yield batch

            def __len__(self):
                return min(self.n, len(self.loader))

        loader = LimitedLoader(loader, n_batches)

    print(f"  Dataset: {dataset}, {n_total} samples, {batch_size} batch_size")
    return loader


def evaluate_at_timestep(model, batch, t_value, device):
    """Evaluate REPA cos_sim at a fixed timestep.

    Returns per-atom cosine similarities and count.
    """
    batch = batch.to(device)

    # Create FlowPath at fixed t
    batch_size = batch["padding_mask"].shape[0]
    t = torch.full((batch_size,), t_value, device=device)
    path = model._create_path(batch, t=t)

    # Forward pass with hidden states
    with torch.no_grad():
        pred = model._call_net(path.x_t, path.t, return_hidden_states=True)

    # Compute encoder targets from clean molecules
    padding_mask = pred["padding_mask"]
    real_mask = ~padding_mask

    repa = model.repa_loss

    try:
        smiles = list(path.x_1.get_non_tensor("smiles"))
    except (KeyError, AttributeError):
        smiles = None

    try:
        lmdb_keys = list(path.x_1.get_non_tensor("lmdb_key"))
    except (KeyError, AttributeError):
        lmdb_keys = None

    with torch.no_grad():
        target_repr = repa.encoder(
            path.x_1["coords"],
            path.x_1["atomics"],
            padding_mask,
            smiles=smiles,
            lmdb_keys=lmdb_keys,
        )

    # Project hidden states
    hs_keys = sorted(k for k in pred.keys() if k.startswith("hidden_states_"))
    h_fused = torch.cat([pred[k] for k in hs_keys], dim=-1)

    with torch.no_grad():
        projected = repa.projector(h_fused)

    cos_sim = F.cosine_similarity(projected[real_mask], target_repr[real_mask], dim=-1)

    return cos_sim.cpu(), real_mask.sum().item()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate REPA cosine similarity stratified by timestep"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .ckpt file"
    )
    parser.add_argument("--dataset", type=str, default="geom", choices=["geom", "qm9"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--n-batches", type=int, default=None, help="Limit eval batches"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--timesteps", type=float, nargs="+", default=TIMESTEPS)
    args = parser.parse_args()

    print(f"{'='*60}")
    print("  Timestep-Stratified REPA Evaluation")
    print(f"{'='*60}")

    # Load model
    print("\n  Loading model...")
    model, encoder_name = load_model(args.checkpoint, device=args.device)

    # Load data
    print("\n  Loading data...")
    loader = load_dataloader(args.dataset, args.batch_size, args.n_batches)

    # Evaluate at each timestep
    results = {}
    for t_val in args.timesteps:
        print(f"\n  t = {t_val:.2f}:")
        all_sims = []
        n_atoms_total = 0
        t0 = time.time()

        for batch in loader:
            cos_sims, n_atoms = evaluate_at_timestep(model, batch, t_val, args.device)
            all_sims.append(cos_sims)
            n_atoms_total += n_atoms

        all_sims = torch.cat(all_sims)
        elapsed = time.time() - t0

        results[t_val] = {
            "mean": all_sims.mean().item(),
            "std": all_sims.std().item(),
            "median": all_sims.median().item(),
            "n_atoms": n_atoms_total,
        }

        print(
            f"    cos_sim = {results[t_val]['mean']:.4f} +/- {results[t_val]['std']:.4f}"
        )
        print(f"    median  = {results[t_val]['median']:.4f}")
        print(f"    {n_atoms_total} atoms, {elapsed:.1f}s")

    # Summary table
    identity_baseline = IDENTITY_BASELINES.get(
        encoder_name.replace("Cached", "").replace("Encoder", ""),
        None,
    )

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {encoder_name}")
    print(f"{'='*60}")
    print(f"  {'t':>6} | {'cos_sim':>8} | {'std':>6} | {'vs ID baseline':>14}")
    print(f"  {'-'*42}")
    for t_val in sorted(results.keys()):
        r = results[t_val]
        gap_str = ""
        if identity_baseline is not None:
            gap = r["mean"] - identity_baseline
            gap_str = f"{gap:+.4f}"
        print(f"  {t_val:>6.2f} | {r['mean']:>8.4f} | {r['std']:>6.4f} | {gap_str:>14}")

    if identity_baseline is not None:
        print(f"\n  Identity baseline (test): {identity_baseline:.4f}")
        best_t = max(results.keys(), key=lambda t: results[t]["mean"])
        print(f"  Best cos_sim at t={best_t:.2f}: {results[best_t]['mean']:.4f}")
        if results[best_t]["mean"] > identity_baseline:
            print(
                f"  -> REPA exceeds identity at t={best_t:.2f} (gap = {results[best_t]['mean'] - identity_baseline:+.4f})"
            )
        else:
            print(
                f"  -> REPA below identity even at best t (gap = {results[best_t]['mean'] - identity_baseline:+.4f})"
            )

    # Save results
    output_path = Path(__file__).parent / "timestep_results.json"
    save_data = {
        "encoder": encoder_name,
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "identity_baseline": identity_baseline,
        "timesteps": {str(k): v for k, v in results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
