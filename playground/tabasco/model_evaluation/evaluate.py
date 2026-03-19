"""Evaluate a trained molecular generation checkpoint.

Generates molecules, computes metrics with bootstrap CIs, and saves results.

Usage:
    # Stripped checkpoint (from strip_checkpoint.py):
    python playground/tabasco/model_evaluation/evaluate.py \
        --checkpoint evaluation_checkpoints/tabasco/baseline.ckpt \
        --num_mols 1000 --output_dir evaluation_results/baseline/

    # Full training checkpoint:
    python playground/tabasco/model_evaluation/evaluate.py \
        --checkpoint /rds/.../last.ckpt \
        --num_mols 1000 --output_dir evaluation_results/baseline/

    # Batch evaluate all stripped checkpoints:
    python playground/tabasco/model_evaluation/evaluate.py --all
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

# --- Project imports ---
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "tabasco" / "src"))

from tabasco.chem.convert import MoleculeConverter
from tabasco.flow.interpolate import SDEMetricInterpolant, DiscreteInterpolant
from tabasco.flow.time_factor import InverseTimeFactor
from tabasco.models.components.transformer_module import TransformerModule
from tabasco.models.flow_model import FlowMatchingModel
from tabasco.models.lightning_tabasco import LightningTabasco
from tabasco.sample.noise_schedule import SampleNoiseSchedule
from tabasco.utils.metrics import (
    MolecularValidity,
    MolecularConnectivity,
    MolecularUniqueness,
    MolecularNovelty,
    MolecularDiversity,
    MolecularQEDValue,
    MolecularLogP,
    MolecularLipinski,
    AtomTypeDistribution,
)


# ---------------------------------------------------------------------------
# Model reconstruction for stripped checkpoints
# ---------------------------------------------------------------------------


def _build_time_factor():
    """Build InverseTimeFactor matching flow_model.yaml defaults."""
    return InverseTimeFactor(max_value=100.0, min_value=0.05, zero_before=0.0, eps=1e-6)


def _build_model_from_config(model_config: dict, data_stats: dict) -> FlowMatchingModel:
    """Reconstruct a FlowMatchingModel from stripped checkpoint config.

    All GEOM runs share the same interpolant/schedule config (flow_model.yaml),
    only the TransformerModule params differ.
    """
    net = TransformerModule(
        spatial_dim=data_stats.get("spatial_dim", 3),
        atom_dim=data_stats.get("atom_dim", 9),
        hidden_dim=model_config.get("hidden_dim", 128),
        num_layers=model_config.get("num_layers", 16),
        num_heads=model_config.get("num_heads", 8),
        activation="SiLU",
        implementation="reimplemented",
        cross_attention=model_config.get("cross_attention", True),
    )

    coords_interpolant = SDEMetricInterpolant(
        key="coords",
        loss_weight=1.0,
        scale_noise_by_log_num_atoms=False,
        noise_scale=1.0,
        langevin_sampling_schedule=SampleNoiseSchedule(cutoff=0.9),
        white_noise_sampling_scale=0.01,
        time_factor=_build_time_factor(),
    )

    atomics_interpolant = DiscreteInterpolant(
        key="atomics",
        loss_weight=0.1,
        time_factor=_build_time_factor(),
    )

    model = FlowMatchingModel(
        net=net,
        coords_interpolant=coords_interpolant,
        atomics_interpolant=atomics_interpolant,
        time_distribution="beta",
        time_alpha_factor=1.8,
        num_random_augmentations=7,
        sample_schedule="log",
        compile=False,
    )
    return model


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a checkpoint (stripped or full) and return (model, data_stats, meta).

    Returns:
        model: FlowMatchingModel in eval mode
        data_stats: dict with sampling metadata
        meta: dict with epoch, global_step, etc.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if ckpt.get("stripped", False):
        # --- Stripped checkpoint: reconstruct model ---
        model_config = ckpt.get("model_config", {})
        data_stats = ckpt["data_stats"]
        model = _build_model_from_config(model_config, data_stats)

        # Load state dict — keys are "model.net.*", strip "model." prefix
        net_state = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("model.net."):
                # Strip "model." prefix so keys become "net.*"
                net_state[k[len("model.") :]] = v
        model.load_state_dict(net_state, strict=False)
        print(
            f"  Loaded stripped checkpoint (epoch {ckpt.get('epoch')}, "
            f"step {ckpt.get('global_step')})"
        )
    else:
        # --- Full training checkpoint: use Lightning ---
        lightning_module = LightningTabasco.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
        model = lightning_module.model
        data_stats = ckpt.get("data_stats", {})
        print(
            f"  Loaded full checkpoint (epoch {ckpt.get('epoch')}, "
            f"step {ckpt.get('global_step')})"
        )

    model.set_data_stats(data_stats)
    model.eval()
    model.to(device)

    meta = {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "stripped": ckpt.get("stripped", False),
    }
    return model, data_stats, meta


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_molecules(
    model: FlowMatchingModel,
    num_mols: int,
    num_steps: int = 100,
    batch_size: int = 256,
) -> list:
    """Generate molecules in batches.

    Returns:
        List[Chem.Mol | None] — None for failed conversions.
    """
    converter = MoleculeConverter()
    all_mols = []
    remaining = num_mols

    while remaining > 0:
        bs = min(batch_size, remaining)
        print(f"  Sampling batch of {bs} ({num_mols - remaining}/{num_mols} done)...")
        batch = model.sample(batch_size=bs, num_steps=num_steps)
        mols = converter.from_batch(batch, sanitize=True)
        all_mols.extend(mols)
        remaining -= bs

    return all_mols[:num_mols]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    molecules: list, train_smiles: list = None, exclude: set = None
) -> dict:
    """Compute all metrics on a list of generated molecules.

    Args:
        exclude: set of metric names to skip (e.g. {"diversity"} for bootstrap speed).

    Returns:
        dict mapping metric_name -> float value
    """
    exclude = exclude or set()
    metrics = {
        "validity": MolecularValidity(),
        "connectivity": MolecularConnectivity(),
        "uniqueness": MolecularUniqueness(),
        "qed": MolecularQEDValue(),
        "logP": MolecularLogP(),
        "lipinski": MolecularLipinski(),
        "diversity": MolecularDiversity(),
    }

    if train_smiles is not None:
        metrics["novelty"] = MolecularNovelty(original_smiles=train_smiles)
        metrics["atom_type_dist"] = AtomTypeDistribution(original_smiles=train_smiles)

    results = {}
    for name, metric in metrics.items():
        if name in exclude:
            continue
        metric.update(molecules)
        val = metric.compute()
        results[name] = val.item() if hasattr(val, "item") else float(val)

    return results


def compute_posebusters(molecules: list) -> dict:
    """Run PoseBusters validation on generated molecules.

    Returns:
        dict mapping pb_<check_name> -> float fraction passing
    """
    from posebusters import PoseBusters
    from yaml import safe_load

    config_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "tabasco"
        / "src"
        / "tabasco"
        / "utils"
        / "posebusters_no_strain.yaml"
    )
    cfg = safe_load(open(config_path, encoding="utf-8"))
    pb = PoseBusters(config=cfg)

    valid_mols = [mol for mol in molecules if mol is not None]
    total = len(molecules)

    if not valid_mols:
        return {}

    try:
        pb_results = pb.bust(mol_pred=valid_mols)
    except RuntimeError as e:
        print(f"  PoseBusters error: {e}")
        return {}

    results = {}
    pb_pass = 0.0
    for _, row in pb_results.iterrows():
        pb_pass += 0 if row.isin([False]).any() else 1
    results["pb_intersection"] = pb_pass / total

    for column in pb_results.columns:
        fraction = (1.0 * pb_results[column].sum()) / total
        results[f"pb_{column}"] = fraction

    return results


def compute_fcd(molecules: list, train_smiles: list, device: str = None) -> float:
    """Compute Frechet ChemNet Distance between generated and training molecules."""
    from fcd_torch import FCD

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_smiles = []
    for mol in molecules:
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    gen_smiles.append(smi)
            except Exception:
                pass

    if len(gen_smiles) < 10:
        print("  Warning: fewer than 10 valid SMILES for FCD")
        return float("nan")

    fcd_calc = FCD(device=device, n_jobs=1)
    return float(fcd_calc(train_smiles, gen_smiles))


def bootstrap_ci(
    molecules: list,
    train_smiles: list = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence intervals for all metrics.

    Returns:
        dict mapping metric_name -> {"mean": float, "ci_low": float, "ci_high": float}
    """
    rng = np.random.RandomState(seed)
    n = len(molecules)
    alpha = (1 - ci) / 2

    # Collect bootstrap samples
    bootstrap_results = []
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = [molecules[j] for j in indices]
        results = compute_metrics(sample, train_smiles, exclude={"diversity"})
        bootstrap_results.append(results)

    # Aggregate
    metric_names = bootstrap_results[0].keys()
    summary = {}
    for name in metric_names:
        values = [r[name] for r in bootstrap_results]
        summary[name] = {
            "mean": float(np.mean(values)),
            "ci_low": float(np.percentile(values, 100 * alpha)),
            "ci_high": float(np.percentile(values, 100 * (1 - alpha))),
            "std": float(np.std(values)),
        }

    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_molecules_sdf(molecules: list, path: str):
    """Save valid molecules to an SDF file."""
    writer = Chem.SDWriter(path)
    for mol in molecules:
        if mol is not None:
            writer.write(mol)
    writer.close()


def save_per_molecule_csv(molecules: list, path: str):
    """Save per-molecule properties to CSV."""
    from rdkit.Chem import Descriptors, Lipinski

    rows = []
    for i, mol in enumerate(molecules):
        if mol is None:
            rows.append({"idx": i, "valid": False})
            continue
        try:
            smi = Chem.MolToSmiles(mol)
            rows.append(
                {
                    "idx": i,
                    "valid": True,
                    "smiles": smi,
                    "qed": Descriptors.qed(mol),
                    "logP": Descriptors.MolLogP(mol),
                    "mol_weight": Descriptors.MolWt(mol),
                    "num_atoms": mol.GetNumAtoms(),
                    "num_bonds": mol.GetNumBonds(),
                    "num_rings": mol.GetRingInfo().NumRings(),
                    "hbd": Lipinski.NumHDonors(mol),
                    "hba": Lipinski.NumHAcceptors(mol),
                }
            )
        except Exception:
            rows.append({"idx": i, "valid": False})

    import csv

    fieldnames = [
        "idx",
        "valid",
        "smiles",
        "qed",
        "logP",
        "mol_weight",
        "num_atoms",
        "num_bonds",
        "num_rings",
        "hbd",
        "hba",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("evaluation_checkpoints/tabasco")
RESULTS_DIR = Path("evaluation_results")

ALL_MODELS = {
    "baseline": CHECKPOINT_DIR / "baseline.ckpt",
    "additive_fused": CHECKPOINT_DIR / "additive_fused.ckpt",
    "additive_same": CHECKPOINT_DIR / "additive_same.ckpt",
    "tradeoff_fused": CHECKPOINT_DIR / "tradeoff_fused.ckpt",
    "tradeoff_same": CHECKPOINT_DIR / "tradeoff_same.ckpt",
}


def evaluate_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    num_mols: int = 1000,
    num_steps: int = 100,
    batch_size: int = 256,
    train_smiles_path: str = None,
    n_bootstrap: int = 1000,
    device: str = "cpu",
):
    """Full evaluation pipeline for one checkpoint."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load training SMILES if available
    train_smiles = None
    if train_smiles_path and Path(train_smiles_path).exists():
        print(f"Loading training SMILES from {train_smiles_path}...")
        with open(train_smiles_path) as f:
            train_smiles = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(train_smiles)} training SMILES")

    # Load model
    model, data_stats, meta = load_checkpoint(checkpoint_path, device=device)

    # Generate molecules
    t0 = time.time()
    molecules = generate_molecules(model, num_mols, num_steps, batch_size)
    gen_time = time.time() - t0
    print(f"  Generated {len(molecules)} molecules in {gen_time:.1f}s")

    # Point estimates
    print("Computing point-estimate metrics...")
    point_metrics = compute_metrics(molecules, train_smiles)
    for k, v in point_metrics.items():
        print(f"  {k}: {v:.4f}")

    # PoseBusters geometric quality checks
    print("Computing PoseBusters checks...")
    t0 = time.time()
    pb_metrics = compute_posebusters(molecules)
    pb_time = time.time() - t0
    print(f"  PoseBusters done in {pb_time:.1f}s")
    for k, v in pb_metrics.items():
        print(f"  {k}: {v:.4f}")
    point_metrics.update(pb_metrics)

    # FCD (Frechet ChemNet Distance)
    fcd_value = float("nan")
    if train_smiles is not None:
        print("Computing FCD...")
        t0 = time.time()
        fcd_value = compute_fcd(molecules, train_smiles, device=device)
        fcd_time = time.time() - t0
        print(f"  FCD: {fcd_value:.4f} (computed in {fcd_time:.1f}s)")
        point_metrics["fcd"] = fcd_value

    # Bootstrap CIs (on core metrics only — PB and FCD are too slow to bootstrap)
    print(f"Computing bootstrap CIs ({n_bootstrap} resamples)...")
    t0 = time.time()
    ci_metrics = bootstrap_ci(molecules, train_smiles, n_bootstrap=n_bootstrap)
    ci_time = time.time() - t0
    print(f"  Bootstrap done in {ci_time:.1f}s")
    for k, v in ci_metrics.items():
        print(f"  {k}: {v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

    # Save outputs
    summary = {
        "checkpoint": str(checkpoint_path),
        "meta": meta,
        "generation": {
            "num_mols": num_mols,
            "num_steps": num_steps,
            "time_seconds": gen_time,
        },
        "metrics": point_metrics,
        "bootstrap_ci": ci_metrics,
    }
    json_path = out / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved metrics to {json_path}")

    sdf_path = out / "molecules.sdf"
    save_molecules_sdf(molecules, str(sdf_path))
    print(f"  Saved molecules to {sdf_path}")

    csv_path = out / "molecules.csv"
    save_per_molecule_csv(molecules, str(csv_path))
    print(f"  Saved per-molecule properties to {csv_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate molecular generation checkpoints"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_mols", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--train_smiles",
        type=str,
        default="evaluation_checkpoints/tabasco/geom_train_smiles.txt",
        help="Path to training SMILES for novelty/FCD",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models in evaluation_checkpoints/tabasco/",
    )
    args = parser.parse_args()

    if args.all:
        for name, ckpt_path in ALL_MODELS.items():
            if not ckpt_path.exists():
                print(f"SKIP {name}: {ckpt_path} not found")
                continue
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")
            evaluate_checkpoint(
                checkpoint_path=str(ckpt_path),
                output_dir=str(RESULTS_DIR / name),
                num_mols=args.num_mols,
                num_steps=args.num_steps,
                batch_size=args.batch_size,
                train_smiles_path=args.train_smiles,
                n_bootstrap=args.n_bootstrap,
                device=args.device,
            )
        print(f"\nAll done. Results in {RESULTS_DIR}/")
    else:
        if not args.checkpoint or not args.output_dir:
            parser.error("Provide --checkpoint and --output_dir, or use --all")
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_mols=args.num_mols,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            train_smiles_path=args.train_smiles,
            n_bootstrap=args.n_bootstrap,
            device=args.device,
        )


if __name__ == "__main__":
    main()
