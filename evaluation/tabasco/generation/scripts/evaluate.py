"""Evaluate a trained molecular generation checkpoint.

Generates molecules, computes evaluation metrics, and saves results.

"Evaluation" metrics = post-hoc generation from final checkpoints (1000 mols, on disk).
"Validation" metrics = computed during training by callbacks (100 mols/epoch, logged to WandB).
See evaluation/tabasco/generation/scripts/geom/compile_wandb_curves.py for validation metrics.

Usage:
    # Stripped checkpoint (from strip_checkpoint.py):
    python evaluation/tabasco/generation/scripts/evaluate.py \
        --checkpoint evaluation/tabasco/checkpoints/geom/baseline.ckpt \
        --num_mols 1000 \
        --output_dir evaluation/tabasco/generation/results/geom/evaluation/baseline/

    # Full training checkpoint:
    python evaluation/tabasco/generation/scripts/evaluate.py \
        --checkpoint /rds/.../last.ckpt \
        --num_mols 1000 \
        --output_dir evaluation/tabasco/generation/results/geom/evaluation/baseline/

    # Batch evaluate all stripped checkpoints:
    python evaluation/tabasco/generation/scripts/evaluate.py --all
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

# parents[4] = repo root. `src/tabasco/src/` is where the tabasco package lives;
# tabasco is typically pip-installed via pyproject, but this fallback ensures
# imports work even from a fresh venv that hasn't installed the editable pkg.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src" / "tabasco" / "src"))

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


def _default_device():
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _infer_net_config(state_dict: dict) -> dict:
    """Infer TransformerModule architecture from state_dict weight shapes."""
    keys = {}
    for k, v in state_dict.items():
        if "model.net" not in k:
            continue
        clean = (
            k.replace("model.net.", "")
            .replace("._orig_mod.", ".")
            .replace("._orig_mod", "")
            .lstrip(".")
        )
        keys[clean] = v

    if not keys:
        return {}

    config = {}
    for k, v in keys.items():
        if "mha.in_proj_weight" in k and v.dim() == 2:
            config["hidden_dim"] = v.shape[1]
            config["num_heads"] = v.shape[1] // 16
            break
    layer_indices = set()
    for k in keys:
        if "transformer.layers." in k:
            for part in k.split("."):
                if part.isdigit():
                    layer_indices.add(int(part))
                    break
    if layer_indices:
        config["num_layers"] = max(layer_indices) + 1
    config["cross_attention"] = any("cross_attention" in k for k in keys)
    return config


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


def load_checkpoint(checkpoint_path: str, device: str = None):
    """Load a checkpoint (stripped or full) and return (model, data_stats, meta).

    Returns:
        model: FlowMatchingModel in eval mode
        data_stats: dict with sampling metadata
        meta: dict with epoch, global_step, etc.
    """
    if device is None:
        device = _default_device()
    print(f"Loading checkpoint: {checkpoint_path}")
    # Load to CPU first to avoid GPU OOM from full checkpoint, then .to(device) below
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if ckpt.get("stripped", False):
        # --- Stripped checkpoint: reconstruct model ---
        model_config = ckpt.get("model_config", {})
        data_stats = ckpt["data_stats"]
        model = _build_model_from_config(model_config, data_stats)

        # Load state dict - keys are "model.net.*", strip "model." prefix
        # Also strip _orig_mod. prefix from torch.compile (handles old checkpoints)
        net_state = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("model.net."):
                clean_key = k[len("model.") :].replace("._orig_mod", "")
                net_state[clean_key] = v

        # Sanity check: strict=True to catch key mismatches immediately
        missing, unexpected = model.load_state_dict(net_state, strict=False)
        if missing:
            print(
                f"  WARNING: {len(missing)} missing keys in model (not in checkpoint):"
            )
            for k in missing[:10]:
                print(f"    {k}")
        if unexpected:
            print(
                f"  WARNING: {len(unexpected)} unexpected keys in checkpoint (not in model):"
            )
            for k in unexpected[:10]:
                print(f"    {k}")
        loaded_count = len(net_state) - len(unexpected)
        model_count = len(model.state_dict())
        print(
            f"  Loaded stripped checkpoint (epoch {ckpt.get('epoch')}, "
            f"step {ckpt.get('global_step')}, "
            f"{loaded_count}/{model_count} params matched)"
        )
        if loaded_count == 0:
            raise RuntimeError(
                "No parameters were loaded! Check state_dict key format. "
                f"Checkpoint keys look like: {list(ckpt['state_dict'].keys())[:3]}"
            )
    else:
        # --- Full training checkpoint ---
        # Try Lightning's load_from_checkpoint first (works when model was saved
        # in hyper_parameters). Falls back to manual reconstruction if model was
        # excluded via save_hyperparameters(ignore=['model']).
        hyper_params = ckpt.get("hyper_parameters", {})
        if "model" in hyper_params:
            lightning_module = LightningTabasco.load_from_checkpoint(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model = lightning_module.model
        else:
            # Model not in hparams (ignore=['model'] checkpoints, e.g. MACE runs).
            # Reconstruct from state_dict, same as stripped path.
            model_config = _infer_net_config(ckpt["state_dict"])
            data_stats_raw = ckpt.get("data_stats", {})
            model = _build_model_from_config(model_config, data_stats_raw)
            net_state = {}
            for k, v in ckpt["state_dict"].items():
                if k.startswith("model.net."):
                    clean_key = k[len("model.") :].replace("._orig_mod", "")
                    net_state[clean_key] = v
            model.load_state_dict(net_state, strict=False)
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
# Generation
# ---------------------------------------------------------------------------


def generate_molecules(model, num_mols, num_steps, batch_size):
    """Generate molecules via flow matching sampling.

    Returns:
        list of (rdkit.Mol, dict) tuples, or (None, dict) if invalid.
    """
    converter = MoleculeConverter()
    all_mols = []

    remaining = num_mols
    generated = 0
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        print(f"  Sampling batch of {this_batch} ({generated}/{num_mols} done)...")
        with torch.no_grad():
            sample_out = model.sample(batch_size=this_batch, num_steps=num_steps)

        for i in range(this_batch):
            try:
                mol = converter.from_tensor(sample_out[i].cpu())
            except Exception:
                mol = None
            all_mols.append(mol)

        remaining -= this_batch
        generated += this_batch

    return all_mols


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


# PoseBusters-based validity check
def _compute_posebusters(mol_list, num_mols):
    """Compute PoseBusters checks on generated molecules."""
    try:
        from posebusters import PoseBusters
    except ImportError:
        print("  PoseBusters not installed, skipping")
        return {}

    # Write molecules to SDF for PoseBusters
    import tempfile
    import os

    results = {}
    valid_mols = [m for m in mol_list if m is not None]

    if not valid_mols:
        return {}

    pb_metrics = {}
    for mol in valid_mols:
        try:
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
                writer = Chem.SDWriter(f.name)
                writer.write(mol)
                writer.close()

                buster = PoseBusters(config="mol", top_n=None)
                df = buster.bust(f.name, None)

                for col in df.columns:
                    if col not in pb_metrics:
                        pb_metrics[col] = []
                    pb_metrics[col].append(bool(df[col].iloc[0]))
            os.unlink(f.name)
        except Exception:
            continue

    # Compute fractions
    for col, vals in pb_metrics.items():
        results[f"pb_{col}"] = sum(vals) / num_mols

    # Intersection = all checks pass
    if pb_metrics:
        all_pass = []
        for i in range(len(list(pb_metrics.values())[0])):
            passed = all(vals[i] for vals in pb_metrics.values() if i < len(vals))
            all_pass.append(passed)
        results["pb_intersection"] = sum(all_pass) / num_mols

    return results


def _compute_fcd(gen_smiles, train_smiles):
    """Compute Frechet ChemNet Distance between generated and training molecules."""
    try:
        from fcd_torch import FCD
    except ImportError:
        print("  fcd-torch not installed, skipping FCD")
        return None

    if not gen_smiles or not train_smiles:
        return None

    device = _default_device()
    fcd = FCD(device=device, n_jobs=1)
    score = fcd(gen_smiles, train_smiles)
    return float(score)


def compute_metrics(
    mol_list,
    data_stats,
    train_smiles=None,
    n_bootstrap=1000,
    skip_posebusters=False,
):
    """Compute all metrics on a list of generated molecules.

    Args:
        mol_list: list of (rdkit.Mol or None) - one per generated sample
        data_stats: dict with dataset statistics
        train_smiles: list of training SMILES for novelty/FCD (optional)
        n_bootstrap: number of bootstrap samples for CIs (0 = skip)
    """
    num_mols = len(mol_list)

    # Initialize metrics
    validity_metric = MolecularValidity(sync_on_compute=False)
    connectivity_metric = MolecularConnectivity(sync_on_compute=False)
    uniqueness_metric = MolecularUniqueness(sync_on_compute=False)
    qed_metric = MolecularQEDValue(sync_on_compute=False)
    logp_metric = MolecularLogP(sync_on_compute=False)
    lipinski_metric = MolecularLipinski(sync_on_compute=False)
    diversity_metric = MolecularDiversity(sync_on_compute=False)
    atom_type_metric = None
    if train_smiles:
        atom_type_metric = AtomTypeDistribution(
            original_smiles=train_smiles, sync_on_compute=False
        )
        novelty_metric = MolecularNovelty(
            original_smiles=train_smiles, sync_on_compute=False
        )

    # Compute on all molecules
    all_metrics = [
        validity_metric,
        connectivity_metric,
        uniqueness_metric,
        qed_metric,
        logp_metric,
        lipinski_metric,
        diversity_metric,
    ]
    if atom_type_metric:
        all_metrics.append(atom_type_metric)
    if novelty_metric:
        all_metrics.append(novelty_metric)

    for m in all_metrics:
        m.update(mol_list)

    # Point estimates
    print("Computing point-estimate metrics...")
    results = {}
    metric_names = [
        ("validity", validity_metric),
        ("connectivity", connectivity_metric),
        ("uniqueness", uniqueness_metric),
        ("qed", qed_metric),
        ("logP", logp_metric),
        ("lipinski", lipinski_metric),
        ("diversity", diversity_metric),
    ]
    if novelty_metric:
        metric_names.append(("novelty", novelty_metric))
    if atom_type_metric:
        metric_names.append(("atom_type_dist", atom_type_metric))

    for name, metric in metric_names:
        try:
            val = metric.compute()
            results[name] = float(val)
            print(f"  {name}: {val:.4f}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # PoseBusters
    print("Computing PoseBusters checks...")
    valid_mols = [m for m in mol_list if m is not None]
    t0 = time.time()
    pb_results = _compute_posebusters(valid_mols, num_mols)
    print(f"  PoseBusters done in {time.time() - t0:.1f}s")
    for k, v in sorted(pb_results.items()):
        results[k] = v
        print(f"  {k}: {v:.4f}")

    # FCD
    print("Computing FCD...")
    gen_smiles = []
    for mol in mol_list:
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    gen_smiles.append(smi)
            except Exception:
                pass

    t0 = time.time()
    fcd_score = _compute_fcd(gen_smiles, train_smiles)
    if fcd_score is not None:
        results["fcd"] = fcd_score
        print(f"  FCD: {fcd_score:.4f} (computed in {time.time() - t0:.1f}s)")
    else:
        print("  FCD: skipped (no data)")

    # Bootstrap CIs
    bootstrap_ci = {}
    if n_bootstrap > 0:
        print(f"Computing bootstrap CIs ({n_bootstrap} samples)...")
        bootstrap_ci = _bootstrap_metrics(
            mol_list, data_stats, train_smiles, n_bootstrap
        )
        for name, ci in bootstrap_ci.items():
            print(f"  {name}: [{ci['low']:.4f}, {ci['high']:.4f}]")
    else:
        print("Skipping bootstrap CIs (--no_bootstrap)")

    return results, bootstrap_ci


def _bootstrap_metrics(mol_list, data_stats, train_smiles, n_bootstrap):
    """Compute bootstrap confidence intervals for key metrics."""
    rng = np.random.default_rng(42)
    n = len(mol_list)

    # Metrics to bootstrap (simple per-molecule ones)
    bootstrap_metrics = {
        "validity": lambda mols: sum(1 for m in mols if m is not None) / len(mols),
        "connectivity": lambda mols: sum(
            1
            for m in mols
            if m is not None and Chem.GetMolFrags(m, asMols=False).__len__() == 1
        )
        / max(1, sum(1 for m in mols if m is not None)),
    }

    # Add QED
    def _qed(mols):
        from rdkit.Chem import Descriptors

        vals = [Descriptors.qed(m) for m in mols if m is not None]
        return np.mean(vals) if vals else 0.0

    bootstrap_metrics["qed"] = _qed

    results = {name: [] for name in bootstrap_metrics}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample = [mol_list[i] for i in indices]
        for name, fn in bootstrap_metrics.items():
            results[name].append(fn(sample))

    ci = {}
    for name, vals in results.items():
        vals = np.array(vals)
        ci[name] = {
            "low": float(np.percentile(vals, 2.5)),
            "high": float(np.percentile(vals, 97.5)),
        }
    return ci


def _save_molecules_sdf(mol_list, output_path):
    """Save generated molecules to SDF file."""
    writer = Chem.SDWriter(str(output_path))
    for i, mol in enumerate(mol_list):
        if mol is not None:
            mol.SetProp("_Name", f"mol_{i}")
            writer.write(mol)
    writer.close()


def _save_molecules_csv(mol_list, output_path):
    """Save per-molecule properties to CSV."""
    import csv

    from rdkit.Chem import Descriptors

    rows = []
    for i, mol in enumerate(mol_list):
        row = {"index": i, "valid": mol is not None}
        if mol is not None:
            try:
                row["smiles"] = Chem.MolToSmiles(mol)
                row["qed"] = Descriptors.qed(mol)
                row["logP"] = Descriptors.MolLogP(mol)
                row["num_atoms"] = mol.GetNumAtoms()
                row["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
            except Exception:
                pass
        rows.append(row)

    fieldnames = [
        "index",
        "valid",
        "smiles",
        "qed",
        "logP",
        "num_atoms",
        "num_heavy_atoms",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("evaluation/tabasco/checkpoints/geom")
RESULTS_DIR = Path("evaluation/tabasco/generation/results/geom/evaluation")

ALL_MODELS = {
    "baseline": CHECKPOINT_DIR / "baseline.ckpt",
    "chemeleon_additive_fused": CHECKPOINT_DIR / "chemeleon_additive_fused.ckpt",
    "chemeleon_additive_same": CHECKPOINT_DIR / "chemeleon_additive_same.ckpt",
    "chemeleon_tradeoff_fused": CHECKPOINT_DIR / "chemeleon_tradeoff_fused.ckpt",
    "chemeleon_tradeoff_same": CHECKPOINT_DIR / "chemeleon_tradeoff_same.ckpt",
    "mace_additive": CHECKPOINT_DIR / "mace_additive.ckpt",
    "mace_tradeoff": CHECKPOINT_DIR / "mace_tradeoff.ckpt",
}


def evaluate_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    num_mols: int = 1000,
    num_steps: int = 100,
    batch_size: int = 256,
    train_smiles_path: str = None,
    n_bootstrap: int = 1000,
    device: str = None,
):
    """Full evaluation pipeline for one checkpoint."""
    if device is None:
        device = _default_device()
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

    # Generate molecules (cache to SDF so we don't regenerate on metric failures)
    cached_sdf = out / "molecules.sdf"
    if cached_sdf.exists():
        print(f"  Loading cached molecules from {cached_sdf}...")
        supplier = Chem.SDMolSupplier(str(cached_sdf), removeHs=False)
        molecules = [mol for mol in supplier]
        gen_time = 0.0
        print(f"  Loaded {len(molecules)} cached molecules")
    else:
        t0 = time.time()
        molecules = generate_molecules(model, num_mols, num_steps, batch_size)
        gen_time = time.time() - t0
        print(f"  Generated {len(molecules)} molecules in {gen_time:.1f}s")
        # Save immediately so we don't lose them if metrics crash
        _save_molecules_sdf(molecules, cached_sdf)
        print(f"  Cached molecules to {cached_sdf}")

    # Compute metrics
    metrics, bootstrap_ci = compute_metrics(
        molecules, data_stats, train_smiles, n_bootstrap
    )

    # Save results
    result = {
        "checkpoint": checkpoint_path,
        "meta": meta,
        "generation": {
            "num_mols": num_mols,
            "num_steps": num_steps,
            "time_seconds": gen_time,
        },
        "metrics": metrics,
        "bootstrap_ci": bootstrap_ci,
    }

    with open(out / "metrics.json", "w") as f:
        json.dump(result, f, indent=4)
    print(f"  Saved metrics to {out / 'metrics.json'}")

    # Save molecules
    _save_molecules_sdf(molecules, out / "molecules.sdf")
    print(f"  Saved molecules to {out / 'molecules.sdf'}")

    _save_molecules_csv(molecules, out / "molecules.csv")
    print(f"  Saved per-molecule properties to {out / 'molecules.csv'}")

    return result


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
        default="evaluation/data/tabasco/geom/geom_train_smiles.txt",
        help="Path to training SMILES for novelty/FCD",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument(
        "--no_bootstrap",
        action="store_true",
        help="Skip bootstrap CI computation",
    )
    parser.add_argument(
        "--skip_posebusters",
        action="store_true",
        help="Skip PoseBusters checks (very slow on CPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models in evaluation/tabasco/checkpoints/geom/",
    )
    args = parser.parse_args()

    n_bootstrap = 0 if args.no_bootstrap else args.n_bootstrap
    device = args.device if args.device is not None else _default_device()

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
                n_bootstrap=n_bootstrap,
                device=device,
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
            n_bootstrap=n_bootstrap,
            device=device,
        )


if __name__ == "__main__":
    main()
