"""
Rigorous projector analysis across all three encoders: CheMeleon, MACE, GearNet.

Tests whether encoder target spaces are trivially matchable (projector saturation)
or require genuine structural information from the transformer.

Three tests per encoder (all with proper 80/20 train/test splits):
  1. Mean-direction baseline: cosine sim between mean embedding and all targets
  2. Identity-input generalization: one-hot type → MLP → targets (test set)
  3. Random-input generalization: random vectors → MLP → targets (test set)

Run (recommended via srun to avoid login node kill):
  srun --partition=ampere --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash
  source .venv/bin/activate
  export PROJECT_ROOT=$(pwd)/src/tabasco
  python -u playground/analysis/projector_analysis.py

Or in two phases:
  python -u playground/analysis/projector_analysis.py --phase embeddings  # expensive
  python -u playground/analysis/projector_analysis.py --phase analysis    # cheap
"""

import argparse
import os
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).parent / "cached_embeddings"
FIGURES_DIR = Path(__file__).parent / "figures"

# ── Embedding computation (Phase 1) ────────────────────────────────────────


def compute_chemeleon_embeddings(n_mols=200):
    """Compute CheMeleon per-atom embeddings from QM9."""
    print("\n=== Computing CheMeleon embeddings ===")
    project_root = os.environ.get("PROJECT_ROOT", str(REPO_ROOT / "src" / "tabasco"))
    sys.path.insert(0, os.path.join(project_root, "src"))

    from tabasco.models.components.encoders import ChemPropEncoder
    from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset

    encoder = ChemPropEncoder(pretrained="chemeleon")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    ds = UnconditionalLMDBDataset(
        data_dir=os.path.join(project_root, "data", "processed_qm9_train.pt"),
        split="train",
        add_random_rotation=False,
        add_random_permutation=False,
        reorder_to_smiles_order=False,
        remove_hydrogens=True,
        lmdb_dir=os.path.join(project_root, "data", "lmdb_qm9"),
    )

    ATOM_NAMES = ["C", "N", "O", "F", "S", "Cl", "Br", "I"]

    rng = np.random.RandomState(42)
    indices = rng.choice(len(ds), size=min(n_mols, len(ds)), replace=False)

    # Collect items and smiles
    smiles_list, items = [], []
    for idx in indices:
        item = ds[int(idx)]
        try:
            smi = item.get_non_tensor("smiles")
        except (KeyError, AttributeError):
            continue
        smiles_list.append(smi)
        items.append(item)

    print(f"  {len(smiles_list)} molecules with SMILES")

    # Batch encode (using coords/atomics from dataset, not regenerated)
    all_emb, all_types = [], []
    batch_size = 64
    for start in range(0, len(smiles_list), batch_size):
        end = min(start + batch_size, len(smiles_list))
        batch_smiles = smiles_list[start:end]
        batch_items = items[start:end]

        coords = torch.stack([item["coords"] for item in batch_items])
        atomics = torch.stack([item["atomics"] for item in batch_items])
        masks = torch.stack([item["padding_mask"] for item in batch_items])

        with torch.no_grad():
            emb = encoder(coords, atomics, masks, smiles=batch_smiles)

        for i in range(emb.shape[0]):
            n_real = (~masks[i]).sum().item()
            for j in range(n_real):
                all_emb.append(emb[i, j].clone())
                atom_idx = atomics[i, j].argmax().item()
                all_types.append(atom_idx)

        if (start + batch_size) % 100 < batch_size:
            print(f"  {min(end, len(smiles_list))}/{len(smiles_list)} molecules")

    result = {
        "embeddings": torch.stack(all_emb),
        "types": torch.tensor(all_types),
        "type_names": ATOM_NAMES,
        "encoder_name": "CheMeleon",
        "embed_dim": all_emb[0].shape[0],
    }
    print(f"  {result['embeddings'].shape[0]} atoms, dim={result['embed_dim']}")
    return result


def compute_mace_embeddings(n_mols=200):
    """Compute MACE per-atom embeddings from GEOM."""
    print("\n=== Computing MACE embeddings ===")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    import ase
    from e3nn import o3
    from mace import data
    from mace.calculators import mace_off
    from mace.modules.utils import extract_invariant
    from mace.tools import torch_geometric
    from rdkit import Chem
    import lmdb

    project_root = os.environ.get("PROJECT_ROOT", str(REPO_ROOT / "src" / "tabasco"))
    lmdb_path = os.path.join(project_root, "data", "lmdb_geom", "train.lmdb")
    db = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    mols = []
    with db.begin() as txn:
        for i, (k, v) in enumerate(txn.cursor()):
            if i >= n_mols:
                break
            mols.append(pickle.loads(v)["molecule"])
    db.close()
    print(f"  Loaded {len(mols)} molecules")

    calc = mace_off("small", device="cpu")
    model = calc.models[0]
    model.eval()

    num_interactions = int(model.num_interactions)
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_inv = irreps_out.dim // (l_max + 1) ** 2
    per_layer = [irreps_out.dim] * num_interactions
    per_layer[-1] = num_inv
    total_feats = int(np.sum(per_layer))

    ATOM_NAMES = ["C", "N", "O", "F", "S", "Cl", "Br", "I"]
    ATOM_Z = [6, 7, 8, 9, 16, 17, 35, 53]

    all_emb, all_types = [], []
    keyspec = data.KeySpecification(
        info_keys=calc.info_keys, arrays_keys=calc.arrays_keys
    )

    for count, mol in enumerate(mols):
        mol_noh = Chem.RemoveAllHs(mol)
        conf = mol_noh.GetConformer()
        positions = conf.GetPositions()
        atomic_numbers = [a.GetAtomicNum() for a in mol_noh.GetAtoms()]
        atoms = ase.Atoms(numbers=atomic_numbers, positions=positions)

        config = data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=calc.head
        )
        atomic_data = data.AtomicData.from_config(
            config,
            z_table=calc.z_table,
            cutoff=calc.r_max,
            heads=calc.available_heads,
        )
        loader = torch_geometric.dataloader.DataLoader(
            [atomic_data], batch_size=1, shuffle=False
        )

        with torch.no_grad():
            for batch in loader:
                output = model(batch.to_dict(), compute_force=False)
                desc = extract_invariant(
                    output["node_feats"],
                    num_layers=num_interactions,
                    num_features=num_inv,
                    l_max=l_max,
                )[:, :total_feats]

        for i, atom in enumerate(mol_noh.GetAtoms()):
            z = atom.GetAtomicNum()
            if z in ATOM_Z:
                all_emb.append(desc[i].cpu().float())
                all_types.append(ATOM_Z.index(z))

        if (count + 1) % 50 == 0:
            print(f"  {count + 1}/{len(mols)} molecules")

    result = {
        "embeddings": torch.stack(all_emb),
        "types": torch.tensor(all_types),
        "type_names": ATOM_NAMES,
        "encoder_name": "MACE",
        "embed_dim": all_emb[0].shape[0],
    }
    print(f"  {result['embeddings'].shape[0]} atoms, dim={result['embed_dim']}")
    return result


def compute_gearnet_embeddings(n_proteins=200):
    """Compute GearNet per-residue embeddings from PDB LMDB."""
    print("\n=== Computing GearNet embeddings ===")
    sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))
    import proteinfoundation.repa.pyg_compat  # noqa: F401

    from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder
    import lmdb

    DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
    encoder = GearNetPerResidueEncoder(
        ckpt_path=os.path.join(DATA_PATH, "metric_factory/model_weights/gearnet_ca.pth")
    )
    encoder.eval()

    lmdb_path = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")
    db = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )
    proteins = []
    with db.begin() as txn:
        for key, value in txn.cursor():
            if key == b"__ids__":
                continue
            if len(proteins) >= n_proteins:
                break
            graph = pickle.loads(value)
            if hasattr(graph, "coords") and graph.coords.shape[0] >= 10:
                proteins.append(graph)
    db.close()
    print(f"  Loaded {len(proteins)} proteins")

    RESIDUE_NAMES = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]

    all_emb, all_types = [], []
    for count, graph in enumerate(proteins):
        ca_coords = graph.coords[:, 1, :]
        ca_mask = graph.coord_mask[:, 1].bool()
        ca_nm = ca_coords.float() / 10.0

        with torch.no_grad():
            emb = encoder(ca_nm.unsqueeze(0), ca_mask.float().unsqueeze(0)).squeeze(0)

        valid_emb = emb[ca_mask]
        valid_types = graph.residue_type[ca_mask]
        all_emb.append(valid_emb)
        all_types.append(valid_types)

        if (count + 1) % 50 == 0:
            print(f"  {count + 1}/{len(proteins)} proteins")

    result = {
        "embeddings": torch.cat(all_emb, dim=0),
        "types": torch.cat(all_types, dim=0),
        "type_names": RESIDUE_NAMES,
        "encoder_name": "GearNet",
        "embed_dim": all_emb[0].shape[1],
    }
    print(f"  {result['embeddings'].shape[0]} residues, dim={result['embed_dim']}")
    return result


# ── Analysis (Phase 2) ─────────────────────────────────────────────────────


def test_mean_direction(embeddings):
    """Test 1: Cosine similarity between mean embedding and all targets."""
    mean_vec = embeddings.mean(dim=0, keepdim=True)  # [1, dim]
    cos_sims = F.cosine_similarity(mean_vec, embeddings, dim=-1)  # [n]
    return {
        "mean": cos_sims.mean().item(),
        "std": cos_sims.std().item(),
        "min": cos_sims.min().item(),
        "max": cos_sims.max().item(),
    }


def train_and_evaluate_mlp(
    inp_train,
    inp_test,
    target_train,
    target_test,
    hidden_dim=256,
    n_epochs=200,
    lr=1e-3,
):
    """Train MLP and evaluate on both train and test sets. Returns per-epoch curves."""
    input_dim = inp_train.shape[1]
    encoder_dim = target_train.shape[1]

    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, encoder_dim),
    ).float()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    train_curve, test_curve = [], []
    for epoch in range(n_epochs):
        # Train step
        mlp.train()
        pred = mlp(inp_train.float())
        cos = F.cosine_similarity(pred, target_train, dim=-1)
        loss = -cos.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            mlp.eval()
            with torch.no_grad():
                train_cos = (
                    F.cosine_similarity(mlp(inp_train.float()), target_train, dim=-1)
                    .mean()
                    .item()
                )
                test_cos = (
                    F.cosine_similarity(mlp(inp_test.float()), target_test, dim=-1)
                    .mean()
                    .item()
                )
            train_curve.append((epoch + 1, train_cos))
            test_curve.append((epoch + 1, test_cos))

    return train_curve, test_curve


def run_analysis(data):
    """Run all three tests for one encoder."""
    name = data["encoder_name"]
    embeddings = data["embeddings"].float()
    types = data["types"]
    n = embeddings.shape[0]

    print(f"\n{'='*60}")
    print(f"  {name}: {n} samples, dim={data['embed_dim']}")
    print(f"{'='*60}")

    # --- Test 1: Mean direction ---
    mean_result = test_mean_direction(embeddings)
    print("\n  Test 1 — Mean-direction baseline:")
    print(f"    cos_sim = {mean_result['mean']:.4f} +/- {mean_result['std']:.4f}")
    print(f"    range: [{mean_result['min']:.4f}, {mean_result['max']:.4f}]")

    # --- Stratified 80/20 split ---
    from sklearn.model_selection import train_test_split

    valid_mask = types < len(data["type_names"])
    embeddings_valid = embeddings[valid_mask]
    types_valid = types[valid_mask]

    train_idx, test_idx = train_test_split(
        np.arange(len(embeddings_valid)),
        test_size=0.2,
        random_state=42,
        stratify=types_valid.numpy(),
    )
    target_train = embeddings_valid[train_idx]
    target_test = embeddings_valid[test_idx]
    types_train = types_valid[train_idx]
    types_test = types_valid[test_idx]

    print(f"\n  Split: {len(train_idx)} train, {len(test_idx)} test")

    # --- Test 2: Identity input ---
    n_classes = len(data["type_names"])
    onehot_train = F.one_hot(types_train.long(), n_classes).float()
    onehot_test = F.one_hot(types_test.long(), n_classes).float()

    print(f"\n  Test 2 — Identity-input (one-hot, {n_classes} classes):")
    id_train_curve, id_test_curve = train_and_evaluate_mlp(
        onehot_train, onehot_test, target_train, target_test
    )
    print(f"    Train cos_sim: {id_train_curve[-1][1]:.4f}")
    print(f"    Test cos_sim:  {id_test_curve[-1][1]:.4f}")
    print(f"    Overfit gap:   {id_train_curve[-1][1] - id_test_curve[-1][1]:.4f}")

    # --- Test 3: Random input ---
    torch.manual_seed(42)
    random_train = torch.randn(len(train_idx), 128)
    random_test = torch.randn(len(test_idx), 128)

    print("\n  Test 3 — Random-input (128-d Gaussian):")
    rand_train_curve, rand_test_curve = train_and_evaluate_mlp(
        random_train, random_test, target_train, target_test
    )
    print(f"    Train cos_sim: {rand_train_curve[-1][1]:.4f}")
    print(f"    Test cos_sim:  {rand_test_curve[-1][1]:.4f}")
    print(f"    Overfit gap:   {rand_train_curve[-1][1] - rand_test_curve[-1][1]:.4f}")

    return {
        "name": name,
        "n_samples": n,
        "embed_dim": data["embed_dim"],
        "mean_direction": mean_result,
        "identity": {
            "train_curve": id_train_curve,
            "test_curve": id_test_curve,
            "final_train": id_train_curve[-1][1],
            "final_test": id_test_curve[-1][1],
        },
        "random": {
            "train_curve": rand_train_curve,
            "test_curve": rand_test_curve,
            "final_train": rand_train_curve[-1][1],
            "final_test": rand_test_curve[-1][1],
        },
    }


# ── Figures (Phase 3) ──────────────────────────────────────────────────────


def generate_figures(results_list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    names = [r["name"] for r in results_list]

    # --- Fig 1: Mean direction baseline ---
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [r["mean_direction"]["mean"] for r in results_list]
    stds = [r["mean_direction"]["std"] for r in results_list]
    bars = ax.bar(
        names,
        means,
        yerr=stds,
        capsize=5,
        color=["#e07a5f", "#3d405b", "#81b29a"],
        alpha=0.8,
    )
    ax.set_ylabel("Cosine similarity")
    ax.set_title(
        "Test 1: Mean-Direction Baseline\n(cosine sim between mean embedding and all targets)"
    )
    ax.set_ylim(0, 1)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.3f}",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_01_mean_direction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'fig_01_mean_direction.png'}")

    # --- Fig 2: Train vs test grouped bars ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    w = 0.18
    colors = {
        "rand_train": "#d4a373",
        "rand_test": "#ccd5ae",
        "id_train": "#e07a5f",
        "id_test": "#81b29a",
    }

    for i, r in enumerate(results_list):
        ax.bar(
            x[i] - 1.5 * w,
            r["random"]["final_train"],
            w,
            color=colors["rand_train"],
            label="Random (train)" if i == 0 else "",
        )
        ax.bar(
            x[i] - 0.5 * w,
            r["random"]["final_test"],
            w,
            color=colors["rand_test"],
            label="Random (test)" if i == 0 else "",
        )
        ax.bar(
            x[i] + 0.5 * w,
            r["identity"]["final_train"],
            w,
            color=colors["id_train"],
            label="Identity (train)" if i == 0 else "",
        )
        ax.bar(
            x[i] + 1.5 * w,
            r["identity"]["final_test"],
            w,
            color=colors["id_test"],
            label="Identity (test)" if i == 0 else "",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Test 2 & 3: Projector Train vs Test Cosine Similarity")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "fig_02_projector_traintest.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'fig_02_projector_traintest.png'}")

    # --- Fig 3: Learning curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, (r, ax) in enumerate(zip(results_list, axes)):
        # Random
        epochs_r = [p[0] for p in r["random"]["train_curve"]]
        ax.plot(
            epochs_r,
            [p[1] for p in r["random"]["train_curve"]],
            "o-",
            color="#d4a373",
            markersize=3,
            label="Random (train)",
        )
        ax.plot(
            epochs_r,
            [p[1] for p in r["random"]["test_curve"]],
            "s--",
            color="#ccd5ae",
            markersize=3,
            label="Random (test)",
        )
        # Identity
        epochs_i = [p[0] for p in r["identity"]["train_curve"]]
        ax.plot(
            epochs_i,
            [p[1] for p in r["identity"]["train_curve"]],
            "o-",
            color="#e07a5f",
            markersize=3,
            label="Identity (train)",
        )
        ax.plot(
            epochs_i,
            [p[1] for p in r["identity"]["test_curve"]],
            "s--",
            color="#81b29a",
            markersize=3,
            label="Identity (test)",
        )
        # Mean direction
        ax.axhline(
            r["mean_direction"]["mean"],
            color="gray",
            linestyle=":",
            linewidth=1,
            label="Mean direction",
        )

        ax.set_title(r["name"])
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Cosine similarity")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(-0.1, 1.0)

    fig.suptitle("Projector Learning Curves (Train vs Test)", y=1.02)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "fig_03_learning_curves.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'fig_03_learning_curves.png'}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["embeddings", "analysis", "all"], default="all"
    )
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Phase 1: Compute and cache embeddings
    if args.phase in ("embeddings", "all"):
        t0 = time.time()

        chemeleon_data = compute_chemeleon_embeddings()
        torch.save(chemeleon_data, CACHE_DIR / "chemeleon.pt")

        mace_data = compute_mace_embeddings()
        torch.save(mace_data, CACHE_DIR / "mace.pt")

        gearnet_data = compute_gearnet_embeddings()
        torch.save(gearnet_data, CACHE_DIR / "gearnet.pt")

        print(f"\nEmbeddings cached in {time.time() - t0:.1f}s")

    # Phase 2: Analysis
    if args.phase in ("analysis", "all"):
        chemeleon_data = torch.load(CACHE_DIR / "chemeleon.pt", weights_only=False)
        mace_data = torch.load(CACHE_DIR / "mace.pt", weights_only=False)
        gearnet_data = torch.load(CACHE_DIR / "gearnet.pt", weights_only=False)

        results = []
        for data in [chemeleon_data, mace_data, gearnet_data]:
            results.append(run_analysis(data))

        # Summary table
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(
            f"{'Encoder':>12} | {'Mean Dir':>8} | {'Rand Train':>10} | {'Rand Test':>10} | {'ID Train':>10} | {'ID Test':>10}"
        )
        print("-" * 75)
        for r in results:
            print(
                f"{r['name']:>12} | {r['mean_direction']['mean']:>8.4f} | "
                f"{r['random']['final_train']:>10.4f} | {r['random']['final_test']:>10.4f} | "
                f"{r['identity']['final_train']:>10.4f} | {r['identity']['final_test']:>10.4f}"
            )

        # Phase 3: Figures
        generate_figures(results)

        print(f"\n{'='*60}")
        print("  DONE")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
