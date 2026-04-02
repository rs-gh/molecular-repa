"""
Unified REPA projector analysis across all three encoders: CheMeleon, MACE, GearNet.

Canonical source of truth for projector saturation tests, effective rank,
sparsity, and atom/residue discrimination metrics. Replaces the scattered
analysis scripts in playground/tabasco/{chemeleon,mace}/ and playground/proteina/gearnet/.

Projector hidden_dim matches the ACTUAL training configs:
  - CheMeleon/MACE: hidden_dim=128 (tabasco model.net.hidden_dim)
  - GearNet: hidden_dim=512 (proteina model.nn.token_dim)

Run (recommended via srun for Phase 1):
  srun --partition=ampere --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash
  source .venv/bin/activate
  export PROJECT_ROOT=$(pwd)/src/tabasco
  python -u playground/projector/encoder_analysis.py

Or in two phases:
  python -u playground/projector/encoder_analysis.py --phase embeddings  # GPU, ~10 min
  python -u playground/projector/encoder_analysis.py --phase analysis    # CPU, ~2 min
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).parent / "cached_embeddings"
FIGURES_DIR = Path(__file__).parent / "figures"

# Actual projector hidden_dim from training configs
PROJECTOR_HIDDEN_DIM = {
    "CheMeleon": 128,  # model.net.hidden_dim in tabasco GEOM configs
    "MACE": 128,
    "GearNet": 512,  # model.nn.token_dim in proteina configs
}


# ── Embedding computation (Phase 1) ──────────────────────────────────────────


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


# ── Standardized metrics (Phase 2) ───────────────────────────────────────────


def compute_distribution_stats(embeddings):
    """Basic distribution statistics."""
    return {
        "mean": embeddings.mean().item(),
        "std": embeddings.std().item(),
        "min": embeddings.min().item(),
        "max": embeddings.max().item(),
        "pct_negative": (embeddings < 0).float().mean().item() * 100,
        "pct_exact_zero": (embeddings == 0).float().mean().item() * 100,
        "pct_near_zero_1e6": (embeddings.abs() < 1e-6).float().mean().item() * 100,
        "pct_near_zero_1e10": (embeddings.abs() < 1e-10).float().mean().item() * 100,
    }


def compute_effective_rank(embeddings):
    """Compute effective rank under ALL four definitions.

    Returns dict with clearly labeled metrics so there is no ambiguity.
    """
    # Center the embeddings
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(centered.float())

    # 1. Threshold: count(SV > 1% of max SV)
    threshold_rank = int((S > 0.01 * S[0]).sum().item())

    # 2. Entropy on normalized singular values: exp(-sum(p * log(p)))
    Sn = S / S.sum()
    Sn_pos = Sn[Sn > 0]
    entropy_sv = torch.exp(-(Sn_pos * torch.log(Sn_pos)).sum()).item()

    # 3. Entropy on normalized squared singular values (variance):
    S2 = S**2
    S2n = S2 / S2.sum()
    S2n_pos = S2n[S2n > 0]
    entropy_var = torch.exp(-(S2n_pos * torch.log(S2n_pos)).sum()).item()

    # 4. PCA 90%: min components for 90% cumulative variance
    cumvar = torch.cumsum(S2, dim=0) / S2.sum()
    pca_90 = int((cumvar < 0.90).sum().item()) + 1

    return {
        "threshold_1pct": threshold_rank,
        "entropy_sv": round(entropy_sv, 1),
        "entropy_var": round(entropy_var, 1),
        "pca_90pct": pca_90,
        "total_dims": embeddings.shape[1],
    }


def compute_discrimination(embeddings, types, type_names):
    """Atom/residue discrimination: linear probe + within/between cosine sim."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    valid_mask = types < len(type_names)
    emb = embeddings[valid_mask].numpy()
    typ = types[valid_mask].numpy()

    # Linear probe
    X_train, X_test, y_train, y_test = train_test_split(
        emb, typ, test_size=0.2, random_state=42, stratify=typ
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    probe_accuracy = clf.score(X_test, y_test)

    # Within-type vs between-type cosine similarity (sampled)
    emb_t = torch.from_numpy(emb).float()
    typ_t = torch.from_numpy(typ)
    rng = np.random.RandomState(42)
    n_pairs = min(5000, len(emb_t) * (len(emb_t) - 1) // 2)

    within_sims, between_sims = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(emb_t), size=2, replace=False)
        sim = F.cosine_similarity(emb_t[i : i + 1], emb_t[j : j + 1], dim=-1).item()
        if typ_t[i] == typ_t[j]:
            within_sims.append(sim)
        else:
            between_sims.append(sim)

    within_mean = np.mean(within_sims) if within_sims else float("nan")
    between_mean = np.mean(between_sims) if between_sims else float("nan")

    return {
        "linear_probe_accuracy": round(probe_accuracy, 4),
        "within_type_cos_sim": round(within_mean, 4),
        "between_type_cos_sim": round(between_mean, 4),
        "discrimination_gap": round(within_mean - between_mean, 4),
        "n_classes": len(set(typ.tolist())),
    }


def compute_mean_direction_floor(embeddings):
    """Cosine similarity between the mean embedding and all targets."""
    mean_vec = embeddings.mean(dim=0, keepdim=True)
    cos_sims = F.cosine_similarity(mean_vec, embeddings, dim=-1)
    return {
        "mean": round(cos_sims.mean().item(), 4),
        "std": round(cos_sims.std().item(), 4),
    }


def train_and_evaluate_mlp(
    inp_train,
    inp_test,
    target_train,
    target_test,
    hidden_dim,
    n_epochs=200,
    lr=1e-3,
):
    """Train 2-layer MLP (matching actual projector architecture) and evaluate."""
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
        mlp.train()
        pred = mlp(inp_train.float())
        cos = F.cosine_similarity(pred, target_train, dim=-1)
        loss = -cos.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def compute_projector_saturation(embeddings, types, type_names, encoder_name):
    """Projector saturation test with per-encoder hidden_dim matching training."""
    from sklearn.model_selection import train_test_split

    hidden_dim = PROJECTOR_HIDDEN_DIM[encoder_name]

    valid_mask = types < len(type_names)
    emb = embeddings[valid_mask].float()
    typ = types[valid_mask]

    train_idx, test_idx = train_test_split(
        np.arange(len(emb)),
        test_size=0.2,
        random_state=42,
        stratify=typ.numpy(),
    )
    target_train = emb[train_idx]
    target_test = emb[test_idx]
    types_train = typ[train_idx]
    types_test = typ[test_idx]

    n_classes = len(type_names)

    # Identity input (one-hot)
    onehot_train = F.one_hot(types_train.long(), n_classes).float()
    onehot_test = F.one_hot(types_test.long(), n_classes).float()

    print(
        f"    Projector saturation (hidden_dim={hidden_dim}, matching training config)"
    )
    print(f"    Split: {len(train_idx)} train, {len(test_idx)} test")

    id_train_curve, id_test_curve = train_and_evaluate_mlp(
        onehot_train,
        onehot_test,
        target_train,
        target_test,
        hidden_dim=hidden_dim,
    )

    # Random input (128-d)
    torch.manual_seed(42)
    random_train = torch.randn(len(train_idx), 128)
    random_test = torch.randn(len(test_idx), 128)

    rand_train_curve, rand_test_curve = train_and_evaluate_mlp(
        random_train,
        random_test,
        target_train,
        target_test,
        hidden_dim=hidden_dim,
    )

    return {
        "hidden_dim_used": hidden_dim,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "identity": {
            "train": round(id_train_curve[-1][1], 4),
            "test": round(id_test_curve[-1][1], 4),
            "train_curve": id_train_curve,
            "test_curve": id_test_curve,
        },
        "random": {
            "train": round(rand_train_curve[-1][1], 4),
            "test": round(rand_test_curve[-1][1], 4),
            "train_curve": rand_train_curve,
            "test_curve": rand_test_curve,
        },
    }


def run_analysis(data):
    """Run all standardized metrics for one encoder."""
    name = data["encoder_name"]
    embeddings = data["embeddings"].float()
    types = data["types"]
    n = embeddings.shape[0]

    print(f"\n{'='*60}")
    print(f"  {name}: {n} samples, dim={data['embed_dim']}")
    print(f"{'='*60}")

    # Distribution
    dist = compute_distribution_stats(embeddings)
    print("\n  Distribution:")
    print(f"    mean={dist['mean']:.4f}, std={dist['std']:.4f}")
    print(f"    negative: {dist['pct_negative']:.1f}%")
    print(f"    exact zero: {dist['pct_exact_zero']:.1f}%")
    print(f"    near-zero (<1e-6): {dist['pct_near_zero_1e6']:.1f}%")

    # Effective rank
    rank = compute_effective_rank(embeddings)
    print(f"\n  Effective rank (dim={rank['total_dims']}):")
    print(f"    Threshold (SV > 1% max):     {rank['threshold_1pct']}")
    print(f"    Entropy (norm SVs):          {rank['entropy_sv']}")
    print(f"    Entropy (norm variance):     {rank['entropy_var']}")
    print(f"    PCA 90% variance:            {rank['pca_90pct']}")

    # Discrimination
    discrim = compute_discrimination(embeddings, types, data["type_names"])
    print(f"\n  Discrimination ({discrim['n_classes']} classes):")
    print(f"    Linear probe accuracy: {discrim['linear_probe_accuracy']:.4f}")
    print(f"    Within-type cos_sim:   {discrim['within_type_cos_sim']:.4f}")
    print(f"    Between-type cos_sim:  {discrim['between_type_cos_sim']:.4f}")
    print(f"    Gap:                   {discrim['discrimination_gap']:.4f}")

    # Mean direction floor
    floor = compute_mean_direction_floor(embeddings)
    print(f"\n  Mean-direction floor: {floor['mean']:.4f} +/- {floor['std']:.4f}")

    # Projector saturation
    print("\n  Projector saturation:")
    sat = compute_projector_saturation(embeddings, types, data["type_names"], name)
    print(
        f"    Identity — train: {sat['identity']['train']:.4f}, test: {sat['identity']['test']:.4f}"
    )
    print(
        f"    Random   — train: {sat['random']['train']:.4f}, test: {sat['random']['test']:.4f}"
    )

    return {
        "name": name,
        "n_samples": n,
        "embed_dim": data["embed_dim"],
        "distribution": dist,
        "effective_rank": rank,
        "discrimination": discrim,
        "mean_direction_floor": floor,
        "projector_saturation": sat,
    }


# ── Figures (Phase 3) ────────────────────────────────────────────────────────


def generate_figures(results_list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    names = [r["name"] for r in results_list]
    colors = ["#e07a5f", "#3d405b", "#81b29a"]

    # Fig 1: Effective rank comparison (all four definitions)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    rank_metrics = [
        ("threshold_1pct", "Threshold\n(SV > 1% max)"),
        ("entropy_sv", "Entropy\n(norm SVs)"),
        ("entropy_var", "Entropy\n(norm variance)"),
        ("pca_90pct", "PCA\n(90% variance)"),
    ]
    for ax, (key, label) in zip(axes, rank_metrics):
        vals = [r["effective_rank"][key] for r in results_list]
        bars = ax.bar(names, vals, color=colors, alpha=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Effective rank")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v}",
                ha="center",
                fontsize=9,
            )
    fig.suptitle("Effective Rank (all definitions)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_01_effective_rank.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'fig_01_effective_rank.png'}")

    # Fig 2: Projector saturation (train vs test, per encoder)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    w = 0.18
    bar_colors = {
        "rand_train": "#d4a373",
        "rand_test": "#ccd5ae",
        "id_train": "#e07a5f",
        "id_test": "#81b29a",
        "floor": "#888888",
    }

    for i, r in enumerate(results_list):
        sat = r["projector_saturation"]
        ax.bar(
            x[i] - 2 * w,
            r["mean_direction_floor"]["mean"],
            w,
            color=bar_colors["floor"],
            label="Floor" if i == 0 else "",
        )
        ax.bar(
            x[i] - 1 * w,
            sat["random"]["train"],
            w,
            color=bar_colors["rand_train"],
            label="Random (train)" if i == 0 else "",
        )
        ax.bar(
            x[i] + 0 * w,
            sat["random"]["test"],
            w,
            color=bar_colors["rand_test"],
            label="Random (test)" if i == 0 else "",
        )
        ax.bar(
            x[i] + 1 * w,
            sat["identity"]["train"],
            w,
            color=bar_colors["id_train"],
            label="Identity (train)" if i == 0 else "",
        )
        ax.bar(
            x[i] + 2 * w,
            sat["identity"]["test"],
            w,
            color=bar_colors["id_test"],
            label="Identity (test)" if i == 0 else "",
        )

    # Annotate hidden_dim used
    for i, r in enumerate(results_list):
        hd = r["projector_saturation"]["hidden_dim_used"]
        ax.text(x[i], -0.08, f"h={hd}", ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(
        "Projector Saturation Test\n(hidden_dim matching actual training configs)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.15, 1)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "fig_02_projector_saturation.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'fig_02_projector_saturation.png'}")

    # Fig 3: Learning curves
    fig, axes = plt.subplots(
        1, len(results_list), figsize=(5 * len(results_list), 4), sharey=True
    )
    if len(results_list) == 1:
        axes = [axes]
    for i, (r, ax) in enumerate(zip(results_list, axes)):
        sat = r["projector_saturation"]
        # Random
        epochs_r = [p[0] for p in sat["random"]["train_curve"]]
        ax.plot(
            epochs_r,
            [p[1] for p in sat["random"]["train_curve"]],
            "o-",
            color="#d4a373",
            markersize=3,
            label="Random (train)",
        )
        ax.plot(
            epochs_r,
            [p[1] for p in sat["random"]["test_curve"]],
            "s--",
            color="#ccd5ae",
            markersize=3,
            label="Random (test)",
        )
        # Identity
        epochs_i = [p[0] for p in sat["identity"]["train_curve"]]
        ax.plot(
            epochs_i,
            [p[1] for p in sat["identity"]["train_curve"]],
            "o-",
            color="#e07a5f",
            markersize=3,
            label="Identity (train)",
        )
        ax.plot(
            epochs_i,
            [p[1] for p in sat["identity"]["test_curve"]],
            "s--",
            color="#81b29a",
            markersize=3,
            label="Identity (test)",
        )
        # Floor
        ax.axhline(
            r["mean_direction_floor"]["mean"],
            color="gray",
            linestyle=":",
            linewidth=1,
            label="Floor",
        )

        hd = sat["hidden_dim_used"]
        ax.set_title(f"{r['name']} (h={hd})")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Cosine similarity")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(-0.1, 1.0)

    fig.suptitle("Learning Curves (hidden_dim matching training)", y=1.02)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "fig_03_learning_curves.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'fig_03_learning_curves.png'}")

    # Fig 4: Distribution comparison
    fig, axes = plt.subplots(1, len(results_list), figsize=(5 * len(results_list), 4))
    if len(results_list) == 1:
        axes = [axes]
    for i, (r, ax) in enumerate(zip(results_list, axes)):
        dist = r["distribution"]
        metrics = ["pct_exact_zero", "pct_near_zero_1e6", "pct_negative"]
        labels = ["Exact zero", "Near-zero\n(<1e-6)", "Negative"]
        vals = [dist[m] for m in metrics]
        ax.bar(labels, vals, color=colors[i], alpha=0.8)
        ax.set_title(r["name"])
        ax.set_ylabel("% of values")
        ax.set_ylim(0, 100)
        for bar, v in zip(ax.patches, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{v:.1f}%",
                ha="center",
                fontsize=9,
            )
    fig.suptitle("Value Distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_04_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'fig_04_distribution.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unified REPA projector analysis across CheMeleon, MACE, GearNet"
    )
    parser.add_argument(
        "--phase",
        choices=["embeddings", "analysis", "all"],
        default="all",
        help="embeddings=compute & cache (GPU), analysis=metrics from cache (CPU)",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=["chemeleon", "mace", "gearnet"],
        help="Which encoders to analyze (default: all three)",
    )
    parser.add_argument(
        "--n-mols",
        type=int,
        default=200,
        help="Number of molecules/proteins to sample (default: 200)",
    )
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    encoder_fns = {
        "chemeleon": (
            "chemeleon.pt",
            lambda: compute_chemeleon_embeddings(args.n_mols),
        ),
        "mace": ("mace.pt", lambda: compute_mace_embeddings(args.n_mols)),
        "gearnet": ("gearnet.pt", lambda: compute_gearnet_embeddings(args.n_mols)),
    }

    # Phase 1: Compute and cache embeddings
    if args.phase in ("embeddings", "all"):
        t0 = time.time()
        for enc in args.encoders:
            cache_file, compute_fn = encoder_fns[enc]
            data = compute_fn()
            torch.save(data, CACHE_DIR / cache_file)
        print(f"\nEmbeddings cached in {time.time() - t0:.1f}s")

    # Phase 2: Analysis
    if args.phase in ("analysis", "all"):
        results = []
        for enc in args.encoders:
            cache_file, _ = encoder_fns[enc]
            cache_path = CACHE_DIR / cache_file
            if not cache_path.exists():
                print(f"  WARNING: {cache_path} not found, skipping {enc}")
                continue
            data = torch.load(cache_path, weights_only=False)
            results.append(run_analysis(data))

        # Summary table
        print(f"\n{'='*80}")
        print("  SUMMARY TABLE")
        print(f"{'='*80}")
        header = (
            f"{'Encoder':>12} | {'Dim':>4} | {'Floor':>6} | "
            f"{'Rand(tr)':>8} | {'Rand(te)':>8} | {'ID(tr)':>8} | {'ID(te)':>8} | "
            f"{'h_dim':>5} | {'Eff.Rank':>8}"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            sat = r["projector_saturation"]
            print(
                f"{r['name']:>12} | {r['embed_dim']:>4} | "
                f"{r['mean_direction_floor']['mean']:>6.4f} | "
                f"{sat['random']['train']:>8.4f} | {sat['random']['test']:>8.4f} | "
                f"{sat['identity']['train']:>8.4f} | {sat['identity']['test']:>8.4f} | "
                f"{sat['hidden_dim_used']:>5} | "
                f"{r['effective_rank']['entropy_var']:>8.1f}"
            )

        print("\n  EFFECTIVE RANK (all definitions)")
        print(
            f"  {'Encoder':>12} | {'Threshold':>9} | {'Ent(SV)':>8} | {'Ent(var)':>8} | {'PCA90%':>6} | {'Dim':>4}"
        )
        print("  " + "-" * 65)
        for r in results:
            er = r["effective_rank"]
            print(
                f"  {r['name']:>12} | {er['threshold_1pct']:>9} | "
                f"{er['entropy_sv']:>8.1f} | {er['entropy_var']:>8.1f} | "
                f"{er['pca_90pct']:>6} | {er['total_dims']:>4}"
            )

        # Phase 3: Figures
        if len(results) > 0:
            generate_figures(results)

        # Save JSON
        json_path = Path(__file__).parent / "results.json"

        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return round(float(obj), 6)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            return obj

        with open(json_path, "w") as f:
            json.dump(make_serializable(results), f, indent=2)
        print(f"\n  Results saved to {json_path}")

        print(f"\n{'='*80}")
        print("  DONE")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
