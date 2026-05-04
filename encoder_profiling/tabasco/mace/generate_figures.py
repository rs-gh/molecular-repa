"""
Generate figures for MACE investigation findings.

Run:
  source .venv/bin/activate
  export PROJECT_ROOT=$(pwd)/src/tabasco
  python encoder_profiling/tabasco/mace/generate_figures.py
"""

import os
import sys
import pickle

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import ase
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from e3nn import o3
from mace import data
from mace.calculators import mace_off
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.join(os.path.dirname(__file__), "../../../src/tabasco"),
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# -- Helpers ------------------------------------------------------------------


def load_geom_molecules(n=200):
    lmdb_path = os.path.join(PROJECT_ROOT, "data", "lmdb_geom", "train.lmdb")
    db = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    mols = []
    with db.begin() as txn:
        for i, (k, v) in enumerate(txn.cursor()):
            if i >= n:
                break
            mols.append(pickle.loads(v)["molecule"])
    db.close()
    return mols


def setup_mace(model_name="small"):
    device = "cpu"
    calc = mace_off(model_name, device=device)
    model = calc.models[0]
    model.eval()
    ni = int(model.num_interactions)
    irreps = o3.Irreps(str(model.products[0].linear.irreps_out))
    num_inv = irreps.dim // (irreps.lmax + 1) ** 2
    pl = [irreps.dim] * ni
    pl[-1] = num_inv
    total = int(np.sum(pl))
    config = {
        "num_interactions": ni,
        "l_max": irreps.lmax,
        "num_inv_feats": num_inv,
        "total_feats": total,
    }
    return calc, model, config


def get_mace_embedding(mol_noh, calc, model, config):
    """Get MACE embedding for a single heavy-atom molecule."""
    conf = mol_noh.GetConformer()
    nums = [a.GetAtomicNum() for a in mol_noh.GetAtoms()]
    atoms = ase.Atoms(numbers=nums, positions=conf.GetPositions())
    ks = data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    cfg = data.config_from_atoms(atoms, key_specification=ks, head_name=calc.head)
    ad = data.AtomicData.from_config(
        cfg,
        z_table=calc.z_table,
        cutoff=calc.r_max,
        heads=calc.available_heads,
    )
    loader = torch_geometric.dataloader.DataLoader([ad], batch_size=1, shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch.to_dict(), compute_force=False)
        nf = out["node_feats"]
        desc = extract_invariant(
            nf,
            num_layers=config["num_interactions"],
            num_features=config["num_inv_feats"],
            l_max=config["l_max"],
        )
        desc = desc[:, : config["total_feats"]].float().cpu()
    return desc


def get_chemeleon_embedding(mol_noh, smi):
    """Get CheMeleon embedding for a single heavy-atom molecule."""
    from tabasco.models.components.encoders import ChemPropEncoder
    from tabasco.chem.convert import MoleculeConverter

    torch.set_default_dtype(torch.float32)
    converter = MoleculeConverter()
    encoder = ChemPropEncoder(pretrained="chemeleon")
    encoder.eval()

    n = mol_noh.GetNumAtoms()
    td = converter.to_tensor(mol_noh, pad_to_size=n)
    coords = td["coords"].unsqueeze(0)
    atomics = td["atomics"].unsqueeze(0)
    padding = td["padding_mask"].unsqueeze(0)
    with torch.no_grad():
        emb = encoder(coords, atomics, padding, smiles=[smi])
    return emb.squeeze(0).cpu()


# -- Figure generation --------------------------------------------------------


def fig_01_value_distribution(mace_embs, chem_embs):
    """Compare value distributions of MACE vs CheMeleon."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mace_flat = torch.cat(mace_embs, dim=0).numpy().flatten()
    chem_flat = torch.cat(chem_embs, dim=0).numpy().flatten()

    axes[0].hist(mace_flat, bins=100, alpha=0.8, color="steelblue", density=True)
    axes[0].set_title("MACE Value Distribution")
    axes[0].set_xlabel("Feature value")
    axes[0].set_ylabel("Density")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)

    axes[1].hist(chem_flat, bins=100, alpha=0.8, color="coral", density=True)
    axes[1].set_title("CheMeleon Value Distribution")
    axes[1].set_xlabel("Feature value")
    axes[1].set_ylabel("Density")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)

    mace_sparsity = (np.abs(mace_flat) < 1e-10).mean()
    chem_sparsity = (np.abs(chem_flat) < 1e-10).mean()
    axes[0].text(
        0.95,
        0.95,
        f"Sparsity: {mace_sparsity:.1%}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )
    axes[1].text(
        0.95,
        0.95,
        f"Sparsity: {chem_sparsity:.1%}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_01_value_distribution.png"), dpi=150)
    plt.close()
    print("  Saved fig_01_value_distribution.png")


def fig_02_singular_values(mace_embs, chem_embs):
    """Compare singular value spectra."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, embs, color in [
        ("MACE", mace_embs, "steelblue"),
        ("CheMeleon", chem_embs, "coral"),
    ]:
        flat = torch.cat(embs, dim=0).float()
        S = torch.linalg.svdvals(flat).numpy()
        cum_var = np.cumsum(S**2) / np.sum(S**2)

        ax.plot(
            range(1, len(cum_var) + 1),
            cum_var,
            label=f"{name} (dim={flat.shape[1]})",
            color=color,
            linewidth=2,
        )

    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="90% variance")
    ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5, label="99% variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Singular Value Spectrum: MACE vs CheMeleon")
    ax.legend()
    ax.set_xlim(0, 200)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_02_singular_values.png"), dpi=150)
    plt.close()
    print("  Saved fig_02_singular_values.png")


def fig_03_conformer_sensitivity(mols, calc, model, config):
    """Show MACE conformer sensitivity."""
    results = []

    for mol in mols[:30]:
        smi = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
        mol_h = Chem.AddHs(Chem.MolFromSmiles(smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=5, params=params)
        if len(conf_ids) < 2:
            continue
        AllChem.MMFFOptimizeMoleculeConfs(mol_h)

        embs = []
        rmsds = []
        for conf_id in range(mol_h.GetNumConformers()):
            conf = mol_h.GetConformer(conf_id)
            nums = [a.GetAtomicNum() for a in mol_h.GetAtoms()]
            # Use heavy atom positions from this conformer
            heavy_idx = [
                i for i, a in enumerate(mol_h.GetAtoms()) if a.GetAtomicNum() > 1
            ]
            heavy_nums = [nums[i] for i in heavy_idx]
            heavy_pos = conf.GetPositions()[heavy_idx]
            atoms_heavy = ase.Atoms(numbers=heavy_nums, positions=heavy_pos)

            ks = data.KeySpecification(
                info_keys=calc.info_keys, arrays_keys=calc.arrays_keys
            )
            cfg = data.config_from_atoms(
                atoms_heavy, key_specification=ks, head_name=calc.head
            )
            ad = data.AtomicData.from_config(
                cfg,
                z_table=calc.z_table,
                cutoff=calc.r_max,
                heads=calc.available_heads,
            )
            loader = torch_geometric.dataloader.DataLoader(
                [ad], batch_size=1, shuffle=False
            )
            batch = next(iter(loader))
            with torch.no_grad():
                out = model(batch.to_dict(), compute_force=False)
                nf = out["node_feats"]
                desc = extract_invariant(
                    nf,
                    num_layers=config["num_interactions"],
                    num_features=config["num_inv_feats"],
                    l_max=config["l_max"],
                )
                desc = desc[:, : config["total_feats"]].float().cpu()
            embs.append(desc)

        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                cos = F.cosine_similarity(embs[i], embs[j], dim=-1).mean().item()
                # RMSD between heavy atoms
                pos_i = mol_h.GetConformer(i).GetPositions()[heavy_idx]
                pos_j = mol_h.GetConformer(j).GetPositions()[heavy_idx]
                rmsd = np.sqrt(np.mean((pos_i - pos_j) ** 2))
                results.append((rmsd, cos, len(heavy_idx)))

    if not results:
        print("  No conformer pairs found, skipping fig_03")
        return

    rmsds, cos_sims, sizes = zip(*results)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(rmsds, cos_sims, c=sizes, cmap="viridis", alpha=0.6, s=30)
    ax.set_xlabel("Conformer RMSD (Angstroms)")
    ax.set_ylabel("MACE Cosine Similarity")
    ax.set_title("MACE Conformer Sensitivity\n(CheMeleon: all points at cos=1.0)")
    ax.axhline(
        1.0,
        color="coral",
        linestyle="--",
        alpha=0.7,
        label="CheMeleon (all conformers identical)",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label="Number of heavy atoms")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_03_conformer_sensitivity.png"), dpi=150)
    plt.close()
    print("  Saved fig_03_conformer_sensitivity.png")


def fig_04_atom_discrimination(mols, calc, model, config):
    """Show per-atom-type embedding similarity."""
    from collections import defaultdict

    type_embeddings = defaultdict(list)

    for mol in mols[:50]:
        mol_noh = Chem.RemoveAllHs(mol)
        emb = get_mace_embedding(mol_noh, calc, model, config)
        symbols = [a.GetSymbol() for a in mol_noh.GetAtoms()]
        for i, sym in enumerate(symbols):
            type_embeddings[sym].append(emb[i])

    # Compute inter-type cosine similarities
    types = sorted(type_embeddings.keys())
    n = len(types)
    sim_matrix = np.zeros((n, n))

    for i, t1 in enumerate(types):
        embs1 = torch.stack(type_embeddings[t1])
        mean1 = embs1.mean(dim=0, keepdim=True)
        for j, t2 in enumerate(types):
            embs2 = torch.stack(type_embeddings[t2])
            mean2 = embs2.mean(dim=0, keepdim=True)
            sim_matrix[i, j] = F.cosine_similarity(mean1, mean2, dim=-1).item()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(types, fontsize=11)
    ax.set_yticklabels(types, fontsize=11)
    ax.set_title("MACE Mean Embedding Similarity by Atom Type")

    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9
            )

    plt.colorbar(im, label="Cosine similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_04_atom_discrimination.png"), dpi=150)
    plt.close()
    print("  Saved fig_04_atom_discrimination.png")


def fig_05_comparison_table(mace_embs, chem_embs):
    """Summary comparison bar chart."""
    mace_flat = torch.cat(mace_embs, dim=0).float()
    chem_flat = torch.cat(chem_embs, dim=0).float()

    metrics = {}
    for name, flat in [("MACE", mace_flat), ("CheMeleon", chem_flat)]:
        sparsity = (flat == 0).float().mean().item()
        # RankMe: Roy-Vetterli effective rank on raw singular values of
        # the uncentered embedding matrix (Garrido et al. 2023, ICML).
        S = torch.linalg.svdvals(flat)
        Sn = S / S.sum()
        rankme = torch.exp(-(Sn * torch.log(Sn + 1e-10)).sum()).item()
        cum_var = torch.cumsum(S**2, 0) / (S**2).sum()
        dims_90 = (cum_var < 0.9).sum().item() + 1
        metrics[name] = {
            "Sparsity (%)": sparsity * 100,
            "RankMe": rankme,
            "Dims for 90% var": dims_90,
            "Dimension": flat.shape[1],
        }

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    metric_names = list(list(metrics.values())[0].keys())
    x = np.arange(2)
    colors = ["steelblue", "coral"]

    for i, metric in enumerate(metric_names):
        values = [metrics["MACE"][metric], metrics["CheMeleon"][metric]]
        axes[i].bar(x, values, color=colors, width=0.5)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(["MACE", "CheMeleon"])
        axes[i].set_title(metric)
        for j, v in enumerate(values):
            axes[i].text(j, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("MACE vs CheMeleon: Representation Properties", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_05_comparison_summary.png"), dpi=150)
    plt.close()
    print("  Saved fig_05_comparison_summary.png")


def fig_06_heavy_vs_all_atom(mols, calc, model, config):
    """Show that removing H doesn't change heavy atom embeddings."""
    cos_sims = []

    for mol in mols[:30]:
        mol_noh = Chem.RemoveAllHs(mol)

        # Heavy only
        emb_heavy = get_mace_embedding(mol_noh, calc, model, config)

        # All atoms
        conf = mol.GetConformer()
        nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
        atoms_all = ase.Atoms(numbers=nums, positions=conf.GetPositions())
        ks = data.KeySpecification(
            info_keys=calc.info_keys, arrays_keys=calc.arrays_keys
        )
        cfg = data.config_from_atoms(
            atoms_all, key_specification=ks, head_name=calc.head
        )
        ad = data.AtomicData.from_config(
            cfg,
            z_table=calc.z_table,
            cutoff=calc.r_max,
            heads=calc.available_heads,
        )
        loader = torch_geometric.dataloader.DataLoader(
            [ad], batch_size=1, shuffle=False
        )
        batch = next(iter(loader))
        with torch.no_grad():
            out = model(batch.to_dict(), compute_force=False)
            nf = out["node_feats"]
            desc = extract_invariant(
                nf,
                num_layers=config["num_interactions"],
                num_features=config["num_inv_feats"],
                l_max=config["l_max"],
            )
            emb_all = desc[:, : config["total_feats"]].float().cpu()

        heavy_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
        emb_all_heavy = emb_all[heavy_idx]

        cos = F.cosine_similarity(emb_all_heavy, emb_heavy, dim=-1)
        cos_sims.extend(cos.tolist())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cos_sims, bins=50, color="steelblue", alpha=0.8)
    ax.set_xlabel("Cosine Similarity (all-atom vs heavy-only)")
    ax.set_ylabel("Count")
    ax.set_title("Effect of Removing H on Heavy Atom Embeddings\n(1.0 = no effect)")
    ax.axvline(
        np.mean(cos_sims),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(cos_sims):.6f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_06_heavy_vs_all_atom.png"), dpi=150)
    plt.close()
    print("  Saved fig_06_heavy_vs_all_atom.png")


# -- Main ---------------------------------------------------------------------


def main():
    print("Loading data and models...")
    mols = load_geom_molecules(n=200)
    calc, model, config = setup_mace("small")

    # Get MACE embeddings
    print("Computing MACE embeddings...")
    mace_embs = []
    for mol in mols[:50]:
        mol_noh = Chem.RemoveAllHs(mol)
        emb = get_mace_embedding(mol_noh, calc, model, config)
        mace_embs.append(emb)

    # Get CheMeleon embeddings
    print("Computing CheMeleon embeddings...")
    torch.set_default_dtype(torch.float32)
    chem_embs = []
    for mol in mols[:50]:
        mol_noh = Chem.RemoveAllHs(mol)
        smi = Chem.MolToSmiles(mol_noh)
        emb = get_chemeleon_embedding(mol_noh, smi)
        chem_embs.append(emb)

    print(f"\nGenerating figures to {FIGURES_DIR}/...")

    fig_01_value_distribution(mace_embs, chem_embs)
    fig_02_singular_values(mace_embs, chem_embs)

    # MACE needs float64 globally; reset after CheMeleon forced float32
    torch.set_default_dtype(torch.float64)
    fig_03_conformer_sensitivity(mols, calc, model, config)
    fig_04_atom_discrimination(mols, calc, model, config)
    fig_05_comparison_table(mace_embs, chem_embs)
    fig_06_heavy_vs_all_atom(mols, calc, model, config)

    print("\nDone! All figures saved.")


if __name__ == "__main__":
    main()
