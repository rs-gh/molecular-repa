#!/usr/bin/env python
"""CheMeleon characterisation for REPA target selection.

This script is the source of every Q1/Q2/Q3 number in
``encoder_profiling/tabasco/chemeleon/FINDINGS.md``. The framing
(Q1 information / Q2 saturation / Q3 conditioning) mirrors
``encoder_profiling/proteina/_probes/lib.py`` — see the cross-encoder
summary at ``encoder_profiling/tabasco/FINDINGS.md`` for the full
selection rubric.

Q1 — WHAT INFORMATION DOES THE ENCODER ENCODE?
    1.1 Atom identity     -> Section 3 (linear probe; per-atom-type cosine
                             distributions; t-SNE)
    1.2 3D sensitivity    -> Section 6 (GEOM conformer L2 distances —
                             the section where CheMeleon fails: L2 = 0.000
                             across all conformer pairs, by construction)
    1.3 Chemical context  -> Section 1 (known-molecule probe: benzene
                             symmetry, aromatic vs aliphatic clustering)
                             plus within-mol vs between-mol cosine
    1.4 Molecule identity -> Section 3 (mol-level cosine distributions)

Q2 — HOW MUCH IS REACHABLE FROM CHEAP INPUTS?
    -> Section 5 (projection feasibility: 2-layer MLP from
       random / atom-type / atom-type+context to encoder embedding).
       Saturation gap = best - random_floor. Random-input cos (~0.43)
       is the mean-direction floor; the +0.04 lift to atom-type+context
       (~0.47) is the structural headroom for any input to extract more.

Q3 — IS THE ENCODER A TRACTABLE OPTIMISATION TARGET?
    -> Section 2 (sparsity, dead dims, effective rank, participation
       ratio, dims-for-90/99-variance). The headline Q3 failures are
       93.8% sparsity (ReLU) and 500 dims-for-90%-variance vs a
       128-d projector input.

Pipeline-integrity checks (Section 4 — atom-order match, padding
all-zero, fast/slow path divergence) are operational sanity rather
than Q1/Q2/Q3 diagnostics; they live here because they share the same
data-loading code. The fast/slow path divergence (mean cos 0.23) is
documented in FINDINGS.md as an inference-path caveat, not a verdict
input.

Usage:
    source .venv/bin/activate
    export PROJECT_ROOT=$(pwd)/src/tabasco
    python encoder_profiling/tabasco/chemeleon/investigate.py
"""

import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "tabasco" / "src"))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from tabasco.models.components.encoders import ChemPropEncoder, Projector  # noqa: E402
from tabasco.chem.convert import MoleculeConverter  # noqa: E402
from tabasco.chem.constants import ATOM_NAMES  # noqa: E402
from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset  # noqa: E402

DATA_DIR = REPO_ROOT / "src" / "tabasco" / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Suppress RDKit warnings
from rdkit import RDLogger  # noqa: E402

RDLogger.logger().setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
CONVERTER = MoleculeConverter()
ENCODER = None  # lazy-loaded
MAX_ATOMS_QM9 = 9
MAX_ATOMS_GEOM = 71


def get_encoder():
    global ENCODER
    if ENCODER is None:
        print("Loading CheMeleon encoder...")
        ENCODER = ChemPropEncoder(pretrained="chemeleon")
        ENCODER.eval()
        for p in ENCODER.parameters():
            p.requires_grad = False
        print(f"  encoder_dim = {ENCODER.encoder_dim}")
    return ENCODER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smiles_to_mol(smi: str) -> Chem.Mol:
    """SMILES -> RDKit mol with 3D coords (hydrogens removed)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return Chem.RemoveAllHs(mol)


def create_batch(smiles_list: list[str], max_atoms: int = 10):
    """Create a batched TensorDict from SMILES (reuses test pattern)."""
    from tensordict import TensorDict

    tensordicts = []
    for s in smiles_list:
        mol = smiles_to_mol(s)
        td = CONVERTER.to_tensor(mol, pad_to_size=max_atoms)
        tensordicts.append(td)
    return TensorDict(
        {
            "coords": torch.stack([td["coords"] for td in tensordicts]),
            "atomics": torch.stack([td["atomics"] for td in tensordicts]),
            "padding_mask": torch.stack([td["padding_mask"] for td in tensordicts]),
        },
        batch_size=[len(smiles_list)],
    )


def embed_smiles(smiles_list: list[str], max_atoms: int = 10):
    """Compute CheMeleon embeddings for a list of SMILES."""
    encoder = get_encoder()
    batch = create_batch(smiles_list, max_atoms=max_atoms)
    with torch.no_grad():
        emb = encoder(
            batch["coords"], batch["atomics"], batch["padding_mask"], smiles=smiles_list
        )
    return emb, batch["padding_mask"]


def mol_level_embedding(emb, mask):
    """Mean-pool atom embeddings (ignoring padding) -> [B, D]."""
    # emb: [B, N, D], mask: [B, N] (True = padding)
    real = ~mask
    emb_masked = emb * real.unsqueeze(-1).float()
    counts = real.sum(dim=1, keepdim=True).float().clamp(min=1)
    return emb_masked.sum(dim=1) / counts


def load_dataset(name: str):
    """Load QM9 or GEOM dataset."""
    if name == "qm9":
        lmdb_dir = str(DATA_DIR / "lmdb_qm9")
        data_dir = str(DATA_DIR / "processed_qm9_train.pt")
    elif name == "geom":
        lmdb_dir = str(DATA_DIR / "lmdb_geom")
        data_dir = str(DATA_DIR / "processed_geom_train.pt")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    ds = UnconditionalLMDBDataset(
        data_dir=data_dir,
        split="train",
        add_random_rotation=False,
        add_random_permutation=False,
        reorder_to_smiles_order=False,
        remove_hydrogens=True,
        lmdb_dir=lmdb_dir,
    )
    return ds


def sample_smiles_from_dataset(ds, n: int, seed: int = 42):
    """Sample n items from dataset, return (smiles_list, items_list)."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    smiles_list = []
    items = []
    n_skipped = 0
    for idx in indices:
        item = ds[int(idx)]
        try:
            smi = item.get_non_tensor("smiles")
        except (KeyError, AttributeError):
            n_skipped += 1
            continue
        smiles_list.append(smi)
        items.append(item)
    if n_skipped > 0:
        print(f"  WARNING: {n_skipped}/{len(indices)} items had no SMILES (skipped)")
    return smiles_list, items


def savefig(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# SECTION 1: Known Molecule Probe
# ===================================================================
def section_1_known_molecules():
    print("\n" + "=" * 70)
    print("SECTION 1: Known Molecule Probe")
    print("=" * 70)

    known_mols = {
        "methane": "C",
        "ethane": "CC",
        "propane": "CCC",
        "butane": "CCCC",
        "methanol": "CO",
        "ethanol": "CCO",
        "benzene": "c1ccccc1",
        "toluene": "Cc1ccccc1",
        "pyridine": "c1ccncc1",
        "phenol": "Oc1ccccc1",
        "naphthalene": "c1ccc2ccccc2c1",
        "formaldehyde": "C=O",
        "acetone": "CC(=O)C",
        "acetic_acid": "CC(=O)O",
        "methylamine": "CN",
        "aniline": "Nc1ccccc1",
        "furan": "c1ccoc1",
        "thiophene": "c1ccsc1",
        "cyclopentane": "C1CCCC1",
        "cyclohexane": "C1CCCCC1",
    }

    names = list(known_mols.keys())
    smiles = list(known_mols.values())

    # Determine max atoms needed
    max_atoms = max(Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in smiles) + 1

    emb, mask = embed_smiles(smiles, max_atoms=max_atoms)
    mol_emb = mol_level_embedding(emb, mask)  # [N_mols, D]

    # ---- Molecule-level cosine similarity heatmap ----
    mol_emb_norm = F.normalize(mol_emb, dim=-1)
    cos_sim = (mol_emb_norm @ mol_emb_norm.T).numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cos_sim,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("CheMeleon Molecule-Level Cosine Similarity (Known Molecules)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    savefig(fig, "fig_01_known_mol_similarity.png")

    # ---- Within-molecule atom similarity for select molecules ----
    select_mols = ["benzene", "ethanol", "acetic_acid", "naphthalene"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, mol_name in zip(axes.flat, select_mols):
        idx = names.index(mol_name)
        n_real = (~mask[idx]).sum().item()
        atom_emb = emb[idx, :n_real]  # [n_real, D]
        atom_norm = F.normalize(atom_emb, dim=-1)
        atom_sim = (atom_norm @ atom_norm.T).numpy()

        # Get atom labels
        mol = smiles_to_mol(smiles[idx])
        atom_labels = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_real)]

        sns.heatmap(
            atom_sim,
            xticklabels=atom_labels,
            yticklabels=atom_labels,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0.5,
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f"{mol_name} ({smiles[idx]})")
    fig.suptitle("Within-Molecule Atom Cosine Similarity", fontsize=14)
    plt.tight_layout()
    savefig(fig, "fig_02_atom_similarity_matrices.png")

    # ---- Atom norm by type ----
    norms_by_type = defaultdict(list)
    for i, mol_name in enumerate(names):
        n_real = (~mask[i]).sum().item()
        mol = smiles_to_mol(smiles[i])
        for j in range(n_real):
            atom_type = mol.GetAtomWithIdx(j).GetSymbol()
            norm_val = emb[i, j].norm().item()
            norms_by_type[atom_type].append(norm_val)

    fig, ax = plt.subplots(figsize=(10, 6))
    types_sorted = sorted(norms_by_type.keys(), key=lambda x: -len(norms_by_type[x]))
    data_for_box = [norms_by_type[t] for t in types_sorted]
    bp = ax.boxplot(data_for_box, labels=types_sorted, patch_artist=True)
    for patch, color in zip(bp["boxes"], sns.color_palette("Set2", len(types_sorted))):
        patch.set_facecolor(color)
    ax.set_xlabel("Atom Type")
    ax.set_ylabel("L2 Norm of CheMeleon Embedding")
    ax.set_title("Embedding Norm by Atom Type (Known Molecules)")
    savefig(fig, "fig_03_atom_norm_by_type.png")

    # ---- Print summary ----
    print(f"\n  Molecules: {len(names)}")
    print(f"  Embedding dim: {emb.shape[-1]}")
    print(
        f"  Cosine sim range: [{cos_sim[np.triu_indices_from(cos_sim, k=1)].min():.3f}, "
        f"{cos_sim[np.triu_indices_from(cos_sim, k=1)].max():.3f}]"
    )
    print(
        f"  Mean off-diagonal cosine sim: {cos_sim[np.triu_indices_from(cos_sim, k=1)].mean():.3f}"
    )

    # Check benzene symmetry
    idx_benz = names.index("benzene")
    n_benz = (~mask[idx_benz]).sum().item()
    benz_emb = emb[idx_benz, :n_benz]
    benz_norm = F.normalize(benz_emb, dim=-1)
    benz_sim = (benz_norm @ benz_norm.T).numpy()
    min_benz_sim = benz_sim[np.triu_indices_from(benz_sim, k=1)].min()
    print(
        f"  Benzene carbon pairwise min cosine sim: {min_benz_sim:.4f} (should be ~1.0 if symmetric)"
    )

    # Sparsity quick check
    all_real_emb = []
    for i in range(len(names)):
        n_real = (~mask[i]).sum().item()
        all_real_emb.append(emb[i, :n_real])
    all_real = torch.cat(all_real_emb, dim=0)
    sparsity = (all_real == 0).float().mean().item()
    print(f"  Sparsity (fraction of exact zeros): {sparsity:.3f}")
    print(f"  Mean embedding norm: {all_real.norm(dim=-1).mean().item():.2f}")


# ===================================================================
# SECTION 2: Sparsity & Dimensionality
# ===================================================================
def section_2_sparsity_dimensionality():
    print("\n" + "=" * 70)
    print("SECTION 2: Sparsity & Dimensionality Analysis")
    print("=" * 70)

    results = {}

    for ds_name, max_atoms in [("qm9", MAX_ATOMS_QM9), ("geom", MAX_ATOMS_GEOM)]:
        print(f"\n  Loading {ds_name} dataset...")
        ds = load_dataset(ds_name)
        smiles_list, items = sample_smiles_from_dataset(ds, n=2000)
        print(f"  Got {len(smiles_list)} molecules from {ds_name}")

        # Compute embeddings in batches (to avoid memory issues)
        all_atom_embs = []
        all_atom_types = []
        batch_size = 64
        encoder = get_encoder()

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
                atom_emb = emb[i, :n_real]
                all_atom_embs.append(atom_emb)
                # Get atom types
                for j in range(n_real):
                    atom_idx = atomics[i, j].argmax().item()
                    all_atom_types.append(ATOM_NAMES[atom_idx])

        all_emb = torch.cat(all_atom_embs, dim=0).numpy()  # [total_atoms, 2048]
        print(f"  Total atoms: {all_emb.shape[0]}, Dims: {all_emb.shape[1]}")

        # Check for all-zero embeddings (failed conversions)
        per_atom_norms = np.linalg.norm(all_emb, axis=1)
        n_zero_atoms = (per_atom_norms == 0).sum()
        if n_zero_atoms > 0:
            print(
                f"  WARNING: {n_zero_atoms}/{all_emb.shape[0]} real atoms have all-zero embeddings!"
            )
            print("    This indicates failed MolFromSmiles conversions.")
        else:
            print(f"  All {all_emb.shape[0]} real atoms have non-zero embeddings (OK)")

        # Sparsity
        sparsity = (all_emb == 0).mean()
        near_zero = (np.abs(all_emb) < 0.01).mean()
        print(f"  Exact zero fraction: {sparsity:.4f}")
        print(f"  Near-zero (<0.01) fraction: {near_zero:.4f}")

        # Per-dimension activation frequency
        dim_activation = (all_emb > 0.01).mean(axis=0)  # [2048]
        dead_dims = (dim_activation == 0).sum()
        rarely_active = (dim_activation < 0.01).sum()
        frequently_active = (dim_activation > 0.5).sum()
        print(f"  Dead dimensions (never active): {dead_dims}")
        print(f"  Rarely active (<1%): {rarely_active}")
        print(f"  Frequently active (>50%): {frequently_active}")

        # PCA
        # Note that we do not standardize (whiten) before PCA, since we want to see the raw variance structure.
        all_emb_centered = all_emb - all_emb.mean(axis=0)
        n_components = min(500, *all_emb_centered.shape)
        pca = PCA(n_components=n_components)
        pca.fit(all_emb_centered)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(
            f"  PCA explained variance: "
            f"top-10={cumvar[9]:.3f}, top-50={cumvar[49]:.3f}, "
            f"top-100={cumvar[99]:.3f}, top-200={cumvar[min(199, n_components-1)]:.3f}"
        )

        # Effective rank (1%-threshold variant; kept for backwards compat
        # with prior CheMeleon FINDINGS numbers).
        sv = pca.singular_values_
        effective_rank_1pct = (sv > 0.01 * sv[0]).sum()
        # Participation ratio (Gao et al. 2017): on covariance eigenvalues.
        eigenvals = pca.explained_variance_
        participation_ratio = eigenvals.sum() ** 2 / (eigenvals**2).sum()
        # RankMe (Garrido et al. 2023, ICML): Roy-Vetterli effective rank
        # on raw singular values of the *uncentered* embedding matrix.
        # This is the canonical cross-encoder metric; the proteina probes
        # use the same definition. Mirror their max_atoms=30000 cap to
        # keep SVD memory/runtime bounded on GEOM (~28M atoms).
        max_atoms_rankme = 30000
        if all_emb.shape[0] > max_atoms_rankme:
            rng = np.random.default_rng(0)
            idx_sub = rng.choice(all_emb.shape[0], max_atoms_rankme, replace=False)
            emb_sub = all_emb[idx_sub]
        else:
            emb_sub = all_emb
        S_raw = np.linalg.svd(emb_sub, compute_uv=False)
        p_raw = S_raw / S_raw.sum()
        p_raw = p_raw[p_raw > 0]
        rankme = float(np.exp(-np.sum(p_raw * np.log(p_raw))))
        print(f"  Effective rank (1% of max SV): {effective_rank_1pct}")
        print(f"  Participation ratio: {participation_ratio:.1f}")
        print(f"  RankMe: {rankme:.1f} / {all_emb.shape[1]}")

        results[ds_name] = {
            "all_emb": all_emb,
            "all_atom_types": all_atom_types,
            "sparsity": sparsity,
            "dim_activation": dim_activation,
            "pca": pca,
            "cumvar": cumvar,
            "sv": sv,
            "effective_rank": effective_rank_1pct,
            "participation_ratio": participation_ratio,
            "rankme": rankme,
            "smiles_list": smiles_list,
            "items": items,
        }

    # ---- Figure: PCA explained variance ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds_name, res in results.items():
        n_comp = len(res["cumvar"])
        ax.plot(range(1, n_comp + 1), res["cumvar"], label=ds_name.upper(), linewidth=2)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95%")
    ax.axhline(y=0.99, color="gray", linestyle="-.", alpha=0.5, label="99%")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("CheMeleon Embedding Dimensionality (PCA)")
    ax.legend()
    ax.set_xlim(0, 300)
    savefig(fig, "fig_04_pca_explained_variance.png")

    # ---- Figure: Dimension activation histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (ds_name, res) in zip(axes, results.items()):
        ax.hist(res["dim_activation"], bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--", label="Dead dims")
        ax.set_xlabel("Activation Frequency (fraction of atoms)")
        ax.set_ylabel("Number of Dimensions")
        ax.set_title(f"{ds_name.upper()} Dimension Activation")
        ax.legend()
    plt.tight_layout()
    savefig(fig, "fig_05_dimension_activation.png")

    # ---- Figure: Singular value spectrum ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds_name, res in results.items():
        sv_norm = res["sv"] / res["sv"][0]
        ax.semilogy(
            range(1, len(sv_norm) + 1), sv_norm, label=ds_name.upper(), linewidth=2
        )
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="1% cutoff")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Normalized Singular Value")
    ax.set_title("Singular Value Spectrum")
    ax.legend()
    savefig(fig, "fig_06_singular_values.png")

    return results


# ===================================================================
# SECTION 3: Discriminativeness
# ===================================================================
def section_3_discriminativeness(results: dict):
    print("\n" + "=" * 70)
    print("SECTION 3: Discriminativeness Analysis")
    print("=" * 70)

    for ds_name, res in results.items():
        print(f"\n  --- {ds_name.upper()} ---")
        all_emb = res["all_emb"]
        all_atom_types = res["all_atom_types"]
        smiles_list = res["smiles_list"]  # noqa: F841
        items = res["items"]

        # ---- Molecule-level cosine similarity distribution ----
        # Recompute molecule-level embeddings
        mol_embs = []
        for item in items:
            n_real = (~item["padding_mask"]).sum().item()
            # We need to re-embed here to get per-molecule embeddings
            pass

        # Instead, compute from stored atom embeddings grouped by molecule
        mol_embs = []
        atom_offset = 0
        for item in items:
            n_real = (~item["padding_mask"]).sum().item()
            atom_chunk = all_emb[atom_offset : atom_offset + n_real]
            mol_embs.append(atom_chunk.mean(axis=0))
            atom_offset += n_real

        mol_embs = np.array(mol_embs)  # [N_mols, D]
        mol_norms = mol_embs / (np.linalg.norm(mol_embs, axis=1, keepdims=True) + 1e-10)
        mol_cos_sim = mol_norms @ mol_norms.T

        # Upper triangle values
        triu_idx = np.triu_indices_from(mol_cos_sim, k=1)
        mol_sims = mol_cos_sim[triu_idx]
        print(
            f"  Molecule-level cosine sim: mean={mol_sims.mean():.3f}, "
            f"std={mol_sims.std():.3f}, range=[{mol_sims.min():.3f}, {mol_sims.max():.3f}]"
        )

        # ---- Within-molecule vs between-molecule atom similarity ----
        within_sims = []
        atom_offset = 0
        for item in items:
            n_real = (~item["padding_mask"]).sum().item()
            if n_real < 2:
                atom_offset += n_real
                continue
            chunk = all_emb[atom_offset : atom_offset + n_real]
            chunk_norm = chunk / (np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-10)
            sim_mat = chunk_norm @ chunk_norm.T
            triu = sim_mat[np.triu_indices_from(sim_mat, k=1)]
            within_sims.extend(triu.tolist())
            atom_offset += n_real

        # Sample some between-molecule pairs
        rng = np.random.RandomState(42)
        between_sims = []
        n_between = min(len(within_sims), 50000)
        all_emb_norm = all_emb / (
            np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-10
        )
        for _ in range(n_between):
            i, j = rng.choice(all_emb.shape[0], size=2, replace=False)
            between_sims.append(np.dot(all_emb_norm[i], all_emb_norm[j]))

        print(
            f"  Within-molecule atom sim: mean={np.mean(within_sims):.3f}, std={np.std(within_sims):.3f}"
        )
        print(
            f"  Between-molecule atom sim: mean={np.mean(between_sims):.3f}, std={np.std(between_sims):.3f}"
        )

        # ---- Linear probe: atom type classification ----
        atom_type_to_idx = {
            name: i for i, name in enumerate(ATOM_NAMES[:-1])
        }  # exclude *
        X_probe, y_probe = [], []
        for i, (emb_row, at) in enumerate(zip(all_emb, all_atom_types)):
            if at in atom_type_to_idx:
                X_probe.append(emb_row)
                y_probe.append(atom_type_to_idx[at])
        X_probe = np.array(X_probe)
        y_probe = np.array(y_probe)

        if len(np.unique(y_probe)) > 1:
            clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
            clf.fit(X_probe, y_probe)
            train_acc = clf.score(X_probe, y_probe)
            n_classes = len(np.unique(y_probe))
            random_baseline = 1.0 / n_classes
            print(
                f"  Linear probe: accuracy={train_acc:.3f} (random={random_baseline:.3f}, {n_classes} classes)"
            )
            # Per-class accuracy
            from sklearn.metrics import classification_report

            pred = clf.predict(X_probe)
            unique_labels = sorted(np.unique(y_probe))
            label_names = [ATOM_NAMES[i] for i in unique_labels]
            report = classification_report(
                y_probe,
                pred,
                labels=unique_labels,
                target_names=label_names,
                output_dict=True,
            )
            for lab in label_names:
                r = report[lab]
                print(
                    f"    {lab:>3s}: precision={r['precision']:.3f}, recall={r['recall']:.3f}, "
                    f"f1={r['f1-score']:.3f}, support={int(r['support'])}"
                )

        # Store for plotting
        res["mol_sims"] = mol_sims
        res["within_sims"] = within_sims
        res["between_sims"] = between_sims
        res["mol_embs"] = mol_embs
        res["probe_accuracy"] = train_acc if len(np.unique(y_probe)) > 1 else None

    # ---- Figures (use QM9 for clarity) ----
    qm9 = results["qm9"]

    # Molecule sim distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(qm9["mol_sims"], bins=100, alpha=0.7, edgecolor="black", density=True)
    ax.axvline(
        x=qm9["mol_sims"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean={qm9['mol_sims'].mean():.3f}",
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("QM9 Molecule-Level Pairwise Cosine Similarity")
    ax.legend()
    savefig(fig, "fig_07_molecule_sim_distribution.png")

    # Within vs between
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        qm9["within_sims"],
        bins=100,
        alpha=0.6,
        density=True,
        label="Within-molecule",
        color="blue",
    )
    ax.hist(
        qm9["between_sims"],
        bins=100,
        alpha=0.6,
        density=True,
        label="Between-molecule",
        color="orange",
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("QM9 Within vs Between Molecule Atom Cosine Similarity")
    ax.legend()
    savefig(fig, "fig_08_within_vs_between.png")

    # Atom type separation (QM9)
    all_emb_qm9 = qm9["all_emb"]
    all_types_qm9 = qm9["all_atom_types"]
    all_emb_qm9_norm = all_emb_qm9 / (
        np.linalg.norm(all_emb_qm9, axis=1, keepdims=True) + 1e-10
    )

    type_groups = defaultdict(list)
    for i, t in enumerate(all_types_qm9):
        type_groups[t].append(i)

    pair_data = {}
    types_present = [t for t in ATOM_NAMES[:-1] if t in type_groups]
    rng = np.random.RandomState(42)
    for i, t1 in enumerate(types_present):
        for t2 in types_present[i:]:
            key = f"{t1}-{t2}"
            sims = []
            idx1 = type_groups[t1]
            idx2 = type_groups[t2]
            n_sample = min(2000, len(idx1) * len(idx2))
            for _ in range(n_sample):
                a = rng.choice(idx1)
                b = rng.choice(idx2)
                if a != b:
                    sims.append(np.dot(all_emb_qm9_norm[a], all_emb_qm9_norm[b]))
            if sims:
                pair_data[key] = sims

    fig, ax = plt.subplots(figsize=(14, 6))
    keys_sorted = sorted(pair_data.keys())
    box_data = [pair_data[k] for k in keys_sorted]
    bp = ax.boxplot(box_data, labels=keys_sorted, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], sns.color_palette("husl", len(keys_sorted))):
        patch.set_facecolor(color)
    ax.set_xlabel("Atom Type Pair")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("QM9 Atom Cosine Similarity by Element Pair")
    plt.xticks(rotation=45, ha="right")
    savefig(fig, "fig_09_atom_type_separation.png")

    # t-SNE of molecule embeddings (QM9)
    print("\n  Computing t-SNE (QM9 molecules)...")
    mol_embs_qm9 = qm9["mol_embs"]
    n_tsne = min(1000, len(mol_embs_qm9))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(mol_embs_qm9[:n_tsne])

    # Color by number of atoms
    n_atoms_list = []
    for item in qm9["items"][:n_tsne]:
        n_atoms_list.append((~item["padding_mask"]).sum().item())

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1], c=n_atoms_list, cmap="viridis", s=10, alpha=0.7
    )
    plt.colorbar(scatter, label="Number of atoms")
    ax.set_title("t-SNE of QM9 CheMeleon Embeddings (colored by #atoms)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    savefig(fig, "fig_10_tsne.png")


# ===================================================================
# SECTION 4: Pipeline Integrity
# ===================================================================
def section_4_pipeline_integrity():
    print("\n" + "=" * 70)
    print("SECTION 4: Pipeline Integrity")
    print("=" * 70)

    encoder = get_encoder()
    ds_qm9 = load_dataset("qm9")

    smiles_list, items = sample_smiles_from_dataset(ds_qm9, n=100)
    print(f"  Testing {len(smiles_list)} QM9 molecules")

    fast_embs = []
    slow_embs = []
    is_aromatic = []
    atom_order_matches = []
    padding_all_zero = []

    for smi, item in zip(smiles_list, items):
        coords = item["coords"].unsqueeze(0)
        atomics = item["atomics"].unsqueeze(0)
        mask = item["padding_mask"].unsqueeze(0)
        n_real = (~item["padding_mask"]).sum().item()

        # Fast path (SMILES)
        with torch.no_grad():
            fast = encoder(coords, atomics, mask, smiles=[smi])

        # Slow path (from_tensor, no SMILES)
        # Clear cache to force slow path
        old_cache = encoder._molgraph_cache.copy()
        encoder._molgraph_cache.clear()
        with torch.no_grad():
            slow = encoder(coords, atomics, mask, smiles=None)
        encoder._molgraph_cache = old_cache

        fast_embs.append(fast[0, :n_real].numpy())
        slow_embs.append(slow[0, :n_real].numpy())

        # Check aromaticity
        mol = Chem.MolFromSmiles(smi)
        is_aromatic.append(
            any(atom.GetIsAromatic() for atom in mol.GetAtoms()) if mol else False
        )

        # Check atom order: compare tensor atom types to MolFromSmiles atom types
        if mol is not None:
            smi_atoms = [
                mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumHeavyAtoms())
            ]
            tensor_atoms = [
                ATOM_NAMES[atomics[0, j].argmax().item()] for j in range(n_real)
            ]
            atom_order_matches.append(smi_atoms == tensor_atoms)
        else:
            atom_order_matches.append(False)

        # Check padding is zero
        if mask.any():
            pad_emb = fast[0, n_real:]
            padding_all_zero.append((pad_emb == 0).all().item())
        else:
            padding_all_zero.append(True)

    # Compute per-atom cosine similarity between fast and slow
    per_mol_cos = []
    for fast, slow in zip(fast_embs, slow_embs):
        if fast.shape[0] == 0:
            continue
        f_norm = fast / (np.linalg.norm(fast, axis=1, keepdims=True) + 1e-10)
        s_norm = slow / (np.linalg.norm(slow, axis=1, keepdims=True) + 1e-10)
        cos = (f_norm * s_norm).sum(axis=1)  # per-atom cos sim
        per_mol_cos.append(cos.mean())

    per_mol_cos = np.array(per_mol_cos)

    print("\n  Fast vs Slow path cosine similarity:")
    print(f"    Mean: {per_mol_cos.mean():.4f}")
    print(f"    Min:  {per_mol_cos.min():.4f}")
    print(
        f"    Molecules with cos_sim < 0.99: {(per_mol_cos < 0.99).sum()}/{len(per_mol_cos)}"
    )

    aromatic_cos = per_mol_cos[np.array(is_aromatic[: len(per_mol_cos)])]
    non_aromatic_cos = per_mol_cos[~np.array(is_aromatic[: len(per_mol_cos)])]
    if len(aromatic_cos) > 0:
        print(
            f"    Aromatic molecules: mean cos_sim = {aromatic_cos.mean():.4f} (n={len(aromatic_cos)})"
        )
    if len(non_aromatic_cos) > 0:
        print(
            f"    Non-aromatic molecules: mean cos_sim = {non_aromatic_cos.mean():.4f} (n={len(non_aromatic_cos)})"
        )

    print("\n  Atom order verification:")
    n_match = sum(atom_order_matches)
    print(
        f"    Matching: {n_match}/{len(atom_order_matches)} ({100*n_match/len(atom_order_matches):.1f}%)"
    )

    print("\n  Padding verification:")
    n_pad_ok = sum(padding_all_zero)
    print(
        f"    All-zero padding: {n_pad_ok}/{len(padding_all_zero)} ({100*n_pad_ok/len(padding_all_zero):.1f}%)"
    )

    # ---- Figure: Fast vs Slow scatter ----
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["red" if a else "blue" for a in is_aromatic[: len(per_mol_cos)]]
    ax.scatter(range(len(per_mol_cos)), per_mol_cos, c=colors, s=20, alpha=0.7)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect match")
    ax.axhline(
        y=0.99, color="orange", linestyle="--", alpha=0.5, label="0.99 threshold"
    )
    ax.set_xlabel("Molecule Index")
    ax.set_ylabel("Mean Atom Cosine Similarity (Fast vs Slow)")
    ax.set_title("Pipeline: Fast Path vs Slow Path\n(Red=aromatic, Blue=non-aromatic)")
    ax.legend()
    savefig(fig, "fig_11_fast_vs_slow.png")


# ===================================================================
# SECTION 5: Projection Feasibility
# ===================================================================
def section_5_projection_feasibility(results: dict):
    print("\n" + "=" * 70)
    print("SECTION 5: Projection Feasibility")
    print("=" * 70)

    # Use QM9 embeddings as targets
    qm9_emb = torch.tensor(results["qm9"]["all_emb"], dtype=torch.float32)
    n_atoms_total = qm9_emb.shape[0]
    enc_dim = qm9_emb.shape[1]

    # Target manifold dimensionality
    pca = results["qm9"]["pca"]  # noqa: F841
    cumvar = results["qm9"]["cumvar"]
    k_90 = np.searchsorted(cumvar, 0.90) + 1
    k_95 = np.searchsorted(cumvar, 0.95) + 1
    k_99 = np.searchsorted(cumvar, 0.99) + 1
    print("\n  Target manifold dimensionality:")
    print(f"    90% variance: {k_90} components")
    print(f"    95% variance: {k_95} components")
    print(f"    99% variance: {k_99} components")
    print("    Projector input dim: 128 (single head) or 256 (cross-attention)")
    feasible_str = "YES" if k_99 < 256 else "MARGINAL" if k_99 < 512 else "NO"
    print(f"    Feasibility assessment: {feasible_str}")

    # ---- Random baseline ----
    hidden_dim = 128
    projector_random = Projector(
        hidden_dim=hidden_dim, encoder_dim=enc_dim, num_layers=2
    )
    random_input = torch.randn(min(1000, n_atoms_total), hidden_dim)
    targets = qm9_emb[: random_input.shape[0]]
    targets_norm = F.normalize(targets, dim=-1)

    with torch.no_grad():
        proj_out = projector_random(random_input)
        proj_norm = F.normalize(proj_out, dim=-1)
        baseline_cos = (proj_norm * targets_norm).sum(dim=-1)
    print("\n  Random projector baseline:")
    print(f"    Mean cosine sim: {baseline_cos.mean().item():.4f}")
    print(f"    Std: {baseline_cos.std().item():.4f}")

    # ---- Train projector with (a) random inputs, (b) atom-type embeddings ----
    n_train = min(5000, n_atoms_total)
    target_train = qm9_emb[:n_train]
    target_train_norm = F.normalize(target_train, dim=-1)

    # Atom type indices for training data
    atom_types_train = results["qm9"]["all_atom_types"][:n_train]
    atom_type_to_idx = {name: i for i, name in enumerate(ATOM_NAMES)}

    def train_projector(input_data, targets_norm, label, n_epochs=200, lr=1e-3):
        proj = Projector(hidden_dim=hidden_dim, encoder_dim=enc_dim, num_layers=2)
        optimizer = torch.optim.Adam(proj.parameters(), lr=lr)
        losses = []
        for epoch in range(n_epochs):
            proj.train()
            out = proj(input_data)
            out_norm = F.normalize(out, dim=-1)
            cos_sim = (out_norm * targets_norm).sum(dim=-1)
            loss = -cos_sim.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(-loss.item())
            if (epoch + 1) % 50 == 0:
                print(
                    f"    [{label}] Epoch {epoch+1}: mean cos_sim = {-loss.item():.4f}"
                )
        return losses

    # (a) Random inputs
    print("\n  Training projector with random inputs:")
    random_inputs = torch.randn(n_train, hidden_dim)
    losses_random = train_projector(
        random_inputs, target_train_norm, "random", n_epochs=200
    )

    # (b) Atom-type one-hot embeddings expanded to hidden_dim
    print("\n  Training projector with atom-type embeddings:")
    atom_type_emb = torch.zeros(n_train, hidden_dim)
    for i, at in enumerate(atom_types_train):
        idx = atom_type_to_idx.get(at, len(ATOM_NAMES) - 1)
        atom_type_emb[i, idx] = 1.0
        # Also add some random noise for variety
        atom_type_emb[i, len(ATOM_NAMES) : len(ATOM_NAMES) + 10] = torch.randn(10) * 0.1
    losses_atomtype = train_projector(
        atom_type_emb, target_train_norm, "atom-type", n_epochs=200
    )

    # (c) Atom-type embedding + random neighbor context
    print("\n  Training projector with atom-type + random context:")
    atom_context = torch.randn(n_train, hidden_dim)
    for i, at in enumerate(atom_types_train):
        idx = atom_type_to_idx.get(at, len(ATOM_NAMES) - 1)
        atom_context[i, idx] = 5.0  # Strong atom-type signal
    losses_context = train_projector(
        atom_context, target_train_norm, "atom+context", n_epochs=200
    )

    print("\n  Final cosine similarities:")
    print(f"    Random inputs: {losses_random[-1]:.4f}")
    print(f"    Atom-type emb: {losses_atomtype[-1]:.4f}")
    print(f"    Atom + context: {losses_context[-1]:.4f}")

    # ---- Gradient analysis with sparse targets ----
    print("\n  Sparsity impact on cosine similarity gradient:")
    # A single gradient step
    proj_test = Projector(hidden_dim=hidden_dim, encoder_dim=enc_dim, num_layers=2)
    test_input = torch.randn(100, hidden_dim)
    test_target = target_train_norm[:100]

    proj_test.train()
    out = proj_test(test_input)
    out_norm = F.normalize(out, dim=-1)
    cos_loss = -(out_norm * test_target).sum(dim=-1).mean()
    cos_loss.backward()

    grad_norms = []
    for p in proj_test.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print(f"    Gradient norms: {[f'{g:.6f}' for g in grad_norms]}")
    print(f"    Mean grad norm: {np.mean(grad_norms):.6f}")

    # ---- Figure: Learning curves ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses_random, label="Random inputs", linewidth=2)
    ax.plot(losses_atomtype, label="Atom-type embeddings", linewidth=2)
    ax.plot(losses_context, label="Atom-type + context", linewidth=2)
    ax.axhline(
        y=0, color="gray", linestyle="--", alpha=0.5, label="Zero (random baseline)"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Projector Training: Can a 2-Layer MLP Learn the Mapping?")
    ax.legend()
    ax.set_ylim(-0.1, max(losses_context[-1], losses_atomtype[-1], 0.5) * 1.2)
    savefig(fig, "fig_12_projector_learning.png")


# ===================================================================
# SECTION 6: 2D vs 3D Mismatch
# ===================================================================
def section_6_2d_3d_mismatch():
    print("\n" + "=" * 70)
    print("SECTION 6: 2D vs 3D Mismatch (GEOM Conformer Analysis)")
    print("=" * 70)

    ds_geom = load_dataset("geom")
    encoder = get_encoder()

    # Find molecules with multiple conformers (same SMILES)
    print("  Scanning GEOM for conformer groups...")
    smiles_to_indices = defaultdict(list)
    n_scan = min(10000, len(ds_geom))
    for idx in range(n_scan):
        item = ds_geom[idx]
        try:
            smi = item.get_non_tensor("smiles")
            smiles_to_indices[smi].append(idx)
        except (KeyError, AttributeError):
            continue

    # Find groups with >= 3 conformers
    multi_conformer = {
        smi: idxs for smi, idxs in smiles_to_indices.items() if len(idxs) >= 3
    }
    print(
        f"  Found {len(multi_conformer)} molecules with >= 3 conformers in first {n_scan} entries"
    )

    if len(multi_conformer) == 0:
        print("  SKIPPING conformer analysis (no multi-conformer molecules found)")
        return

    # Analyze up to 50 molecules
    rmsds = []
    emb_dists = []
    n_analyzed = 0

    for smi, idxs in list(multi_conformer.items())[:50]:
        # Get items for all conformers
        conformer_items = [
            ds_geom[i] for i in idxs[:5]
        ]  # max 5 conformers per molecule

        # Compute embeddings for all conformers
        coords_list = [item["coords"].unsqueeze(0) for item in conformer_items]
        atomics_list = [item["atomics"].unsqueeze(0) for item in conformer_items]
        mask_list = [item["padding_mask"].unsqueeze(0) for item in conformer_items]

        coords_batch = torch.cat(coords_list, dim=0)
        atomics_batch = torch.cat(atomics_list, dim=0)
        mask_batch = torch.cat(mask_list, dim=0)

        with torch.no_grad():
            embs = encoder(
                coords_batch,
                atomics_batch,
                mask_batch,
                smiles=[smi] * len(conformer_items),
            )

        n_real = (~mask_batch[0]).sum().item()

        # Check embedding distance between conformers (should be 0 for 2D encoder)
        for i in range(len(conformer_items)):
            for j in range(i + 1, len(conformer_items)):
                emb_i = embs[i, :n_real]
                emb_j = embs[j, :n_real]
                dist = (emb_i - emb_j).norm().item()
                emb_dists.append(dist)

                # Compute RMSD between 3D coordinates
                c_i = coords_batch[i, :n_real].numpy()
                c_j = coords_batch[j, :n_real].numpy()
                rmsd = np.sqrt(((c_i - c_j) ** 2).sum(axis=1).mean())
                rmsds.append(rmsd)

        n_analyzed += 1

    rmsds = np.array(rmsds)
    emb_dists = np.array(emb_dists)

    print(f"\n  Analyzed {n_analyzed} molecules, {len(rmsds)} conformer pairs")
    print(
        f"  3D RMSD: mean={rmsds.mean():.3f}, range=[{rmsds.min():.3f}, {rmsds.max():.3f}]"
    )
    print(
        f"  CheMeleon embedding L2 distance: mean={emb_dists.mean():.6f}, "
        f"max={emb_dists.max():.6f}"
    )
    print(f"  Embedding distance == 0 for all pairs: {(emb_dists < 1e-6).all()}")

    if (emb_dists < 1e-6).all():
        print("  CONFIRMED: CheMeleon gives identical embeddings for all conformers")
        print(
            "  IMPLICATION: REPA forces the flow model to produce conformation-invariant"
        )
        print("    hidden states. This means REPA can help with atom identity/topology")
        print("    but CANNOT provide 3D geometric guidance.")
    else:
        print("  WARNING: CheMeleon embeddings differ across conformers!")
        print("  This is unexpected for a 2D encoder - investigate further.")

    # ---- Figure: Conformer analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(rmsds, bins=40, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("3D RMSD (normalized coords)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Conformer Pair 3D Distances")

    axes[1].hist(emb_dists, bins=40, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("CheMeleon Embedding L2 Distance")
    axes[1].set_ylabel("Count")
    axes[1].set_title("CheMeleon Embedding Distance\n(Should be 0 for 2D encoder)")

    plt.tight_layout()
    savefig(fig, "fig_13_conformer_analysis.png")


# ===================================================================
# SECTION 7: Summary
# ===================================================================
def section_7_summary(results: dict):
    print("\n" + "=" * 70)
    print("SECTION 7: Summary")
    print("=" * 70)

    qm9 = results.get("qm9", {})
    geom = results.get("geom", {})

    def fmt(val, fmt_str=".3f"):
        if val is None:
            return "N/A"
        return f"{val:{fmt_str}}"

    rows = [
        ("Embedding dimension", "2048", "2048"),
        ("Sparsity (exact zeros)", fmt(qm9.get("sparsity")), fmt(geom.get("sparsity"))),
        (
            "Effective rank (1% SV)",
            str(qm9.get("effective_rank", "N/A")),
            str(geom.get("effective_rank", "N/A")),
        ),
        (
            "RankMe",
            fmt(qm9.get("rankme"), ".1f"),
            fmt(geom.get("rankme"), ".1f"),
        ),
        (
            "Participation ratio",
            fmt(qm9.get("participation_ratio"), ".1f"),
            fmt(geom.get("participation_ratio"), ".1f"),
        ),
        (
            "Mol-level cos sim (mean)",
            fmt(
                qm9.get("mol_sims", np.array([None])).mean()
                if "mol_sims" in qm9
                else None
            ),
            fmt(
                geom.get("mol_sims", np.array([None])).mean()
                if "mol_sims" in geom
                else None
            ),
        ),
        (
            "Within-mol atom sim (mean)",
            fmt(
                np.mean(qm9.get("within_sims", [None]))
                if qm9.get("within_sims")
                else None
            ),
            fmt(
                np.mean(geom.get("within_sims", [None]))
                if geom.get("within_sims")
                else None
            ),
        ),
        (
            "Between-mol atom sim (mean)",
            fmt(
                np.mean(qm9.get("between_sims", [None]))
                if qm9.get("between_sims")
                else None
            ),
            fmt(
                np.mean(geom.get("between_sims", [None]))
                if geom.get("between_sims")
                else None
            ),
        ),
        (
            "Linear probe accuracy",
            fmt(qm9.get("probe_accuracy")),
            fmt(geom.get("probe_accuracy")),
        ),
    ]

    print(f"\n  {'Metric':<35s} {'QM9':>12s} {'GEOM':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    for metric, v1, v2 in rows:
        print(f"  {metric:<35s} {v1:>12s} {v2:>12s}")

    print("\n  --- Assessment ---")
    print()

    sparsity_qm9 = qm9.get("sparsity", 0)
    eff_rank = qm9.get("effective_rank", 0)
    probe_acc = qm9.get("probe_accuracy", 0)

    # Sparsity
    if sparsity_qm9 > 0.9:
        print(
            "  [CONCERN] High sparsity (>90%): cosine similarity with 93% zeros produces"
        )
        print(
            "    noisy gradients. Consider aligning in PCA-reduced space or using pre-ReLU outputs."
        )
    elif sparsity_qm9 > 0.5:
        print("  [OK] Moderate sparsity: standard cosine similarity should work.")
    else:
        print("  [GOOD] Low sparsity: dense embeddings, good gradient signal.")

    # Dimensionality
    if eff_rank < 50:
        print(
            "  [CONCERN] Very low effective rank: embeddings may lack expressiveness."
        )
    elif eff_rank < 200:
        print(
            "  [OK] Moderate effective rank: projector bottleneck (128/256 -> ~{eff_rank}) is feasible."
        )
    else:
        print("  [CONCERN] High effective rank: 128-dim projector may be a bottleneck.")

    # Discriminativeness
    if probe_acc and probe_acc > 0.9:
        print(
            "  [GOOD] High linear probe accuracy: embeddings encode atom identity well."
        )
    elif probe_acc and probe_acc > 0.5:
        print("  [OK] Moderate probe accuracy: embeddings carry useful information.")
    else:
        print(
            "  [CONCERN] Low probe accuracy: embeddings may not encode useful features."
        )

    # 2D vs 3D
    print(
        "\n  [FUNDAMENTAL] CheMeleon is 2D-only: identical embeddings for all conformers."
    )
    print("    -> REPA can guide atom identity and topology learning")
    print("    -> REPA CANNOT guide 3D geometry (bond lengths, angles, torsions)")
    print("    -> For QM9 (max 9 atoms), topology signal may be redundant")
    print("    -> For GEOM (up to 71 atoms), topology guidance could be more valuable")

    print("\n" + "=" * 70)
    print("Investigation complete. Check figures in:")
    print(f"  {FIG_DIR}")
    print("=" * 70)


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 70)
    print("CheMeleon Embedding Investigation for REPA Alignment")
    print("=" * 70)

    # Section 1: Known molecules (standalone)
    section_1_known_molecules()

    # Section 2: Sparsity & dimensionality (loads datasets)
    results = section_2_sparsity_dimensionality()

    # Section 3: Discriminativeness (reuses Section 2 data)
    section_3_discriminativeness(results)

    # Section 4: Pipeline integrity (standalone)
    section_4_pipeline_integrity()

    # Section 5: Projection feasibility (uses Section 2 PCA)
    section_5_projection_feasibility(results)

    # Section 6: 2D vs 3D mismatch (loads GEOM)
    section_6_2d_3d_mismatch()

    # Section 7: Summary
    section_7_summary(results)


if __name__ == "__main__":
    main()
