"""MACE-OFF small characterisation for REPA target selection.

Companion to ``encoder_profiling/tabasco/mace/probe_and_saturation.py``;
together they produce every Q1/Q2/Q3 number in
``encoder_profiling/tabasco/mace/FINDINGS.md``. Framing
(Q1 information / Q2 saturation / Q3 conditioning) mirrors
``encoder_profiling/proteina/_probes/lib.py`` and the cross-encoder
selection rubric at ``encoder_profiling/tabasco/FINDINGS.md``.

This file covers the parts of the rubric that are *encoder-structural*
(no labelled split / no MLP training):

Q1 — WHAT INFORMATION DOES THE ENCODER ENCODE?
    1.2 3D sensitivity     -> conformer L2 / cosine across GEOM pairs
                              (the metric where MACE differs sharply
                              from CheMeleon; cos 0.998 vs 1.000)
    1.4 H-atom invariance  -> all-atom vs heavy-only embeddings on
                              GEOM (heavy-only path is what tabasco
                              actually feeds; verify equivalence)

Q3 — IS THE ENCODER A TRACTABLE OPTIMISATION TARGET?
    3.1 Sparsity / value distribution
    3.2 Effective rank, dims-for-variance breakdown
    3.3 Norm distributions

Q1.1 (atom identity probe) and Q2 (projector saturation) live in the
companion ``probe_and_saturation.py`` because they require labelled
splits + an MLP training loop, which is cleaner separated from the
exploratory dumps here.

Run:
  source .venv/bin/activate
  export PROJECT_ROOT=$(pwd)/src/tabasco
  python encoder_profiling/tabasco/mace/explore_mace.py
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


# Atom name -> atomic number mapping (matches ATOM_NAMES order in constants.py)
ATOM_NAME_TO_Z = {
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "*": 0,
}


# -- Data loading -------------------------------------------------------------


def load_geom_molecules(n=100):
    """Load n molecules from the GEOM LMDB."""
    lmdb_path = os.path.join(PROJECT_ROOT, "data", "lmdb_geom", "train.lmdb")
    if not os.path.exists(lmdb_path):
        print(f"GEOM LMDB not found at {lmdb_path}, trying QM9...")
        lmdb_path = os.path.join(PROJECT_ROOT, "data", "lmdb_qm9", "train.lmdb")

    print(f"Loading molecules from {lmdb_path}...")
    db = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)

    mols = []
    with db.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= n:
                break
            data_dict = pickle.loads(value)
            mol = data_dict["molecule"]
            mols.append(mol)

    db.close()
    print(f"Loaded {len(mols)} molecules")
    return mols


def rdkit_mol_to_ase_atoms(mol, remove_h=False):
    """Convert an RDKit Mol (with 3D coords) to an ASE Atoms object."""
    if remove_h:
        mol = Chem.RemoveAllHs(mol)

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return ase.Atoms(numbers=atomic_numbers, positions=positions)


# -- MACE helpers -------------------------------------------------------------


def setup_mace(model_name="small", device=None):
    """Load MACE model and return (calc, model, config_dict)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading MACE-OFF {model_name} on {device}...")
    calc = mace_off(model_name, device=device)
    model = calc.models[0]
    model.eval()

    num_interactions = int(model.num_interactions)
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_inv = irreps_out.dim // (l_max + 1) ** 2

    per_layer = [irreps_out.dim] * num_interactions
    per_layer[-1] = num_inv
    total_feats = int(np.sum(per_layer))

    config = {
        "num_interactions": num_interactions,
        "irreps_out": irreps_out,
        "l_max": l_max,
        "num_inv_feats": num_inv,
        "total_feats": total_feats,
        "per_layer": per_layer,
    }
    print(
        f"  {num_interactions} layers, {total_feats} invariant features "
        f"({num_inv}/layer), l_max={l_max}"
    )
    return calc, model, config


def get_mace_embeddings(atoms_list, calc, model, config, batch_size=64):
    """Extract per-atom MACE embeddings. Returns list of [n_atoms, feat_dim] tensors."""
    keyspec = data.KeySpecification(
        info_keys=calc.info_keys, arrays_keys=calc.arrays_keys
    )
    configs = [
        data.config_from_atoms(x, key_specification=keyspec, head_name=calc.head)
        for x in atoms_list
    ]
    dataset = [
        data.AtomicData.from_config(
            c,
            z_table=calc.z_table,
            cutoff=calc.r_max,
            heads=calc.available_heads,
        )
        for c in configs
    ]

    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_descriptors = []
    all_batch_idx = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.to_dict(), compute_force=False)
            node_feats = output["node_feats"]

            descriptors = extract_invariant(
                node_feats,
                num_layers=config["num_interactions"],
                num_features=config["num_inv_feats"],
                l_max=config["l_max"],
            )
            descriptors = descriptors[:, : config["total_feats"]]
            all_descriptors.append(descriptors.cpu())
            all_batch_idx.append(batch["batch"].cpu())

    all_descriptors = torch.cat(all_descriptors, dim=0)
    all_batch_idx = torch.cat(all_batch_idx, dim=0)

    # Split by molecule
    per_mol = []
    for i in range(len(atoms_list)):
        mask = all_batch_idx == i
        per_mol.append(all_descriptors[mask])

    return per_mol


# -- Analysis -----------------------------------------------------------------


def analyze_representation(embeddings, name=""):
    """Analyze embedding matrix properties."""
    if isinstance(embeddings, list):
        flat = torch.cat(embeddings, dim=0).float()
    else:
        flat = embeddings.float()

    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"{'-'*60}")
    print(f"  Shape: {flat.shape}")
    print(f"  Mean: {flat.mean():.4f}, Std: {flat.std():.4f}")
    print(f"  Min: {flat.min():.4f}, Max: {flat.max():.4f}")

    sparsity = (flat == 0).float().mean().item()
    print(f"  Sparsity (exact zeros): {sparsity*100:.1f}%")

    near_zero = (flat.abs() < 1e-6).float().mean().item()
    print(f"  Near-zero (<1e-6): {near_zero*100:.1f}%")

    if flat.shape[0] > 1 and flat.shape[1] > 1:
        # RankMe: Roy-Vetterli effective rank on raw singular values
        # of the uncentered embedding matrix (Garrido et al. 2023, ICML).
        S = torch.linalg.svdvals(flat)
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        rankme = torch.exp(entropy).item()
        print(f"  RankMe: {rankme:.1f} / {min(flat.shape)}")

        cum_var = torch.cumsum(S**2, dim=0) / (S**2).sum()
        for k in [10, 50, 100]:
            if k <= len(cum_var):
                print(f"  Variance in top {k} dims: {cum_var[k-1]*100:.1f}%")

    return flat


def test_conformer_sensitivity(mols, calc, model, config, n_mols=10, n_conf=5):
    """Test whether MACE gives different embeddings for different 3D conformers."""
    print(f"\n{'='*60}")
    print("  CONFORMER SENSITIVITY TEST")
    print("  (CheMeleon gives IDENTICAL embeddings for all conformers)")
    print(f"{'='*60}")

    results = []

    for mol in mols[:n_mols]:
        smi = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
        mol_h = Chem.AddHs(Chem.MolFromSmiles(smi))

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conf, params=params)
        if len(conf_ids) < 2:
            continue
        AllChem.MMFFOptimizeMoleculeConfs(mol_h)

        atoms_list = []
        for conf_id in range(mol_h.GetNumConformers()):
            conf = mol_h.GetConformer(conf_id)
            positions = conf.GetPositions()
            numbers = [a.GetAtomicNum() for a in mol_h.GetAtoms()]
            atoms_list.append(ase.Atoms(numbers=numbers, positions=positions))

        per_mol = get_mace_embeddings(atoms_list, calc, model, config)

        # Pairwise cosine similarity between conformer embeddings
        # Average over atoms, then over pairs
        cos_sims = []
        l2_dists = []
        for i in range(len(per_mol)):
            for j in range(i + 1, len(per_mol)):
                cos = F.cosine_similarity(per_mol[i], per_mol[j], dim=-1).mean().item()
                l2 = (per_mol[i] - per_mol[j]).norm(dim=-1).mean().item()
                cos_sims.append(cos)
                l2_dists.append(l2)

        # Coordinate RMSD
        rmsds = []
        for i in range(len(atoms_list)):
            for j in range(i + 1, len(atoms_list)):
                rmsd = np.sqrt(
                    np.mean((atoms_list[i].positions - atoms_list[j].positions) ** 2)
                )
                rmsds.append(rmsd)

        mean_cos = np.mean(cos_sims)
        mean_l2 = np.mean(l2_dists)
        mean_rmsd = np.mean(rmsds)
        n_atoms = per_mol[0].shape[0]

        print(
            f"  {smi:30s} | {n_atoms:2d} atoms | {len(per_mol)} confs | "
            f"cos={mean_cos:.4f} | L2={mean_l2:.4f} | RMSD={mean_rmsd:.2f}A"
        )
        results.append((smi, mean_cos, mean_l2, mean_rmsd))

    if results:
        mean_cos_all = np.mean([r[1] for r in results])
        print(f"\n  Overall mean cosine sim between conformers: {mean_cos_all:.4f}")
        print("  (1.0 = identical like CheMeleon, <1.0 = geometry-sensitive)")

    return results


def test_heavy_atom_extraction(mols, calc, model, config, n_mols=5):
    """Show the H vs heavy atom issue for REPA alignment."""
    print(f"\n{'='*60}")
    print("  HEAVY ATOM vs ALL-ATOM ANALYSIS")
    print("  (Our model uses heavy atoms only; MACE includes H)")
    print(f"{'='*60}")

    for mol in mols[:n_mols]:
        mol_noh = Chem.RemoveAllHs(mol)
        smi = Chem.MolToSmiles(mol_noh)
        n_heavy = mol_noh.GetNumAtoms()

        # All atoms (with H)
        atoms_all = rdkit_mol_to_ase_atoms(mol, remove_h=False)
        # Heavy atoms only
        atoms_heavy = rdkit_mol_to_ase_atoms(mol, remove_h=True)

        emb_all = get_mace_embeddings([atoms_all], calc, model, config)[0]
        emb_heavy = get_mace_embeddings([atoms_heavy], calc, model, config)[0]

        print(f"\n  {smi}")
        print(
            f"    All atoms: {atoms_all.get_chemical_symbols()} -> emb shape {emb_all.shape}"
        )
        print(
            f"    Heavy only: {atoms_heavy.get_chemical_symbols()} -> emb shape {emb_heavy.shape}"
        )

        # The embeddings will be DIFFERENT because MACE uses neighbor info
        # Heavy-only MACE won't see H neighbors, changing the embedding
        # Compare the heavy atom embeddings from both
        if n_heavy == emb_heavy.shape[0]:
            # Find heavy atom indices in the all-atom molecule
            heavy_idx = [
                i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1
            ]
            emb_all_heavy = emb_all[heavy_idx]

            cos = F.cosine_similarity(emb_all_heavy, emb_heavy, dim=-1)
            print(
                f"    Cosine sim (heavy from all-atom vs heavy-only): "
                f"mean={cos.mean():.4f}, min={cos.min():.4f}, max={cos.max():.4f}"
            )
            print(
                f"    -> {'Similar' if cos.mean() > 0.9 else 'DIFFERENT'}: "
                f"H neighbors {'do' if cos.mean() < 0.9 else 'do not'} significantly affect heavy-atom embeddings"
            )


def test_atom_discrimination(mols, calc, model, config):
    """Check if MACE can discriminate different atom types and chemical environments."""
    print(f"\n{'='*60}")
    print("  ATOM-TYPE & ENVIRONMENT DISCRIMINATION")
    print(f"{'='*60}")

    # Use a molecule with diverse atom types
    # Find one in our dataset
    for mol in mols:
        mol_noh = Chem.RemoveAllHs(mol)
        symbols = set(a.GetSymbol() for a in mol_noh.GetAtoms())
        if len(symbols) >= 3 and mol_noh.GetNumAtoms() >= 5:
            break

    smi = Chem.MolToSmiles(mol_noh)
    atoms_heavy = rdkit_mol_to_ase_atoms(mol, remove_h=True)
    emb = get_mace_embeddings([atoms_heavy], calc, model, config)[0]

    atom_symbols = [a.GetSymbol() for a in mol_noh.GetAtoms()]
    print(f"\n  Molecule: {smi}")
    print(f"  Atoms: {atom_symbols}")
    print("\n  Pairwise cosine similarities:")

    n = len(atom_symbols)
    # Print as a matrix (first few atoms)
    show_n = min(n, 8)
    header = "        " + "".join(
        f"{atom_symbols[j]}({j}):".rjust(8) for j in range(show_n)
    )
    print(header)
    for i in range(show_n):
        row = f"  {atom_symbols[i]}({i}):".ljust(8)
        for j in range(show_n):
            cos = F.cosine_similarity(emb[i : i + 1], emb[j : j + 1], dim=-1).item()
            row += f"{cos:8.3f}"
        print(row)

    # Same element, different environment
    from collections import defaultdict

    by_type = defaultdict(list)
    for i, s in enumerate(atom_symbols):
        by_type[s].append(i)

    print("\n  Same-element similarities:")
    for elem, indices in by_type.items():
        if len(indices) >= 2:
            sims = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    cos = F.cosine_similarity(
                        emb[indices[i] : indices[i] + 1],
                        emb[indices[j] : indices[j] + 1],
                        dim=-1,
                    ).item()
                    sims.append(cos)
            print(
                f"    {elem} ({len(indices)} atoms): mean cos={np.mean(sims):.4f} "
                f"(range {np.min(sims):.4f}-{np.max(sims):.4f})"
            )
            print(
                f"      -> {'Distinguishes' if np.mean(sims) < 0.95 else 'Groups'} "
                f"different {elem} environments"
            )


def compare_with_chemeleon(mols, calc, model, config, n_mols=20):
    """Side-by-side comparison of representation properties."""
    print(f"\n{'='*60}")
    print("  MACE vs CheMeleon: REPRESENTATION COMPARISON")
    print(f"{'='*60}")

    # MACE sets global dtype to float64; reset for CheMeleon
    torch.set_default_dtype(torch.float32)

    try:
        from tabasco.models.components.encoders import ChemPropEncoder
        from tabasco.chem.convert import MoleculeConverter
    except ImportError as e:
        print(f"  Skipping CheMeleon comparison: {e}")
        return

    converter = MoleculeConverter()
    chem_encoder = ChemPropEncoder(pretrained="chemeleon")
    chem_encoder.eval()

    mace_embs = []
    chem_embs = []

    for mol in mols[:n_mols]:
        mol_noh = Chem.RemoveAllHs(mol)
        smi = Chem.MolToSmiles(mol_noh)
        n_heavy = mol_noh.GetNumAtoms()

        # MACE (heavy atoms only, for fair comparison)
        atoms_heavy = rdkit_mol_to_ase_atoms(mol, remove_h=True)
        m_emb = get_mace_embeddings([atoms_heavy], calc, model, config)[0]
        mace_embs.append(m_emb)

        # CheMeleon
        td = converter.to_tensor(mol_noh, pad_to_size=n_heavy)
        coords = td["coords"].unsqueeze(0)
        atomics = td["atomics"].unsqueeze(0)
        padding = td["padding_mask"].unsqueeze(0)
        with torch.no_grad():
            c_emb = chem_encoder(coords, atomics, padding, smiles=[smi])
        chem_embs.append(c_emb.squeeze(0).cpu())

    analyze_representation(mace_embs, "MACE (heavy atoms only)")
    analyze_representation(chem_embs, "CheMeleon")

    # Per-molecule comparison
    print("\n  Per-molecule embedding norms:")
    print(
        f"  {'SMILES':30s} | {'MACE norm':>10s} | {'Chem norm':>10s} | {'MACE dim':>8s} | {'Chem dim':>8s}"
    )
    for i, mol in enumerate(mols[: min(n_mols, 10)]):
        smi = Chem.MolToSmiles(Chem.RemoveAllHs(mol))[:30]
        m_norm = mace_embs[i].norm(dim=-1).mean().item()
        c_norm = chem_embs[i].norm(dim=-1).mean().item()
        print(
            f"  {smi:30s} | {m_norm:10.4f} | {c_norm:10.4f} | {mace_embs[i].shape[-1]:>8d} | {chem_embs[i].shape[-1]:>8d}"
        )


# -- Main ---------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc, model, config = setup_mace("small", device=device)

    # Load real GEOM molecules
    mols = load_geom_molecules(n=200)

    # 1. Basic stats on a batch of molecules
    print(f"\n{'='*60}")
    print("  1. BASIC MACE EMBEDDINGS ON GEOM DATA")
    print(f"{'='*60}")

    atoms_list = [rdkit_mol_to_ase_atoms(m, remove_h=True) for m in mols[:50]]
    sizes = [len(a) for a in atoms_list]
    print(
        f"  Heavy atom counts: min={min(sizes)}, max={max(sizes)}, "
        f"mean={np.mean(sizes):.1f}"
    )

    embeddings = get_mace_embeddings(atoms_list, calc, model, config)
    analyze_representation(embeddings, "MACE on 50 GEOM molecules (heavy atoms)")

    # 2. Conformer sensitivity
    test_conformer_sensitivity(mols, calc, model, config, n_mols=15, n_conf=5)

    # 3. Heavy atom vs all-atom
    test_heavy_atom_extraction(mols, calc, model, config, n_mols=5)

    # 4. Atom discrimination
    test_atom_discrimination(mols, calc, model, config)

    # 5. Compare with CheMeleon
    compare_with_chemeleon(mols, calc, model, config, n_mols=30)

    # 6. Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print("""
  Key things to look at in the output above:
  1. Conformer cosine sim: how far below 1.0? (CheMeleon = exactly 1.0)
     -> Lower = more 3D-sensitive = better for REPA geometry guidance
  2. Sparsity: CheMeleon has 93.8% zeros (ReLU). MACE?
     -> Lower sparsity = denser gradients = easier optimization
  3. Effective rank: CheMeleon has ~500 but bottlenecked to 128 by projector
     -> Higher effective rank = richer signal for alignment
  4. Heavy-atom sensitivity: does removing H change heavy-atom embeddings?
     -> If yes, need to decide: include H in MACE input or not
  5. Atom discrimination: does MACE distinguish chemical environments?
     -> Important for atom-type-aware alignment
    """)


if __name__ == "__main__":
    main()
