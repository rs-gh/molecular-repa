"""MACE Q1.1 (atom identity) and Q2 (projector saturation) probes.

Companion to ``explore_mace.py``; together they produce every
Q1/Q2/Q3 number in ``encoder_profiling/tabasco/mace/FINDINGS.md``.
This file owns the two probes that need a train/test split + MLP
training loop:

Q1.1 — ATOM IDENTITY
    Linear probe (logistic regression) from MACE embedding to atom
    type. Reports accuracy + within-type / between-type cosine
    distributions and their delta. Cross-checked against CheMeleon
    and GearNet (proteins) in the FINDINGS table.

Q2 — PROJECTOR SATURATION
    Train a small MLP from each cheap input condition to the MACE
    embedding (cosine loss). Conditions:
      - random 128-d  (mean-direction floor: what the projector
                       reaches with no input information)
      - atom-type one-hot (cheap-input ceiling)

    Saturation gap = best - random_floor. For MACE this is the
    headline finding: ~+0.005 vs CheMeleon ~+0.04 vs GearNet +0.32
    (proteins). A small gap on a high floor (~0.86) is what makes
    MACE "saturated" — it is the structural reason any MACE-REPA
    training reporting cos > 0.86 should be cross-checked against
    this floor before claiming alignment progress.

The pre-existing CheMeleon / GearNet numbers used in the comparison
tables come from each encoder's own probe scripts, not this file.

Run:
  source .venv/bin/activate
  export PROJECT_ROOT=$(pwd)/src/tabasco
  python encoder_profiling/tabasco/mace/probe_and_saturation.py
"""

import os
import sys
import pickle
import random

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ase
from e3nn import o3
from mace import data
from mace.calculators import mace_off
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric
from rdkit import Chem

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.join(os.path.dirname(__file__), "../../../src/tabasco"),
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Atom name -> symbol mapping (matches tabasco ATOM_NAMES)
ATOM_NAMES = ["C", "N", "O", "F", "S", "Cl", "Br", "I"]
ATOM_Z = [6, 7, 8, 9, 16, 17, 35, 53]


# -- Data Loading (reused from explore_mace.py) -----------------------------


def load_geom_molecules(n=200):
    lmdb_path = os.path.join(PROJECT_ROOT, "data", "lmdb_geom", "train.lmdb")
    if not os.path.exists(lmdb_path):
        lmdb_path = os.path.join(PROJECT_ROOT, "data", "lmdb_qm9", "train.lmdb")
    print(f"Loading molecules from {lmdb_path}...")
    db = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    mols = []
    with db.begin() as txn:
        for i, (k, v) in enumerate(txn.cursor()):
            if i >= n:
                break
            mols.append(pickle.loads(v)["molecule"])
    db.close()
    print(f"Loaded {len(mols)} molecules")
    return mols


def rdkit_mol_to_ase_atoms(mol, remove_h=False):
    if remove_h:
        mol = Chem.RemoveAllHs(mol)
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return ase.Atoms(numbers=atomic_numbers, positions=positions)


def setup_mace(model_name="small", device=None):
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
    print(f"  {num_interactions} layers, {total_feats} features, l_max={l_max}")
    return calc, model, config


def get_mace_embeddings(atoms_list, calc, model, config, batch_size=64):
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

    all_desc, all_batch = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.to_dict(), compute_force=False)
            desc = extract_invariant(
                output["node_feats"],
                num_layers=config["num_interactions"],
                num_features=config["num_inv_feats"],
                l_max=config["l_max"],
            )[:, : config["total_feats"]]
            all_desc.append(desc.cpu())
            all_batch.append(batch["batch"].cpu())

    all_desc = torch.cat(all_desc, dim=0)
    all_batch = torch.cat(all_batch, dim=0)

    per_mol = []
    for i in range(len(atoms_list)):
        per_mol.append(all_desc[all_batch == i])
    return per_mol


# -- Analysis 1: Linear Probe -----------------------------------------------


def test_linear_probe(mols, calc, model, config):
    print("\n" + "=" * 60)
    print("LINEAR PROBE: Atom-Type Discrimination")
    print("=" * 60)

    # Get embeddings and atom types
    all_emb, all_types = [], []
    for mol in mols:
        mol_noh = Chem.RemoveAllHs(mol)
        atoms = rdkit_mol_to_ase_atoms(mol, remove_h=True)
        emb = get_mace_embeddings([atoms], calc, model, config)[0]

        for i, atom in enumerate(mol_noh.GetAtoms()):
            z = atom.GetAtomicNum()
            if z in ATOM_Z:
                all_emb.append(emb[i])
                all_types.append(ATOM_Z.index(z))

    X = torch.stack(all_emb).numpy()
    y = np.array(all_types)

    print(f"Total atoms: {len(y)}")
    print(
        f"Type distribution: {dict(zip(ATOM_NAMES, [int((y==i).sum()) for i in range(len(ATOM_NAMES))]))}"
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Only keep types with enough samples
    type_counts = np.bincount(y, minlength=len(ATOM_NAMES))
    valid_types = type_counts >= 5
    mask = np.array([valid_types[t] for t in y])
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\nLinear probe accuracy: {acc:.4f}")
    print("  (CheMeleon: 1.000, GearNet: 0.154)")

    # Within-type vs between-type
    n_sample = min(5000, len(X) * (len(X) - 1) // 2)
    within, between = [], []
    for _ in range(n_sample):
        i, j = random.sample(range(len(X)), 2)
        cos = float(
            F.cosine_similarity(
                torch.tensor(X[i]).unsqueeze(0),
                torch.tensor(X[j]).unsqueeze(0),
            )
        )
        if y[i] == y[j]:
            within.append(cos)
        else:
            between.append(cos)
    print(
        f"\nWithin-type cosine sim:  {np.mean(within):.4f} +/- {np.std(within):.4f} (n={len(within)})"
    )
    print(
        f"Between-type cosine sim: {np.mean(between):.4f} +/- {np.std(between):.4f} (n={len(between)})"
    )
    print(f"Delta: {np.mean(within) - np.mean(between):.4f}")
    print("  (GearNet delta: 0.013)")

    return X, y


# -- Analysis 2: Projector Saturation Test ----------------------------------


def test_projector_saturation(mols, calc, model, config):
    print("\n" + "=" * 60)
    print("PROJECTOR SATURATION TEST")
    print("=" * 60)

    # Collect all embeddings
    all_emb, all_types = [], []
    for mol in mols:
        mol_noh = Chem.RemoveAllHs(mol)
        atoms = rdkit_mol_to_ase_atoms(mol, remove_h=True)
        emb = get_mace_embeddings([atoms], calc, model, config)[0]
        for i, atom in enumerate(mol_noh.GetAtoms()):
            z = atom.GetAtomicNum()
            if z in ATOM_Z:
                all_emb.append(emb[i])
                all_types.append(ATOM_Z.index(z))

    target = torch.stack(all_emb).float()  # MACE returns float64, cast to float32
    types = torch.tensor(all_types)
    n = target.shape[0]
    encoder_dim = target.shape[1]
    print(f"Target: {target.shape} (n_atoms x encoder_dim)")

    train_n = min(10000, n)
    idx = torch.randperm(n)[:train_n]
    target_train = target[idx]

    conditions = {
        "random (128-d)": torch.randn(n, 128)[idx],
        "atom-type one-hot": F.one_hot(
            types.clamp(max=len(ATOM_NAMES) - 1).long(), len(ATOM_NAMES)
        ).float()[idx],
    }

    for name, inp in conditions.items():
        input_dim = inp.shape[1]
        inp = inp.float()
        mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, encoder_dim),
        ).float()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        sims = []
        for epoch in range(200):
            pred = mlp(inp)
            cos = F.cosine_similarity(pred, target_train, dim=-1)
            loss = -cos.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                sims.append((epoch + 1, cos.mean().item()))

        print(
            f"  {name:25s}: final cos_sim={sims[-1][1]:.4f} "
            f"(epoch 1: {sims[0][1]:.4f})"
        )

    print("\n  Comparison:")
    print("    CheMeleon random: ~0.43, trained: ~0.47 (saturated)")
    print("    GearNet   random:  0.46, trained:  0.78 (not saturated)")


# -- Main --------------------------------------------------------------------


def main():
    mols = load_geom_molecules(200)
    calc, model, config = setup_mace()

    test_linear_probe(mols, calc, model, config)
    test_projector_saturation(mols, calc, model, config)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
