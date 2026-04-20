"""Shared utilities for tabasco representation-quality probes.

Provides:
  - `load_molecules(n)`             : read RDKit mols + SMILES from LMDB
  - `build_batch(mols, smiles)`     : pack into (coords, atomics, padding_mask, smiles)
  - `extract_encoder_embeddings`    : [B, N, D] per-atom reps from a frozen encoder
  - `extract_hidden_states`         : [B, N, D] per-atom reps from a trained FlowMatchingModel
  - `atom_type_labels(mols)`        : per-atom ATOM_Z index (filters padding + '*')
  - `descriptor_labels(mols)`       : per-molecule RDKit descriptors (MolWt, LogP, NumRings, NumRotatableBonds)
  - `linear_classify(X, y)`         : sklearn logistic regression, returns test accuracy / F1
  - `linear_regress(X, y)`          : sklearn ridge, returns per-target Pearson r, R²

Design notes:
  - Works at T=1 (clean data endpoint) for hidden-state extraction: we set x_t = x_1, t=1.
  - Pads every molecule to the same length so we can batch across encoder/model.
  - Drops padding atoms + '*' dummies before any probing.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import lmdb
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    # evaluation/tabasco/representation/lib/utils.py → parents[4] = repo root.
    str(Path(__file__).resolve().parents[4] / "src" / "tabasco"),
)

# Matches tabasco ATOM_NAMES (encoders.py / convert.py). Index 8 is the '*' dummy used for padding.
ATOM_NAMES = ["C", "N", "O", "F", "S", "Cl", "Br", "I", "*"]
ATOM_Z = [6, 7, 8, 9, 16, 17, 35, 53, 0]
DUMMY_INDEX = 8

DESCRIPTOR_NAMES = ["MolWt", "MolLogP", "NumRings", "NumRotatableBonds"]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_molecules(
    n: int,
    dataset: str = "geom",
    split: str = "val",
    lmdb_path: str = None,
) -> List[Chem.Mol]:
    """Read `n` mols from an LMDB. Pass an explicit `lmdb_path` to bypass Lustre
    path probing entirely. Default uses the val split (smaller, less contended).
    """
    if lmdb_path is None:
        lmdb_path = str(
            Path(PROJECT_ROOT) / "data" / f"lmdb_{dataset}" / f"{split}.lmdb"
        )
    print(f"Loading {n} molecules from {lmdb_path}...")
    db = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    mols = []
    with db.begin() as txn:
        for i, (_, v) in enumerate(txn.cursor()):
            if i >= n:
                break
            mol = pickle.loads(v)["molecule"]
            mols.append(Chem.RemoveAllHs(mol))
    db.close()
    return mols


# --------------------------------------------------------------------------- #
# Batching (coords, atomics, padding_mask, smiles)
# --------------------------------------------------------------------------- #


def build_batch(
    mols: List[Chem.Mol],
    max_atoms: int,
    device: str = None,
) -> dict:
    """Pack a list of RDKit mols into the tabasco batch format.

    Returns dict with: coords [B, N, 3], atomics [B, N, 9], padding_mask [B, N],
    smiles (list[str]), atom_count (list[int]).
    """
    if device is None:
        device = _default_device()

    B = len(mols)
    N = max_atoms
    coords = torch.zeros(B, N, 3, dtype=torch.float32)
    atomics = torch.zeros(B, N, len(ATOM_NAMES), dtype=torch.float32)
    # Initialise all slots as padding (True). We flip False for real atoms.
    padding_mask = torch.ones(B, N, dtype=torch.bool)
    atom_counts = []
    smiles = []

    for i, mol in enumerate(mols):
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        n_atoms = mol.GetNumAtoms()
        assert n_atoms <= N, f"mol {i} has {n_atoms} atoms, max_atoms={N}"
        coords[i, :n_atoms] = torch.from_numpy(pos).float()
        for j, atom in enumerate(mol.GetAtoms()):
            z = atom.GetAtomicNum()
            idx = ATOM_Z.index(z) if z in ATOM_Z else DUMMY_INDEX
            atomics[i, j, idx] = 1.0
        # Pad atoms: one-hot on '*' dummy (matches MoleculeConverter._pad_to_size).
        for j in range(n_atoms, N):
            atomics[i, j, DUMMY_INDEX] = 1.0
        padding_mask[i, :n_atoms] = False
        atom_counts.append(n_atoms)
        smiles.append(Chem.MolToSmiles(mol))

    return {
        "coords": coords.to(device),
        "atomics": atomics.to(device),
        "padding_mask": padding_mask.to(device),
        "smiles": smiles,
        "atom_counts": atom_counts,
    }


# --------------------------------------------------------------------------- #
# Representation extraction
# --------------------------------------------------------------------------- #


@torch.no_grad()
def extract_encoder_embeddings(
    encoder, batch: dict, chunk_size: int = 64
) -> torch.Tensor:
    """Run a frozen `MolecularEncoder` over the batch. Returns [B, N, D] on CPU."""
    encoder.eval()
    B, N, _ = batch["coords"].shape
    outs = []
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        smi_chunk = batch["smiles"][s:e] if batch["smiles"] is not None else None
        out = encoder(
            batch["coords"][s:e],
            batch["atomics"][s:e],
            batch["padding_mask"][s:e],
            smiles=smi_chunk,
        )
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)  # [B, N, D]


@torch.no_grad()
def extract_hidden_states(
    model,
    batch: dict,
    key: str = "hidden_states_coord",
    chunk_size: int = 64,
    t_value: float = 1.0,
) -> torch.Tensor:
    """Extract [B, N, D] hidden states from a trained `FlowMatchingModel`.

    Runs `_call_net` with x_t = x_1 (clean data) and t = t_value (default 1.0,
    the data endpoint). Returns the tensor at `key` on CPU.
    """
    from tensordict import TensorDict

    model.eval()
    device = next(model.parameters()).device
    B, N, _ = batch["coords"].shape
    outs = []
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        bs = e - s
        x_t = TensorDict(
            {
                "coords": batch["coords"][s:e].to(device),
                "atomics": batch["atomics"][s:e].to(device),
                "padding_mask": batch["padding_mask"][s:e].to(device),
            },
            batch_size=bs,
        )
        t = torch.full((bs,), float(t_value), device=device)
        pred = model._call_net(x_t, t, return_hidden_states=True)
        if key not in pred.keys():
            # Fall back to first hidden_states_* key
            hskeys = [k for k in pred.keys() if str(k).startswith("hidden_states_")]
            if not hskeys:
                raise KeyError(
                    f"No hidden_states_* keys found in pred: {list(pred.keys())}"
                )
            key = hskeys[0]
        outs.append(pred[key].detach().cpu())
    return torch.cat(outs, dim=0)  # [B, N, D]


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def atom_type_labels(mols: List[Chem.Mol], max_atoms: int) -> torch.Tensor:
    """Per-atom labels, shape [B, N]. Padding atoms get -1."""
    B = len(mols)
    y = -torch.ones(B, max_atoms, dtype=torch.long)
    for i, mol in enumerate(mols):
        for j, atom in enumerate(mol.GetAtoms()):
            z = atom.GetAtomicNum()
            if z in ATOM_Z[:DUMMY_INDEX]:  # exclude '*'
                y[i, j] = ATOM_Z.index(z)
    return y


def descriptor_labels(mols: List[Chem.Mol]) -> Tuple[np.ndarray, List[str]]:
    """Per-molecule descriptors, shape [B, len(DESCRIPTOR_NAMES)]."""
    ys = []
    for mol in mols:
        ys.append(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                rdMolDescriptors.CalcNumRings(mol),
                Lipinski.NumRotatableBonds(mol),
            ]
        )
    return np.asarray(ys, dtype=np.float32), DESCRIPTOR_NAMES


# --------------------------------------------------------------------------- #
# Flattening / masking
# --------------------------------------------------------------------------- #


def flatten_unmasked(
    reps: torch.Tensor, labels: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Given [B, N, D] reps and [B, N] labels (labels == -1 marks padding / exclude),
    return flat (X, y) numpy arrays over kept positions only."""
    keep = labels != -1
    X = reps[keep].float().numpy()
    y = labels[keep].long().numpy()
    return X, y


def mean_pool(reps: torch.Tensor, padding_mask: torch.Tensor) -> np.ndarray:
    """Mean-pool [B, N, D] reps over unmasked atoms. padding_mask: True = pad."""
    real = (~padding_mask).float().unsqueeze(-1)  # [B, N, 1]
    summed = (reps * real).sum(dim=1)  # [B, D]
    n = real.sum(dim=1).clamp(min=1.0)  # [B, 1]
    return (summed / n).float().numpy()


# --------------------------------------------------------------------------- #
# Probe trainers
# --------------------------------------------------------------------------- #


@dataclass
class ClassResult:
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    n_classes: int


def linear_classify(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> ClassResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    # Drop classes with < 5 samples (stratify will fail).
    uniq, cnts = np.unique(y, return_counts=True)
    keep_classes = uniq[cnts >= 5]
    mask = np.isin(y, keep_classes)
    X, y = X[mask], y[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return ClassResult(
        accuracy=float(accuracy_score(y_te, pred)),
        macro_f1=float(f1_score(y_te, pred, average="macro")),
        n_train=len(X_tr),
        n_test=len(X_te),
        n_classes=len(keep_classes),
    )


@dataclass
class RegResult:
    target: str
    pearson: float
    r2: float
    n_train: int
    n_test: int


def linear_regress(
    X: np.ndarray,
    Y: np.ndarray,
    target_names: List[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> List[RegResult]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )
    results = []
    for k, name in enumerate(target_names):
        reg = Ridge(alpha=1.0, random_state=seed)
        reg.fit(X_tr, Y_tr[:, k])
        pred = reg.predict(X_te)
        r2 = float(r2_score(Y_te[:, k], pred))
        r = float(pearsonr(Y_te[:, k], pred)[0])
        results.append(
            RegResult(
                target=name,
                pearson=r,
                r2=r2,
                n_train=len(X_tr),
                n_test=len(X_te),
            )
        )
    return results
