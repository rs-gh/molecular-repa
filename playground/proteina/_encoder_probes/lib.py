"""Shared encoder characterization pipeline.

Public surface:
    - load_proteins(lmdb_path, n, seed)        — LMDB loader with train_keys.pkl fast path
    - graph_to_inputs(graph)                   — graph -> (ca_nm, mask, residue_type) on CPU
    - EncoderProbe                             — declares an encoder + its capabilities
    - run_pipeline(probe, proteins, device)    — runs the standard battery
    - analyze_*                                — individual analyses (callable directly)

Each encoder driver supplies an `embed_fn(ca_nm, mask, residue_type=None) -> [B,n,D]`
closure (typically `torch.no_grad()` + `.to(device)` + `.cpu()` wrapper around its
encoder's forward) and declares whether the encoder is 3D-aware / takes residue
type. The lib runs only the analyses that make sense for those capabilities.

Convention: `embed_fn` receives CPU tensors and returns CPU tensors. Internal
device transfers are the driver's responsibility.
"""

from __future__ import annotations

import os
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    "UNK",
]


# ── EncoderProbe ────────────────────────────────────────────────────────────


EmbedFn = Callable[..., torch.Tensor]
LayerwiseFn = Callable[[nn.Module, list, torch.device], None]


def make_embed_fn(encoder: nn.Module, device) -> EmbedFn:
    """Standard embed_fn closure: CPU in, CPU out, no_grad, device-managed.

    Works for any encoder with the signature
        encoder(ca_nm[B,n,3], mask[B,n], residue_type=[B,n] or None) -> [B,n,D]
    which is the contract every encoder in src/proteina/proteinfoundation/repa/
    follows.
    """

    @torch.no_grad()
    def embed_fn(ca_nm, mask, residue_type=None):
        ca_nm_d = ca_nm.to(device)
        mask_d = mask.to(device)
        rt_d = residue_type.to(device) if residue_type is not None else None
        out = encoder(ca_nm_d, mask_d, residue_type=rt_d)
        return out.cpu()

    return embed_fn


@dataclass
class EncoderProbe:
    """Declares an encoder + which analyses make sense for it.

    embed_fn signature:
        embed_fn(ca_nm[B,n,3], mask[B,n], residue_type=[B,n] or None) -> [B,n,D]
    All inputs/outputs on CPU; the closure handles device transfers internally.

    Capability flags drive analysis selection:
        is_3d_aware           — runs perturbation + rotation tests when True
        accepts_residue_type  — runs residue-type-shuffle test when True
        context_mode          — "structural" (uses CA-CA-CA angles to bin SS),
                                "sequence" (perturb flanks, keep center fixed),
                                or None (skip context test)
    """

    name: str
    encoder: nn.Module
    embed_fn: EmbedFn
    is_3d_aware: bool = True
    accepts_residue_type: bool = True
    context_mode: Optional[str] = "structural"  # "structural" | "sequence" | None
    layerwise_fn: Optional[LayerwiseFn] = None
    notes: list = field(default_factory=list)


# ── Data loading ────────────────────────────────────────────────────────────


def load_proteins(lmdb_path: str, n: int, seed: Optional[int] = None) -> list:
    """Load `n` proteins from a PDB LMDB.

    Uses the pre-enumerated `train_keys.pkl` next to the LMDB when present —
    cursor walk over Lustre is pathologically slow.
    """
    keys_path = os.path.join(os.path.dirname(lmdb_path), "train_keys.pkl")
    print(
        f"Loading proteins from {lmdb_path} "
        f"({'random seed=' + str(seed) if seed is not None else 'first-in-order'})..."
    )
    db = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )

    def _accept(graph):
        return (
            hasattr(graph, "coords")
            and hasattr(graph, "residue_type")
            and graph.coords.shape[0] >= 10
        )

    keys = None
    if os.path.exists(keys_path):
        with open(keys_path, "rb") as f:
            keys = pickle.load(f)
        print(f"  loaded {len(keys)} pre-enumerated keys from train_keys.pkl")

    proteins = []
    with db.begin() as txn:
        if keys is not None:
            keys = list(keys)
            if seed is not None:
                random.Random(seed).shuffle(keys)
            for key in keys:
                if len(proteins) >= n:
                    break
                try:
                    graph = pickle.loads(txn.get(key))
                    if _accept(graph):
                        proteins.append(graph)
                except Exception:
                    continue
        else:
            cursor = txn.cursor()
            enum = []
            if seed is not None:
                for k, _ in cursor:
                    if k != b"__ids__":
                        enum.append(k)
                random.Random(seed).shuffle(enum)
                for k in enum:
                    if len(proteins) >= n:
                        break
                    try:
                        graph = pickle.loads(txn.get(k))
                        if _accept(graph):
                            proteins.append(graph)
                    except Exception:
                        continue
            else:
                for k, v in cursor:
                    if k == b"__ids__":
                        continue
                    if len(proteins) >= n:
                        break
                    try:
                        graph = pickle.loads(v)
                        if _accept(graph):
                            proteins.append(graph)
                    except Exception:
                        continue
    db.close()
    print(f"Loaded {len(proteins)} proteins")
    return proteins


def graph_to_inputs(graph):
    """Return (ca_nm[n,3], mask[n], residue_type[n]) — all CPU tensors.

    CA is at atom index 1 in OpenFold ordering. Coords stored in Å; converted
    to nm here since all encoder forward passes expect nm.
    """
    ca_coords = graph.coords[:, 1, :].float()
    ca_nm = ca_coords / 10.0
    mask = graph.coord_mask[:, 1].bool()
    residue_type = graph.residue_type.long()
    return ca_nm, mask, residue_type


def collect_all_embeddings(embed_fn: EmbedFn, proteins: list):
    """Drive `embed_fn` over each protein. Returns (all_emb, all_types, per_protein).

    Filters to valid (CA-mask True) residues; everything stays on CPU.
    """
    all_emb, all_types, per_protein = [], [], []
    t0 = time.time()
    for i, graph in enumerate(proteins):
        ca_nm, mask, rtype = graph_to_inputs(graph)
        emb = embed_fn(
            ca_nm.unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0)
        ).squeeze(0)  # [n, D]
        valid_emb = emb[mask].float()
        valid_types = rtype[mask]
        all_emb.append(valid_emb)
        all_types.append(valid_types)
        per_protein.append(valid_emb)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(proteins)} proteins ({time.time()-t0:.1f}s)")
    all_emb_cat = torch.cat(all_emb, dim=0)
    all_types_cat = torch.cat(all_types, dim=0)
    print(
        f"Total: {all_emb_cat.shape[0]} residues from {len(proteins)} proteins "
        f"({time.time()-t0:.1f}s)"
    )
    return all_emb_cat, all_types_cat, per_protein


# ── Analyses ───────────────────────────────────────────────────────────────


def _header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def analyze_distribution(all_emb: torch.Tensor):
    _header("Value Distribution & Sparsity")
    vals = all_emb.numpy().flatten()
    n = vals.size
    exact_zero = int((vals == 0.0).sum())
    near_zero = int((np.abs(vals) < 1e-6).sum())
    negative = int((vals < 0).sum())
    print(f"Shape: {tuple(all_emb.shape)}")
    print(f"Mean: {vals.mean():.6f}, Std: {vals.std():.6f}")
    print(f"Min: {vals.min():.6f}, Max: {vals.max():.6f}")
    print(f"Exact zeros: {exact_zero}/{n} ({100*exact_zero/n:.4f}%)")
    print(f"Near-zero (<1e-6): {near_zero}/{n} ({100*near_zero/n:.4f}%)")
    print(f"Negative values: {negative}/{n} ({100*negative/n:.2f}%)")


def analyze_dimensionality(all_emb: torch.Tensor, max_residues: int = 30000):
    _header("Dimensionality & Singular Values")
    emb = all_emb
    if emb.shape[0] > max_residues:
        idx = torch.randperm(emb.shape[0])[:max_residues]
        emb = emb[idx]
        print(f"  (subsampled {max_residues}/{all_emb.shape[0]} residues for SVD)")
    centered = emb - emb.mean(dim=0)
    S = torch.linalg.svdvals(centered).numpy()
    var = S**2
    cum_var = np.cumsum(var) / var.sum()
    p = var / var.sum()
    p = p[p > 0]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    part_ratio = float((var.sum() ** 2) / np.sum(var**2))
    print(f"Effective rank: {eff_rank:.1f} / {all_emb.shape[1]}")
    print(f"Participation ratio: {part_ratio:.1f}")
    print(f"Dims for 90% variance: {int(np.searchsorted(cum_var, 0.90)) + 1}")
    print(f"Dims for 95% variance: {int(np.searchsorted(cum_var, 0.95)) + 1}")
    print(f"Dims for 99% variance: {int(np.searchsorted(cum_var, 0.99)) + 1}")
    print(f"Top singular value: {S[0]:.2f}")
    print(f"S[0]/S[-1] ratio: {S[0] / max(S[-1], 1e-12):.2e}")
    return S, cum_var


def analyze_perturbation(
    embed_fn: EmbedFn, proteins, n_test: int = 50, sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)
):
    _header("3D Sensitivity — Gaussian Coordinate Perturbation")
    n_test = min(n_test, len(proteins))
    print("Cosine sim between original and perturbed embeddings:")
    for sigma in sigmas:
        sims = []
        for graph in proteins[:n_test]:
            ca_nm, mask, rtype = graph_to_inputs(graph)
            orig = embed_fn(
                ca_nm.unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0)
            ).squeeze(0)[mask]
            noise = torch.randn_like(ca_nm) * (sigma / 10.0)  # Å -> nm
            pert = embed_fn(
                (ca_nm + noise).unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0)
            ).squeeze(0)[mask]
            sims.append(F.cosine_similarity(orig, pert, dim=-1).mean().item())
        print(
            f"  sigma={sigma:.1f}A: cos_sim={np.mean(sims):.4f} +/- {np.std(sims):.4f}"
        )


def analyze_rotation_invariance(embed_fn: EmbedFn, proteins, n_test: int = 50):
    _header("3D Sensitivity — Rotation Invariance")
    n_test = min(n_test, len(proteins))
    sims = []
    for graph in proteins[:n_test]:
        ca_nm, mask, rtype = graph_to_inputs(graph)
        orig = embed_fn(
            ca_nm.unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0)
        ).squeeze(0)[mask]
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        if torch.det(R) < 0:
            R[:, 0] *= -1
        ca_rot = ca_nm @ R.T
        rot = embed_fn(
            ca_rot.unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0)
        ).squeeze(0)[mask]
        sims.append(F.cosine_similarity(orig, rot, dim=-1).mean().item())
    print(f"Random rotation: cos_sim={np.mean(sims):.6f} +/- {np.std(sims):.6f}")
    print("  (1.0 = perfectly invariant)")


def analyze_residue_shuffle(embed_fn: EmbedFn, proteins, n_test: int = 50):
    _header("Residue-Type Shuffle (coords fixed, residue labels permuted)")
    n_test = min(n_test, len(proteins))
    sims = []
    for graph in proteins[:n_test]:
        ca_nm, mask, rtype = graph_to_inputs(graph)
        perm = torch.randperm(rtype.shape[0])
        rt_shuf = rtype[perm]
        orig = embed_fn(
            ca_nm.unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0)
        ).squeeze(0)[mask]
        shuf = embed_fn(
            ca_nm.unsqueeze(0), mask.unsqueeze(0), rt_shuf.unsqueeze(0)
        ).squeeze(0)[mask]
        sims.append(F.cosine_similarity(orig, shuf, dim=-1).mean().item())
    print(f"Shuffle residue types: cos_sim={np.mean(sims):.4f} +/- {np.std(sims):.4f}")
    print("  (low = identity-driven, high = geometry-driven)")


def analyze_residue_discrimination(all_emb, all_types, max_residues: int = 30000):
    _header("Residue-Type Discrimination")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X = all_emb.numpy()
    y = all_types.numpy()
    keep = (y >= 0) & (y < 20)
    X, y = X[keep], y[keep]
    if X.shape[0] > max_residues:
        idx = np.random.RandomState(0).permutation(X.shape[0])[:max_residues]
        X, y = X[idx], y[idx]
        print(f"  (subsampled {max_residues} residues for probe)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Linear probe accuracy: {acc:.4f}  (chance ~5%)")

    unique_types = np.unique(y)
    n_types = len(unique_types)
    mean_embs = np.stack([X[y == t].mean(axis=0) for t in unique_types])
    mean_embs_t = torch.from_numpy(mean_embs).float()
    sim_matrix = F.cosine_similarity(
        mean_embs_t.unsqueeze(0), mean_embs_t.unsqueeze(1), dim=-1
    ).numpy()
    upper = sim_matrix[np.triu_indices(n_types, k=1)]
    print(f"\nMean cos-sim between AA centroids: {upper.mean():.4f}")
    print(f"Min/max off-diagonal: {upper.min():.4f} / {upper.max():.4f}")

    rs = random.Random(0)
    Xt = torch.from_numpy(X).float()
    within, between = [], []
    for _ in range(5000):
        i, j = rs.sample(range(len(X)), 2)
        cos = F.cosine_similarity(Xt[i : i + 1], Xt[j : j + 1]).item()
        (within if y[i] == y[j] else between).append(cos)
    print(
        f"\nWithin-type:  {np.mean(within):.4f} +/- {np.std(within):.4f} (n={len(within)})"
    )
    print(
        f"Between-type: {np.mean(between):.4f} +/- {np.std(between):.4f} (n={len(between)})"
    )


def analyze_structural_context(all_emb, all_types, proteins, test_aas=(0, 7, 10, 19)):
    _header("Structural Context Sensitivity (helix/sheet/loop via CA-CA-CA angle)")
    # Bin (aa, ss) -> list of embeddings
    embs_by_context: dict = {}
    offset = 0
    for graph in proteins:
        ca_nm, mask, rt = graph_to_inputs(graph)
        valid_idx = mask.nonzero(as_tuple=True)[0]
        n_valid = valid_idx.shape[0]
        if n_valid < 3:
            offset += n_valid
            continue
        valid_coords = ca_nm[valid_idx] * 10.0  # back to Å for angle calc
        valid_types = rt[valid_idx].numpy()
        for k in range(1, n_valid - 1):
            v1 = valid_coords[k - 1] - valid_coords[k]
            v2 = valid_coords[k + 1] - valid_coords[k]
            cos_ang = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            angle = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            if 80 < angle < 100:
                ss = "helix"
            elif 110 < angle < 135:
                ss = "sheet"
            else:
                ss = "loop"
            aa = int(valid_types[k])
            if aa >= 20:
                continue
            embs_by_context.setdefault((aa, ss), []).append(all_emb[offset + k])
        offset += n_valid

    rs = random.Random(0)
    for aa in test_aas:
        contexts = {}
        for ss in ["helix", "sheet", "loop"]:
            key = (aa, ss)
            if key in embs_by_context and len(embs_by_context[key]) >= 10:
                contexts[ss] = torch.stack(embs_by_context[key])
        if len(contexts) < 2:
            print(f"  {RESIDUE_NAMES[aa]:3s}: insufficient data for context comparison")
            continue
        within, between = [], []
        for ss, embs in contexts.items():
            n = min(100, len(embs))
            for _ in range(200):
                i, j = rs.sample(range(n), 2)
                within.append(
                    F.cosine_similarity(embs[i : i + 1], embs[j : j + 1]).item()
                )
        ss_keys = list(contexts.keys())
        for si in range(len(ss_keys)):
            for sj in range(si + 1, len(ss_keys)):
                e1, e2 = contexts[ss_keys[si]], contexts[ss_keys[sj]]
                n1, n2 = min(100, len(e1)), min(100, len(e2))
                for _ in range(200):
                    i, j = rs.randint(0, n1 - 1), rs.randint(0, n2 - 1)
                    between.append(
                        F.cosine_similarity(e1[i : i + 1], e2[j : j + 1]).item()
                    )
        print(
            f"  {RESIDUE_NAMES[aa]:3s}: within={np.mean(within):.4f}, "
            f"between={np.mean(between):.4f}, "
            f"delta={np.mean(within)-np.mean(between):.4f} "
            f"(contexts: {', '.join(f'{k}={len(v)}' for k, v in contexts.items())})"
        )


def analyze_sequence_context(embed_fn: EmbedFn, proteins, n_test: int = 30):
    """Hold the residue at the center fixed; vary flanking residues. Used for
    sequence-only encoders (ESM) where structural context is not meaningful."""
    _header("Sequence Context Sensitivity (center fixed, flanks varied)")
    n_test = min(n_test, len(proteins))
    rs = np.random.RandomState(0)
    shuf_sims, rand_sims = [], []
    for graph in proteins[:n_test]:
        ca_nm, mask, rtype = graph_to_inputs(graph)
        valid_idx = mask.nonzero(as_tuple=True)[0].tolist()
        if len(valid_idx) < 21:
            continue
        center = valid_idx[len(valid_idx) // 2]

        emb_orig = (
            embed_fn(ca_nm.unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0))
            .squeeze(0)[center]
            .float()
        )

        # Shuffle flanking valid residues, keep center fixed
        rt_shuf = rtype.clone()
        others = torch.tensor([v for v in valid_idx if v != center], dtype=torch.long)
        perm = torch.tensor(rs.permutation(len(others)), dtype=torch.long)
        rt_shuf[others] = rtype[others[perm]]
        emb_shuf = (
            embed_fn(ca_nm.unsqueeze(0), mask.unsqueeze(0), rt_shuf.unsqueeze(0))
            .squeeze(0)[center]
            .float()
        )

        # Fully randomize flanks
        rt_rand = torch.randint(0, 20, rtype.shape)
        rt_rand[center] = rtype[center]
        emb_rand = (
            embed_fn(ca_nm.unsqueeze(0), mask.unsqueeze(0), rt_rand.unsqueeze(0))
            .squeeze(0)[center]
            .float()
        )

        shuf_sims.append(
            F.cosine_similarity(emb_orig.unsqueeze(0), emb_shuf.unsqueeze(0)).item()
        )
        rand_sims.append(
            F.cosine_similarity(emb_orig.unsqueeze(0), emb_rand.unsqueeze(0)).item()
        )

    if not shuf_sims:
        print("  Not enough valid residues for context test.")
        return
    print(
        f"Original vs shuffled-flanks: {np.mean(shuf_sims):.4f} +/- {np.std(shuf_sims):.4f}"
    )
    print(
        f"Original vs random-flanks:   {np.mean(rand_sims):.4f} +/- {np.std(rand_sims):.4f}"
    )
    print(
        "  (1.0 = context doesn't matter; lower = context strongly modulates the center)"
    )


def analyze_norms(all_emb, all_types):
    _header("Embedding Norms & Conditioning")
    norms = torch.norm(all_emb, dim=-1).numpy()
    print(
        f"L2 norm: mean={norms.mean():.4f}, std={norms.std():.4f}, "
        f"min={norms.min():.4f}, max={norms.max():.4f}"
    )
    dim_stds = all_emb.std(dim=0).numpy()
    dead_dims = int((dim_stds < 1e-6).sum())
    print(f"Dead dimensions (std < 1e-6): {dead_dims} / {all_emb.shape[1]}")
    print(f"Dimension std range: [{dim_stds.min():.6f}, {dim_stds.max():.6f}]")
    y = all_types.numpy()
    print("\nPer-AA L2 norms:")
    for aa in range(20):
        m = y == aa
        if m.sum() > 0:
            an = norms[m]
            print(
                f"  {RESIDUE_NAMES[aa]:3s}: mean={an.mean():.4f} "
                f"+/- {an.std():.4f} (n={int(m.sum())})"
            )


def analyze_projector_saturation(
    all_emb, all_types, device, epochs: int = 300, hidden: int = 512
):
    """Train a 3-layer MLP projector from {random, AA-onehot, onehot+pos} to the
    encoder's embedding. 80/20 train/test split. If random input reaches the
    cos-sim that REPA training hits, the projector is absorbing the signal."""
    _header("Projector Saturation Test")
    target = all_emb.to(device)
    n, encoder_dim = target.shape
    print(f"  target: n={n}, dim={encoder_dim}")

    mean_vec = target.mean(dim=0, keepdim=True)
    mean_cos = (
        F.cosine_similarity(mean_vec.expand_as(target), target, dim=-1).mean().item()
    )
    print(f"  mean-direction baseline (test cos): {mean_cos:.4f}")

    onehot = F.one_hot(all_types.clamp(min=0, max=20).long(), 21).float().to(device)
    pos = torch.arange(n, device=device).float().unsqueeze(1) / n
    conditions = {
        "random": torch.randn(n, 128, device=device),
        "onehot": onehot,
        "onehot+pos": torch.cat([onehot, pos], dim=-1),
    }

    perm = torch.randperm(n, device=device)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    for name, inp in conditions.items():
        mlp = nn.Sequential(
            nn.Linear(inp.shape[1], hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, encoder_dim),
        ).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        for _ in range(epochs):
            pred = mlp(inp[train_idx])
            loss = -F.cosine_similarity(pred, target[train_idx], dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            train_cos = (
                F.cosine_similarity(mlp(inp[train_idx]), target[train_idx], dim=-1)
                .mean()
                .item()
            )
            test_cos = (
                F.cosine_similarity(mlp(inp[test_idx]), target[test_idx], dim=-1)
                .mean()
                .item()
            )
        print(f"  {name:12s}: train={train_cos:.4f}  test={test_cos:.4f}")


def analyze_protein_similarity(per_protein, n_proteins: int = 50):
    _header("Within-Protein vs Between-Protein Similarity")
    n_proteins = min(n_proteins, len(per_protein))
    rs = random.Random(0)
    within = []
    for emb in per_protein[:n_proteins]:
        if len(emb) < 2:
            continue
        m = min(50, len(emb))
        for _ in range(100):
            i, j = rs.sample(range(m), 2)
            within.append(F.cosine_similarity(emb[i : i + 1], emb[j : j + 1]).item())
    between = []
    for _ in range(5000):
        p1, p2 = rs.sample(range(n_proteins), 2)
        e1, e2 = per_protein[p1], per_protein[p2]
        i = rs.randint(0, len(e1) - 1)
        j = rs.randint(0, len(e2) - 1)
        between.append(F.cosine_similarity(e1[i : i + 1], e2[j : j + 1]).item())
    print(f"Within-protein:  mean={np.mean(within):.4f} +/- {np.std(within):.4f}")
    print(f"Between-protein: mean={np.mean(between):.4f} +/- {np.std(between):.4f}")
    print(f"Delta: {np.mean(within) - np.mean(between):.4f}")


# ── Pipeline ───────────────────────────────────────────────────────────────


def run_pipeline(probe: EncoderProbe, proteins: list, device, *, skip: tuple = ()):
    """Run the standard battery of analyses for a given probe.

    `skip` is a tuple of analysis names to omit. Names:
        distribution, dimensionality, perturbation, rotation, residue_shuffle,
        residue_probe, context, norms, projector, protein_sim, layerwise.
    """
    skip_set = set(skip)
    print("\n" + "#" * 70)
    print(f"# Encoder probe: {probe.name}")
    print(
        f"#   is_3d_aware={probe.is_3d_aware}, "
        f"accepts_residue_type={probe.accepts_residue_type}, "
        f"context_mode={probe.context_mode}"
    )
    for note in probe.notes:
        print(f"#   note: {note}")
    print("#" * 70)

    all_emb, all_types, per_protein = collect_all_embeddings(probe.embed_fn, proteins)

    if "distribution" not in skip_set:
        analyze_distribution(all_emb)
    if "dimensionality" not in skip_set:
        analyze_dimensionality(all_emb)
    if probe.is_3d_aware:
        if "perturbation" not in skip_set:
            analyze_perturbation(probe.embed_fn, proteins)
        if "rotation" not in skip_set:
            analyze_rotation_invariance(probe.embed_fn, proteins)
    if probe.accepts_residue_type and "residue_shuffle" not in skip_set:
        analyze_residue_shuffle(probe.embed_fn, proteins)
    if "residue_probe" not in skip_set:
        analyze_residue_discrimination(all_emb, all_types)
    if "context" not in skip_set:
        if probe.context_mode == "structural":
            analyze_structural_context(all_emb, all_types, proteins)
        elif probe.context_mode == "sequence":
            analyze_sequence_context(probe.embed_fn, proteins)
    if "norms" not in skip_set:
        analyze_norms(all_emb, all_types)
    if "projector" not in skip_set:
        analyze_projector_saturation(all_emb, all_types, device)
    if "protein_sim" not in skip_set:
        analyze_protein_similarity(per_protein)
    if probe.layerwise_fn is not None and "layerwise" not in skip_set:
        probe.layerwise_fn(probe.encoder, proteins, device)

    print("\n" + "=" * 70)
    print(f"DONE — {probe.name} characterization complete")
    print("=" * 70)
