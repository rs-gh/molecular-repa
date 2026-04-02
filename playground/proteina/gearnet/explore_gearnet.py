"""
Explore GearNet CA-only encoder embeddings on real PDB proteins.

Key questions:
1. What does the representation look like? (sparsity, effective rank, distribution)
2. How sensitive are embeddings to 3D coordinate changes?
3. Can we discriminate residue types from embeddings?
4. Is the projector saturating? (can random inputs reach 0.75 cosine sim?)
5. Do embeddings capture protein-specific context beyond amino acid identity?

Run:
  source .venv/bin/activate
  python playground/proteina/gearnet/explore_gearnet.py
"""

import os
import sys
import pickle
import random
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Must import pyg_compat FIRST to patch torch_scatter/torch_cluster
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))
import proteinfoundation.repa.pyg_compat  # noqa: F401, E402

from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────

DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_PATH = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")
GEARNET_CKPT = os.path.join(DATA_PATH, "metric_factory/model_weights/gearnet_ca.pth")
N_PROTEINS = 200

# OpenFold residue type names (from openfold.np.residue_constants)
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


# ── Data Loading ────────────────────────────────────────────────────────────


def load_proteins(n=N_PROTEINS):
    """Load n proteins from the PDB LMDB."""
    print(f"Loading proteins from {LMDB_PATH}...")
    db = lmdb.open(
        LMDB_PATH,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )
    proteins = []
    with db.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if key == b"__ids__":
                continue
            if len(proteins) >= n:
                break
            try:
                graph = pickle.loads(value)
                # Filter: need coords and reasonable length
                if hasattr(graph, "coords") and graph.coords.shape[0] >= 10:
                    proteins.append(graph)
            except Exception:
                continue
    db.close()
    print(f"Loaded {len(proteins)} proteins")
    return proteins


def get_ca_coords_and_mask(graph):
    """Extract CA coordinates (Angstroms) and mask from a PyG graph.

    Returns:
        ca_coords: [n_residues, 3] in Angstroms
        mask: [n_residues] boolean
    """
    # CA is at index 1 in OpenFold atom ordering
    ca_coords = graph.coords[:, 1, :]  # [n, 3]
    ca_mask = graph.coord_mask[:, 1]  # [n]
    return ca_coords, ca_mask.bool()


def setup_encoder():
    """Load frozen GearNet encoder."""
    print(f"Loading GearNet from {GEARNET_CKPT}...")
    encoder = GearNetPerResidueEncoder(ckpt_path=GEARNET_CKPT)
    encoder.eval()
    print(f"Encoder dim: {encoder.encoder_dim}")
    return encoder


@torch.no_grad()
def get_embeddings(encoder, graph):
    """Get per-residue GearNet embeddings for a single protein.

    Returns:
        embeddings: [n_valid, 512] float32
        residue_types: [n_valid] long
    """
    ca_coords, mask = get_ca_coords_and_mask(graph)
    # Convert Angstroms to nm (encoder multiplies by 10 internally)
    ca_nm = ca_coords.float() / 10.0
    # Reshape to [1, n, 3] for encoder
    emb = encoder(ca_nm.unsqueeze(0), mask.unsqueeze(0).float())  # [1, n, 512]
    emb = emb.squeeze(0)  # [n, 512]
    # Return only valid residues
    valid_emb = emb[mask]
    valid_types = graph.residue_type[mask] if hasattr(graph, "residue_type") else None
    return valid_emb, valid_types


def collect_all_embeddings(encoder, proteins):
    """Collect embeddings from all proteins. Returns (all_emb, all_types, per_protein_embs)."""
    all_emb = []
    all_types = []
    per_protein = []
    t0 = time.time()
    for i, graph in enumerate(proteins):
        emb, types = get_embeddings(encoder, graph)
        all_emb.append(emb)
        if types is not None:
            all_types.append(types)
        per_protein.append(emb)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(proteins)} proteins ({elapsed:.1f}s)")
    all_emb = torch.cat(all_emb, dim=0)
    all_types = torch.cat(all_types, dim=0) if all_types else None
    print(
        f"Total: {all_emb.shape[0]} residues from {len(proteins)} proteins ({time.time()-t0:.1f}s)"
    )
    return all_emb, all_types, per_protein


# ── Analysis 1: Value Distribution & Sparsity ──────────────────────────────


def analyze_distribution(all_emb):
    """Analyze embedding value distribution and sparsity."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Value Distribution & Sparsity")
    print("=" * 70)

    vals = all_emb.numpy().flatten()
    n_total = vals.size

    exact_zero = np.sum(vals == 0.0)
    near_zero = np.sum(np.abs(vals) < 1e-6)
    negative = np.sum(vals < 0)

    print(f"Shape: {all_emb.shape}")
    print(f"Mean: {vals.mean():.6f}, Std: {vals.std():.6f}")
    print(f"Min: {vals.min():.6f}, Max: {vals.max():.6f}")
    print(f"Exact zeros: {exact_zero}/{n_total} ({100*exact_zero/n_total:.2f}%)")
    print(f"Near-zero (<1e-6): {near_zero}/{n_total} ({100*near_zero/n_total:.2f}%)")
    print(f"Negative values: {negative}/{n_total} ({100*negative/n_total:.2f}%)")
    print("  (CheMeleon: 93.8% zeros, MACE: 0.0% zeros)")


# ── Analysis 2: Dimensionality & Singular Values ───────────────────────────


def analyze_dimensionality(all_emb):
    """SVD analysis of embedding matrix."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Dimensionality & Singular Values")
    print("=" * 70)

    # Center the data
    emb_centered = all_emb - all_emb.mean(dim=0)
    U, S, V = torch.linalg.svd(emb_centered, full_matrices=False)
    S = S.numpy()

    # Cumulative variance
    var = S**2
    cum_var = np.cumsum(var) / np.sum(var)

    # Effective rank (entropy-based)
    p = var / var.sum()
    p = p[p > 0]
    eff_rank = np.exp(-np.sum(p * np.log(p)))

    # Participation ratio
    part_ratio = (np.sum(var) ** 2) / np.sum(var**2)

    # Dims for thresholds
    dims_90 = np.searchsorted(cum_var, 0.90) + 1
    dims_95 = np.searchsorted(cum_var, 0.95) + 1
    dims_99 = np.searchsorted(cum_var, 0.99) + 1

    print(f"Effective rank: {eff_rank:.1f} / {all_emb.shape[1]}")
    print(f"Participation ratio: {part_ratio:.1f}")
    print(f"Dims for 90% variance: {dims_90}")
    print(f"Dims for 95% variance: {dims_95}")
    print(f"Dims for 99% variance: {dims_99}")
    print(f"Top singular value: {S[0]:.2f}")
    print(f"S[0]/S[-1] ratio: {S[0]/S[-1]:.1f}")

    return S, cum_var


# ── Analysis 3: 3D Sensitivity ─────────────────────────────────────────────


def analyze_3d_sensitivity(encoder, proteins):
    """Test how embeddings change with coordinate perturbations."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: 3D Sensitivity via Perturbation")
    print("=" * 70)

    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]  # Angstroms
    n_test = min(50, len(proteins))

    # Perturbation test
    print("\nGaussian noise perturbation (cosine sim between original and perturbed):")
    for sigma in sigmas:
        sims = []
        for graph in proteins[:n_test]:
            ca_coords, mask = get_ca_coords_and_mask(graph)
            ca_nm = ca_coords.float() / 10.0
            mask_f = mask.float()

            # Original
            with torch.no_grad():
                orig = encoder(ca_nm.unsqueeze(0), mask_f.unsqueeze(0)).squeeze(0)[mask]

            # Perturbed (sigma in Angstroms, convert to nm)
            noise = torch.randn_like(ca_nm) * (sigma / 10.0)
            ca_perturbed = ca_nm + noise
            with torch.no_grad():
                pert = encoder(ca_perturbed.unsqueeze(0), mask_f.unsqueeze(0)).squeeze(
                    0
                )[mask]

            cos = F.cosine_similarity(orig, pert, dim=-1).mean().item()
            sims.append(cos)
        print(
            f"  sigma={sigma:.1f}A: cos_sim={np.mean(sims):.4f} +/- {np.std(sims):.4f}"
        )

    # Rotation invariance sanity check
    print("\nRotation invariance check:")
    sims_rot = []
    for graph in proteins[:n_test]:
        ca_coords, mask = get_ca_coords_and_mask(graph)
        ca_nm = ca_coords.float() / 10.0
        mask_f = mask.float()

        with torch.no_grad():
            orig = encoder(ca_nm.unsqueeze(0), mask_f.unsqueeze(0)).squeeze(0)[mask]

        # Random rotation
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        if torch.det(R) < 0:
            R[:, 0] *= -1
        ca_rot = ca_nm @ R.T
        with torch.no_grad():
            rot_emb = encoder(ca_rot.unsqueeze(0), mask_f.unsqueeze(0)).squeeze(0)[mask]

        cos = F.cosine_similarity(orig, rot_emb, dim=-1).mean().item()
        sims_rot.append(cos)
    print(
        f"  Random rotation: cos_sim={np.mean(sims_rot):.6f} +/- {np.std(sims_rot):.6f}"
    )
    print("  (1.0 = perfectly invariant)")


# ── Analysis 4: Residue-Type Discrimination ─────────────────────────────────


def analyze_residue_discrimination(all_emb, all_types):
    """Test if embeddings discriminate amino acid types."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Residue-Type Discrimination")
    print("=" * 70)

    if all_types is None:
        print("  No residue type info available, skipping.")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = all_emb.numpy()
    y = all_types.numpy()

    # Filter out UNK (type 20) if present
    valid = y < 20
    X, y = X[valid], y[valid]

    # Linear probe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Linear probe accuracy: {acc:.4f}")

    # Per-type mean embeddings and similarity
    unique_types = np.unique(y)
    n_types = len(unique_types)
    mean_embs = []
    for t in unique_types:
        mean_embs.append(X[y == t].mean(axis=0))
    mean_embs = np.stack(mean_embs)
    mean_embs_t = torch.from_numpy(mean_embs).float()

    # Cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        mean_embs_t.unsqueeze(0), mean_embs_t.unsqueeze(1), dim=-1
    ).numpy()
    print(
        f"\nMean cosine sim between AA types: {sim_matrix[np.triu_indices(n_types, k=1)].mean():.4f}"
    )
    print(f"Min off-diagonal: {sim_matrix[np.triu_indices(n_types, k=1)].min():.4f}")
    print(f"Max off-diagonal: {sim_matrix[np.triu_indices(n_types, k=1)].max():.4f}")

    # Within-type vs between-type
    n_sample = 5000
    within_sims = []
    between_sims = []
    for _ in range(n_sample):
        i, j = random.sample(range(len(X)), 2)
        cos = float(
            F.cosine_similarity(
                torch.tensor(X[i]).unsqueeze(0), torch.tensor(X[j]).unsqueeze(0)
            )
        )
        if y[i] == y[j]:
            within_sims.append(cos)
        else:
            between_sims.append(cos)
    print(
        f"\nWithin-type cosine sim: {np.mean(within_sims):.4f} +/- {np.std(within_sims):.4f} (n={len(within_sims)})"
    )
    print(
        f"Between-type cosine sim: {np.mean(between_sims):.4f} +/- {np.std(between_sims):.4f} (n={len(between_sims)})"
    )


# ── Analysis 5: Structural Context Sensitivity ─────────────────────────────


def analyze_structural_context(all_emb, all_types, proteins):
    """Test if same AA type has different embeddings in different structural contexts."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Structural Context Sensitivity")
    print("=" * 70)

    if all_types is None:
        print("  No residue type info available, skipping.")
        return

    # Collect (embedding, residue_type, ss_proxy) for each residue
    # SS proxy: CA-CA-CA angle for consecutive triples
    embs_by_context = {}  # {(aa_type, ss_class): [embeddings]}
    offset = 0
    for graph in proteins:
        ca_coords, mask = get_ca_coords_and_mask(graph)
        valid_idx = mask.nonzero(as_tuple=True)[0]
        n_valid = valid_idx.shape[0]

        if n_valid < 3:
            offset += n_valid
            continue

        valid_coords = ca_coords[valid_idx].float()
        valid_types = graph.residue_type[valid_idx].numpy()

        # Compute CA-CA-CA angles for consecutive residues
        for k in range(1, n_valid - 1):
            v1 = valid_coords[k - 1] - valid_coords[k]
            v2 = valid_coords[k + 1] - valid_coords[k]
            cos_angle = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            # Classify: helix ~91, sheet ~120, loop = rest
            if 80 < angle_deg < 100:
                ss = "helix"
            elif 110 < angle_deg < 135:
                ss = "sheet"
            else:
                ss = "loop"

            aa = int(valid_types[k])
            if aa >= 20:
                continue
            key = (aa, ss)
            if key not in embs_by_context:
                embs_by_context[key] = []
            embs_by_context[key].append(all_emb[offset + k])

        offset += n_valid

    # For common AAs, compare within-context vs between-context similarity
    test_aas = [0, 7, 10, 19]  # ALA, GLY, LEU, VAL
    for aa in test_aas:
        aa_name = RESIDUE_NAMES[aa]
        contexts = {}
        for ss in ["helix", "sheet", "loop"]:
            key = (aa, ss)
            if key in embs_by_context and len(embs_by_context[key]) >= 10:
                contexts[ss] = torch.stack(embs_by_context[key])

        if len(contexts) < 2:
            print(f"  {aa_name}: insufficient data for context comparison")
            continue

        # Within-context similarity
        within_sims = []
        for ss, embs in contexts.items():
            n = min(100, len(embs))
            for _ in range(200):
                i, j = random.sample(range(n), 2)
                within_sims.append(
                    F.cosine_similarity(embs[i : i + 1], embs[j : j + 1]).item()
                )

        # Between-context similarity
        between_sims = []
        ss_keys = list(contexts.keys())
        for si in range(len(ss_keys)):
            for sj in range(si + 1, len(ss_keys)):
                e1, e2 = contexts[ss_keys[si]], contexts[ss_keys[sj]]
                n1, n2 = min(100, len(e1)), min(100, len(e2))
                for _ in range(200):
                    i, j = random.randint(0, n1 - 1), random.randint(0, n2 - 1)
                    between_sims.append(
                        F.cosine_similarity(e1[i : i + 1], e2[j : j + 1]).item()
                    )

        print(
            f"  {aa_name}: within-context={np.mean(within_sims):.4f}, "
            f"between-context={np.mean(between_sims):.4f}, "
            f"delta={np.mean(within_sims)-np.mean(between_sims):.4f} "
            f"(contexts: {', '.join(f'{k}={len(v)}' for k, v in contexts.items())})"
        )


# ── Analysis 6: Embedding Norms ────────────────────────────────────────────


def analyze_norms(all_emb, all_types):
    """Analyze per-residue embedding norms."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Embedding Norms & Conditioning")
    print("=" * 70)

    norms = torch.norm(all_emb, dim=-1).numpy()
    print(
        f"L2 norm: mean={norms.mean():.4f}, std={norms.std():.4f}, "
        f"min={norms.min():.4f}, max={norms.max():.4f}"
    )

    # Per-dimension statistics
    dim_stds = all_emb.std(dim=0).numpy()
    dead_dims = np.sum(dim_stds < 1e-6)
    print(f"Dead dimensions (std < 1e-6): {dead_dims} / {all_emb.shape[1]}")
    print(f"Dimension std range: [{dim_stds.min():.6f}, {dim_stds.max():.6f}]")

    # Per AA type norms
    if all_types is not None:
        print("\nPer-AA L2 norms:")
        for aa in range(20):
            aa_mask = all_types.numpy() == aa
            if aa_mask.sum() > 0:
                aa_norms = norms[aa_mask]
                print(
                    f"  {RESIDUE_NAMES[aa]:3s}: mean={aa_norms.mean():.4f} +/- {aa_norms.std():.4f} (n={aa_mask.sum()})"
                )


# ── Analysis 7: Projector Saturation Test ──────────────────────────────────


def analyze_projector_saturation(all_emb, all_types):
    """Test if a random MLP can reach high cosine sim with GearNet targets."""
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Projector Saturation Test")
    print("=" * 70)

    target = all_emb.clone()
    n = target.shape[0]
    encoder_dim = target.shape[1]

    conditions = {}

    # Condition 1: Random input
    conditions["random"] = torch.randn(n, 128)

    # Condition 2: Residue-type one-hot
    if all_types is not None:
        onehot = F.one_hot(all_types.clamp(max=20).long(), 21).float()
        conditions["onehot"] = onehot

        # Condition 3: One-hot + positional encoding (use embedding index as proxy)
        pos = torch.arange(n).float().unsqueeze(1) / n
        conditions["onehot+pos"] = torch.cat([onehot, pos], dim=-1)

    for name, inp in conditions.items():
        input_dim = inp.shape[1]
        mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, encoder_dim),
        )
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        # Use a random subset for training
        train_n = min(10000, n)
        idx = torch.randperm(n)[:train_n]
        inp_train = inp[idx]
        target_train = target[idx]

        cos_sims = []
        for epoch in range(200):
            pred = mlp(inp_train)
            cos = F.cosine_similarity(pred, target_train, dim=-1)
            loss = -cos.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                cos_sims.append((epoch + 1, cos.mean().item()))

        final_cos = cos_sims[-1][1]
        print(
            f"  {name:15s}: final cos_sim={final_cos:.4f} "
            f"(epoch 1={cos_sims[0][1]:.4f})"
        )

    print("\n  (Training REPA reaches ~0.75-0.80 cos_sim)")
    print("  If random input reaches similar levels, projector is absorbing the signal")


# ── Analysis 8: Within vs Between Protein Similarity ───────────────────────


def analyze_protein_similarity(per_protein):
    """Compare within-protein vs between-protein residue similarities."""
    print("\n" + "=" * 70)
    print("ANALYSIS 8: Within-Protein vs Between-Protein Similarity")
    print("=" * 70)

    n_proteins = min(50, len(per_protein))

    within_sims = []
    for emb in per_protein[:n_proteins]:
        if len(emb) < 2:
            continue
        n = min(50, len(emb))
        for _ in range(100):
            i, j = random.sample(range(n), 2)
            within_sims.append(
                F.cosine_similarity(emb[i : i + 1], emb[j : j + 1]).item()
            )

    between_sims = []
    for _ in range(5000):
        p1, p2 = random.sample(range(n_proteins), 2)
        e1, e2 = per_protein[p1], per_protein[p2]
        i = random.randint(0, len(e1) - 1)
        j = random.randint(0, len(e2) - 1)
        between_sims.append(F.cosine_similarity(e1[i : i + 1], e2[j : j + 1]).item())

    print(
        f"Within-protein:  mean={np.mean(within_sims):.4f} +/- {np.std(within_sims):.4f}"
    )
    print(
        f"Between-protein: mean={np.mean(between_sims):.4f} +/- {np.std(between_sims):.4f}"
    )
    print(f"Delta: {np.mean(within_sims) - np.mean(between_sims):.4f}")


# ── Analysis 9: Layer-wise Representation ──────────────────────────────────


def analyze_layerwise(encoder, proteins):
    """Capture GearNet's internal representations at each of its 8 layers."""
    print("\n" + "=" * 70)
    print("ANALYSIS 9: GearNet Layer-wise Representation")
    print("=" * 70)

    n_test = min(30, len(proteins))
    n_layers = len(encoder.gearnet.layers)
    print(f"GearNet has {n_layers} layers")

    # Collect per-layer embeddings
    layer_stats = {
        i: {"norms": [], "sparsity": [], "eff_rank": None, "embs": []}
        for i in range(n_layers)
    }

    for graph in proteins[:n_test]:
        ca_coords, mask = get_ca_coords_and_mask(graph)
        ca_nm = ca_coords.float() / 10.0
        ca_ang = ca_nm * 10.0
        mask_f = mask.float()

        with torch.no_grad():
            # Need [1, n, 3] for _dense_to_gearnet_inputs
            coords, atom_type, atom_seq_pos, atom2batch = (
                encoder._dense_to_gearnet_inputs(
                    ca_ang.unsqueeze(0), mask_f.unsqueeze(0)
                )
            )
            coords = coords.float()

            h_v = encoder.gearnet.node_feature(atom_type, atom_seq_pos)
            edge_list = encoder.gearnet.construct_graph(
                atom_seq_pos, coords, atom2batch
            )
            h_e = encoder.gearnet.edge_feature(
                edge_list, atom_seq_pos, coords, atom2batch
            )

            for i, layer in enumerate(encoder.gearnet.layers):
                h_v = layer(h_v, edge_list, h_e)
                layer_stats[i]["norms"].append(h_v.norm(dim=-1).mean().item())
                layer_stats[i]["sparsity"].append((h_v == 0).float().mean().item())
                layer_stats[i]["embs"].append(h_v.clone())

    # Compute effective rank per layer
    for i in range(n_layers):
        all_h = torch.cat(layer_stats[i]["embs"], dim=0)
        centered = all_h - all_h.mean(dim=0)
        S = torch.linalg.svdvals(centered).numpy()
        p = (S**2) / (S**2).sum()
        p = p[p > 0]
        layer_stats[i]["eff_rank"] = np.exp(-np.sum(p * np.log(p)))

    print(f"\n{'Layer':>5} | {'Eff Rank':>9} | {'Mean Norm':>10} | {'Sparsity':>9}")
    print("-" * 50)
    for i in range(n_layers):
        print(
            f"  {i:>3d} | {layer_stats[i]['eff_rank']:>9.1f} | "
            f"{np.mean(layer_stats[i]['norms']):>10.4f} | "
            f"{np.mean(layer_stats[i]['sparsity']):>8.4f}"
        )

    # Inter-layer cosine similarity
    print("\nInter-layer cosine similarity (how much does each layer change?):")
    for graph in proteins[:5]:
        ca_coords, mask = get_ca_coords_and_mask(graph)
        ca_nm = ca_coords.float() / 10.0
        ca_ang = ca_nm * 10.0
        mask_f = mask.float()

        with torch.no_grad():
            coords, atom_type, atom_seq_pos, atom2batch = (
                encoder._dense_to_gearnet_inputs(
                    ca_ang.unsqueeze(0), mask_f.unsqueeze(0)
                )
            )
            coords = coords.float()

            h_v = encoder.gearnet.node_feature(atom_type, atom_seq_pos)
            edge_list = encoder.gearnet.construct_graph(
                atom_seq_pos, coords, atom2batch
            )
            h_e = encoder.gearnet.edge_feature(
                edge_list, atom_seq_pos, coords, atom2batch
            )

            prev = h_v.clone()
            sims = []
            for layer in encoder.gearnet.layers:
                h_v = layer(h_v, edge_list, h_e)
                cos = F.cosine_similarity(prev, h_v, dim=-1).mean().item()
                sims.append(cos)
                prev = h_v.clone()

        print(
            f"  Protein (n={coords.shape[0]}): " + " -> ".join(f"{s:.3f}" for s in sims)
        )


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    proteins = load_proteins(N_PROTEINS)
    encoder = setup_encoder()

    all_emb, all_types, per_protein = collect_all_embeddings(encoder, proteins)

    analyze_distribution(all_emb)
    S, cum_var = analyze_dimensionality(all_emb)
    analyze_3d_sensitivity(encoder, proteins)
    analyze_residue_discrimination(all_emb, all_types)
    analyze_structural_context(all_emb, all_types, proteins)
    analyze_norms(all_emb, all_types)
    analyze_projector_saturation(all_emb, all_types)
    analyze_protein_similarity(per_protein)
    analyze_layerwise(encoder, proteins)

    print("\n" + "=" * 70)
    print("DONE — all analyses complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
