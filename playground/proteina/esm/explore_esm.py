"""
Explore ESM-2 (650M) per-residue encoder embeddings on real PDB proteins.

Key questions:
1. What does the representation look like? (sparsity, effective rank, distribution)
2. Which of the 33 ESM-2 layers is best for REPA? (layer-wise analysis)
3. How strongly does ESM discriminate amino-acid identity?
4. How much does sequence context matter? (mutate flanking residues)
5. Is the projector saturating? (can random inputs reach training cos sim?)
6. Do embeddings capture protein-specific context?

Companion to playground/proteina/gearnet/explore_gearnet.py — ESM is a
sequence-only language model, so 3D-sensitivity / rotation-invariance
analyses are skipped (ESM's `forward` ignores coordinates).

Run:
  source .venv/bin/activate
  python playground/proteina/esm/explore_esm.py [--quick] [--n-proteins 200] [--layer 33]
"""

import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))

from proteinfoundation.repa.esm_encoder import ESMPerResidueEncoder  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────

DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_PATH = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")
MODEL_ID = os.environ.get("ESM_MODEL_ID", "facebook/esm2_t33_650M_UR50D")

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


def load_proteins(n, seed=None):
    """Load n proteins from the PDB LMDB.

    Uses the pre-enumerated ``train_keys.pkl`` alongside the LMDB when present
    — iterating the cursor over Lustre is pathologically slow (see
    feedback_lmdb_local_nvme.md). Falls back to cursor walk otherwise.
    """
    keys_path = os.path.join(os.path.dirname(LMDB_PATH), "train_keys.pkl")
    print(
        f"Loading proteins from {LMDB_PATH} "
        f"({'random seed=' + str(seed) if seed is not None else 'first-in-order'})..."
    )

    db = lmdb.open(
        LMDB_PATH,
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

    if os.path.exists(keys_path):
        with open(keys_path, "rb") as f:
            keys = pickle.load(f)
        print(f"  loaded {len(keys)} pre-enumerated keys from train_keys.pkl")
    else:
        keys = None

    proteins = []
    with db.begin() as txn:
        if keys is not None:
            if seed is not None:
                keys = list(keys)
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
            # Fallback: cursor walk. For random seed this enumerates all keys
            # first — slow on Lustre, fine on local NVMe.
            if seed is not None:
                enum = []
                cursor = txn.cursor()
                for key, _ in cursor:
                    if key != b"__ids__":
                        enum.append(key)
                random.Random(seed).shuffle(enum)
                for key in enum:
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
                for key, value in cursor:
                    if key == b"__ids__":
                        continue
                    if len(proteins) >= n:
                        break
                    try:
                        graph = pickle.loads(value)
                        if _accept(graph):
                            proteins.append(graph)
                    except Exception:
                        continue
    db.close()
    print(f"Loaded {len(proteins)} proteins")
    return proteins


def get_ca_mask_and_types(graph):
    """Return (ca_mask, residue_type) — CA mask is at atom index 1 in OpenFold."""
    ca_mask = graph.coord_mask[:, 1].bool()
    residue_type = graph.residue_type.long()
    return ca_mask, residue_type


def setup_encoder(device, layer=None):
    """Load frozen ESM-2 encoder."""
    print(f"Loading ESM-2 from {MODEL_ID}... (layer={layer})")
    t0 = time.time()
    encoder = ESMPerResidueEncoder(model_id=MODEL_ID, layer=layer).to(device)
    encoder.eval()
    print(
        f"Encoder dim: {encoder.encoder_dim}, "
        f"num layers: {encoder.esm.config.num_hidden_layers}, "
        f"loaded in {time.time()-t0:.1f}s"
    )
    return encoder


@torch.no_grad()
def get_embeddings(encoder, graph, device):
    """Return (embeddings [n_valid, D], residue_types [n_valid])."""
    ca_mask, residue_type = get_ca_mask_and_types(graph)
    n = residue_type.shape[0]
    # Dummy coords (ESM ignores them) but must match shape + dtype expected.
    coords = torch.zeros(1, n, 3, device=device)
    mask = ca_mask.unsqueeze(0).to(device)
    rtype = residue_type.unsqueeze(0).to(device)
    emb = encoder(coords, mask, residue_type=rtype).squeeze(0)  # [n, D]
    valid_emb = emb[ca_mask.to(device)].cpu().float()
    valid_types = residue_type[ca_mask].cpu()
    return valid_emb, valid_types


def collect_all_embeddings(encoder, proteins, device):
    """Collect per-residue embeddings from all proteins."""
    all_emb, all_types, per_protein = [], [], []
    t0 = time.time()
    for i, graph in enumerate(proteins):
        emb, types = get_embeddings(encoder, graph, device)
        all_emb.append(emb)
        all_types.append(types)
        per_protein.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(proteins)} proteins ({time.time()-t0:.1f}s)")
    all_emb = torch.cat(all_emb, dim=0)
    all_types = torch.cat(all_types, dim=0)
    print(
        f"Total: {all_emb.shape[0]} residues from {len(proteins)} proteins "
        f"({time.time()-t0:.1f}s)"
    )
    return all_emb, all_types, per_protein


# ── Analysis 1: Value Distribution & Sparsity ──────────────────────────────


def analyze_distribution(all_emb):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Value Distribution & Sparsity")
    print("=" * 70)

    vals = all_emb.numpy().flatten()
    n_total = vals.size

    exact_zero = np.sum(vals == 0.0)
    near_zero = np.sum(np.abs(vals) < 1e-6)
    negative = np.sum(vals < 0)

    print(f"Shape: {tuple(all_emb.shape)}")
    print(f"Mean: {vals.mean():.6f}, Std: {vals.std():.6f}")
    print(f"Min: {vals.min():.6f}, Max: {vals.max():.6f}")
    print(f"Exact zeros: {exact_zero}/{n_total} ({100*exact_zero/n_total:.2f}%)")
    print(f"Near-zero (<1e-6): {near_zero}/{n_total} ({100*near_zero/n_total:.2f}%)")
    print(f"Negative values: {negative}/{n_total} ({100*negative/n_total:.2f}%)")
    print("  (GearNet: 0.00% zeros, MACE: 0.0% zeros, CheMeleon: 93.8% zeros)")


# ── Analysis 2: Dimensionality & Singular Values ───────────────────────────


def analyze_dimensionality(all_emb):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Dimensionality & Singular Values")
    print("=" * 70)

    # Subsample to keep SVD tractable for 1280-dim + many residues.
    n = all_emb.shape[0]
    if n > 20000:
        idx = torch.randperm(n)[:20000]
        emb = all_emb[idx]
        print(f"  (subsampled {20000}/{n} residues for SVD)")
    else:
        emb = all_emb

    emb_centered = emb - emb.mean(dim=0)
    S = torch.linalg.svdvals(emb_centered).numpy()

    var = S**2
    cum_var = np.cumsum(var) / np.sum(var)
    p = var / var.sum()
    p = p[p > 0]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    part_ratio = float((np.sum(var) ** 2) / np.sum(var**2))

    dims_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    dims_95 = int(np.searchsorted(cum_var, 0.95) + 1)
    dims_99 = int(np.searchsorted(cum_var, 0.99) + 1)

    print(f"Effective rank: {eff_rank:.1f} / {all_emb.shape[1]}")
    print(f"Participation ratio: {part_ratio:.1f}")
    print(f"Dims for 90% variance: {dims_90}")
    print(f"Dims for 95% variance: {dims_95}")
    print(f"Dims for 99% variance: {dims_99}")
    print(f"Top singular value: {S[0]:.2f}")
    print(f"S[0]/S[-1] ratio: {S[0] / max(S[-1], 1e-12):.1f}")

    return S, cum_var


# ── Analysis 3: Residue-Type Discrimination ────────────────────────────────


def analyze_residue_discrimination(all_emb, all_types):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Residue-Type Discrimination")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X = all_emb.numpy()
    y = all_types.numpy()

    # Drop UNK (20) and any <0 pad leakage.
    keep = (y >= 0) & (y < 20)
    X, y = X[keep], y[keep]

    # Cap for probe speed.
    if X.shape[0] > 30000:
        idx = np.random.RandomState(0).permutation(X.shape[0])[:30000]
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Linear probe accuracy: {acc:.4f}  (chance ~5%)")

    # Per-type centroid cosine similarity matrix
    unique_types = np.unique(y)
    mean_embs = np.stack([X[y == t].mean(axis=0) for t in unique_types])
    mean_embs_t = torch.from_numpy(mean_embs).float()
    sim_matrix = F.cosine_similarity(
        mean_embs_t.unsqueeze(0), mean_embs_t.unsqueeze(1), dim=-1
    ).numpy()
    off_diag = sim_matrix[np.triu_indices(len(unique_types), k=1)]
    print(f"\nMean cos-sim between AA-type centroids: {off_diag.mean():.4f}")
    print(f"Min off-diagonal: {off_diag.min():.4f}")
    print(f"Max off-diagonal: {off_diag.max():.4f}")

    # Within-type vs between-type sampled sims
    n_sample = 5000
    within_sims, between_sims = [], []
    rs = random.Random(0)
    for _ in range(n_sample):
        i, j = rs.sample(range(len(X)), 2)
        cos = float(
            F.cosine_similarity(
                torch.tensor(X[i]).unsqueeze(0), torch.tensor(X[j]).unsqueeze(0)
            )
        )
        (within_sims if y[i] == y[j] else between_sims).append(cos)
    print(
        f"\nWithin-type:  {np.mean(within_sims):.4f} +/- {np.std(within_sims):.4f} "
        f"(n={len(within_sims)})"
    )
    print(
        f"Between-type: {np.mean(between_sims):.4f} +/- {np.std(between_sims):.4f} "
        f"(n={len(between_sims)})"
    )


# ── Analysis 4: Sequence Context Sensitivity ───────────────────────────────


def analyze_sequence_context(encoder, proteins, device, n_test=30):
    """Measure how strongly sequence context affects the embedding of a center
    residue. For each protein, hold the residue at a chosen center position
    fixed and randomize the flanking residues; compare embeddings."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Sequence Context Sensitivity")
    print("=" * 70)

    deltas = []
    rs = np.random.RandomState(0)
    n_test = min(n_test, len(proteins))

    for graph in proteins[:n_test]:
        ca_mask, rtype = get_ca_mask_and_types(graph)
        n = rtype.shape[0]
        valid_idx = ca_mask.nonzero(as_tuple=True)[0].tolist()
        if len(valid_idx) < 21:
            continue
        center = valid_idx[len(valid_idx) // 2]

        coords = torch.zeros(1, n, 3, device=device)
        mask = ca_mask.unsqueeze(0).to(device)

        # Original
        with torch.no_grad():
            emb_orig = (
                encoder(coords, mask, residue_type=rtype.unsqueeze(0).to(device))
                .squeeze(0)[center]
                .cpu()
                .float()
            )

        # Shuffled context (keep center fixed, scramble flanking valid residues)
        rtype_shuffled = rtype.clone()
        others = torch.tensor([v for v in valid_idx if v != center], dtype=torch.long)
        perm = torch.tensor(rs.permutation(len(others)), dtype=torch.long)
        rtype_shuffled[others] = rtype[others[perm]]
        with torch.no_grad():
            emb_shuf = (
                encoder(
                    coords, mask, residue_type=rtype_shuffled.unsqueeze(0).to(device)
                )
                .squeeze(0)[center]
                .cpu()
                .float()
            )

        # Fully randomized context (uniform over 20 AAs, keep center fixed)
        rtype_random = torch.randint(0, 20, (n,))
        rtype_random[center] = rtype[center]
        with torch.no_grad():
            emb_rand = (
                encoder(coords, mask, residue_type=rtype_random.unsqueeze(0).to(device))
                .squeeze(0)[center]
                .cpu()
                .float()
            )

        deltas.append(
            (
                F.cosine_similarity(
                    emb_orig.unsqueeze(0), emb_shuf.unsqueeze(0)
                ).item(),
                F.cosine_similarity(
                    emb_orig.unsqueeze(0), emb_rand.unsqueeze(0)
                ).item(),
            )
        )

    if not deltas:
        print("  Not enough valid residues for context test.")
        return

    shuf_sims = [d[0] for d in deltas]
    rand_sims = [d[1] for d in deltas]
    print(
        f"Center-residue cos-sim (same AA at center, varied context):\n"
        f"  Original vs shuffled-flanks:  {np.mean(shuf_sims):.4f} +/- {np.std(shuf_sims):.4f}\n"
        f"  Original vs random-flanks:    {np.mean(rand_sims):.4f} +/- {np.std(rand_sims):.4f}\n"
        f"  (1.0 = context doesn't matter, lower = context strongly modulates the center)"
    )


# ── Analysis 5: Embedding Norms & Conditioning ─────────────────────────────


def analyze_norms(all_emb, all_types):
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Embedding Norms & Conditioning")
    print("=" * 70)

    norms = torch.norm(all_emb, dim=-1).numpy()
    print(
        f"L2 norm: mean={norms.mean():.4f}, std={norms.std():.4f}, "
        f"min={norms.min():.4f}, max={norms.max():.4f}"
    )

    dim_stds = all_emb.std(dim=0).numpy()
    dead_dims = int(np.sum(dim_stds < 1e-6))
    print(f"Dead dimensions (std < 1e-6): {dead_dims} / {all_emb.shape[1]}")
    print(f"Dimension std range: [{dim_stds.min():.6f}, {dim_stds.max():.6f}]")

    y = all_types.numpy()
    print("\nPer-AA L2 norms:")
    for aa in range(20):
        aa_mask = y == aa
        if aa_mask.sum() > 0:
            aa_norms = norms[aa_mask]
            print(
                f"  {RESIDUE_NAMES[aa]:3s}: mean={aa_norms.mean():.4f} "
                f"+/- {aa_norms.std():.4f} (n={int(aa_mask.sum())})"
            )


# ── Analysis 6: Projector Saturation Test ──────────────────────────────────


def analyze_projector_saturation(all_emb, all_types):
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Projector Saturation Test")
    print("=" * 70)

    target = all_emb.clone()
    n = target.shape[0]
    encoder_dim = target.shape[1]

    conditions = {
        "random": torch.randn(n, 128),
        "aa_onehot": F.one_hot(all_types.clamp(min=0, max=20).long(), 21).float(),
        "onehot+pos": torch.cat(
            [
                F.one_hot(all_types.clamp(min=0, max=20).long(), 21).float(),
                (torch.arange(n).float().unsqueeze(1) / n),
            ],
            dim=-1,
        ),
    }

    # 80/20 train/test split so we report test-set saturation (matches
    # gearnet/analysis methodology).
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    target_train, target_test = target[train_idx], target[test_idx]

    for name, inp in conditions.items():
        inp_train, inp_test = inp[train_idx], inp[test_idx]
        input_dim = inp.shape[1]
        mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, encoder_dim),
        )
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        for _ in range(300):
            pred = mlp(inp_train)
            loss = -F.cosine_similarity(pred, target_train, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_cos = (
                F.cosine_similarity(mlp(inp_train), target_train, dim=-1).mean().item()
            )
            test_cos = (
                F.cosine_similarity(mlp(inp_test), target_test, dim=-1).mean().item()
            )
        print(f"  {name:12s}: train={train_cos:.4f}  test={test_cos:.4f}")

    # Mean-direction baseline
    mean_dir = target_train.mean(dim=0, keepdim=True).expand_as(target_test)
    mean_cos = F.cosine_similarity(mean_dir, target_test, dim=-1).mean().item()
    print(f"  {'mean-dir':12s}: test={mean_cos:.4f} (projector-free lower bound)")

    print("\n  (A REPA training run reports cos-sim around X.XX — fill in from wandb.)")


# ── Analysis 7: Within vs Between Protein Similarity ───────────────────────


def analyze_protein_similarity(per_protein):
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Within-Protein vs Between-Protein Similarity")
    print("=" * 70)

    n_proteins = min(50, len(per_protein))
    rs = random.Random(0)

    within_sims = []
    for emb in per_protein[:n_proteins]:
        if len(emb) < 2:
            continue
        m = min(50, len(emb))
        for _ in range(100):
            i, j = rs.sample(range(m), 2)
            within_sims.append(
                F.cosine_similarity(emb[i : i + 1], emb[j : j + 1]).item()
            )

    between_sims = []
    for _ in range(5000):
        p1, p2 = rs.sample(range(n_proteins), 2)
        e1, e2 = per_protein[p1], per_protein[p2]
        i = rs.randint(0, len(e1) - 1)
        j = rs.randint(0, len(e2) - 1)
        between_sims.append(F.cosine_similarity(e1[i : i + 1], e2[j : j + 1]).item())

    print(
        f"Within-protein:  mean={np.mean(within_sims):.4f} "
        f"+/- {np.std(within_sims):.4f}"
    )
    print(
        f"Between-protein: mean={np.mean(between_sims):.4f} "
        f"+/- {np.std(between_sims):.4f}"
    )
    print(f"Delta: {np.mean(within_sims) - np.mean(between_sims):.4f}")


# ── Analysis 8: Layer-wise Analysis (KEY for REPA) ─────────────────────────


@torch.no_grad()
def analyze_layerwise(proteins, device, n_test=30, max_residues=10000):
    """For each of the 33 ESM layers, compute effective rank, mean norm,
    sparsity, and linear-probe accuracy for residue type.

    This is the most actionable analysis for REPA: it answers "which layer
    should `repa_layer` target?"."""
    print("\n" + "=" * 70)
    print("ANALYSIS 8: ESM Layer-wise Analysis")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # Load encoder once with output_hidden_states — no `layer` arg means
    # last_hidden_state, but we bypass that by calling the underlying ESM directly.
    print(f"Loading ESM (for layerwise pass) on {device}...")
    encoder = ESMPerResidueEncoder(model_id=MODEL_ID, layer=None).to(device)
    encoder.eval()
    n_layers = encoder.esm.config.num_hidden_layers  # 33 for 650M
    print(f"  {n_layers} layers (+1 embedding layer = {n_layers + 1} hidden states)")

    # Collect per-layer features and residue types. Cap total residues.
    per_layer_feats = [[] for _ in range(n_layers + 1)]
    per_layer_types = []
    per_layer_protein_ids = []
    total_residues = 0

    t0 = time.time()
    for pid, graph in enumerate(proteins[:n_test]):
        ca_mask, rtype = get_ca_mask_and_types(graph)

        # Build the same tokenized input the encoder would, to be able to pull
        # all hidden states in one pass.
        mask_bool = ca_mask.to(device)
        safe_idx = rtype.clamp(min=0, max=len(encoder.aa_to_esm_token) - 1)
        aa_tokens = encoder.aa_to_esm_token[safe_idx.to(device)]
        aa_tokens = torch.where(
            mask_bool, aa_tokens, torch.full_like(aa_tokens, encoder.pad_token_id)
        ).unsqueeze(0)
        cls = torch.full((1, 1), encoder.cls_token_id, dtype=torch.long, device=device)
        eos = torch.full((1, 1), encoder.eos_token_id, dtype=torch.long, device=device)
        input_ids = torch.cat([cls, aa_tokens, eos], dim=1)
        ones = torch.ones((1, 1), dtype=torch.bool, device=device)
        attn = torch.cat([ones, mask_bool.unsqueeze(0), ones], dim=1).long()

        out = encoder.esm(
            input_ids=input_ids, attention_mask=attn, output_hidden_states=True
        )
        # hidden_states is a tuple of length (n_layers + 1): embeddings + each layer.
        valid_indices = ca_mask.to(device).nonzero(as_tuple=True)[0]
        n_valid = valid_indices.shape[0]
        if n_valid == 0:
            continue

        for layer_idx, hs in enumerate(out.hidden_states):
            # Strip CLS/EOS → align with residue positions.
            feats = hs[0, 1:-1, :]  # [n, D]
            per_layer_feats[layer_idx].append(feats[ca_mask.to(device)].cpu().float())
        per_layer_types.append(rtype[ca_mask].cpu())
        per_layer_protein_ids.extend([pid] * n_valid)
        total_residues += n_valid

        if total_residues >= max_residues:
            print(f"  Stopping at {total_residues} residues (max={max_residues})")
            break
        if (pid + 1) % 10 == 0:
            print(
                f"  {pid+1}/{n_test} proteins, {total_residues} residues, "
                f"{time.time()-t0:.1f}s"
            )

    types_flat = torch.cat(per_layer_types)
    keep = (types_flat >= 0) & (types_flat < 20)
    y = types_flat.numpy()[keep.numpy()]

    # Free GPU memory before scikit-learn work.
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    print(
        f"\n{'Layer':>5} | {'Eff Rank':>9} | {'Mean Norm':>10} | "
        f"{'Sparsity':>9} | {'AA Probe':>9}"
    )
    print("-" * 60)

    layer_stats = {}
    for layer_idx in range(n_layers + 1):
        feats = torch.cat(per_layer_feats[layer_idx], dim=0)
        feats_clean = feats[keep]

        centered = feats_clean - feats_clean.mean(dim=0)
        S = torch.linalg.svdvals(centered).numpy()
        p = (S**2) / (S**2).sum()
        p = p[p > 0]
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))

        mean_norm = float(feats_clean.norm(dim=-1).mean())
        sparsity = float((feats_clean.abs() < 1e-6).float().mean())

        # Linear probe on a random subsample for speed
        X = feats_clean.numpy()
        n_probe = min(10000, X.shape[0])
        idx = np.random.RandomState(0).permutation(X.shape[0])[:n_probe]
        Xs, ys = X[idx], y[idx]
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xs, ys, test_size=0.2, random_state=42, stratify=ys
        )
        clf = LogisticRegression(max_iter=400, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        probe_acc = accuracy_score(y_te, clf.predict(X_te))

        layer_stats[layer_idx] = dict(
            eff_rank=eff_rank,
            mean_norm=mean_norm,
            sparsity=sparsity,
            probe_acc=probe_acc,
        )
        print(
            f"  {layer_idx:>3d} | {eff_rank:>9.1f} | {mean_norm:>10.4f} | "
            f"{sparsity:>8.4f} | {probe_acc:>8.4f}"
        )

    # Inter-layer cosine similarity for a handful of representative transitions.
    print("\nInter-layer cosine similarity (sample residues):")
    sample = torch.randperm(total_residues)[: min(500, total_residues)]
    transitions = sorted(set([0, 5, 10, 15, 20, 25, 30, n_layers - 1]))
    for li in transitions:
        if li >= n_layers:
            continue
        a = torch.cat(per_layer_feats[li], dim=0)[sample]
        b = torch.cat(per_layer_feats[li + 1], dim=0)[sample]
        cos = F.cosine_similarity(a, b, dim=-1).mean().item()
        print(f"  layer {li:>2d} -> {li+1:>2d}: cos_sim = {cos:.4f}")

    return layer_stats


# ── Main ────────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    print("WARNING: no CUDA device available — running ESM on CPU will be slow.")
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=200)
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Which hidden-states layer to use for the main analyses "
        "(None = last_hidden_state, i.e. final layer).",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke pass: 20 proteins, skip heavy analyses.",
    )
    ap.add_argument("--skip-layerwise", action="store_true")
    ap.add_argument("--skip-context", action="store_true")
    ap.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="If set, randomize protein selection with this seed "
        "instead of taking the first N LMDB keys in byte order.",
    )
    args = ap.parse_args()

    if args.quick:
        args.n_proteins = min(args.n_proteins, 20)

    device = get_device()
    print(f"Device: {device}")
    print(
        f"Target layer for main analyses: "
        f"{args.layer if args.layer is not None else 'last_hidden_state (33)'}"
    )

    proteins = load_proteins(args.n_proteins, seed=args.random_seed)
    encoder = setup_encoder(device, layer=args.layer)

    all_emb, all_types, per_protein = collect_all_embeddings(encoder, proteins, device)

    analyze_distribution(all_emb)
    analyze_dimensionality(all_emb)
    analyze_residue_discrimination(all_emb, all_types)
    if not args.skip_context and not args.quick:
        analyze_sequence_context(encoder, proteins, device)
    analyze_norms(all_emb, all_types)
    analyze_projector_saturation(all_emb, all_types)
    analyze_protein_similarity(per_protein)

    # Free the single-layer encoder before loading the layerwise one.
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    if not args.skip_layerwise and not args.quick:
        # Cap layerwise at 60 proteins; residue-cap internal to the analysis
        # (10k residues) still kicks in first for normal-length proteins.
        analyze_layerwise(proteins, device, n_test=min(60, len(proteins)))

    print("\n" + "=" * 70)
    print("DONE — all analyses complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
