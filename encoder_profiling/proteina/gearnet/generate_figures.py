"""
Generate figures for GearNet CA-only encoder characterization.

Run:
  source .venv/bin/activate
  python encoder_profiling/proteina/gearnet/generate_figures.py
"""

import os
import sys
import pickle
import random
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Must import pyg_compat FIRST
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))
import proteinfoundation.repa.pyg_compat  # noqa: F401, E402

from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder  # noqa: E402

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_PATH = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")
GEARNET_CKPT = os.path.join(DATA_PATH, "metric_factory/model_weights/gearnet_ca.pth")

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
N_PROTEINS = 200


# -- Helpers (same as explore_gearnet.py) ------------------------------------


def load_proteins(n=N_PROTEINS):
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
        for key, value in txn.cursor():
            if key == b"__ids__":
                continue
            if len(proteins) >= n:
                break
            try:
                graph = pickle.loads(value)
                if hasattr(graph, "coords") and graph.coords.shape[0] >= 10:
                    proteins.append(graph)
            except Exception:
                continue
    db.close()
    return proteins


def get_ca_coords_and_mask(graph):
    ca_coords = graph.coords[:, 1, :]
    ca_mask = graph.coord_mask[:, 1]
    return ca_coords, ca_mask.bool()


def setup_encoder():
    encoder = GearNetPerResidueEncoder(ckpt_path=GEARNET_CKPT)
    encoder.eval()
    return encoder


@torch.no_grad()
def get_embeddings(encoder, graph):
    ca_coords, mask = get_ca_coords_and_mask(graph)
    ca_nm = ca_coords.float() / 10.0
    emb = encoder(ca_nm.unsqueeze(0), mask.unsqueeze(0).float()).squeeze(0)
    return emb[mask], graph.residue_type[mask] if hasattr(
        graph, "residue_type"
    ) else None


def collect_all_embeddings(encoder, proteins):
    all_emb, all_types, per_protein = [], [], []
    for graph in proteins:
        emb, types = get_embeddings(encoder, graph)
        all_emb.append(emb)
        if types is not None:
            all_types.append(types)
        per_protein.append(emb)
    all_emb = torch.cat(all_emb, dim=0)
    all_types = torch.cat(all_types, dim=0) if all_types else None
    return all_emb, all_types, per_protein


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# -- Figure 1: Value Distribution & Sparsity --------------------------------


def fig_01_value_distribution(all_emb):
    print("Fig 01: Value Distribution & Sparsity")
    vals = all_emb.numpy().flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Full distribution
    ax1.hist(vals, bins=200, density=True, color="steelblue", alpha=0.8)
    ax1.axvline(0, color="red", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Embedding value")
    ax1.set_ylabel("Density")
    ax1.set_title(
        f"GearNet Embedding Values\n" f"mean={vals.mean():.3f}, std={vals.std():.3f}"
    )

    # Zoomed near-zero
    near = vals[np.abs(vals) < 1.0]
    ax2.hist(near, bins=200, density=True, color="steelblue", alpha=0.8)
    ax2.axvline(0, color="red", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Embedding value")
    ax2.set_ylabel("Density")
    exact_zero_pct = 100 * np.sum(vals == 0) / len(vals)
    ax2.set_title(
        f"Zoomed [-1, 1]\n"
        f"exact zeros: {exact_zero_pct:.2f}% "
        f"(CheMeleon: 93.8%, MACE: 0%)"
    )

    fig.tight_layout()
    savefig("fig_01_value_distribution.png")


# -- Figure 2: Singular Values & Dimensionality -----------------------------


def fig_02_singular_values(all_emb):
    print("Fig 02: Singular Values & Dimensionality")
    emb_centered = all_emb - all_emb.mean(dim=0)
    S = torch.linalg.svdvals(emb_centered).numpy()
    var = S**2
    cum_var = np.cumsum(var) / np.sum(var)

    p = var / var.sum()
    p = p[p > 0]
    eff_rank = np.exp(-np.sum(p * np.log(p)))
    dims_90 = np.searchsorted(cum_var, 0.90) + 1
    dims_95 = np.searchsorted(cum_var, 0.95) + 1
    dims_99 = np.searchsorted(cum_var, 0.99) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(S, color="steelblue")
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Singular value")
    ax1.set_title(f"Singular Value Spectrum\nEffective rank: {eff_rank:.1f} / {len(S)}")

    ax2.plot(cum_var, color="steelblue")
    ax2.axhline(
        0.90,
        color="orange",
        linestyle="--",
        linewidth=0.8,
        label=f"90% -> {dims_90} dims",
    )
    ax2.axhline(
        0.95, color="red", linestyle="--", linewidth=0.8, label=f"95% -> {dims_95} dims"
    )
    ax2.axhline(
        0.99,
        color="darkred",
        linestyle="--",
        linewidth=0.8,
        label=f"99% -> {dims_99} dims",
    )
    ax2.set_xlabel("Component index")
    ax2.set_ylabel("Cumulative variance explained")
    ax2.set_title("PCA Cumulative Variance")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    savefig("fig_02_singular_values.png")


# -- Figure 3: 3D Sensitivity -----------------------------------------------


def fig_03_3d_sensitivity(encoder, proteins):
    print("Fig 03: 3D Sensitivity")
    sigmas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    n_test = min(50, len(proteins))

    all_sims = {s: [] for s in sigmas}
    for graph in proteins[:n_test]:
        ca_coords, mask = get_ca_coords_and_mask(graph)
        ca_nm = ca_coords.float() / 10.0
        mask_f = mask.float()

        with torch.no_grad():
            orig = encoder(ca_nm.unsqueeze(0), mask_f.unsqueeze(0)).squeeze(0)[mask]

        for sigma in sigmas:
            if sigma == 0:
                all_sims[sigma].append(1.0)
                continue
            noise = torch.randn_like(ca_nm) * (sigma / 10.0)
            with torch.no_grad():
                pert = encoder(
                    (ca_nm + noise).unsqueeze(0), mask_f.unsqueeze(0)
                ).squeeze(0)[mask]
            all_sims[sigma].append(
                F.cosine_similarity(orig, pert, dim=-1).mean().item()
            )

    # Rotation invariance
    rot_sims = []
    for graph in proteins[:n_test]:
        ca_coords, mask = get_ca_coords_and_mask(graph)
        ca_nm = ca_coords.float() / 10.0
        mask_f = mask.float()
        with torch.no_grad():
            orig = encoder(ca_nm.unsqueeze(0), mask_f.unsqueeze(0)).squeeze(0)[mask]
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        if torch.det(R) < 0:
            R[:, 0] *= -1
        with torch.no_grad():
            rot = encoder((ca_nm @ R.T).unsqueeze(0), mask_f.unsqueeze(0)).squeeze(0)[
                mask
            ]
        rot_sims.append(F.cosine_similarity(orig, rot, dim=-1).mean().item())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot of perturbation sensitivity
    data = [all_sims[s] for s in sigmas]
    bp = ax1.boxplot(data, labels=[f"{s}" for s in sigmas], patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("steelblue")
        box.set_alpha(0.6)
    ax1.set_xlabel("Gaussian noise sigma (Angstroms)")
    ax1.set_ylabel("Cosine similarity to original")
    ax1.set_title("3D Sensitivity: Coordinate Perturbation")
    ax1.set_ylim(0, 1.05)

    # Rotation invariance
    ax2.hist(rot_sims, bins=30, color="steelblue", alpha=0.8)
    ax2.axvline(
        np.mean(rot_sims),
        color="red",
        linestyle="--",
        label=f"mean={np.mean(rot_sims):.6f}",
    )
    ax2.set_xlabel("Cosine similarity (original vs rotated)")
    ax2.set_ylabel("Count")
    ax2.set_title("Rotation Invariance Check")
    ax2.legend()

    fig.tight_layout()
    savefig("fig_03_3d_sensitivity.png")


# -- Figure 4: Residue-Type Discrimination -----------------------------------


def fig_04_residue_discrimination(all_emb, all_types):
    print("Fig 04: Residue-Type Discrimination")
    if all_types is None:
        print("  Skipped (no type info)")
        return

    X = all_emb.numpy()
    y = all_types.numpy()
    valid = y < 20
    X, y = X[valid], y[valid]

    # Mean embeddings per AA type
    unique_types = sorted(np.unique(y))
    mean_embs = np.stack([X[y == t].mean(axis=0) for t in unique_types])
    mean_t = torch.from_numpy(mean_embs).float()
    sim = F.cosine_similarity(mean_t.unsqueeze(0), mean_t.unsqueeze(1), dim=-1).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Similarity heatmap
    labels = [RESIDUE_NAMES[t] for t in unique_types]
    im = ax1.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=90, fontsize=7)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title("Mean Embedding Cosine Similarity\nby Amino Acid Type")

    # Within vs between type
    n_sample = 10000
    within, between = [], []
    for _ in range(n_sample):
        i, j = random.sample(range(len(X)), 2)
        cos = float(
            F.cosine_similarity(
                torch.tensor(X[i]).unsqueeze(0), torch.tensor(X[j]).unsqueeze(0)
            )
        )
        if y[i] == y[j]:
            within.append(cos)
        else:
            between.append(cos)

    ax2.hist(
        between,
        bins=80,
        density=True,
        alpha=0.6,
        color="gray",
        label=f"Between-type (n={len(between)})",
    )
    ax2.hist(
        within,
        bins=80,
        density=True,
        alpha=0.6,
        color="steelblue",
        label=f"Within-type (n={len(within)})",
    )
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Density")
    ax2.set_title("Within-type vs Between-type\nResidue Similarity")
    ax2.legend()

    fig.tight_layout()
    savefig("fig_04_residue_discrimination.png")


# -- Figure 5: Structural Context -------------------------------------------


def fig_05_structural_context(all_emb, all_types, proteins):
    print("Fig 05: Structural Context Sensitivity")
    if all_types is None:
        print("  Skipped")
        return

    # Collect per-residue SS proxy
    embs_list, types_list, ss_list = [], [], []
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

        for k in range(1, n_valid - 1):
            v1 = valid_coords[k - 1] - valid_coords[k]
            v2 = valid_coords[k + 1] - valid_coords[k]
            cos_a = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
            if 80 < angle < 100:
                ss = 0  # helix
            elif 110 < angle < 135:
                ss = 1  # sheet
            else:
                ss = 2  # loop

            aa = int(valid_types[k])
            if aa < 20:
                embs_list.append(all_emb[offset + k])
                types_list.append(aa)
                ss_list.append(ss)
        offset += n_valid

    if len(embs_list) < 100:
        print("  Insufficient data")
        return

    embs_arr = torch.stack(embs_list).numpy()
    types_arr = np.array(types_list)
    ss_arr = np.array(ss_list)

    # t-SNE on a subsample
    from sklearn.manifold import TSNE

    n_tsne = min(5000, len(embs_arr))
    idx = np.random.choice(len(embs_arr), n_tsne, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords_2d = tsne.fit_transform(embs_arr[idx])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # t-SNE colored by AA type
    ax1.scatter(
        coords_2d[:, 0], coords_2d[:, 1], c=types_arr[idx], cmap="tab20", s=3, alpha=0.5
    )
    ax1.set_title("t-SNE colored by amino acid type")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")

    # t-SNE colored by SS
    ss_colors = {0: "red", 1: "blue", 2: "gray"}
    ss_names = {0: "helix", 1: "sheet", 2: "loop"}
    for ss_val in [0, 1, 2]:
        ss_mask = ss_arr[idx] == ss_val
        ax2.scatter(
            coords_2d[ss_mask, 0],
            coords_2d[ss_mask, 1],
            c=ss_colors[ss_val],
            s=3,
            alpha=0.4,
            label=ss_names[ss_val],
        )
    ax2.set_title("t-SNE colored by secondary structure proxy")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(markerscale=5)

    fig.tight_layout()
    savefig("fig_05_structural_context.png")


# -- Figure 6: Embedding Norms ----------------------------------------------


def fig_06_norms(all_emb, all_types):
    print("Fig 06: Embedding Norms & Conditioning")
    norms = torch.norm(all_emb, dim=-1).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Norm distribution by AA type
    if all_types is not None:
        y = all_types.numpy()
        for aa in range(20):
            aa_norms = norms[y == aa]
            if len(aa_norms) > 10:
                ax1.boxplot(
                    aa_norms,
                    positions=[aa],
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.5),
                    flierprops=dict(markersize=1),
                )
        ax1.set_xticks(range(20))
        ax1.set_xticklabels(RESIDUE_NAMES, rotation=90, fontsize=7)
    else:
        ax1.hist(norms, bins=50, color="steelblue", alpha=0.8)
    ax1.set_ylabel("L2 norm")
    ax1.set_title(f"Per-residue Embedding Norms\nmean={norms.mean():.2f}")

    # Per-dimension variance
    dim_var = all_emb.var(dim=0).numpy()
    ax2.bar(range(len(dim_var)), np.sort(dim_var)[::-1], color="steelblue", alpha=0.8)
    ax2.set_xlabel("Dimension (sorted by variance)")
    ax2.set_ylabel("Variance")
    dead = np.sum(dim_var < 1e-6)
    ax2.set_title(f"Per-dimension Variance\ndead dims: {dead}/{len(dim_var)}")

    fig.tight_layout()
    savefig("fig_06_norms.png")


# -- Figure 7: Projector Saturation -----------------------------------------


def fig_07_projector_saturation(all_emb, all_types):
    print("Fig 07: Projector Saturation Test")
    target = all_emb.clone()
    n = target.shape[0]
    encoder_dim = target.shape[1]

    train_n = min(10000, n)
    idx = torch.randperm(n)[:train_n]
    target_train = target[idx]

    conditions = {}
    conditions["random (128-d)"] = torch.randn(n, 128)[idx]
    if all_types is not None:
        onehot = F.one_hot(all_types.clamp(max=20).long(), 21).float()
        conditions["AA one-hot (21-d)"] = onehot[idx]
        pos = torch.arange(n).float().unsqueeze(1) / n
        conditions["one-hot + pos (22-d)"] = torch.cat([onehot, pos], dim=-1)[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["gray", "steelblue", "orange"]

    for ci, (name, inp) in enumerate(conditions.items()):
        input_dim = inp.shape[1]
        mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(), nn.Linear(256, encoder_dim)
        )
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        epochs, sims = [], []
        for epoch in range(200):
            pred = mlp(inp)
            cos = F.cosine_similarity(pred, target_train, dim=-1)
            loss = -cos.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochs.append(epoch)
            sims.append(cos.mean().item())

        ax.plot(epochs, sims, color=colors[ci], label=f"{name} (final: {sims[-1]:.3f})")

    ax.axhline(
        0.78,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label="REPA training plateau (~0.78)",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Projector Saturation Test\nCan a simple MLP match GearNet targets?")
    ax.legend()
    ax.set_ylim(0, 1)

    savefig("fig_07_projector_saturation.png")


# -- Figure 8: Within vs Between Protein ------------------------------------


def fig_08_protein_similarity(per_protein):
    print("Fig 08: Within vs Between Protein Similarity")
    n_proteins = min(50, len(per_protein))

    within, between = [], []
    for emb in per_protein[:n_proteins]:
        if len(emb) < 2:
            continue
        n = min(50, len(emb))
        for _ in range(200):
            i, j = random.sample(range(n), 2)
            within.append(F.cosine_similarity(emb[i : i + 1], emb[j : j + 1]).item())

    for _ in range(10000):
        p1, p2 = random.sample(range(n_proteins), 2)
        e1, e2 = per_protein[p1], per_protein[p2]
        i, j = random.randint(0, len(e1) - 1), random.randint(0, len(e2) - 1)
        between.append(F.cosine_similarity(e1[i : i + 1], e2[j : j + 1]).item())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        between,
        bins=80,
        density=True,
        alpha=0.6,
        color="gray",
        label=f"Between-protein (mean={np.mean(between):.3f})",
    )
    ax.hist(
        within,
        bins=80,
        density=True,
        alpha=0.6,
        color="steelblue",
        label=f"Within-protein (mean={np.mean(within):.3f})",
    )
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Within-protein vs Between-protein\nResidue Similarity")
    ax.legend()

    savefig("fig_08_protein_similarity.png")


# -- Figure 9: Layer-wise Analysis ------------------------------------------


def fig_09_layerwise(encoder, proteins):
    print("Fig 09: Layer-wise Analysis")
    n_test = min(30, len(proteins))
    n_layers = len(encoder.gearnet.layers)

    layer_eff_ranks = []
    layer_norms = [[] for _ in range(n_layers)]
    layer_sparsity = [[] for _ in range(n_layers)]
    inter_layer_sims = [[] for _ in range(n_layers)]

    for graph in proteins[:n_test]:
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
            for i, layer in enumerate(encoder.gearnet.layers):
                h_v = layer(h_v, edge_list, h_e)
                layer_norms[i].append(h_v.norm(dim=-1).mean().item())
                layer_sparsity[i].append((h_v == 0).float().mean().item())
                inter_layer_sims[i].append(
                    F.cosine_similarity(prev, h_v, dim=-1).mean().item()
                )
                prev = h_v.clone()

    # Effective rank per layer (need to re-run collecting all h_v)
    for i in range(n_layers):
        all_h = []
        for graph in proteins[:n_test]:
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
                for j in range(i + 1):
                    h_v = encoder.gearnet.layers[j](h_v, edge_list, h_e)
                all_h.append(h_v)
        all_h = torch.cat(all_h, dim=0)
        centered = all_h - all_h.mean(dim=0)
        S = torch.linalg.svdvals(centered).numpy()
        p = (S**2) / (S**2).sum()
        p = p[p > 0]
        layer_eff_ranks.append(np.exp(-np.sum(p * np.log(p))))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Effective rank vs layer
    ax1.plot(range(n_layers), layer_eff_ranks, "o-", color="steelblue")
    ax1.set_xlabel("GearNet layer")
    ax1.set_ylabel("Effective rank")
    ax1.set_title("Effective Rank by Layer")
    ax1.set_xticks(range(n_layers))

    # Mean norm vs layer
    mean_norms = [np.mean(layer_norms[i]) for i in range(n_layers)]
    ax2.plot(range(n_layers), mean_norms, "o-", color="steelblue")
    ax2.set_xlabel("GearNet layer")
    ax2.set_ylabel("Mean L2 norm")
    ax2.set_title("Embedding Norm by Layer")
    ax2.set_xticks(range(n_layers))

    # Inter-layer cosine similarity
    mean_sims = [np.mean(inter_layer_sims[i]) for i in range(n_layers)]
    ax3.plot(range(n_layers), mean_sims, "o-", color="steelblue")
    ax3.set_xlabel("GearNet layer")
    ax3.set_ylabel("Cosine sim (layer i vs i-1)")
    ax3.set_title("Inter-layer Similarity")
    ax3.set_xticks(range(n_layers))
    ax3.set_ylim(0, 1)

    fig.tight_layout()
    savefig("fig_09_layerwise.png")


# -- Main --------------------------------------------------------------------


def main():
    print("Loading data...")
    proteins = load_proteins(N_PROTEINS)
    encoder = setup_encoder()
    print("Computing embeddings...")
    all_emb, all_types, per_protein = collect_all_embeddings(encoder, proteins)
    print(f"Total: {all_emb.shape[0]} residues\n")

    fig_01_value_distribution(all_emb)
    fig_02_singular_values(all_emb)
    fig_03_3d_sensitivity(encoder, proteins)
    fig_04_residue_discrimination(all_emb, all_types)
    fig_05_structural_context(all_emb, all_types, proteins)
    fig_06_norms(all_emb, all_types)
    fig_07_projector_saturation(all_emb, all_types)
    fig_08_protein_similarity(per_protein)
    fig_09_layerwise(encoder, proteins)

    print("\nAll 9 figures generated!")


if __name__ == "__main__":
    main()
