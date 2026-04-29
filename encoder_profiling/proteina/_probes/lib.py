"""Encoder characterisation pipeline for proteina REPA target selection.

This file is organised around three questions. Every analysis function below
maps to exactly one of them, and the cross-encoder narrative in
``encoder_profiling/proteina/FINDINGS.md`` reads them in the same order.

Q1. WHAT INFORMATION DOES THE ENCODER ENCODE?
    Upper bound on what REPA can transfer. If the encoder doesn't carry
    information X, REPA can never teach the student X. We probe four kinds:
    residue identity, 3D geometric sensitivity, structural/sequence context,
    and protein-level identity.

Q2. HOW MUCH OF THAT IS REACHABLE FROM CHEAP INPUTS?
    Saturation floor. A 3-layer MLP from ``(onehot, position) -> encoder``
    - per-residue, no neighbour context - absorbs most of the per-residue
    cosine alignment. The proteina student's projector reads off a
    transformer hidden state computed from *strictly more*:
    ``(onehot, position) + noisy 3D coords + cross-residue attention
    + diffusion timestep``.

    Two distinct quantities, both measurable from this probe::

        floor          = best_(onehot+pos)_test_cos
                       = cosine the projector reaches for free,
                         REPA-or-no-REPA.

        gap_structural = best_(onehot+pos)_test_cos - mean_direction_cos
                       = how much identity + position lifts cosine over
                         a constant prediction. A *structural property
                         of the encoder*; an estimate of what *could*
                         be learned if richer inputs helped further.
                         Small -> encoder variance is tight, no rich
                         input is likely to push much further.

    Neither is REPA's actual operating budget. That quantity is::

        REPA_budget    = training_cos - best_(onehot+pos)_floor

    observable only from wandb logs, not from this probe. The probe
    tells you whether headroom *could* exist; training tells you whether
    the student actually used it. A run that plateaus at the floor is a
    run where REPA contributed nothing the projector wouldn't do alone.

    For 3D-aware encoders (CA-GearNet, PW) what's above the floor is
    mostly coord-driven; for sequence-only encoders (ESM2) it is mostly
    cross-residue attention context (the encoder ignores coords by
    construction). ``analyze_projector_saturation`` measures the floor
    and the structural gap.

Q3. IS THE ENCODER A TRACTABLE OPTIMISATION TARGET?
    Even when the encoder is informative, sparsity, rank collapse, dead
    dims and norm explosions can degrade the gradient signal so much that
    REPA learns nothing. Three conditioning probes: value distribution,
    effective dimensionality, embedding norms.

Diagnostic flow
---------------
The diagnostic value lives in the *gap* between Q1 and Q2 (= REPA headroom),
modulated by Q3 conditioning::

                pretraining quality
                       |
          +------------+------------+
          v                         v
     eff rank up         residue/protein delta up
          |                         |
          v                         |
     mean-dir cos down              |
          |                         |
          v                         v
            projector gap = REPA headroom
                       |
                       v
            empirical val-loss gain

Caveat: gap is necessary but not sufficient. ESM2 has the largest gap
(+0.053) but it is sequence-only - the headroom is *what kind* of
information varies, not just how much. A small Q1 vs Q2 gap with the
*right* kind of information (geometry, for a 3D generative model) can be
worth more than a larger gap of identity/sequence headroom.

Public surface (consumed by encoder_profiling/proteina/*/explore_*.py)
---------------------------------------------------------------------
- ``EncoderProbe``           - declares an encoder + its capabilities
- ``make_embed_fn``          - standard CPU<->device wrapper for forward()
- ``load_proteins``          - LMDB loader with train_keys.pkl fast path
- ``graph_to_inputs``        - graph -> (ca_nm, mask, residue_type) on CPU
- ``run_pipeline``           - runs the standard battery
- ``RESIDUE_NAMES``          - 21-class AA label list
- ``analyze_*``              - individual analyses (callable directly)

Each encoder driver supplies an ``embed_fn(ca_nm, mask, residue_type=None)
-> [B,n,D]`` closure (typically ``torch.no_grad()`` + ``.to(device)`` +
``.cpu()`` wrapper around its encoder's forward) and declares whether the
encoder is 3D-aware / takes residue type. The lib runs only the analyses
that make sense for those capabilities.

Convention: ``embed_fn`` receives CPU tensors and returns CPU tensors.
Internal device transfers are the driver's responsibility.

Stable JSON keys (read by encoder_profiling/proteina/collate.py)
----------------------------------------------------------------
Top-level: ``distribution``, ``dimensionality``, ``perturbation``,
``rotation``, ``residue_shuffle``, ``residue_probe``,
``structural_context``, ``sequence_context``, ``norms``, ``projector``,
``protein_similarity``, ``layerwise``, ``embed_dim``, ``n_residues``,
``n_proteins``, ``name``, ``capabilities``, ``notes``, ``timestamp``.
Renames here will break the collator - reorder freely, rename with care.
"""

from __future__ import annotations

import csv
import json
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


# -- EncoderProbe ------------------------------------------------------------


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
        is_3d_aware           - runs perturbation + rotation tests when True
        accepts_residue_type  - runs residue-type-shuffle test when True
        context_mode          - "structural" (uses CA-CA-CA angles to bin SS),
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
    output_dir: Optional[str] = None  # if set, run_pipeline writes results.json here


# -- Data loading ------------------------------------------------------------


def load_proteins(lmdb_path: str, n: int, seed: Optional[int] = None) -> list:
    """Load `n` proteins from a PDB LMDB.

    Uses the pre-enumerated `train_keys.pkl` next to the LMDB when present -
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
    """Return (ca_nm[n,3], mask[n], residue_type[n]) - all CPU tensors.

    CA is at atom index 1 in OpenFold ordering. Coords stored in A; converted
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


def _header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# Q1. WHAT INFORMATION DOES THE ENCODER ENCODE?
#
# Upper bound on what REPA can transfer. We probe four orthogonal kinds of
# information:
#
#   1.1  Residue identity        -> analyze_residue_discrimination
#                                   analyze_residue_shuffle
#   1.2  3D geometric sensitivity -> analyze_perturbation
#                                   analyze_rotation_invariance
#   1.3  Structural / sequence   -> analyze_structural_context
#        context (depending      analyze_sequence_context
#        on encoder type)
#   1.4  Protein-level identity  -> analyze_protein_similarity
#
# Read the random-init baseline (gearnet_random/) as the architecture-only
# floor: a Q1 metric that doesn't move substantially off that floor is one
# where pretraining isn't contributing and REPA has nothing useful to teach.
# ============================================================================


def analyze_residue_discrimination(all_emb, all_types, max_residues: int = 30000):
    """Q1.1 - Residue-type discrimination (linear probe + cosine geometry).

    What it measures
    ----------------
    Three complementary tests of whether the embedding encodes amino-acid
    identity:

    1. **Linear probe**: multinomial logistic regression on a 80/20 split.
       Reports test accuracy. Chance is ~5% for 20 classes (uniform), but
       real proteins skew toward ALA/LEU/VAL so the practical floor is
       ~12-13%.

    2. **AA centroid cosine**: compute one mean embedding per AA type (21
       centroids), then mean cosine similarity over all 210 off-diagonal
       pairs. This is "how distinguishable are AA classes on average."

    3. **Pairwise within/between sampling**: 5000 random pairs of *individual*
       residue embeddings, bucketed by `same AA` vs `different AA`. Note
       this is *not* anchor-vs-all; sampling is unbiased w.r.t. AA frequency
       (an anchor-based formulation would oversample rare AAs).

    Why it matters for REPA
    -----------------------
    If a downstream task or auxiliary loss needs residue identity (e.g.
    sequence design heads), a high probe accuracy means the encoder
    already provides it. Conversely, an encoder with no input-side residue
    feature (CA-GearNet) cannot encode identity - probe accuracy near
    chance is the expected "well-behaved" answer there, not a defect.

    How to read the output
    ----------------------
    - probe_acc near chance + centroid cos near 1: encoder doesn't see AA.
    - probe_acc ~ 1.0 + centroid cos < 1: encoder is an AA classifier
      (ESM2 last layer: 0.998 / 0.853).
    - within-between Delta is the cosine-side signal, complements the
      probe's linear separability.

    Correlations
    ------------
    - High probe acc with high mean-direction cos (Q2) -> projector
      saturates from `onehot` input alone; little headroom for REPA.
    - Combined with the random-init baseline, Delta_trained - Delta_random
      isolates the contribution of pretraining over architecture.
    """
    _header("Q1.1 - Residue-Type Discrimination")
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
    return {
        "linear_probe_acc": float(acc),
        "centroid_cos_mean": float(upper.mean()),
        "centroid_cos_min": float(upper.min()),
        "centroid_cos_max": float(upper.max()),
        "within_type_cos_mean": float(np.mean(within)),
        "within_type_cos_std": float(np.std(within)),
        "between_type_cos_mean": float(np.mean(between)),
        "between_type_cos_std": float(np.std(between)),
    }


def analyze_residue_shuffle(embed_fn: EmbedFn, proteins, n_test: int = 50):
    """Q1.1 - Residue-type shuffle: identity-driven vs geometry-driven.

    What it measures
    ----------------
    Hold the CA coordinates fixed; permute the residue-type labels within
    each protein; recompute the embedding; report mean cosine similarity
    between original and shuffled.

    Why it matters for REPA
    -----------------------
    Together with analyze_perturbation (Q1.2), this disentangles what the
    encoder is actually using:
        cos low  + perturbation cos low   -> encoder uses both
        cos low  + perturbation cos high  -> encoder is identity-driven
        cos high + perturbation cos low   -> encoder is geometry-driven
        cos high + perturbation cos high  -> encoder is collapsed
                                              (input-insensitive overall)

    How to read the output
    ----------------------
    - cos near 1.0 means identity is irrelevant to the embedding (good
      for a 3D-only encoder; bad for an encoder that *takes* residue
      features and should use them - MC-GearNet's 0.983 here despite
      ingesting one-hot AA features is a red flag for collapse).
    - cos << 1 with high perturbation cos says the encoder is a sequence
      lookup with weak coordinate dependence (ESM-like behaviour, but
      ESM doesn't take coords so this test is skipped).

    Capability gating
    -----------------
    Only meaningful for encoders that accept residue_type. Skipped via
    the EncoderProbe.accepts_residue_type flag.
    """
    _header("Q1.1 - Residue-Type Shuffle (coords fixed, labels permuted)")
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
    return {
        "shuffle_cos_mean": float(np.mean(sims)),
        "shuffle_cos_std": float(np.std(sims)),
    }


def analyze_perturbation(
    embed_fn: EmbedFn, proteins, n_test: int = 50, sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)
):
    """Q1.2 - 3D geometric sensitivity (Gaussian coordinate noise sweep).

    What it measures
    ----------------
    Cosine similarity between the embedding of the original CA backbone
    and a perturbed copy with isotropic Gaussian noise added to each CA.
    Sweeps sigma in {0.1, 0.5, 1.0, 2.0, 5.0} Angstrom.

    Why it matters for REPA
    -----------------------
    When the encoder is 3D-aware, REPA's job is to teach the student
    transformer about geometry. If the embedding doesn't move under
    sub-Angstrom perturbations, alignment teaches nothing geometric -
    the encoder has memoised the input identity and ignored the
    coordinates. We want cos to drop noticeably between 0.1 A (within
    thermal fluctuation) and 0.5-1.0 A (clearly different conformations).

    How to read the output
    ----------------------
    - cos ~1.0 across all sigmas -> encoder is 3D-blind (random-init
      CA-GearNet, MC-GearNet-Edge dominated by exploding norms).
    - cos drops monotonically -> healthy 3D sensitivity (CA-GearNet
      trained: 0.5 A -> 0.37; PW-torsional: 0.5 A -> 0.92).
    - The *slope* between 0.1 A and 1.0 A is the diagnostic. The
      magnitude varies by encoder; what matters is that there is a
      slope at all.

    Correlations
    ------------
    - High 3D sensitivity -> the projector gap (Q2) measures genuinely
      geometric headroom rather than identity headroom.
    - Low 3D sensitivity AND high mean-dir cos -> encoder is collapsed;
      Q2 gap will be near zero, regardless of how high the AA probe (Q1.1)
      runs.

    Capability gating
    -----------------
    Skipped if EncoderProbe.is_3d_aware is False (e.g. ESM2).
    """
    _header("Q1.2 - 3D Sensitivity: Gaussian Coordinate Perturbation")
    n_test = min(n_test, len(proteins))
    print("Cosine sim between original and perturbed embeddings:")
    out = {}
    for sigma in sigmas:
        sims = []
        for graph in proteins[:n_test]:
            ca_nm, mask, rtype = graph_to_inputs(graph)
            orig = embed_fn(
                ca_nm.unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0)
            ).squeeze(0)[mask]
            noise = torch.randn_like(ca_nm) * (sigma / 10.0)  # A -> nm
            pert = embed_fn(
                (ca_nm + noise).unsqueeze(0), mask.unsqueeze(0), rtype.unsqueeze(0)
            ).squeeze(0)[mask]
            sims.append(F.cosine_similarity(orig, pert, dim=-1).mean().item())
        print(
            f"  sigma={sigma:.1f}A: cos_sim={np.mean(sims):.4f} +/- {np.std(sims):.4f}"
        )
        out[f"sigma_{sigma}A"] = {
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
        }
    return out


def analyze_rotation_invariance(embed_fn: EmbedFn, proteins, n_test: int = 50):
    """Q1.2 - 3D rotation-invariance check (sanity test, not a Q1 signal).

    What it measures
    ----------------
    Cosine similarity between the original embedding and the embedding
    after applying a uniformly-random SO(3) rotation to all CA coords.
    The det-correction (`R[:,0] *= -1` if `det(R) < 0`) ensures we sample
    from SO(3) not O(3).

    Why it matters for REPA
    -----------------------
    Every GearNet variant in our shortlist uses pairwise distances and
    angles - SO(3)-invariant edge features by construction. cos near 1.0
    is the expected and required result. Anything else means the encoder
    leaks orientation, and the student would need to learn rotation-
    equivariant alignment, which it should not have to.

    This is a *sanity check*, not an information-content probe. We log
    it to detect implementation bugs (e.g. accidental coordinate-frame
    leakage), not to discriminate good encoders from bad.

    How to read the output
    ----------------------
    - cos > 0.999: invariant by construction (all GearNets, ESM-trivially
      since it ignores coords).
    - cos < 0.99: bug; check that distance/angle features are computed
      from invariant primitives.
    """
    _header("Q1.2 - 3D Sensitivity: Rotation Invariance (sanity)")
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
    return {
        "rotation_cos_mean": float(np.mean(sims)),
        "rotation_cos_std": float(np.std(sims)),
    }


def analyze_structural_context(all_emb, all_types, proteins, test_aas=(0, 7, 10, 19)):
    """Q1.3 - Structural-context sensitivity (helix / sheet / loop).

    What it measures
    ----------------
    For each residue, classify its local secondary-structure context using
    the CA-CA-CA bond angle as a coarse proxy::

        ~91 deg   -> helix
        ~120 deg  -> sheet
        otherwise -> loop

    Then for a few test amino acids (default ALA/GLY/LEU/VAL), bucket
    embeddings by (aa, ss) and compare:
        within-SS  cos   = pairs of same-AA-same-SS embeddings
        between-SS cos   = pairs of same-AA-different-SS embeddings
        Delta            = within - between

    Why it matters for REPA
    -----------------------
    A 3D generative model wants encoder signal that varies with local
    geometry (so REPA can teach 'this residue is in a helix; that
    residue is in a beta strand'). Positive within-SS Delta means SS
    information lives in the embedding. Near-zero or negative Delta
    (MC-GearNet) means the encoder has collapsed and SS-sensitive
    pretraining (PW-torsional) shines through here especially well.

    How to read the output
    ----------------------
    - Delta > 0.05 across most AAs: strong SS signal (PW-torsional).
    - Delta ~0.03 with one outlier (GLY): typical (CA-GearNet).
    - Delta near 0 or negative: collapse (MC-GearNet) or SS-blind encoder.
    - GLY is conformationally promiscuous; near-zero Delta there is
      expected and not diagnostic.

    Limitations
    -----------
    The CA-CA-CA angle classifier is noisy near the helix/sheet boundary
    and conflates short turns with loops. Suitable for averaged signal,
    not per-residue calls. A DSSP-based classifier would be cleaner; the
    angle proxy is a no-extra-deps stand-in.

    Capability gating
    -----------------
    Used when EncoderProbe.context_mode == "structural". Sequence-only
    encoders use analyze_sequence_context instead.
    """
    out = {}
    _header("Q1.3 - Structural Context: helix/sheet/loop via CA-CA-CA angle")
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
        valid_coords = ca_nm[valid_idx] * 10.0  # back to A for angle calc
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
            out[RESIDUE_NAMES[aa]] = {"insufficient_data": True}
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
        out[RESIDUE_NAMES[aa]] = {
            "within_ss_cos_mean": float(np.mean(within)),
            "between_ss_cos_mean": float(np.mean(between)),
            "delta": float(np.mean(within) - np.mean(between)),
            "context_counts": {k: len(v) for k, v in contexts.items()},
        }
    return out


def analyze_sequence_context(embed_fn: EmbedFn, proteins, n_test: int = 30):
    """Q1.3 - Sequence-context sensitivity (sequence-only encoders).

    What it measures
    ----------------
    For each test protein, fix the *center* residue's identity; either
    shuffle the flanking residue labels (preserving the multiset) or
    fully randomise them; measure how much the center embedding moves.

    Why it matters for REPA
    -----------------------
    For sequence-only encoders (ESM2), the 3D-perturbation test is
    meaningless because the encoder ignores coordinates. The relevant
    contextual signal is which residues sit near the focal one.
    Cos << 1 means the encoder is not a per-residue lookup table -
    its embeddings are shaped by neighbourhood.

    How to read the output
    ----------------------
    - cos near 1.0: encoder behaves as an AA-only lookup; shuffling the
      sequence around a residue doesn't change its embedding. Bad sign
      for a 'language model' - means context isn't informing the
      representation.
    - cos in [0.4, 0.7]: typical for ESM-like models; substantial
      neighbourhood contribution.
    - shuffled vs random give similar values: the *fact* of perturbation
      matters more than the specific multiset.

    Caveat
    ------
    Fixing only the center residue conflates 'neighbours provide context'
    with 'neighbours break the local MLM prediction' (sequence models
    are trained to fill in masks; perturbing flanks changes the prior).
    A cleaner test would mutate only *far* neighbours.

    Capability gating
    -----------------
    Used when EncoderProbe.context_mode == "sequence" (ESM driver).
    """
    _header("Q1.3 - Sequence Context: center fixed, flanks varied")
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
        return {"insufficient_data": True}
    print(
        f"Original vs shuffled-flanks: {np.mean(shuf_sims):.4f} +/- {np.std(shuf_sims):.4f}"
    )
    print(
        f"Original vs random-flanks:   {np.mean(rand_sims):.4f} +/- {np.std(rand_sims):.4f}"
    )
    print(
        "  (1.0 = context doesn't matter; lower = context strongly modulates the center)"
    )
    return {
        "shuffled_flanks_cos_mean": float(np.mean(shuf_sims)),
        "shuffled_flanks_cos_std": float(np.std(shuf_sims)),
        "random_flanks_cos_mean": float(np.mean(rand_sims)),
        "random_flanks_cos_std": float(np.std(rand_sims)),
    }


def analyze_protein_similarity(per_protein, n_proteins: int = 50):
    """Q1.4 - Within-protein vs between-protein similarity.

    What it measures
    ----------------
    Pairwise cosine similarity in two buckets:
        within-protein:  100 random pairs of residue embeddings drawn
                         from the *same* protein, repeated over the first
                         n_proteins proteins.
        between-protein: 5000 random pairs drawn from *different*
                         proteins.
    Headline number is Delta = within - between.

    Why it matters for REPA
    -----------------------
    This is the single metric where pretraining most clearly contributes
    to a CA-only encoder (CA-GearNet trained Delta = 0.222 vs random
    Delta = 0.035, a 6x lift). It captures 'protein-level identity':
    residues in the same protein share global geometric context, and
    the embedding reflects that. REPA aligned against a high-Delta
    encoder teaches the student about the protein it's currently
    generating, not just per-residue micro-structure.

    How to read the output
    ----------------------
    - Delta near 0: residues are interchangeable across proteins.
      Encoder is either collapsed (MC-GearNet 0.043) or context-blind.
    - Delta in [0.08, 0.15]: typical sequence/identity-driven encoder
      (ESM2: 0.098, PW-torsional: 0.102).
    - Delta > 0.2: strong protein-level signal (CA-GearNet trained:
      0.222). REPA against this teaches the most 'protein context'.

    Correlations
    ------------
    - High Delta with low Q2 gap means most of the protein-context
      signal is *not* extractable by a cosine-aligned projector from
      (onehot, position) inputs. That can still be useful, but only
      via auxiliary objectives that don't normalise direction.
    """
    _header("Q1.4 - Within-Protein vs Between-Protein Similarity")
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
    return {
        "within_protein_cos_mean": float(np.mean(within)),
        "within_protein_cos_std": float(np.std(within)),
        "between_protein_cos_mean": float(np.mean(between)),
        "between_protein_cos_std": float(np.std(between)),
        "delta": float(np.mean(within) - np.mean(between)),
    }


# ============================================================================
# Q2. HOW MUCH OF THE ENCODER'S INFORMATION IS REACHABLE FROM CHEAP INPUTS?
#
# Saturation / headroom. Train a small projector (3-layer MLP) from various
# 'cheap' inputs to the encoder embedding under a cosine loss. The student
# transformer in proteina has its own (residue-onehot, position) inputs and
# its own projector head, so any cosine alignment a 3-layer MLP can already
# achieve from those inputs alone is alignment that REPA cannot claim
# credit for - it would have happened in the projector regardless.
#
# Headline number:  gap = best_projector_test_cos - mean_direction_cos
#                       = REPA's actual operating budget.
# ============================================================================


def analyze_projector_saturation(
    all_emb, all_types, device, epochs: int = 300, hidden: int = 512
):
    """Q2 - Projector saturation: how much cosine alignment is *free*?

    What it measures
    ----------------
    Trains a 3-layer MLP under cosine loss
        -F.cosine_similarity(mlp(input), target).mean()
    from each of four input conditions to the encoder embedding, on an
    80/20 train/test split, for 300 epochs:

        Q2.1  mean direction      no MLP - cosine to dataset centroid.
                                  Reflects 'how aligned is everything
                                  already.'
        Q2.2  random 128-d        torch.randn(n, 128). The 128 is an
                                  arbitrary 'meaningless feature' dim,
                                  NOT tied to any architectural width.
                                  This row diagnoses memorisation: a
                                  high train_cos with a sharply lower
                                  test_cos means the MLP is memorising
                                  per-residue noise.
        Q2.3  AA one-hot (21d)    F.one_hot(residue_type, 21). Tests
                                  what identity alone can predict.
        Q2.4  onehot + position   onehot concatenated with normalised
                                  index. The most 'cheap' input the
                                  student transformer has access to
                                  before any attention.

    Two derived numbers, both from this probe::

        floor          = max(onehot, onehot+pos)_test_cos
                       = the cosine the projector reaches for free,
                         REPA-or-no-REPA. The student's projector cannot
                         do *worse* than this in steady state.

        gap_structural = floor - mean_direction_cos
                       = how much (onehot, position) lifts cosine over
                         a constant prediction. A *structural property
                         of the encoder*; an estimate of what *could*
                         be learned if richer inputs (coords, attention,
                         timestep) helped further. Small -> encoder
                         variance is tight, no rich input is likely to
                         push much further.

    What this probe is NOT
    ----------------------
    Neither number is REPA's actual operating budget. The budget is::

        REPA_budget = training_cos - floor

    visible only in wandb logs, not measurable here. This probe tells
    you whether headroom *could* exist; training tells you whether the
    student actually used it. A run that plateaus at the floor is a run
    where REPA contributed nothing the projector wouldn't do alone.

    Why it matters for REPA
    -----------------------
    REPA loss is cosine alignment of the student's *projected* hidden
    state to the encoder. The student's hidden state is computed from
    strictly more than (onehot, position): it also has noisy 3D coords,
    cross-residue self-attention context, and the diffusion timestep.
    The cheap-input MLP probe here has none of those - it processes
    each residue independently from identity + index alone, and is
    therefore a clean lower bound for what any input-richer projector
    can achieve.

    Below the floor: projector head doing its job for free. Above the
    floor (only visible at training time): the cosine the student
    extracts via coords / attention / timestep - REPA's actual
    contribution. For 3D-aware encoders that contribution is mostly
    coord-driven; for sequence-only encoders it is mostly cross-residue
    attention.

    How to read the output
    ----------------------
    - gap_structural > 0.05: large estimated room (ESM2 +0.053). Rich
      inputs *could* push above the floor by a meaningful margin -
      training will tell you whether they do.
    - gap_structural in [0.005, 0.020]: tight (CA-GearNet +0.006,
      PW-torsional +0.009). The encoder's variance pattern leaves
      little room above the floor for any input. Expect modest
      training-time gains over the floor; don't expect REPA to
      dominate the loss.
    - gap_structural <= 0: encoder is collapsed (MC-GearNet -0.002).
      Even cheap inputs cannot beat the constant baseline. REPA target
      is unusable.
    - random_test_cos near random_train_cos: meaningless input still
      generalises, projector is genuinely fitting structure.
    - random_test_cos << random_train_cos: memorisation, projector is
      overfitting per-residue noise; train-test gap is interesting but
      doesn't reflect generalisable signal.

    Correlations
    ------------
    - Strongly anti-correlated with eff rank (Q3.2): low rank ->
      embeddings collapse near the centroid -> high mean-dir baseline
      -> small gap_structural.
    - Sparsity (Q3.1) drives down gap_structural by concentrating
      variance on a few non-zero dims.
    - gap_structural is the most predictive *single* probe-side metric
      for empirical REPA val-loss gain, but it's a proxy: it estimates
      the room above the floor, not REPA's actual extraction. Read it
      alongside Q1 to understand *what kind* of information could sit
      in that room (geometry vs sequence vs identity), and read the
      training cosine vs floor to confirm REPA used it.
    """
    _header("Q2 - Projector Saturation Test")
    target = all_emb.to(device)
    n, encoder_dim = target.shape
    print(f"  target: n={n}, dim={encoder_dim}")

    mean_vec = target.mean(dim=0, keepdim=True)
    mean_cos = (
        F.cosine_similarity(mean_vec.expand_as(target), target, dim=-1).mean().item()
    )
    print(f"  mean-direction baseline (test cos): {mean_cos:.4f}")
    out = {"mean_direction_cos": float(mean_cos), "conditions": {}}

    onehot = F.one_hot(all_types.clamp(min=0, max=20).long(), 21).float().to(device)
    pos = torch.arange(n, device=device).float().unsqueeze(1) / n
    conditions = {
        # 128 here is arbitrary - we just need a reasonable-sized
        # information-free vector to probe whether the MLP can memorise
        # per-residue targets without any meaningful signal.
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
        out["conditions"][name] = {
            "train_cos": float(train_cos),
            "test_cos": float(test_cos),
        }
    return out


# ============================================================================
# Q3. IS THE ENCODER A TRACTABLE OPTIMISATION TARGET?
#
# Even when the encoder is informative (Q1) and has projector headroom (Q2),
# numerical pathologies in the embedding can degrade the REPA gradient
# signal so much that nothing gets learned. Three orthogonal probes:
#
#   3.1  Sparsity & value distribution -> analyze_distribution
#        Element-level zero / near-zero / sign / extreme-value statistics.
#        Sparse targets give jittery directions; cosine loss flows only
#        through non-zero target dims.
#
#   3.2  Effective dimensionality      -> analyze_dimensionality
#        SVD spectrum: effective rank, singular-value decay, condition
#        number. Low rank -> embeddings collapse near the centroid -> the
#        projector trivially saturates (Q2).
#
#   3.3  Norms & dead dimensions       -> analyze_norms
#        Per-residue L2 magnitude and per-dim std. Norm explosions
#        co-occur with dead-dim count and indicate ill-conditioned
#        forward passes (BatchNorm + residual chains without final LN).
#
# Q3 failures dominate Q1 / Q2: an encoder that fails Q3 is unusable
# regardless of what else it scores on.
# ============================================================================


def analyze_distribution(all_emb: torch.Tensor):
    """Q3.1 - Element-wise value distribution and sparsity.

    What it measures
    ----------------
    Element-wise statistics across all (residue, dim) entries: mean, std,
    min, max, fraction of exact zeros, fraction near-zero (<1e-6),
    fraction negative.

    Distinct from analyze_norms (per-residue L2 magnitudes, Q3.3) and
    analyze_dimensionality (per-direction variance, Q3.2). All three are
    different views of the same underlying tensor; the failure modes
    they catch are different.

    Why density beats sparsity for REPA
    -----------------------------------
    REPA loss is 1 - cos_sim(student, encoder). Cosine sim only sees the
    active (non-zero) components of the target. If the target is 93%
    zeros (CheMeleon, ReLU output) the gradient flows back through the
    student only where the target dims happen to be non-zero - typically
    a *different* 7-dim subspace per sample. The student sees a
    high-variance, jittery training signal where the supervised
    dimensions change every step.

    Dense targets (CA-GearNet 0% zero, ESM 0%, PW 0%) give every
    dimension a non-trivial signal at every step. This is the headline
    reason LeakyReLU/SiLU/GELU encoders are friendlier REPA targets
    than ReLU encoders.

    How to read the output
    ----------------------
    - exact_zero_frac < 1e-3: dense (good).
    - exact_zero_frac > 0.5: sparse (problematic). Inspect activation
      function; ReLU at the head is usually the culprit.
    - Wildly negative mean (e.g. -1.1 for CA-GearNet) is fine - cosine
      is shift-equivariant after the loss normalises the direction.
    - min/max in millions (MC-GearNet ~5e7) is a norm-explosion signal;
      cross-reference analyze_norms.

    Correlations
    ------------
    - Sparse -> typically high mean-direction cos (variance concentrated
      on few dims, rest near-mean) -> small projector gap (Q2).
      CheMeleon: 93.8% sparse, projector saturates at 0.47 regardless
      of input.
    - Dense alone is necessary but not sufficient. MC-GearNet is fully
      dense yet still collapses on rank (Q3.2).
    """
    _header("Q3.1 - Value Distribution & Sparsity")
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
    return {
        "shape": list(all_emb.shape),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "exact_zero_frac": exact_zero / n,
        "near_zero_frac": near_zero / n,
        "negative_frac": negative / n,
    }


def analyze_dimensionality(all_emb: torch.Tensor, max_residues: int = 30000):
    """Q3.2 - Effective dimensionality and singular-value spectrum.

    What it measures
    ----------------
    SVD of the centered (residues x dim) embedding matrix, then summary
    statistics of the spectrum:

    - **effective rank**:
          exp(-sum_i p_i log p_i)   where  p_i = sigma_i^2 / sum sigma^2
      Equals N if the spectrum is flat (all singular values equal);
      equals 1 if a single direction carries all the variance. This is
      the entropy of the variance spectrum, exponentiated.
    - **participation ratio**:
          (sum sigma^2)^2  /  sum sigma^4
      Similar idea, less sensitive to long tails of small singular
      values.
    - **dims for {90, 95, 99}% variance**: how many PCs you need to
      span that fraction of the spread.
    - **condition number** sigma_max / sigma_min: numerical stability
      of any operation that needs to invert / pseudo-invert the
      embedding.

    Why it matters for REPA
    -----------------------
    1. Anti-collapse signal. Low effective rank means embeddings live
       near a low-dim subspace; the projector can match the target by
       predicting the mean (cf. analyze_projector_saturation - Q2).
       MC-GearNet-Edge's eff rank 1.1/3072 is exactly this pathology -
       mean-dir cos 0.855, gap negative.

    2. Projector bottleneck check. The student's REPA projector has a
       specific input dim (transformer hidden size). If
              eff_rank(encoder)  >  projector_in_dim
       the projector cannot recover all of the encoder's variance even
       in principle. CheMeleon's 500-d effective rank with a 128-d
       projector input is this failure mode.

    3. Discriminability proxy. High eff rank correlates with the
       embedding space being 'spread out' - distinct inputs land in
       distinct outputs.

    How to read the output
    ----------------------
    - eff_rank << total_dim with mean-dir cos near 1: rank collapse.
      Encoder unusable.
    - eff_rank << total_dim but mean-dir cos modest (~0.5): low-rank
      but spread across that subspace; usable.
    - condition_number > 1e6: numerical issues; co-occurs with norm
      explosion (analyze_norms - Q3.3).

    Correlations
    ------------
    Strongly anti-correlated with mean_direction_cos in
    analyze_projector_saturation (Q2); strongly correlated with the
    projector gap. The chain is:
        eff rank up -> mean-dir down -> gap up -> REPA headroom up.

    Implementation note
    -------------------
    Subsample to max_residues=30000 for tractability; SVD is O(n d^2).
    Spectrum tail is preserved in the JSON sidecar (`spectrum.json`)
    in case downstream analysis wants the full curve.
    """
    _header("Q3.2 - Dimensionality & Singular Values")
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
    return {
        "effective_rank": eff_rank,
        "participation_ratio": part_ratio,
        "dim_total": int(all_emb.shape[1]),
        "dims_for_90pct_var": int(np.searchsorted(cum_var, 0.90)) + 1,
        "dims_for_95pct_var": int(np.searchsorted(cum_var, 0.95)) + 1,
        "dims_for_99pct_var": int(np.searchsorted(cum_var, 0.99)) + 1,
        "top_singular_value": float(S[0]),
        "condition_number": float(S[0] / max(S[-1], 1e-12)),
        "singular_values": S.tolist(),
        "cumulative_variance": cum_var.tolist(),
    }


def analyze_norms(all_emb, all_types):
    """Q3.3 - Embedding norms and dead dimensions.

    What it measures
    ----------------
    - per-residue L2 norm: mean, std, min, max.
    - per-dimension std across the dataset; count of 'dead' dims with
      std < 1e-6 (literally constant - they contribute nothing to any
      cosine and bloat the projector).
    - per-AA L2 norm: spotting AA-specific magnitude bias (e.g. CYS
      higher because of disulfide-relevant features).

    Distinct from analyze_distribution (element-level scalars) and
    analyze_dimensionality (variance directions). This is the
    *magnitude / scale* view.

    Why it matters for REPA
    -----------------------
    Cosine similarity normalises away magnitude in the loss, but the
    raw scale still flows through the encoder's projector head during
    optimisation. Norm explosions (MC-GearNet 1.5e6 mean L2) co-occur
    with dead-dim count: a few dimensions dominate the norm, the rest
    are silent, and the optimisation problem is poorly conditioned.

    Dead dims also flag pretraining failures: a 1280-dim ESM with 0
    dead dims is healthier than a 3072-dim MC-GearNet with 507 dead
    dims (close to one full layer-slab worth of nothing).

    How to read the output
    ----------------------
    - mean L2 in [10, 100], std/mean ratio < 1: well-conditioned.
    - mean L2 in millions: norm explosion. Cross-reference dim std
      range and condition number (Q3.2).
    - dead_dims > 5% of dim_total: a slab of the architecture is
      contributing nothing. Common with concat-hidden setups missing
      a final LayerNorm.
    - per-AA norm bias: usually small and biological (CYS, HIS run
      slightly high). Large bias (>2x) points at residue-specific
      learning rather than uniform encoding.

    Correlations
    ------------
    Norm explosion + many dead dims + low effective rank co-occur as
    a single failure cluster (the 'BatchNorm + residual + concat
    without final LN' pathology - MC-GearNet-Edge being the canonical
    case).
    """
    _header("Q3.3 - Embedding Norms & Conditioning")
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
    per_aa = {}
    print("\nPer-AA L2 norms:")
    for aa in range(20):
        m = y == aa
        if m.sum() > 0:
            an = norms[m]
            print(
                f"  {RESIDUE_NAMES[aa]:3s}: mean={an.mean():.4f} "
                f"+/- {an.std():.4f} (n={int(m.sum())})"
            )
            per_aa[RESIDUE_NAMES[aa]] = {
                "mean": float(an.mean()),
                "std": float(an.std()),
                "n": int(m.sum()),
            }
    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "dead_dims": dead_dims,
        "dim_total": int(all_emb.shape[1]),
        "dim_std_min": float(dim_stds.min()),
        "dim_std_max": float(dim_stds.max()),
        "per_aa_norm": per_aa,
    }


# -- Pipeline ---------------------------------------------------------------


def run_pipeline(probe: EncoderProbe, proteins: list, device, *, skip: tuple = ()):
    """Run the standard battery of analyses for a given probe.

    `skip` is a tuple of analysis names to omit. Names:
        distribution, dimensionality, perturbation, rotation, residue_shuffle,
        residue_probe, context, norms, projector, protein_sim, layerwise.

    If `probe.output_dir` is set, writes results.json and (when layerwise_fn
    returns a list of dicts) layerwise.csv.

    `layerwise_fn` may now optionally return a list[dict] with per-layer rows
    (e.g. {"layer": 0, "eff_rank": ..., "mean_norm": ..., "sparsity": ...}).
    Backwards-compatible: a None return is fine and just skips the CSV.

    Analyses run in Q1 -> Q2 -> Q3 order. The output JSON is a dict so
    insertion order is cosmetic; collate.py is order-agnostic.
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

    results: dict = {
        "name": probe.name,
        "capabilities": {
            "is_3d_aware": probe.is_3d_aware,
            "accepts_residue_type": probe.accepts_residue_type,
            "context_mode": probe.context_mode,
        },
        "n_proteins": len(proteins),
        "notes": list(probe.notes),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    all_emb, all_types, per_protein = collect_all_embeddings(probe.embed_fn, proteins)
    results["n_residues"] = int(all_emb.shape[0])
    results["embed_dim"] = int(all_emb.shape[1])

    # ---- Q1. Information content -----------------------------------------
    if "residue_probe" not in skip_set:
        results["residue_probe"] = analyze_residue_discrimination(all_emb, all_types)
    if probe.accepts_residue_type and "residue_shuffle" not in skip_set:
        results["residue_shuffle"] = analyze_residue_shuffle(probe.embed_fn, proteins)
    if probe.is_3d_aware:
        if "perturbation" not in skip_set:
            results["perturbation"] = analyze_perturbation(probe.embed_fn, proteins)
        if "rotation" not in skip_set:
            results["rotation"] = analyze_rotation_invariance(probe.embed_fn, proteins)
    if "context" not in skip_set:
        if probe.context_mode == "structural":
            results["structural_context"] = analyze_structural_context(
                all_emb, all_types, proteins
            )
        elif probe.context_mode == "sequence":
            results["sequence_context"] = analyze_sequence_context(
                probe.embed_fn, proteins
            )
    if "protein_sim" not in skip_set:
        results["protein_similarity"] = analyze_protein_similarity(per_protein)

    # ---- Q2. Saturation / headroom ---------------------------------------
    if "projector" not in skip_set:
        results["projector"] = analyze_projector_saturation(all_emb, all_types, device)

    # ---- Q3. Optimisation conditioning -----------------------------------
    if "distribution" not in skip_set:
        results["distribution"] = analyze_distribution(all_emb)
    if "dimensionality" not in skip_set:
        results["dimensionality"] = analyze_dimensionality(all_emb)
    if "norms" not in skip_set:
        results["norms"] = analyze_norms(all_emb, all_types)

    # ---- Optional: layer-wise --------------------------------------------
    layerwise_rows = None
    if probe.layerwise_fn is not None and "layerwise" not in skip_set:
        layerwise_rows = probe.layerwise_fn(probe.encoder, proteins, device)
        if isinstance(layerwise_rows, list) and layerwise_rows:
            results["layerwise"] = layerwise_rows

    if probe.output_dir:
        os.makedirs(probe.output_dir, exist_ok=True)
        json_path = os.path.join(probe.output_dir, "results.json")
        # Drop large arrays from JSON (keep summary scalars only)
        slim = {k: v for k, v in results.items()}
        if "dimensionality" in slim and isinstance(slim["dimensionality"], dict):
            slim["dimensionality"] = {
                k: v
                for k, v in slim["dimensionality"].items()
                if k not in ("singular_values", "cumulative_variance")
            }
        with open(json_path, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"\nWrote {json_path}")
        # Full SVD spectra to a sidecar file
        if (
            "dimensionality" in results
            and "singular_values" in results["dimensionality"]
        ):
            spec_path = os.path.join(probe.output_dir, "spectrum.json")
            with open(spec_path, "w") as f:
                json.dump(
                    {
                        "singular_values": results["dimensionality"]["singular_values"],
                        "cumulative_variance": results["dimensionality"][
                            "cumulative_variance"
                        ],
                    },
                    f,
                )
        if isinstance(layerwise_rows, list) and layerwise_rows:
            csv_path = os.path.join(probe.output_dir, "layerwise.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(layerwise_rows[0].keys()))
                w.writeheader()
                w.writerows(layerwise_rows)
            print(f"Wrote {csv_path}")

    print("\n" + "=" * 70)
    print(f"DONE - {probe.name} characterization complete")
    print("=" * 70)
    return results
