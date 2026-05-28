"""Experiment B: does the FIXED MPNN encoder represent SS differently on PDB vs AFDB?

Hypothesis (from proteina_narratives Q2): REPA-MPNN pulls AFDB→helix but PDB→sheet
because the same encoder represents different SS classes most saliently depending on
the structure population it sees. Test: train a per-residue linear probe SS→{H,E,C}
on MPNN embeddings, SEPARATELY on PDB and AFDB structures. Compare per-class recall
and a per-class linear separability score.

If on AFDB the helix class is most separable (and on PDB the sheet class is),
that maps onto the directional pull and explains the divergence.

CA-only throughout (MPNN encoder is CA-based; biotite annotate_sse is CA P-SEA).
Runs on GPU for the encoder forward; probe is sklearn (CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
PROFILING_ROOT = HERE.parents[1]  # encoder_profiling/proteina
sys.path.insert(0, str(PROFILING_ROOT))
sys.path.insert(0, str(HERE.parents[3] / "src/proteina"))

from proteinfoundation.repa.mpnn_encoder import ProteinMPNNPerResidueEncoder  # noqa: E402
from _probes.lib import load_proteins, make_embed_fn  # noqa: E402

DATA = "/rds/user/sr2173/hpc-work/proteina/data"
LMDBS = {
    "PDB": f"{DATA}/pdb_train/lmdb/train.lmdb",
    "AFDB": f"{DATA}/afdb_swissprot/lmdb/train.lmdb",
}
MPNN_CKPT = os.path.join(
    os.environ.get(
        "PROTEINMPNN_WEIGHTS_DIR", "/home/sr2173/rds/hpc-work/proteina/ProteinMPNN"
    ),
    "ca_model_weights/v_48_020.pt",
)
SS_CODE = {"a": 0, "b": 1, "c": 2}  # helix, sheet, coil (biotite annotate_sse)
SS_NAME = ["helix", "sheet", "coil"]


def ca_ss(ca_ang: np.ndarray) -> np.ndarray | None:
    """Per-residue SS labels (0/1/2) from CA coords via biotite P-SEA."""
    import biotite.structure as struc
    from biotite.structure import annotate_sse

    n = ca_ang.shape[0]
    atoms = struc.AtomArray(n)
    atoms.coord = ca_ang.astype(np.float32)
    atoms.chain_id = np.full(n, "A")
    atoms.res_id = np.arange(1, n + 1)
    atoms.res_name = np.full(n, "GLY")
    atoms.atom_name = np.full(n, "CA")
    atoms.element = np.full(n, "C")
    try:
        sse = annotate_sse(atoms)  # array of 'a'/'b'/'c', length = n_residues
    except Exception:
        return None
    if len(sse) != n:
        # annotate_sse can return per-residue; pad/truncate defensively
        if len(sse) < n:
            return None
    return np.array([SS_CODE.get(s, 2) for s in sse[:n]], dtype=np.int64)


@torch.no_grad()
def collect(encoder, embed_fn, proteins, device, max_res=40000):
    embs, labels = [], []
    n_res = 0
    for g in proteins:
        ca_ang = g.coords[:, 1, :].float().numpy()  # [n,3] Angstrom
        mask = g.coord_mask[:, 1].bool().numpy()
        ca_ang = ca_ang[mask]
        if ca_ang.shape[0] < 8:
            continue
        ss = ca_ss(ca_ang)
        if ss is None:
            continue
        ca_nm = torch.tensor(
            ca_ang / 10.0, dtype=torch.float32, device=device
        ).unsqueeze(0)
        m = torch.ones(1, ca_ang.shape[0], dtype=torch.float32, device=device)
        emb = embed_fn(ca_nm, m, None)[0].float().cpu().numpy()  # [n, dim]
        if emb.shape[0] != ss.shape[0]:
            k = min(emb.shape[0], ss.shape[0])
            emb, ss = emb[:k], ss[:k]
        embs.append(emb)
        labels.append(ss)
        n_res += emb.shape[0]
        if n_res >= max_res:
            break
    return np.concatenate(embs), np.concatenate(labels)


def probe(X, y, seed=42):
    """3-class linear probe; return overall acc + per-class recall + macro F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, f1_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    rec = recall_score(yte, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "acc": float(accuracy_score(yte, pred)),
        "macro_f1": float(
            f1_score(yte, pred, average="macro", labels=[0, 1, 2], zero_division=0)
        ),
        "recall_helix": float(rec[0]),
        "recall_sheet": float(rec[1]),
        "recall_coil": float(rec[2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=400)
    ap.add_argument("--max-res", type=int, default=40000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = ProteinMPNNPerResidueEncoder(ckpt_path=MPNN_CKPT).eval().to(device)
    embed_fn = make_embed_fn(encoder, device)
    print(f"MPNN encoder dim {encoder.encoder_dim}, device {device}")

    results = {}
    for ds, lmdb in LMDBS.items():
        print(f"\n=== {ds} ({lmdb}) ===")
        proteins = load_proteins(lmdb, args.n_proteins, seed=42)
        X, y = collect(encoder, embed_fn, proteins, device, args.max_res)
        dist = Counter(y.tolist())
        print(
            f"  residues: {len(y)}  SS dist: helix={dist[0]} sheet={dist[1]} coil={dist[2]}"
        )
        m = probe(X, y)
        m["n_res"] = int(len(y))
        m["ss_frac"] = {SS_NAME[k]: dist[k] / len(y) for k in (0, 1, 2)}
        results[ds] = m
        print(
            f"  probe acc={m['acc']:.3f} macroF1={m['macro_f1']:.3f} | "
            f"recall helix={m['recall_helix']:.3f} sheet={m['recall_sheet']:.3f} coil={m['recall_coil']:.3f}"
        )

    out = PROFILING_ROOT / "mpnn/results/ss_probe_cross_dataset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    # Headline comparison
    print("\n=== Hypothesis check ===")
    for ds in ("PDB", "AFDB"):
        r = results[ds]
        best = max(
            [("helix", r["recall_helix"]), ("sheet", r["recall_sheet"])],
            key=lambda x: x[1],
        )
        print(
            f"  {ds}: most-separable SS (helix vs sheet) = {best[0]} (recall {best[1]:.3f}); "
            f"helix-minus-sheet recall = {r['recall_helix']-r['recall_sheet']:+.3f}"
        )
    print(
        "  Predicts: AFDB helix>sheet separability, PDB sheet>helix → explains MPNN α-pull on AFDB, β-pull on PDB."
    )


if __name__ == "__main__":
    main()
