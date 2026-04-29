"""Characterize ESM-2 (650M) per-residue encoder.

Sequence-only model — coords are ignored by the encoder, so 3D-sensitivity and
rotation tests are skipped (set is_3d_aware=False). The standard `context` test
is replaced with the sequence-context variant (mutate flanks, keep center).

The layerwise pass pulls all 33 hidden states in one forward (output_hidden_states),
so it diverges from the per-layer-iteration pattern used by the GearNet drivers.

Run:
  python playground/proteina/esm/explore_esm.py [--quick] [--n-proteins 200] [--layer 33]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))
sys.path.insert(0, str(REPO_ROOT / "playground" / "proteina"))

from _encoder_probes import (  # noqa: E402
    EncoderProbe,
    load_proteins,
    make_embed_fn,
    run_pipeline,
)
from _encoder_probes.lib import graph_to_inputs  # noqa: E402

from proteinfoundation.repa.esm_encoder import ESMPerResidueEncoder  # noqa: E402


DATA_PATH = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
LMDB_PATH = os.path.join(DATA_PATH, "pdb_train/lmdb/train.lmdb")
MODEL_ID = os.environ.get("ESM_MODEL_ID", "facebook/esm2_t33_650M_UR50D")


def setup_encoder(device, layer=None):
    print(f"Loading ESM-2 from {MODEL_ID} (layer={layer})")
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
def layerwise_fn(encoder, proteins, device, n_test=30, max_residues=10000):
    """Pull all (n_layers + 1) hidden states in one forward and report
    eff-rank / mean-norm / sparsity / AA-probe accuracy per layer.

    Most actionable: tells us which layer `repa_layer` should target.
    """
    print("\n" + "=" * 70)
    print("ESM-2 Layer-wise Analysis")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    n_layers = encoder.esm.config.num_hidden_layers
    print(
        f"  {n_layers} transformer layers (+1 embedding => {n_layers + 1} hidden states)"
    )

    per_layer_feats = [[] for _ in range(n_layers + 1)]
    per_layer_types = []
    total_residues = 0
    n_test = min(n_test, len(proteins))
    t0 = time.time()

    for pid, graph in enumerate(proteins[:n_test]):
        ca_nm, mask, rtype = graph_to_inputs(graph)
        mask_d = mask.to(device)
        safe_idx = rtype.clamp(min=0, max=len(encoder.aa_to_esm_token) - 1)
        aa_tokens = encoder.aa_to_esm_token[safe_idx.to(device)]
        aa_tokens = torch.where(
            mask_d, aa_tokens, torch.full_like(aa_tokens, encoder.pad_token_id)
        ).unsqueeze(0)
        cls = torch.full((1, 1), encoder.cls_token_id, dtype=torch.long, device=device)
        eos = torch.full((1, 1), encoder.eos_token_id, dtype=torch.long, device=device)
        input_ids = torch.cat([cls, aa_tokens, eos], dim=1)
        ones = torch.ones((1, 1), dtype=torch.bool, device=device)
        attn = torch.cat([ones, mask_d.unsqueeze(0), ones], dim=1).long()

        out = encoder.esm(
            input_ids=input_ids, attention_mask=attn, output_hidden_states=True
        )
        n_valid = int(mask_d.sum())
        if n_valid == 0:
            continue
        for li, hs in enumerate(out.hidden_states):
            feats = hs[0, 1:-1, :]  # strip CLS/EOS
            per_layer_feats[li].append(feats[mask_d].cpu().float())
        per_layer_types.append(rtype[mask].cpu())
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

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"\n{'Layer':>5} | {'Eff Rank':>9} | {'Mean Norm':>10} | "
        f"{'Sparsity':>9} | {'AA Probe':>9}"
    )
    print("-" * 60)

    for li in range(n_layers + 1):
        feats = torch.cat(per_layer_feats[li], dim=0)[keep]
        centered = feats - feats.mean(dim=0)
        S = torch.linalg.svdvals(centered).numpy()
        p = (S**2) / (S**2).sum()
        p = p[p > 0]
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        mean_norm = float(feats.norm(dim=-1).mean())
        sparsity = float((feats.abs() < 1e-6).float().mean())

        X = feats.numpy()
        n_probe = min(10000, X.shape[0])
        idx = np.random.RandomState(0).permutation(X.shape[0])[:n_probe]
        Xs, ys = X[idx], y[idx]
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xs, ys, test_size=0.2, random_state=42, stratify=ys
        )
        clf = LogisticRegression(max_iter=400, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        probe_acc = accuracy_score(y_te, clf.predict(X_te))
        print(
            f"  {li:>3d} | {eff_rank:>9.1f} | {mean_norm:>10.4f} | "
            f"{sparsity:>8.4f} | {probe_acc:>8.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=200)
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Hidden-states layer for main analyses (None = last_hidden_state).",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke pass: 20 proteins, skip layerwise + context.",
    )
    ap.add_argument("--skip-layerwise", action="store_true")
    ap.add_argument("--random-seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.quick:
        args.n_proteins = min(args.n_proteins, 20)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(
        f"Target layer for main analyses: "
        f"{args.layer if args.layer is not None else 'last_hidden_state'}"
    )

    proteins = load_proteins(LMDB_PATH, args.n_proteins, seed=args.random_seed)
    encoder = setup_encoder(device, layer=args.layer)

    skip = []
    if args.quick or args.skip_layerwise:
        skip.append("layerwise")
    if args.quick:
        skip.append("context")

    probe = EncoderProbe(
        name=f"esm2-650M{f'-L{args.layer}' if args.layer is not None else ''}",
        encoder=encoder,
        embed_fn=make_embed_fn(encoder, device),
        is_3d_aware=False,  # ESM ignores coords
        accepts_residue_type=True,
        context_mode="sequence",  # flanks-mutated context test
        layerwise_fn=layerwise_fn if not (args.quick or args.skip_layerwise) else None,
    )
    run_pipeline(probe, proteins, device, skip=tuple(skip))


if __name__ == "__main__":
    main()
