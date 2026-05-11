# ruff: noqa: E402
"""One-off CATH classifier artifact builder for the generation eval suite.

Trains a single linear logistic-regression CATH head on top of GearNet
``protein_feature`` embeddings of train.lmdb proteins, validates on
val.lmdb, and writes a single pickle bundle that the generation eval
suite (``compute_cath_metrics``) consumes::

    eval_output/_artifacts/cath_classifier/cath_gearnet_T.pkl
        head            sklearn LogisticRegression
        train_meta      dict with vocab, level, in_dim, ...
        train_dist      dict[class_str -> freq] (used for KL in gen eval)
        gearnet_ckpt    path used to embed train proteins
        train_manifest  manifest tag from build_or_load_manifest
        eval_manifest   manifest tag for val.lmdb
        val_metrics     dict of accuracy / macro_f1 / top1_conf_mean on val

Usage (smoke):
    python evaluation/proteina/representation/scripts/build_cath_classifier.py \
        --n_train 200 --n_eval 100 --device cpu

Production (intr GPU):
    sbatch hpc-scripts/proteina/evaluation/representation/build_cath_classifier.sh
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lib.checkpoints import LMDB_PATH
from lib.data import _default_device
from lib.manifest import build_or_load_manifest, load_proteina_batch_from_manifest
from lib.probes.cath_pretrained import (
    eval_cath_probe,
    train_cath_probe,
)


# Default GearNet checkpoint used by the FID / fS / fJSD metric factory. The
# generation eval suite consumes the SAME checkpoint, so train and inference
# embeddings live in the same space - that's the contract this script keeps.
DEFAULT_GEARNET_CKPT = os.environ.get(
    "GEARNET_CA_CKPT",
    "/rds/user/sr2173/hpc-work/proteina/data/metric_factory/model_weights/gearnet_ca.pth",
)


def _lmdb_data_to_gearnet_data(g: Data) -> Data:
    """Convert an LMDB Data record to the field shape GearNet expects.

    LMDB records (per ``bake_cath_labels.py``) store:
        coords      [N, 37, 3]   OpenFold ordering, CA at index 1
        mask        [N]          bool, True = real residue
    GearNet's ``forward`` reads:
        batch.coord_mask  [N, 37] bool
        batch.coords      [N, 37, 3]
        batch.batch       PyG batch index (added by Batch.from_data_list)
        batch.node_id     used by PyG to infer num_nodes when no other
                          *node-level* attribute is set explicitly

    For CA-only GearNet only ``coord_mask[:, 1]`` matters - the model
    explicitly zeros out indices 0, 2..36 inside ``forward``. So we set CA
    valid where the residue is real and leave the rest zero.
    """
    coords = g.coords.float()
    n_res = coords.shape[0]
    coord_mask = torch.zeros(n_res, 37, dtype=torch.bool)
    coord_mask[:, 1] = (
        g.mask.bool() if hasattr(g, "mask") else torch.ones(n_res, dtype=torch.bool)
    )
    return Data(
        coords=coords,
        coord_mask=coord_mask,
        node_id=torch.arange(n_res).unsqueeze(-1),
    )


@torch.no_grad()
def _embed_with_gearnet(
    raw: List[Data],
    gearnet,
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Run GearNet over the raw proteins, return a [B, D] CPU tensor of protein_feature."""
    items = [_lmdb_data_to_gearnet_data(g) for g in raw]
    loader = DataLoader(items, batch_size=batch_size, shuffle=False)
    feats: List[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        out = gearnet(batch)
        feats.append(out["protein_feature"].detach().cpu())
    return torch.cat(feats, dim=0)


def _train_cath_distribution(raw: List[Data], level: str) -> Dict[str, float]:
    """Frequency distribution over level-truncated CATH classes in ``raw``.

    Used as the reference distribution for KL(generated || train) in
    ``compute_cath_metrics``. Unlabelled proteins are dropped from the
    denominator so the distribution is over labelled-only proteins.
    """
    from lib.labels import cath_labels_from_raw

    y, uniq = cath_labels_from_raw(raw, level=level)
    c = Counter(int(v) for v in y if v >= 0)
    total = sum(c.values())
    if total == 0:
        return {}
    return {uniq[i]: cnt / total for i, cnt in c.items()}


def _load_gearnet(ckpt_path: str, device: str):
    """Load the same NoTrainCAGearNet the FID metric uses. Returns model on device."""
    from proteinfoundation.metrics.gearnet_utils import NoTrainCAGearNet

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"GearNet checkpoint not found at {ckpt_path}. Set $GEARNET_CA_CKPT or "
            f"pass --gearnet_ckpt."
        )
    print(f"[gearnet] loading {ckpt_path}")
    model = NoTrainCAGearNet(ckpt_path)
    model = model.to(device)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_size", type=int, default=512)
    ap.add_argument("--cath_level", type=str, default="T", choices=["C", "A", "T", "H"])
    ap.add_argument("--cath_min_per_class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest_seed", type=int, default=42)
    ap.add_argument(
        "--train_lmdb_path",
        type=str,
        default=None,
        help="Path to train.lmdb. Defaults to sibling of $PROBES_LMDB_PATH.",
    )
    ap.add_argument(
        "--eval_lmdb_path",
        type=str,
        default=None,
        help="Path to val.lmdb. Defaults to LMDB_PATH from lib.checkpoints.",
    )
    ap.add_argument("--gearnet_ckpt", type=str, default=DEFAULT_GEARNET_CKPT)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle path. Defaults to "
        "evaluation/proteina/representation/results/cath_classifier/cath_gearnet_<level>.pkl",
    )
    ap.add_argument(
        "--manifest_outdir",
        type=str,
        default=None,
        help="Where to write manifests. Defaults to alongside --output.",
    )
    # Versions encode "labelled-only" so old non-labelled caches don't get
    # silently reused if someone flips require_cath off and on.
    ap.add_argument(
        "--train_manifest_version", type=str, default="cath_train_labelled_v1"
    )
    ap.add_argument(
        "--eval_manifest_version", type=str, default="cath_eval_labelled_v1"
    )
    args = ap.parse_args()

    device = args.device or _default_device()
    train_lmdb = args.train_lmdb_path or str(Path(LMDB_PATH).parent / "train.lmdb")
    eval_lmdb = args.eval_lmdb_path or str(LMDB_PATH)

    out_path = Path(
        args.output
        or HERE.parents[1]
        / "results"
        / "inputs"
        / "cath_classifier"
        / f"cath_gearnet_{args.cath_level}.pkl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_outdir = (
        Path(args.manifest_outdir) if args.manifest_outdir else out_path.parent
    )

    print(f"Train LMDB: {train_lmdb}")
    print(f"Eval  LMDB: {eval_lmdb}")
    print(f"GearNet ckpt: {args.gearnet_ckpt}")
    print(f"N_train={args.n_train}, N_eval={args.n_eval}, max_size={args.max_size}")
    print(f"CATH level: {args.cath_level}, min_per_class={args.cath_min_per_class}")
    print(f"Device: {device}")
    print(f"Output: {out_path}")

    # Build the manifests once. Sample-size is bounded by n_train/n_eval; the
    # build step here is fast (< 1 min for n=5000) and the result is cached on
    # disk for re-runs.
    print("\n[manifest] train (labelled-only)")
    train_manifest = build_or_load_manifest(
        outdir=manifest_outdir,
        version=args.train_manifest_version,
        lmdb_path=train_lmdb,
        n=args.n_train,
        max_size=args.max_size,
        seed=args.manifest_seed,
        require_cath=True,
    )
    print("[manifest] eval (labelled-only)")
    eval_manifest = build_or_load_manifest(
        outdir=manifest_outdir,
        version=args.eval_manifest_version,
        lmdb_path=eval_lmdb,
        n=args.n_eval,
        max_size=args.max_size,
        seed=args.manifest_seed,
        require_cath=True,
    )
    train_manifest["lmdb_path"] = train_lmdb
    eval_manifest["lmdb_path"] = eval_lmdb

    print("\n[load] eval batch + raw...")
    _, raw_eval = load_proteina_batch_from_manifest(eval_manifest, device="cpu")
    print(f"  {len(raw_eval)} proteins")

    print("[load] train batch + raw...")
    _, raw_train = load_proteina_batch_from_manifest(train_manifest, device="cpu")
    print(f"  {len(raw_train)} proteins")

    # Sanity: manifest is labelled-only by construction. This assertion catches
    # a stale cached manifest or a record where bake_cath_labels.py somehow
    # left the cath_code field empty after sampling.
    n_train_lab = sum(1 for g in raw_train if getattr(g, "cath_code", None))
    n_eval_lab = sum(1 for g in raw_eval if getattr(g, "cath_code", None))
    print(
        f"\n[cath] train labelled: {n_train_lab}/{len(raw_train)}; "
        f"eval labelled: {n_eval_lab}/{len(raw_eval)} (expect 100% - labelled manifest)"
    )
    if n_train_lab != len(raw_train) or n_eval_lab != len(raw_eval):
        raise RuntimeError(
            "Labelled-only manifest contains unlabelled records - possibly stale cache. "
            "Delete the manifest JSON in --manifest_outdir and re-run."
        )

    # Embed via GearNet - this is the dominant time cost (~30s / 1000 proteins on intr GPU).
    gearnet = _load_gearnet(args.gearnet_ckpt, device=device)

    print("\n[embed] train...")
    t0 = time.time()
    feats_train = _embed_with_gearnet(
        raw_train, gearnet, device=device, batch_size=args.batch_size
    )
    print(f"  {feats_train.shape} in {time.time() - t0:.1f}s")

    print("[embed] eval...")
    t0 = time.time()
    feats_eval = _embed_with_gearnet(
        raw_eval, gearnet, device=device, batch_size=args.batch_size
    )
    print(f"  {feats_eval.shape} in {time.time() - t0:.1f}s")

    # Free GearNet so the rest of the script doesn't hold the GPU hostage.
    del gearnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[fit] CATH probe...")
    head, train_meta = train_cath_probe(
        feats_train,
        mask=None,
        raw=raw_train,
        level=args.cath_level,
        head_type="linear",
        min_per_class=args.cath_min_per_class,
        seed=args.seed,
    )
    if head is None:
        raise RuntimeError(
            f"train_cath_probe returned no head - too few classes survived "
            f"min_per_class={args.cath_min_per_class} at level={args.cath_level}. "
            f"Increase --n_train or lower --cath_min_per_class."
        )
    print(
        f"  fit on {train_meta['n_train']} proteins, vocab size {len(train_meta['vocab'])}"
    )

    print("[eval] on val.lmdb manifest...")
    res = eval_cath_probe(head, train_meta, feats_eval, mask=None, raw=raw_eval)
    print(
        f"  level={res.level}  acc={res.accuracy:.3f}  macro_f1={res.macro_f1:.3f}  "
        f"top1_conf={res.top1_conf_mean:.3f}  oov={res.n_eval_oov}/{res.n_eval}"
    )

    train_dist = _train_cath_distribution(raw_train, level=args.cath_level)

    bundle = {
        "head": head,
        "train_meta": train_meta,
        "train_dist": train_dist,
        "level": args.cath_level,
        "encoder": "gearnet_ca",
        "gearnet_ckpt": args.gearnet_ckpt,
        "train_lmdb": train_lmdb,
        "eval_lmdb": eval_lmdb,
        "train_manifest": args.train_manifest_version,
        "eval_manifest": args.eval_manifest_version,
        "n_train_proteins": int(feats_train.shape[0]),
        "n_eval_proteins": int(feats_eval.shape[0]),
        "embed_dim": int(feats_train.shape[1]),
        "min_per_class": int(args.cath_min_per_class),
        "seed": int(args.seed),
        "val_metrics": {
            "accuracy": res.accuracy,
            "macro_f1": res.macro_f1,
            "top1_conf_mean": res.top1_conf_mean,
            "n_classes": res.n_classes,
            "n_eval": res.n_eval,
            "n_eval_in_vocab": res.n_eval_in_vocab,
            "n_eval_oov": res.n_eval_oov,
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[out] wrote {out_path}")

    # Also dump a small JSON sidecar for at-a-glance inspection without
    # unpickling - paths, sizes, and val metrics, no tensors.
    sidecar = {
        k: v for k, v in bundle.items() if k not in ("head", "train_meta", "train_dist")
    }
    sidecar["vocab_preview"] = train_meta["vocab"][:10]
    sidecar["vocab_size"] = len(train_meta["vocab"])
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    print(f"[out] wrote {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
