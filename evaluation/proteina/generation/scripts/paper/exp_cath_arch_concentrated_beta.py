"""TODO from proteina_narratives_handoff_2026-05-27.md:
"CATH-A label inspection of REPA's concentrated β-rich folds — Exp 1 confirms
concentration, but which architecture(s)? Likely a single CATH-A class."

For each (run, step) in CASES, take the *β≥25* designable subset, embed each
PDB through the same GearNet checkpoint used by build_cath_classifier, score
via the frozen CATH-T head, then aggregate predictions up to CATH-A (first
two dots of the T code: "2.40.10" → "2.40").

Output: per-case A-class distribution → JSON. If REPA's β-rich subset is
dominated by a single A-class, we have the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[5]
sys.path.insert(0, str(REPO_ROOT / "src/proteina"))
sys.path.insert(0, str(REPO_ROOT / "src/proteina/proteinfoundation"))

# Patch sys.modules with pure-torch scatter/cluster shims BEFORE anything pulls
# in proteinfoundation internals (gearnet_utils does `from torch_scatter import
# ...`, whose CUDA wheel has an ABI mismatch against this cluster's torch).
import proteinfoundation.repa.pyg_compat  # noqa: E402,F401

EVAL_OUT = REPO_ROOT / "eval_output"
SAMPLER_TAG = "sde_n0.45"
BETA_MIN = 0.25
CLF_PATH = (
    REPO_ROOT
    / "evaluation/proteina/representation/results/inputs/cath_classifier/cath_gearnet_T.pkl"
)

# Focus on the configurations confirmed concentrated by Exp 1 (β≥25 pwTM ≥ 0.6).
# Each entry is (case_label, run_step_dir_prefix).
CASES = [
    # PDB GearNet-REPA: heavy concentration
    ("pdb_REPA_L9_GN_700K", "repa_l9_256_per_residue_bs24_2gpu_step700k_step_700000"),
    ("pdb_REPA_L9_GN_900K", "repa_l9_256_per_residue_bs24_2gpu_step900k_step_900000"),
    (
        "pdb_REPA_L9_GN_1000K",
        "repa_l9_256_per_residue_bs24_2gpu_step1000k_step_1000000",
    ),
    ("pdb_REPA_L4_GN_700K", "repa_l4_256_per_residue_bs24_2gpu_step700k_step_700000"),
    ("pdb_REPA_L9_MPNN_700K", "repa_mpnn_l9_256_per_residue_step700k_step_700000"),
    ("pdb_REPA_L9_MPNN_1000K", "repa_mpnn_l9_256_per_residue_step1000k_step_1000000"),
    # AFDB GearNet-REPA: concentrated
    ("afdb_REPA_L9_GN_700K", "repa_l9_afdb_256_step700k_step_700000"),
    ("afdb_REPA_L9_GN_900K", "repa_l9_afdb_256_step900k_step_900000"),
    ("afdb_REPA_L4_GN_700K", "repa_l4_afdb_256_step700k_step_700000"),
    ("afdb_REPA_L4_GN_1.0M", "repa_l4_afdb_256_step1000k_step_1000000"),
    # Baselines as controls (β-rich should be diverse → many A-classes)
    ("pdb_baseline_1.0M", "baseline_256_bs24_2gpu_step1000k_step_1000000"),
    ("pdb_baseline_1.6M", "baseline_256_bs24_2gpu_step1600k_step_1600000"),
    ("afdb_baseline_700K", "baseline_afdb_256_step700k_step_700000"),
    # MPNN-AFDB falsifier (β-rich was NOT concentrated — should be diverse)
    ("afdb_REPA_L9_MPNN_700K", "repa_mpnn_l9_afdb_256_step700k_step_700000"),
    ("afdb_REPA_L9_MPNN_1.0M", "repa_mpnn_l9_afdb_256_step1000k_step_1000000"),
]


def collect_beta_rich_pdbs(run_step_prefix: str) -> list[str]:
    """Return absolute paths to designable PDBs with β-fraction ≥ BETA_MIN."""
    paths = []
    pat = str(
        EVAL_OUT
        / f"inference_paper_inference_fid_60m_paper_sweep_{run_step_prefix}__{SAMPLER_TAG}__rep*"
    )
    for rep_dir in sorted(glob(pat)):
        ss_npz = Path(rep_dir) / "ss_cache/ss_fractions.npz"
        di_csv = Path(rep_dir) / "designability_index.csv"
        if not ss_npz.exists() or not di_csv.exists():
            continue
        npz = np.load(ss_npz, allow_pickle=True)
        fracs = npz["fracs"]
        ss_paths = [os.path.abspath(str(p)) for p in npz["paths"]]
        path_to_beta = {p: float(fracs[i, 1]) for i, p in enumerate(ss_paths)}
        di = pd.read_csv(di_csv)
        for _, r in di.iterrows():
            if not r["designable"]:
                continue
            abs_p = os.path.abspath(os.path.join(rep_dir, r["pdb_path"]))
            b = path_to_beta.get(abs_p)
            if b is not None and b >= BETA_MIN:
                paths.append(abs_p)
    return paths


def to_arch(t_code: str) -> str:
    """CATH T → A: keep first two dots. '2.40.10' → '2.40'."""
    parts = t_code.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else t_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_per_case", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import pickle

    with open(CLF_PATH, "rb") as f:
        bundle = pickle.load(f)
    head = bundle["head"]
    vocab = bundle["train_meta"]["vocab"]
    gearnet_ckpt = bundle["gearnet_ckpt"]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  gearnet_ckpt={gearnet_ckpt}", flush=True)
    print(f"vocab T-classes: {vocab}", flush=True)

    from proteinfoundation.metrics.gearnet_utils import NoTrainCAGearNet
    from proteinfoundation.metrics.metric_factory import DatasetWrapper
    from torch_geometric.loader import DataLoader as _PyGLoader

    gearnet = NoTrainCAGearNet(gearnet_ckpt).to(device).eval()

    results = {}
    for case_label, run_step in CASES:
        pdbs = collect_beta_rich_pdbs(run_step)
        if args.max_per_case is not None:
            pdbs = pdbs[: args.max_per_case]
        if not pdbs:
            print(f"{case_label:<24} <no β-rich designable PDBs found>", flush=True)
            results[case_label] = {"n": 0}
            continue
        # Embed
        ds = DatasetWrapper(pdbs)
        loader = _PyGLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        feats = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = gearnet(batch)
                feats.append(out["protein_feature"].detach().cpu())
        feats = torch.cat(feats, dim=0).numpy()
        proba = head.predict_proba(feats)
        pred_idx = head.classes_[proba.argmax(axis=1)]
        top_conf = proba.max(axis=1)
        t_preds = [vocab[i] for i in pred_idx]
        a_preds = [to_arch(t) for t in t_preds]

        t_dist = Counter(t_preds)
        a_dist = Counter(a_preds)
        n = len(t_preds)
        t_top = t_dist.most_common(5)
        a_top = a_dist.most_common(5)
        print(f"{case_label:<24} n={n}  conf_mean={top_conf.mean():.3f}", flush=True)
        print(f"  A-top: {[(a, c, round(c/n,2)) for a,c in a_top]}", flush=True)
        print(f"  T-top: {[(t, c, round(c/n,2)) for t,c in t_top]}", flush=True)
        results[case_label] = {
            "n": n,
            "conf_mean": float(top_conf.mean()),
            "A_dist": dict(a_dist),
            "T_dist": dict(t_dist),
        }

    out_path = (
        REPO_ROOT
        / "evaluation/proteina/generation/results/variance/cath_arch_concentrated_beta.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
