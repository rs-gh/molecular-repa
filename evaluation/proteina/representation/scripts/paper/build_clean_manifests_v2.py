"""Regenerate cleantrain / xclean train manifests at n=5000.

The original v1 manifests committed under
``evaluation/proteina/representation/results/paper/n{128,256}_*`` were
pre-sampled at n=1000. To match the dirty regime's n_train=5000 (paper-final),
this script writes ``train_clean_v2.json`` (cleantrain) and ``train_v2.json``
(xclean) variants at n=5000, seed=42.

cleantrain
  Filter pool: PDB train chains with NO >=30%-identity / 80%-coverage hit
  to any PDB val chain. Hit source: leakage_audit/pdb_val_vs_pdb_train.m8.

xclean
  No homology filter on the train side (only the eval side is doubly-cleaned).
  Uniform random sample from full train.lmdb.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
AUDIT_DIR = REPO / "evaluation/proteina/representation/results/inputs/leakage_audit"
RESULTS = REPO / "evaluation/proteina/representation/results/paper"

PDB_LMDB = "/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb"
AFDB_LMDB = "/rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/lmdb"

SEED = 42
N_TARGET = 5000


def load_sidecars(lmdb_dir: str):
    """Return (keys: list[bytes], lengths: np.ndarray) in sidecar order.

    Verified ordering: FASTA header at line index N == chain_id for the LMDB
    key at sidecar position N.
    """
    keys = pickle.load(open(f"{lmdb_dir}/train_keys.pkl", "rb"))
    lengths = np.load(f"{lmdb_dir}/train_lengths.npy")
    assert len(keys) == len(
        lengths
    ), f"{lmdb_dir}: keys={len(keys)} lengths={len(lengths)}"
    return keys, lengths


def load_chain_ids(fasta_path: Path, expected_n: int) -> list[str]:
    """Read chain_ids in FASTA order (matches LMDB sidecar order)."""
    ids = []
    with open(fasta_path) as fp:
        for line in fp:
            if line.startswith(">"):
                ids.append(line[1:].strip())
    assert (
        len(ids) == expected_n
    ), f"{fasta_path}: {len(ids)} headers vs expected {expected_n}"
    return ids


def val_homolog_train_chains(m8_path: Path) -> set[str]:
    """Return set of train chain_ids that appear as targets in val→train m8 hits."""
    hits = set()
    with open(m8_path) as fp:
        for line in fp:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 2:
                hits.add(cols[1])  # target = train chain
    return hits


def write_manifest(
    out_path: Path,
    *,
    version: str,
    lmdb_path: str,
    n: int,
    max_size: int,
    seed: int,
    keys: list[str],
    lengths: list[int],
    ids: list[str] | None,
    provenance: dict,
):
    manifest = {
        "manifest_version": version,
        "seed": int(seed),
        "n_proteins": int(n),
        "max_size": int(max_size),
        "lmdb_path": lmdb_path,
        "keys": [k for k in keys],
        "lengths": [int(L) for L in lengths],
    }
    if ids is not None:
        manifest["ids"] = ids
    manifest["_provenance"] = provenance
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {out_path} (n={n}, max_size={max_size})")


def build_cleantrain(max_size: int, out_dir: Path, seed: int = SEED, suffix: str = ""):
    print(f"\n[cleantrain max_size={max_size} seed={seed}] {out_dir}")
    keys, lengths = load_sidecars(PDB_LMDB)
    ids = load_chain_ids(AUDIT_DIR / "pdb_train.fasta", len(keys))
    val_homologs = val_homolog_train_chains(AUDIT_DIR / "pdb_val_vs_pdb_train.m8")
    print(f"  total train chains: {len(keys)}")
    print(f"  val-homologous train chains to drop: {len(val_homologs)}")

    # Indices passing both filters: not in val-homologs AND length<=max_size
    eligible = [
        i
        for i, cid in enumerate(ids)
        if cid not in val_homologs and lengths[i] <= max_size
    ]
    print(f"  eligible (clean, length<=max_size): {len(eligible)}")
    assert len(eligible) >= N_TARGET, f"eligible pool {len(eligible)} < {N_TARGET}"

    rng = random.Random(seed)
    sampled = sorted(rng.sample(eligible, N_TARGET))

    write_manifest(
        out_dir / f"batch_manifest_train_clean_v2{suffix}.json",
        version=f"train_clean_v2{suffix}",
        lmdb_path=f"{PDB_LMDB}/train.lmdb",
        n=N_TARGET,
        max_size=max_size,
        seed=seed,
        keys=[keys[i].decode() for i in sampled],
        lengths=[int(lengths[i]) for i in sampled],
        ids=[ids[i] for i in sampled],
        provenance={
            "audit": "mmseqs easy-search pdb_val.fasta pdb_train.fasta --min-seq-id 0.3 -c 0.8 --cov-mode 0",
            "filter": "train chains with NO >=30% identity / 80% coverage hit from any val chain",
            "audit_date": "2026-05-23",
            "n_leaked_dropped": len(val_homologs),
            "n_eligible_after_filter": len(eligible),
            "seed_note": f"seed={seed} used to subsample N={N_TARGET} from eligible clean pool",
            "regen_date": "2026-05-26",
            "regen_reason": "paper-final: match dirty regime's n_train=5000",
        },
    )


def build_xclean(
    lmdb_dir: str,
    max_size: int,
    out_dir: Path,
    label: str,
    seed: int = SEED,
    suffix: str = "",
):
    print(f"\n[xclean {label} max_size={max_size} seed={seed}] {out_dir}")
    keys, lengths = load_sidecars(lmdb_dir)
    eligible = [i for i, L in enumerate(lengths) if L <= max_size]
    print(f"  eligible (length<=max_size): {len(eligible)} / {len(keys)}")
    assert len(eligible) >= N_TARGET, f"eligible pool {len(eligible)} < {N_TARGET}"

    rng = random.Random(seed)
    sampled = sorted(rng.sample(eligible, N_TARGET))

    write_manifest(
        out_dir / f"batch_manifest_train_v2{suffix}.json",
        version=f"train_v2{suffix}",
        lmdb_path=f"{lmdb_dir}/train.lmdb",
        n=N_TARGET,
        max_size=max_size,
        seed=seed,
        keys=[keys[i].decode() for i in sampled],
        lengths=[int(lengths[i]) for i in sampled],
        ids=None,
        provenance={
            "filter": "uniform random sample from full train.lmdb (no homology filter on train side)",
            "seed_note": f"seed={seed} used to subsample N={N_TARGET}",
            "regen_date": "2026-05-26",
            "regen_reason": "paper-final: match dirty regime's n_train=5000",
            "src": label,
        },
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="subsample seed; default 42 reproduces the paper-final manifests",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="version/filename suffix; default '' for seed 42, else '_s{seed}'",
    )
    ap.add_argument(
        "--figure-only",
        action="store_true",
        help="build only the two n256 regimes used by report Fig 6.2 "
        "(n256_convergence_cleantrain_pdb + n256_xclean_afdb_pdb)",
    )
    args = ap.parse_args()
    seed = args.seed
    suffix = (
        args.suffix if args.suffix is not None else ("" if seed == 42 else f"_s{seed}")
    )

    if args.figure_only:
        # The two n256 regimes that feed report Fig 6.2 (IF/dih from xclean, CATH
        # from cleantrain). Used to add new-seed train manifests for the seed band.
        build_cleantrain(256, RESULTS / "n256_convergence_cleantrain_pdb", seed, suffix)
        build_xclean(
            AFDB_LMDB, 256, RESULTS / "n256_xclean_afdb_pdb", "afdb_train", seed, suffix
        )
        return

    # cleantrain: PDB train minus val-homologs
    build_cleantrain(128, RESULTS / "n128_convergence_cleantrain_pdb", seed, suffix)
    build_cleantrain(256, RESULTS / "n256_convergence_cleantrain_pdb", seed, suffix)

    # xclean: uniform sample of train.lmdb (no filter)
    # n128/n256 xclean_afdb_pdb: PDB-trained model -> AFDB train pool -> AFDB val clean
    build_xclean(
        AFDB_LMDB, 128, RESULTS / "n128_xclean_afdb_pdb", "afdb_train", seed, suffix
    )
    build_xclean(
        AFDB_LMDB, 256, RESULTS / "n256_xclean_afdb_pdb", "afdb_train", seed, suffix
    )
    # n256 xclean_pdb_afdb: AFDB-trained model -> PDB train pool -> PDB val clean
    build_xclean(
        PDB_LMDB, 256, RESULTS / "n256_xclean_pdb_afdb", "pdb_train", seed, suffix
    )


if __name__ == "__main__":
    main()
