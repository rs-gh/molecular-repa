"""Bake CATH labels into PDB and/or AFDB LMDBs in-place.

Reads each LMDB entry, looks up graph.id in the CATH mapping, and
writes back with graph.cath_code set.  Entries that have no CATH
match get graph.cath_code = [].  Entries that already have a
non-empty graph.cath_code are SKIPPED (idempotent re-run safety).

Backcompat guarantee: only adds the cath_code field; no other
field is modified or deleted.  Old code that does not read cath_code
is completely unaffected.

Usage:
    source .venv/bin/activate

    # PDB (all splits):
    python hpc-scripts/proteina/data_prep/bake_cath_labels.py \
        --dataset pdb \
        --lmdb_dir  /rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb \
        --cath_dir  /rds/user/sr2173/hpc-work/proteina/data/pdb_train

    # AFDB (requires TED TSV):
    python hpc-scripts/proteina/data_prep/bake_cath_labels.py \
        --dataset afdb \
        --lmdb_dir  /rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/lmdb \
        --ted_tsv   /rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/ted_365m.domain_summary.cath.globularity.taxid.tsv

    # Smoke-test (first 500 entries of train only):
    python hpc-scripts/proteina/data_prep/bake_cath_labels.py \
        --dataset pdb \
        --lmdb_dir  /rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb \
        --cath_dir  /rds/user/sr2173/hpc-work/proteina/data/pdb_train \
        --splits train --max_entries 500 --dry_run
"""

import argparse
import gzip
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import lmdb


# ---------------------------------------------------------------------------
# CATH lookup builders (same logic as analyse_cath_distribution.py)
# ---------------------------------------------------------------------------


def build_pdb_cath_mapping(cath_dir: Path) -> dict:
    sifts_path = cath_dir / "pdb_chain_cath_uniprot.tsv.gz"
    cath_path = cath_dir / "cath-b-newest-all.gz"

    print("  Loading SIFTS pdb_chain->cath_id mapping...", flush=True)
    pdbchain_to_cathid: dict = defaultdict(list)
    with gzip.open(sifts_path, "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pdb, chain, _, cath_id = parts[:4]
            pdbchain_to_cathid[f"{pdb}_{chain}"].append(cath_id)

    print("  Loading CATH cath_id->cath_code mapping...", flush=True)
    cathid_to_code: dict = {}
    with gzip.open(cath_path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cathid_to_code[parts[0]] = parts[2]

    pdbchain_to_codes: dict = {}
    for key, cath_ids in pdbchain_to_cathid.items():
        codes = [cathid_to_code[c] for c in cath_ids if c in cathid_to_code]
        if codes:
            pdbchain_to_codes[key] = codes

    print(f"  PDB CATH map: {len(pdbchain_to_codes):,} chains with codes", flush=True)
    return pdbchain_to_codes


def build_interpro_gene3d_mapping(interpro_tsv: str) -> dict:
    """Return (uniprot_acc->cath_codes) dict from filtered InterPro Gene3D TSV."""
    print(f"  Loading InterPro Gene3D mapping from {interpro_tsv} ...", flush=True)
    sample_to_codes: dict = defaultdict(list)
    with open(interpro_tsv) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sample_to_codes[parts[0]].append(parts[1])
    result = dict(sample_to_codes)
    print(
        f"  InterPro Gene3D map: {len(result):,} accessions with CATH codes", flush=True
    )
    return result


def build_afdb_cath_mapping(ted_tsv: str) -> dict:
    print(f"  Loading TED AFDB CATH mapping from {ted_tsv} ...", flush=True)
    sample_to_codes: dict = defaultdict(list)
    t0 = time.time()
    with open(ted_tsv) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) <= 13:
                continue
            full_id = parts[0]
            sample_id = "_".join(full_id.split("_")[:-1]) if "_" in full_id else full_id
            cath_codes_str = parts[13]
            if cath_codes_str != "-":
                codes = []
                for c in cath_codes_str.split(","):
                    lvls = c.split(".")
                    codes.append(c + ".x" if len(lvls) == 3 else c)
                sample_to_codes[sample_id].extend(codes)
            if (i + 1) % 5_000_000 == 0:
                print(
                    f"    Parsed {i+1:,} TED lines ({time.time()-t0:.0f}s)", flush=True
                )
    print(
        f"  TED map: {len(sample_to_codes):,} entries ({time.time()-t0:.0f}s)",
        flush=True,
    )
    return dict(sample_to_codes)


def afdb_id_to_ted_key(graph_id: str) -> str:
    """'AF-Q9RUA0-F1-model_v6.pdb' -> 'AF-Q9RUA0-F1'"""
    stem = graph_id.replace(".pdb", "")
    parts = stem.split("-")
    if len(parts) >= 4 and parts[-1].startswith("model"):
        stem = "-".join(parts[:-1])
    return stem


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_split(
    lmdb_path: str,
    cath_lookup: dict,
    id_transform=None,
    max_entries: int = None,
    dry_run: bool = False,
    commit_every: int = 2000,
):
    """Add cath_code to every entry in one LMDB split.

    Skips entries that already have a non-empty cath_code (idempotent).
    Entries with no CATH match get cath_code = [].

    Args:
        lmdb_path:    path to .lmdb file
        cath_lookup:  dict mapping lookup_key -> list[str] of CATH codes
        id_transform: optional callable graph.id -> lookup_key
        max_entries:  process at most this many entries (None = all)
        dry_run:      if True, read and lookup but do not write
        commit_every: commit transaction every N writes
    """
    if not os.path.exists(lmdb_path):
        print(f"  Skipping (not found): {lmdb_path}", flush=True)
        return

    map_size = 100 * (1024**3)  # 100 GB - generous headroom for pickle growth

    db = lmdb.open(
        lmdb_path,
        map_size=map_size,
        create=False,
        subdir=False,
        readonly=dry_run,
        lock=not dry_run,
        readahead=False,
        meminit=False,
    )

    t0 = time.time()
    total = skipped_already = written = with_cath = no_cath = 0

    # Pass 1: collect all keys (keys-only scan - does NOT page in values)
    print("    Pass 1: collecting keys...", flush=True)
    with db.begin(write=False) as txn:
        cursor = txn.cursor()
        all_keys = [
            k for k in cursor.iternext(keys=True, values=False) if k != b"__ids__"
        ]
    print(
        f"    Pass 1: {len(all_keys):,} keys collected in {time.time()-t0:.0f}s",
        flush=True,
    )

    if max_entries:
        all_keys = all_keys[:max_entries]

    # Pass 2: batch read-modify-write, committing every `commit_every` entries
    print("    Pass 2: enriching in batches...", flush=True)
    for batch_start in range(0, len(all_keys), commit_every):
        batch = all_keys[batch_start : batch_start + commit_every]
        with db.begin(write=not dry_run) as txn:
            for key in batch:
                value = txn.get(key)
                if value is None:
                    continue
                graph = pickle.loads(value)
                total += 1

                existing = getattr(graph, "cath_code", None)
                if existing is not None and len(existing) > 0:
                    skipped_already += 1
                    continue

                lookup_key = id_transform(graph.id) if id_transform else graph.id
                codes = cath_lookup.get(lookup_key, [])
                graph.cath_code = codes

                if codes:
                    with_cath += 1
                else:
                    no_cath += 1

                if not dry_run:
                    txn.put(key, pickle.dumps(graph))
                    written += 1
        # txn commits here at end of `with` block

        if total // 50_000 > (total - len(batch)) // 50_000:
            elapsed = time.time() - t0
            print(
                f"    Processed {total:,} | written {written:,} | "
                f"with_cath {with_cath:,} | {elapsed:.0f}s",
                flush=True,
            )

    db.close()

    elapsed = time.time() - t0
    mode = "[DRY RUN] " if dry_run else ""
    print(f"  {mode}Done in {elapsed:.1f}s", flush=True)
    print(f"    Total scanned   : {total:,}", flush=True)
    print(f"    Already had code: {skipped_already:,} (skipped)", flush=True)
    print(f"    Written         : {written:,}", flush=True)
    print(
        f"      With CATH     : {with_cath:,} ({100*with_cath/max(written,1):.1f}%)",
        flush=True,
    )
    print(f"      No CATH match : {no_cath:,}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Bake CATH labels into LMDB in-place")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["pdb", "afdb"],
        help="Which dataset to enrich",
    )
    parser.add_argument(
        "--lmdb_dir", required=True, help="Directory containing {split}.lmdb files"
    )
    parser.add_argument(
        "--cath_dir",
        default=None,
        help="Dir with pdb_chain_cath_uniprot.tsv.gz + cath-b-newest-all.gz (PDB only)",
    )
    parser.add_argument(
        "--ted_tsv",
        default=None,
        help="(Deprecated) Path to TED domain summary TSV (AFDB only)",
    )
    parser.add_argument(
        "--interpro_tsv",
        default=None,
        help="Path to interpro_gene3d_swissprot.tsv (preferred for AFDB)",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Process at most N entries per split (smoke-test)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Read and lookup but do not write back to LMDB",
    )
    parser.add_argument(
        "--commit_every",
        type=int,
        default=2000,
        help="Commit transaction every N entries",
    )
    args = parser.parse_args()

    # --- Build lookup ---
    print("\n[1/2] Building CATH lookup...", flush=True)
    if args.dataset == "pdb":
        if not args.cath_dir:
            print("ERROR: --cath_dir required for --dataset pdb", file=sys.stderr)
            sys.exit(1)
        cath_lookup = build_pdb_cath_mapping(Path(args.cath_dir))
        id_transform = None
    else:
        if args.interpro_tsv:
            cath_lookup = build_interpro_gene3d_mapping(args.interpro_tsv)
        elif args.ted_tsv:
            cath_lookup = build_afdb_cath_mapping(args.ted_tsv)
        else:
            print(
                "ERROR: --interpro_tsv (or --ted_tsv) required for --dataset afdb",
                file=sys.stderr,
            )
            sys.exit(1)

        # InterPro lookup key is the raw UniProt accession: Q9RUA0
        # AFDB graph.id is AF-Q9RUA0-F1-model_v6.pdb -> extract middle part
        def id_transform(gid):
            return gid.replace(".pdb", "").split("-")[1]

    # --- Enrich splits ---
    print(
        f"\n[2/2] Enriching {args.dataset.upper()} LMDB splits "
        f"{'(DRY RUN) ' if args.dry_run else ''}...",
        flush=True,
    )
    for split in args.splits:
        lmdb_path = os.path.join(args.lmdb_dir, f"{split}.lmdb")
        print(f"\n  Split: {split} ({lmdb_path})", flush=True)
        enrich_split(
            lmdb_path,
            cath_lookup,
            id_transform=id_transform,
            max_entries=args.max_entries,
            dry_run=args.dry_run,
            commit_every=args.commit_every,
        )

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
