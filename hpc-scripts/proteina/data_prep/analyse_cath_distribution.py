"""Characterise PDB and AFDB datasets by CATH structural family.

Reads train+val splits from both LMDBs, applies CATH label lookups,
and reports coverage + C/A/T-level distributions. Saves results to a
JSON summary and matplotlib figures.

Usage (login node, read-only, no GPU needed):
    source .venv/bin/activate
    python hpc-scripts/proteina/data_prep/analyse_cath_distribution.py \
        --pdb_lmdb_dir  /rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb \
        --afdb_lmdb_dir /rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/lmdb \
        --cath_dir      /rds/user/sr2173/hpc-work/proteina/data/pdb_train \
        --out_dir       evaluation/proteina/cath_distribution \
        [--ted_tsv      /path/to/ted_365m.domain_summary.cath.globularity.taxid.tsv]
        [--splits train val]
        [--max_entries  N]   # limit per split for quick smoke-test

The --ted_tsv argument is optional. If omitted, AFDB CATH stats are skipped.
"""

import argparse
import gzip
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import lmdb


# ---------------------------------------------------------------------------
# CATH lookup helpers (mirrors CATHLabelTransform logic without PyG dep)
# ---------------------------------------------------------------------------


def build_pdb_cath_mapping(cath_dir: Path):
    """Return (pdbchain->cath_codes) dict from SIFTS + CATH files."""
    sifts_path = cath_dir / "pdb_chain_cath_uniprot.tsv.gz"
    cath_path = cath_dir / "cath-b-newest-all.gz"

    if not sifts_path.exists():
        raise FileNotFoundError(f"SIFTS file not found: {sifts_path}")
    if not cath_path.exists():
        raise FileNotFoundError(f"CATH file not found: {cath_path}")

    print("  Loading SIFTS pdb_chain->cath_id mapping...", flush=True)
    pdbchain_to_cathid: dict[str, list[str]] = defaultdict(list)
    with gzip.open(sifts_path, "rt") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pdb, chain, _, cath_id = parts[:4]
            pdbchain_to_cathid[f"{pdb}_{chain}"].append(cath_id)

    print("  Loading CATH cath_id->cath_code mapping...", flush=True)
    cathid_to_code: dict[str, str] = {}
    with gzip.open(cath_path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cathid_to_code[parts[0]] = parts[2]

    # Compose: pdbchain -> list of CATH codes
    pdbchain_to_codes: dict[str, list[str]] = {}
    for key, cath_ids in pdbchain_to_cathid.items():
        codes = [cathid_to_code[c] for c in cath_ids if c in cathid_to_code]
        if codes:
            pdbchain_to_codes[key] = codes

    print(f"  PDB CATH map: {len(pdbchain_to_codes):,} chains with codes", flush=True)
    return pdbchain_to_codes


def build_interpro_gene3d_mapping(interpro_tsv: str) -> dict:
    """Return (uniprot_acc->cath_codes) dict from filtered InterPro Gene3D TSV.

    Expects the output of download_interpro_gene3d.sh:
        uniprot_acc  cath_code  start  end
    """
    print(f"  Loading InterPro Gene3D mapping from {interpro_tsv} ...", flush=True)
    sample_to_codes: dict[str, list[str]] = defaultdict(list)
    with open(interpro_tsv) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            acc, cath_code = parts[0], parts[1]
            sample_to_codes[acc].append(cath_code)
    result = dict(sample_to_codes)
    print(
        f"  InterPro Gene3D map: {len(result):,} accessions with CATH codes", flush=True
    )
    return result


def build_afdb_cath_mapping(ted_tsv: str):
    """Return (afdb_sample_id->cath_codes) dict from TED TSV.

    sample_id = 'AF-Q9RUA0-F1' (drops trailing '-model_v6.pdb').
    """
    print(f"  Loading TED AFDB CATH mapping from {ted_tsv} ...", flush=True)
    sample_to_codes: dict[str, list[str]] = defaultdict(list)
    t0 = time.time()
    with open(ted_tsv) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) <= 13:
                continue
            full_id = parts[0]
            # full_id looks like AF-Q9RUA0-F1-model_v6 or AF-Q9RUA0-F1
            sample_id = "_".join(full_id.split("_")[:-1]) if "_" in full_id else full_id
            cath_codes_str = parts[13]
            if cath_codes_str != "-":
                codes = cath_codes_str.split(",")
                # Pad to 4 levels if needed (TED sometimes gives CAT-level only)
                padded = []
                for c in codes:
                    lvls = c.split(".")
                    if len(lvls) == 3:
                        c = c + ".x"
                    padded.append(c)
                sample_to_codes[sample_id].extend(padded)
            if (i + 1) % 5_000_000 == 0:
                print(
                    f"    Parsed {i+1:,} TED lines ({time.time()-t0:.0f}s)", flush=True
                )
    print(
        f"  TED map: {len(sample_to_codes):,} AFDB entries with CATH codes "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )
    return dict(sample_to_codes)


# ---------------------------------------------------------------------------
# LMDB scan
# ---------------------------------------------------------------------------


def scan_lmdb(lmdb_path: str, cath_lookup: dict, id_transform=None, max_entries=None):
    """Scan one LMDB split and collect per-entry CATH codes.

    Args:
        lmdb_path:    path to .lmdb file
        cath_lookup:  dict mapping entry id -> list[str] of CATH codes
        id_transform: optional callable to convert graph.id -> lookup key
        max_entries:  if set, stop after this many entries (for smoke-test)

    Returns:
        dict with keys: total, with_cath, codes_flat, codes_per_entry
    """
    db = lmdb.open(
        lmdb_path,
        map_size=50 * (1024**3),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,  # sequential full-scan - readahead helps on NVMe
        meminit=False,
    )

    total = 0
    with_cath = 0
    codes_flat: list[str] = []  # one entry per CATH code occurrence
    multi_domain: list[int] = []  # number of domains per entry

    with db.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key == b"__ids__":
                continue
            if max_entries and total >= max_entries:
                break

            graph = pickle.loads(value)
            lookup_key = id_transform(graph.id) if id_transform else graph.id
            codes = cath_lookup.get(lookup_key, [])

            total += 1
            if codes:
                with_cath += 1
                codes_flat.extend(codes)
                multi_domain.append(len(codes))
            else:
                multi_domain.append(0)

            if total % 50_000 == 0:
                print(f"    Scanned {total:,} entries ...", flush=True)

    db.close()
    return {
        "total": total,
        "with_cath": with_cath,
        "codes_flat": codes_flat,
        "multi_domain": multi_domain,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cath_level_counts(codes: list[str], level: int) -> Counter:
    """Extract counts at a given CATH hierarchy level (0=C,1=A,2=T,3=H)."""
    prefixes = []
    for c in codes:
        parts = c.split(".")
        if len(parts) > level:
            prefixes.append(".".join(parts[: level + 1]))
    return Counter(prefixes)


CLASS_NAMES = {
    "1": "Mainly Alpha",
    "2": "Mainly Beta",
    "3": "Alpha/Beta",
    "4": "Few Secondary Structures",
}


def summarise(scan_result: dict, label: str) -> dict:
    total = scan_result["total"]
    covered = scan_result["with_cath"]
    codes = scan_result["codes_flat"]

    class_counts = cath_level_counts(codes, 0)
    arch_counts = cath_level_counts(codes, 1)
    topo_counts = cath_level_counts(codes, 2)

    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total entries   : {total:,}", flush=True)
    print(
        f"  With CATH label : {covered:,} ({100*covered/max(total,1):.1f}%)", flush=True
    )
    print(
        f"  Total CATH codes: {len(codes):,} (multi-domain entries counted once each)",
        flush=True,
    )

    print("\n  Class (C-level) distribution:", flush=True)
    for cls, cnt in sorted(class_counts.items()):
        name = CLASS_NAMES.get(cls, "Other")
        print(
            f"    {cls} ({name}): {cnt:,} ({100*cnt/max(len(codes),1):.1f}%)",
            flush=True,
        )

    print("\n  Top-20 Architectures (A-level):", flush=True)
    for arch, cnt in arch_counts.most_common(20):
        print(f"    {arch}: {cnt:,}", flush=True)

    print("\n  Top-20 Topologies (T-level):", flush=True)
    for topo, cnt in topo_counts.most_common(20):
        print(f"    {topo}: {cnt:,}", flush=True)

    return {
        "label": label,
        "total": total,
        "with_cath": covered,
        "coverage_pct": round(100 * covered / max(total, 1), 2),
        "total_cath_codes": len(codes),
        "class_counts": dict(class_counts),
        "top_architectures": dict(arch_counts.most_common(50)),
        "top_topologies": dict(topo_counts.most_common(50)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distributions(summaries: list[dict], out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available - skipping plots", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # --- C-level class distribution ---
    fig, axes = plt.subplots(
        1, len(summaries), figsize=(6 * len(summaries), 5), sharey=False
    )
    if len(summaries) == 1:
        axes = [axes]
    for ax, s in zip(axes, summaries):
        classes = sorted(s["class_counts"].keys())
        counts = [s["class_counts"][c] for c in classes]
        labels = [f"{c}\n{CLASS_NAMES.get(c,'Other')}" for c in classes]
        bars = ax.bar(labels, counts, color=colors[: len(classes)])
        ax.set_title(s["label"], fontsize=13)
        ax.set_ylabel("CATH domain count")
        ax.set_xlabel("CATH Class")
        total_codes = max(s["total_cath_codes"], 1)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{100*cnt/total_codes:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("CATH Class Distribution", fontsize=14, y=1.02)
    fig.tight_layout()
    path = out_dir / "cath_class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)

    # --- A-level top-20 architectures side by side ---
    n_top = 20
    fig, axes = plt.subplots(1, len(summaries), figsize=(10 * len(summaries), 6))
    if len(summaries) == 1:
        axes = [axes]
    for ax, s in zip(axes, summaries):
        arch_items = list(s["top_architectures"].items())[:n_top]
        archs, counts = zip(*arch_items) if arch_items else ([], [])
        y = np.arange(len(archs))
        ax.barh(y, counts, color="#4C72B0", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(archs, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(s["label"], fontsize=13)
        ax.set_xlabel("CATH domain count")
    fig.suptitle(f"Top-{n_top} CATH Architectures (A-level)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = out_dir / "cath_architecture_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)

    # --- T-level top-20 topologies ---
    fig, axes = plt.subplots(1, len(summaries), figsize=(10 * len(summaries), 6))
    if len(summaries) == 1:
        axes = [axes]
    for ax, s in zip(axes, summaries):
        topo_items = list(s["top_topologies"].items())[:n_top]
        topos, counts = zip(*topo_items) if topo_items else ([], [])
        y = np.arange(len(topos))
        ax.barh(y, counts, color="#55A868", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(topos, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(s["label"], fontsize=13)
        ax.set_xlabel("CATH domain count")
    fig.suptitle(f"Top-{n_top} CATH Topologies (T-level)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = out_dir / "cath_topology_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)

    # --- Coverage comparison bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [s["label"] for s in summaries]
    coverages = [s["coverage_pct"] for s in summaries]
    ax.bar(labels, coverages, color=colors[: len(summaries)])
    ax.set_ylabel("% entries with CATH label")
    ax.set_ylim(0, 100)
    for i, (lbl, cov) in enumerate(zip(labels, coverages)):
        ax.text(i, cov + 1, f"{cov:.1f}%", ha="center", fontsize=11)
    ax.set_title("CATH Label Coverage by Dataset")
    fig.tight_layout()
    path = out_dir / "cath_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Characterise PDB/AFDB by CATH family")
    parser.add_argument(
        "--pdb_lmdb_dir",
        default=None,
        help="Dir containing PDB {split}.lmdb files (omit to skip PDB)",
    )
    parser.add_argument(
        "--afdb_lmdb_dir",
        default=None,
        help="Dir containing AFDB {split}.lmdb files (omit to skip AFDB)",
    )
    parser.add_argument(
        "--cath_dir", default="/rds/user/sr2173/hpc-work/proteina/data/pdb_train"
    )
    parser.add_argument(
        "--ted_tsv",
        default=None,
        help="(Deprecated - low coverage) Path to TED domain summary TSV",
    )
    parser.add_argument(
        "--interpro_tsv",
        default=None,
        help="Path to interpro_gene3d_swissprot.tsv (preferred for AFDB)",
    )
    parser.add_argument("--out_dir", default="evaluation/proteina/cath_distribution")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Limit entries per split (for smoke-test)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build lookup tables ---
    print("\n[1/3] Building CATH lookup tables...", flush=True)
    pdb_cath = build_pdb_cath_mapping(Path(args.cath_dir)) if args.pdb_lmdb_dir else {}
    afdb_cath = None
    afdb_id_transform = None
    if args.interpro_tsv:
        if not os.path.exists(args.interpro_tsv):
            print(
                f"  WARNING: --interpro_tsv not found: {args.interpro_tsv} - skipping AFDB",
                flush=True,
            )
        else:
            afdb_cath = build_interpro_gene3d_mapping(args.interpro_tsv)

            # InterPro key is the bare UniProt accession (e.g. 'Q9RUA0')
            def afdb_id_transform(gid):
                return gid.replace(".pdb", "").split("-")[1]
    elif args.ted_tsv:
        if not os.path.exists(args.ted_tsv):
            print(
                f"  WARNING: --ted_tsv not found: {args.ted_tsv} - skipping AFDB",
                flush=True,
            )
        else:
            afdb_cath = build_afdb_cath_mapping(args.ted_tsv)

            # TED key is e.g. 'AF-Q9RUA0-F1'
            def _ted_key(graph_id: str) -> str:
                stem = graph_id.replace(".pdb", "")
                parts = stem.split("-")
                if len(parts) >= 4 and parts[-1].startswith("model"):
                    stem = "-".join(parts[:-1])
                return stem

            afdb_id_transform = _ted_key

    # --- Scan LMDBs ---
    print("\n[2/3] Scanning LMDBs...", flush=True)
    summaries = []

    if args.pdb_lmdb_dir:
        for split in args.splits:
            pdb_path = os.path.join(args.pdb_lmdb_dir, f"{split}.lmdb")
            if os.path.exists(pdb_path):
                print(f"\n  PDB {split}...", flush=True)
                result = scan_lmdb(pdb_path, pdb_cath, max_entries=args.max_entries)
                summaries.append(summarise(result, f"PDB {split}"))
            else:
                print(f"  Skipping PDB {split}: {pdb_path} not found", flush=True)

    if afdb_cath is not None:
        for split in args.splits:
            afdb_path = os.path.join(args.afdb_lmdb_dir, f"{split}.lmdb")
            if os.path.exists(afdb_path):
                print(f"\n  AFDB {split}...", flush=True)
                result = scan_lmdb(
                    afdb_path,
                    afdb_cath,
                    id_transform=afdb_id_transform,
                    max_entries=args.max_entries,
                )
                summaries.append(summarise(result, f"AFDB {split}"))
            else:
                print(f"  Skipping AFDB {split}: {afdb_path} not found", flush=True)
    else:
        print("\n  Skipping AFDB (no --interpro_tsv or --ted_tsv provided)", flush=True)

    # --- Save JSON (merge with existing so parallel PDB/AFDB jobs coexist) ---
    json_path = out_dir / "cath_distribution_summary.json"
    existing: list[dict] = []
    if json_path.exists():
        try:
            with open(json_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    new_labels = {s["label"] for s in summaries}
    merged = [s for s in existing if s["label"] not in new_labels] + summaries
    summaries = sorted(merged, key=lambda s: s["label"])
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n[3/3] Saved JSON summary: {json_path}", flush=True)

    # --- Plots ---
    print("\n  Generating plots...", flush=True)
    plot_distributions(summaries, out_dir)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
