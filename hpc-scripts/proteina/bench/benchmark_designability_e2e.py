"""End-to-end designability wall on a small subset, with the TF32 patch in.

Picks K PDBs at each length in --lengths and runs batch_designability through
the full ProteinMPNN -> ESMFold -> scRMSD/TM/pLDDT chain. Reports total wall
and per-PDB wall, broken down by length, vs the historical 0.155*L estimate
from reference_proteina_eval_budget.md.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "proteina",
        "proteinfoundation",
    ),
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "proteina")
)

import proteinfoundation.repa.pyg_compat  # noqa: F401

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

from proteinfoundation.metrics.designability import batch_designability  # noqa: E402


def pick_pdbs(samples_dir, L, k):
    out = []
    for fname in sorted(os.listdir(samples_dir)):
        if not fname.endswith(".pdb"):
            continue
        p = os.path.join(samples_dir, fname)
        with open(p) as f:
            n = sum(1 for line in f if line.startswith("ATOM "))
        if n == L:
            out.append(p)
            if len(out) >= k:
                return out
    raise FileNotFoundError(
        f"Only found {len(out)} PDBs of length {L} in {samples_dir}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[100, 150, 200, 250])
    ap.add_argument("--per_length", type=int, default=5)
    ap.add_argument("--tmp_root", default="/tmp/desig_e2e")
    args = ap.parse_args()
    os.makedirs(args.tmp_root, exist_ok=True)

    # Gather PDB list, tracking length per PDB so we can break down timing.
    pdbs = []
    pdb_to_L = {}
    for L in args.lengths:
        picked = pick_pdbs(args.samples_dir, L, args.per_length)
        for p in picked:
            pdbs.append(p)
            pdb_to_L[p] = L
    print(
        f"Running designability on {len(pdbs)} PDBs ({args.per_length}/length x {len(args.lengths)} lengths)",
        flush=True,
    )

    # batch_designability times individual PDBs internally via the per-PDB log
    # line, but doesn't return them. To get per-length walls we monkey-patch
    # logger.info to capture timestamps around each "Designability [i/N]:" log.
    import proteinfoundation.metrics.designability as D

    orig_info = D.logger.info
    pdb_timestamps = []

    def info_hook(msg, *a, **kw):
        if "Designability [" in msg:
            pdb_timestamps.append(time.perf_counter())
        return orig_info(msg, *a, **kw)

    D.logger.info = info_hook

    t0 = time.perf_counter()
    results = batch_designability(pdbs, tmp_root=args.tmp_root)
    total = time.perf_counter() - t0
    pdb_timestamps.append(time.perf_counter())  # end-cap

    # Per-PDB durations: timestamp[i+1] - timestamp[i]
    per_pdb = []
    for i, p in enumerate(pdbs):
        if i + 1 < len(pdb_timestamps):
            per_pdb.append((p, pdb_to_L[p], pdb_timestamps[i + 1] - pdb_timestamps[i]))

    # Per-length aggregates
    print("\n=== PER-LENGTH WALL ===", flush=True)
    print(
        f"{'L':>5s}{'n':>4s}{'mean s/PDB':>14s}{'historical (0.155L)':>22s}{'speedup':>10s}"
    )
    by_L = {}
    for _, L, t in per_pdb:
        by_L.setdefault(L, []).append(t)
    for L in sorted(by_L.keys()):
        ts = by_L[L]
        mean = sum(ts) / len(ts)
        hist = 0.155 * L
        print(f"{L:5d}{len(ts):4d}{mean:14.2f}{hist:22.2f}{hist/mean:10.2f}x")

    print(
        f"\n=== TOTAL WALL: {total:.1f}s for {len(pdbs)} PDBs ({total/len(pdbs):.2f}s/PDB avg) ==="
    )
    print(
        f"scRMSD_mean: {results['scRMSD_mean']:.3f}, designability_rate: {results['designability_rate']:.2f}"
    )


if __name__ == "__main__":
    main()
