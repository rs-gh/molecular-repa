"""Micro-benchmark: ESMFold fp32 vs fp16(+chunking) on real generated backbones.

For each length L in --lengths, pick one PDB, run ProteinMPNN once to get 8 seqs,
then time three ESMFold configs on the *same* sequence set:
  A. fp32, no chunking            (= current pipeline)
  B. fp32, trunk chunk_size=64
  C. fp16 (esm.half()) + chunk_size=64

Reports per-config median wall (3 trials, 1 warmup) and the cumulative scRMSD
of the refolds against the original backbone so we can spot quality regressions.
"""

import argparse
import os
import sys
import time
from statistics import median

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

from transformers import AutoTokenizer, EsmForProteinFolding  # noqa: E402

from proteinfoundation.metrics.designability import (  # noqa: E402
    convert_outputs_to_pdb,
    load_pdb,
    rmsd_metric,
    run_proteinmpnn,
)


def pick_pdb_for_length(samples_dir: str, L: int) -> str:
    for fname in sorted(os.listdir(samples_dir)):
        if not fname.endswith(".pdb"):
            continue
        path = os.path.join(samples_dir, fname)
        with open(path) as f:
            n = sum(1 for line in f if line.startswith("ATOM "))
        if n == L:
            return path
    raise FileNotFoundError(f"No PDB of length {L} in {samples_dir}")


def fold_once(model, tokenizer, seqs):
    """One forward pass on all `seqs` as a single batch."""
    inputs = tokenizer(
        seqs, return_tensors="pt", add_special_tokens=False, padding=True
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
    pdbs, plddts = convert_outputs_to_pdb(out)
    torch.cuda.synchronize()
    return pdbs, plddts


def time_config(label, model, tokenizer, seqs, ref_pdb_path, n_trials=3, warmup=1):
    times = []
    pdbs = None
    for i in range(warmup + n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pdbs, _ = fold_once(model, tokenizer, seqs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        print(f"  [{label}] trial {i}: {dt:.2f}s", flush=True)

    # Quality check: scRMSD against the source backbone
    gen = load_pdb(ref_pdb_path)
    gen_coors = torch.Tensor(gen.atom_positions)
    rmsds = []
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as td:
            for k, pdb_str in enumerate(pdbs):
                p = os.path.join(td, f"r_{k}.pdb")
                with open(p, "w") as f:
                    f.write(pdb_str)
                rec = load_pdb(p)
                rec_coors = torch.Tensor(rec.atom_positions)
                n = min(len(gen_coors), len(rec_coors))
                rmsds.append(rmsd_metric(gen_coors[:n], rec_coors[:n]))
        best = min(rmsds)
    except Exception as e:
        print(f"  [{label}] quality check failed: {e}", flush=True)
        best = float("nan")
    return median(times), best


def load_model(precision: str, chunk_size: int | None):
    cache_dir = os.environ.get("CACHE_DIR")
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=cache_dir)
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", cache_dir=cache_dir
    )
    model = model.cuda().eval()
    if precision == "fp16":
        model.esm = model.esm.half()
    if chunk_size is not None:
        model.trunk.set_chunk_size(chunk_size)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[100, 150, 200, 250])
    ap.add_argument("--num_seq", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tmp_root", default="/tmp/esm_bench")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.tmp_root, exist_ok=True)

    # ---- Pick PDBs + run MPNN ONCE per length (same seqs across all configs) ----
    seqs_by_L = {}
    pdb_by_L = {}
    for L in args.lengths:
        pdb_path = pick_pdb_for_length(args.samples_dir, L)
        pdb_by_L[L] = pdb_path
        print(f"L={L}: {pdb_path}", flush=True)
        seqs = run_proteinmpnn(
            pdb_path,
            os.path.join(args.tmp_root, f"L{L}"),
            num_seq_per_target=args.num_seq,
            sampling_temp=0.1,
        )
        assert len(seqs) >= args.num_seq, f"got {len(seqs)} seqs for L={L}"
        seqs_by_L[L] = seqs[: args.num_seq]
        print(f"  -> {len(seqs_by_L[L])} sequences", flush=True)

    configs = [
        ("fp32-nochunk (current)", "fp32", None),
        ("fp32+chunk64", "fp32", 64),
        ("fp16+chunk64", "fp16", 64),
    ]

    rows = []
    for label, prec, chunk in configs:
        print(f"\n=== {label} ===", flush=True)
        torch.cuda.empty_cache()
        model, tok = load_model(prec, chunk)
        for L in args.lengths:
            print(f"  L={L}", flush=True)
            try:
                med, best_rmsd = time_config(
                    label, model, tok, seqs_by_L[L], pdb_by_L[L], n_trials=args.trials
                )
            except Exception as e:
                print(f"  [{label}] L={L} FAILED: {e}", flush=True)
                med, best_rmsd = float("nan"), float("nan")
            rows.append((label, L, med, best_rmsd))
        del model, tok
        torch.cuda.empty_cache()

    # ---- Summary ----
    print(
        "\n\n=== SUMMARY (median wall over {} trials, batch-of-8 seqs) ===".format(
            args.trials
        )
    )
    print(f"{'config':28s}{'L':>5s}{'wall_s':>10s}{'best_scRMSD':>14s}")
    for label, L, t, rmsd in rows:
        print(f"{label:28s}{L:5d}{t:10.2f}{rmsd:14.3f}")

    # Speedup table
    print("\n=== SPEEDUP vs fp32-nochunk ===")
    baseline = {L: t for (label, L, t, _) in rows if label.startswith("fp32-nochunk")}
    print(f"{'config':28s}" + "".join(f"{('L=' + str(L)):>10s}" for L in args.lengths))
    for label, _, _, _ in [c for c in rows if c[1] == args.lengths[0]]:
        line = f"{label:28s}"
        for L in args.lengths:
            t = next(t for (lab, LL, t, _) in rows if lab == label and LL == L)
            line += f"{baseline[L]/t:>10.2f}x"
        print(line)


if __name__ == "__main__":
    main()
