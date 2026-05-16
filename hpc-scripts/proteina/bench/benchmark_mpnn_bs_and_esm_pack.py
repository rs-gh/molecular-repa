"""Test two further designability optimizations on top of the TF32 patch:

  (A) MPNN --batch_size 8 (vs current --batch_size 1).
  (B) ESMFold packs 8 seqs in one batch (vs the current 2 batches of 4).

For each of N PDBs at L=200 we run all four combinations and report:
  - mean wall per PDB (MPNN + ESMFold + scoring)
  - scRMSD/pLDDT distribution, designability rate
  - speedup vs the post-TF32 baseline (config A)

For (B), inputs are identical seq lists across the 2x4 and 1x8 paths, so
refolds should be numerically near-equal -- we report max |scRMSD diff| as
a sanity check. For (A), seqs differ between bs=1 and bs=8 (different MPNN
sampling), so we compare the *distribution* of scRMSDs (best-of-8 per PDB,
designability rate across the set).
"""

import argparse
import os
import sys
import time
from statistics import mean

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


import proteinfoundation.metrics.designability as D  # noqa: E402
from proteinfoundation.metrics.designability import (  # noqa: E402
    convert_outputs_to_pdb,
    load_pdb,
    pdb_name_from_path,
    extract_gen_seqs,
    rmsd_metric,
)


def run_mpnn(pdb_path, out_dir, mpnn_batch_size, num_seq=8, seed=42):
    """Local copy of run_proteinmpnn with parameterized --batch_size."""
    name = pdb_name_from_path(pdb_path)
    python_exec = os.environ.get("PYTHON_EXEC", "python")
    mpnn_dir = os.environ["PROTEINMPNN_DIR"]
    mpnn_weights_root = os.environ["PROTEINMPNN_WEIGHTS_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        f"{python_exec} {mpnn_dir}/protein_mpnn_run.py "
        f"--pdb_path {pdb_path} --pdb_path_chains A "
        f"--out_folder {out_dir} "
        f"--num_seq_per_target {num_seq} --sampling_temp 0.1 "
        f"--batch_size {mpnn_batch_size} --suppress_print 1 "
        f"--ca_only --path_to_model_weights {mpnn_weights_root}/ca_model_weights "
        f"--seed {seed} > /dev/null 2>&1"
    )
    t0 = time.perf_counter()
    os.system(cmd)
    dt = time.perf_counter() - t0
    seqs = extract_gen_seqs(os.path.join(out_dir, "seqs", name + ".fa"))[:num_seq]
    return seqs, dt


def fold_seqs(model, tok, seqs, batch_size, num_batches):
    """Run ESMFold over `seqs` using batch_size x num_batches partitioning.
    Mirrors run_and_store_esm but returns refolded coords + pLDDTs and timing."""
    out_pdbs = []
    out_plddts = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(num_batches):
        chunk = seqs[i * batch_size : (i + 1) * batch_size]
        if not chunk:
            continue
        inputs = tok(chunk, return_tensors="pt", add_special_tokens=False, padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)
        pdbs, plddts = convert_outputs_to_pdb(out)
        out_pdbs.extend(pdbs)
        out_plddts.extend(plddts)
    torch.cuda.synchronize()
    return out_pdbs, out_plddts, time.perf_counter() - t0


def score_against_gen(gen_pdb_path, refolded_pdb_strs, tmp_dir):
    """Compute scRMSD list for the refolds against the source backbone."""
    gen = load_pdb(gen_pdb_path)
    gen_coors = torch.Tensor(gen.atom_positions)
    os.makedirs(tmp_dir, exist_ok=True)
    rmsds = []
    for k, pdb_str in enumerate(refolded_pdb_strs):
        p = os.path.join(tmp_dir, f"r_{k}.pdb")
        with open(p, "w") as f:
            f.write(pdb_str)
        rec = load_pdb(p)
        rec_coors = torch.Tensor(rec.atom_positions)
        n = min(len(gen_coors), len(rec_coors))
        rmsds.append(rmsd_metric(gen_coors[:n], rec_coors[:n]))
    return rmsds


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
    raise FileNotFoundError(f"Only found {len(out)} PDBs of length {L}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--n_pdbs", type=int, default=5)
    ap.add_argument("--tmp_root", default="/tmp/mpnn_esm_pack")
    args = ap.parse_args()
    os.makedirs(args.tmp_root, exist_ok=True)

    # Trigger TF32 + ESMFold load (this exercises the patched load_esmfold).
    esm_model, tok = D.load_esmfold()
    esm_model = esm_model.eval()

    pdbs = pick_pdbs(args.samples_dir, args.L, args.n_pdbs)
    print(f"Benchmarking on {len(pdbs)} PDBs at L={args.L}", flush=True)

    # ---- Phase 1: get sequences from both MPNN configs for every PDB
    mpnn_results = {}  # (config_name) -> {pdb -> (seqs, time)}
    for bs in (1, 8):
        label = f"mpnn_bs{bs}"
        mpnn_results[label] = {}
        print(f"\n--- {label} ---", flush=True)
        for p in pdbs:
            seqs, dt = run_mpnn(
                p, os.path.join(args.tmp_root, f"{label}_{pdb_name_from_path(p)}"), bs
            )
            mpnn_results[label][p] = (seqs, dt)
            print(
                f"  {pdb_name_from_path(p)}: {dt:.2f}s, len(seq)={len(seqs)}",
                flush=True,
            )
        avg = mean(v[1] for v in mpnn_results[label].values())
        print(f"  mean MPNN time: {avg:.2f}s/PDB", flush=True)

    # ---- Phase 2: ESMFold over each (mpnn_bs, esm_pack) combo
    esm_configs = [
        ("esm_2x4", 4, 2),  # current
        ("esm_1x8", 8, 1),  # packed
    ]

    # Aggregate results: (mpnn_label, esm_label) -> dict
    bench = {}
    refold_cache = {}  # (mpnn_label, esm_label, pdb) -> list of refolded PDB strings

    for mpnn_label in ("mpnn_bs1", "mpnn_bs8"):
        for esm_label, bsz, nbatch in esm_configs:
            key = (mpnn_label, esm_label)
            scrmsds_per_pdb = []
            wall_esm_per_pdb = []
            best_scrmsds = []
            print(f"\n=== {key} ===", flush=True)
            for p in pdbs:
                seqs, _ = mpnn_results[mpnn_label][p]
                # Warm one fold call once per config (first PDB) to flush cudnn benchmarking.
                pdbs_out, plddts, dt = fold_seqs(esm_model, tok, seqs, bsz, nbatch)
                # Re-run once for stable timing (drop warm-up)
                pdbs_out, plddts, dt = fold_seqs(esm_model, tok, seqs, bsz, nbatch)
                rmsds = score_against_gen(
                    p,
                    pdbs_out,
                    os.path.join(
                        args.tmp_root,
                        f"score_{mpnn_label}_{esm_label}_{pdb_name_from_path(p)}",
                    ),
                )
                scrmsds_per_pdb.append(rmsds)
                best_scrmsds.append(min(rmsds))
                wall_esm_per_pdb.append(dt)
                refold_cache[(mpnn_label, esm_label, p)] = pdbs_out
                print(
                    f"  {pdb_name_from_path(p)}: ESMFold {dt:.2f}s, best scRMSD={min(rmsds):.3f}",
                    flush=True,
                )
            bench[key] = {
                "esm_wall_mean": mean(wall_esm_per_pdb),
                "best_scrmsd_mean": mean(best_scrmsds),
                "des_rate": sum(1 for r in best_scrmsds if r < 2.0) / len(best_scrmsds),
                "best_scrmsds": best_scrmsds,
            }

    # ---- Phase 3: numerical equivalence of ESMFold 2x4 vs 1x8 (same seqs)
    print("\n\n=== ESMFold 2x4 vs 1x8 numerical agreement (same seqs) ===", flush=True)
    for mpnn_label in ("mpnn_bs1", "mpnn_bs8"):
        max_diffs = []
        for p in pdbs:
            r_24 = score_against_gen(
                p,
                refold_cache[(mpnn_label, "esm_2x4", p)],
                os.path.join(args.tmp_root, "_diff_a"),
            )
            r_18 = score_against_gen(
                p,
                refold_cache[(mpnn_label, "esm_1x8", p)],
                os.path.join(args.tmp_root, "_diff_b"),
            )
            diffs = [abs(a - b) for a, b in zip(r_24, r_18)]
            max_diffs.append(max(diffs))
        print(
            f"  {mpnn_label}: max |scRMSD diff| across PDBs: {max(max_diffs):.4f} A (mean per-PDB max: {mean(max_diffs):.4f})",
            flush=True,
        )

    # ---- Summary
    print("\n\n========== SUMMARY ==========")
    print(
        f"{'config':24s}{'mpnn s/PDB':>14s}{'esm s/PDB':>12s}{'total s/PDB':>14s}{'best_scRMSD':>14s}{'des_rate':>10s}"
    )
    baseline_total = None
    rows = []
    for mpnn_label in ("mpnn_bs1", "mpnn_bs8"):
        m_mean = mean(v[1] for v in mpnn_results[mpnn_label].values())
        for esm_label, _, _ in esm_configs:
            e = bench[(mpnn_label, esm_label)]
            total = m_mean + e["esm_wall_mean"]
            label = f"{mpnn_label}+{esm_label}"
            rows.append(
                (
                    label,
                    m_mean,
                    e["esm_wall_mean"],
                    total,
                    e["best_scrmsd_mean"],
                    e["des_rate"],
                )
            )
            if mpnn_label == "mpnn_bs1" and esm_label == "esm_2x4":
                baseline_total = total
    for label, m, e, t, r, d in rows:
        speedup = baseline_total / t
        print(
            f"{label:24s}{m:14.2f}{e:12.2f}{t:14.2f}{r:14.3f}{d:10.2f}  ({speedup:.2f}x vs current)"
        )


if __name__ == "__main__":
    main()
