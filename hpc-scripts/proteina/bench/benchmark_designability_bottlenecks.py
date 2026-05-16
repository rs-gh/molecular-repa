"""Follow-up to benchmark_esmfold_precision.py: probe the remaining suspects.

(1) ProteinMPNN subprocess overhead: per-call wall for `os.system` invocations
    vs an in-process call to `protein_mpnn_run.main` (same args, no fork). The
    delta is the Python startup + weight load cost we pay per PDB.
(2) TF32 on vs off in ESMFold. Production `load_esmfold` does not touch
    `torch.backends.cuda.matmul.allow_tf32`. We measure fp32 with TF32 off
    (true production path) vs on.
(3) Cross-PDB ESMFold batching: 4 PDBs at L=200, 8 seqs each. Time as
    4 sequential calls of bs=8 (current pipeline) vs 1 call of bs=32 (packed).
"""

import argparse
import io
import os
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from statistics import median
from types import SimpleNamespace

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
    run_proteinmpnn,
)


def pick_pdb_for_length(samples_dir, L, n=1):
    out = []
    for fname in sorted(os.listdir(samples_dir)):
        if not fname.endswith(".pdb"):
            continue
        path = os.path.join(samples_dir, fname)
        with open(path) as f:
            cnt = sum(1 for line in f if line.startswith("ATOM "))
        if cnt == L:
            out.append(path)
            if len(out) >= n:
                return out
    raise FileNotFoundError(
        f"Need {n} PDBs of length {L} in {samples_dir}, found {len(out)}"
    )


# -------------------------------------------------------------------- (1) MPNN


def bench_mpnn_subprocess(pdb, tmp_root, n_calls=5):
    """Current pipeline: os.system per PDB call. Includes Python startup + weight load."""
    times = []
    for i in range(n_calls):
        t0 = time.perf_counter()
        seqs = run_proteinmpnn(
            pdb, os.path.join(tmp_root, f"sub_{i}"), num_seq_per_target=8
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        print(
            f"  [mpnn-subprocess] call {i}: {dt:.2f}s (got {len(seqs)} seqs)",
            flush=True,
        )
    return times


def bench_mpnn_inprocess(pdb, tmp_root, n_calls=5):
    """In-process: call protein_mpnn_run.main directly. Still reloads the model each call
    (paper code-shape), but skips Python startup + import overhead."""
    mpnn_dir = os.environ["PROTEINMPNN_DIR"]
    mpnn_weights = os.environ["PROTEINMPNN_WEIGHTS_DIR"]
    sys.path.insert(0, mpnn_dir)
    import protein_mpnn_run as pmpnn

    times = []
    for i in range(n_calls):
        out_dir = os.path.join(tmp_root, f"inp_{i}")
        os.makedirs(out_dir, exist_ok=True)
        # Build the same Namespace as the argparse default + our overrides.
        # Defaults pulled from protein_mpnn_run.py argparser block.
        args = SimpleNamespace(
            suppress_print=1,
            ca_only=True,
            path_to_model_weights=f"{mpnn_weights}/ca_model_weights",
            model_name="v_48_020",
            use_soluble_model=False,
            seed=0,
            save_score=0,
            save_probs=0,
            score_only=0,
            path_to_fasta="",
            conditional_probs_only=0,
            conditional_probs_only_backbone=0,
            unconditional_probs_only=0,
            backbone_noise=0.0,
            num_seq_per_target=8,
            batch_size=1,
            max_length=20000,
            sampling_temp="0.1",
            out_folder=out_dir,
            pdb_path=pdb,
            pdb_path_chains="A",
            jsonl_path="",
            chain_id_jsonl="",
            fixed_positions_jsonl="",
            omit_AAs="X",
            bias_AA_jsonl="",
            bias_by_res_jsonl="",
            omit_AA_jsonl="",
            pssm_jsonl="",
            pssm_multi=0.0,
            pssm_threshold=0.0,
            pssm_log_odds_flag=0,
            pssm_bias_flag=0,
            tied_positions_jsonl="",
        )
        t0 = time.perf_counter()
        # MPNN main() prints a lot. Silence stdout/stderr to mimic the > /dev/null in subprocess.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            pmpnn.main(args)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  [mpnn-inprocess]  call {i}: {dt:.2f}s", flush=True)
    return times


# -------------------------------------------------------------------- (2) TF32


def load_esmfold():
    cache_dir = os.environ.get("CACHE_DIR")
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=cache_dir)
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", cache_dir=cache_dir
    )
    model = model.cuda().eval()
    return model, tok


def fold(model, tok, seqs):
    inputs = tok(seqs, return_tensors="pt", add_special_tokens=False, padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
    convert_outputs_to_pdb(out)
    torch.cuda.synchronize()


def bench_tf32(model, tok, seqs, tf32, n_trials=3, warmup=1):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    times = []
    for i in range(warmup + n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fold(model, tok, seqs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        print(f"  [tf32={tf32}] trial {i}: {dt:.2f}s", flush=True)
    return median(times)


# -------------------------------------------------------------------- (3) Cross-PDB batching


def bench_crosspdb(model, tok, seqs_per_pdb, n_trials=3, warmup=1):
    """seqs_per_pdb: list of 4 lists, each 8 seqs of length L=200.
    Current path: 4 sequential calls of bs=8.
    Packed:        1 call of bs=32.
    """
    flat = [s for group in seqs_per_pdb for s in group]
    times_seq = []
    times_packed = []
    for i in range(warmup + n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for group in seqs_per_pdb:
            fold(model, tok, group)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times_seq.append(dt)
        print(f"  [seq 4x bs=8] trial {i}: {dt:.2f}s", flush=True)
    for i in range(warmup + n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fold(model, tok, flat)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times_packed.append(dt)
        print(f"  [packed 1x bs=32] trial {i}: {dt:.2f}s", flush=True)
    return median(times_seq), median(times_packed)


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--tmp_root", default="/tmp/desig_bottle")
    args = ap.parse_args()
    os.makedirs(args.tmp_root, exist_ok=True)

    # Pick one L=200 PDB for MPNN/TF32, plus 4 L=200 PDBs for cross-PDB.
    pdb_l200_one = pick_pdb_for_length(args.samples_dir, 200, n=1)[0]
    pdbs_l200_four = pick_pdb_for_length(args.samples_dir, 200, n=4)
    print(f"Single PDB (L=200): {pdb_l200_one}")
    print(f"Four PDBs (L=200):  {pdbs_l200_four}")

    # ---------------- (1) MPNN subprocess vs in-process
    print(
        "\n========== (1) ProteinMPNN subprocess vs in-process ==========", flush=True
    )
    sub_times = bench_mpnn_subprocess(pdb_l200_one, args.tmp_root, n_calls=5)
    inp_times = bench_mpnn_inprocess(pdb_l200_one, args.tmp_root, n_calls=5)
    sub_med = median(sub_times)
    inp_med = median(inp_times)
    print(f"\n  subprocess median: {sub_med:.2f}s/PDB")
    print(f"  in-process median: {inp_med:.2f}s/PDB")
    print(
        f"  saved per PDB:    {sub_med - inp_med:+.2f}s ({sub_med/max(inp_med,1e-3):.2f}x)"
    )
    # Pull MPNN seqs once for the ESMFold benches below
    mpnn_seqs_one = run_proteinmpnn(
        pdb_l200_one, os.path.join(args.tmp_root, "_for_esm"), num_seq_per_target=8
    )[:8]
    mpnn_seqs_four = [
        run_proteinmpnn(
            p, os.path.join(args.tmp_root, f"_for_esm_{k}"), num_seq_per_target=8
        )[:8]
        for k, p in enumerate(pdbs_l200_four)
    ]

    # ---------------- (2) TF32 on/off
    print(
        "\n========== (2) ESMFold TF32 on vs off (L=200, bs=8) ==========", flush=True
    )
    model, tok = load_esmfold()
    med_off = bench_tf32(model, tok, mpnn_seqs_one, tf32=False)
    med_on = bench_tf32(model, tok, mpnn_seqs_one, tf32=True)
    print(f"\n  TF32 OFF (production): {med_off:.2f}s")
    print(f"  TF32 ON:               {med_on:.2f}s")
    print(f"  speedup:               {med_off/med_on:.2f}x")

    # ---------------- (3) Cross-PDB packing
    print(
        "\n========== (3) Cross-PDB batching (L=200, 4 PDBs * 8 seqs) ==========",
        flush=True,
    )
    torch.backends.cuda.matmul.allow_tf32 = (
        True  # keep TF32 on so we isolate batching effect
    )
    seq_med, packed_med = bench_crosspdb(model, tok, mpnn_seqs_four)
    print(f"\n  4 sequential calls of bs=8: {seq_med:.2f}s ({seq_med/4:.2f}s/PDB)")
    print(f"  1 packed call of bs=32:     {packed_med:.2f}s ({packed_med/4:.2f}s/PDB)")
    print(f"  speedup:                    {seq_med/packed_med:.2f}x")

    # ---------------- Summary
    print("\n\n========== SUMMARY ==========")
    print(
        f"(1) MPNN subprocess overhead: {sub_med:.2f}s -> {inp_med:.2f}s in-process ({(sub_med-inp_med):.2f}s/PDB saved)"
    )
    print(
        f"(2) TF32 on ESMFold:          {med_off:.2f}s -> {med_on:.2f}s ({med_off/med_on:.2f}x speedup)"
    )
    print(
        f"(3) Cross-PDB ESMFold pack:   {seq_med:.2f}s -> {packed_med:.2f}s for 4 PDBs ({seq_med/packed_med:.2f}x)"
    )


if __name__ == "__main__":
    main()
