# ruff: noqa: E402, E702
"""Sweep representation-quality probes across the FID-style step schedule,
at ALL transformer layers per checkpoint, with JSONL append-resume.

What this produces (for each of ~48 checkpoints × ~10 layers, plus encoders):
  P1 — long-range contact P@L/5 (headline)
  P2 — CATH fold classification (accuracy + macro-F1)

Columns in the resulting CSV: `run, step, layer, dim, p_at_L, p_at_L_2, p_at_L_5, ...`
— directly sliceable into:

  • Training-progression curve (y = metric, x = step, one curve per run/layer)
  • Layer-wise curve (y = metric, x = layer, one curve per run × step) —
    the REPA-paper Fig. 3a/b analogue

Resume behaviour:
  Results are appended to `sweep_results.jsonl` (one probe per line) as they
  complete. On rerun we read this file first and skip any (run, step, layer)
  tuple already present, so SLURM preemption / OOM / Lustre hiccups never
  waste more than one in-flight probe. Removing the JSONL forces a clean run.

Usage:
  sbatch hpc-scripts/proteina/evaluation/run_probes.sh --sweep
  python playground/proteina/probes/run_sweep.py --n_proteins 200
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from utils import (
    RUN_SCHEDULES,
    _default_device,
    extract_gearnet_embeddings,
    extract_model_hidden_states_multilayer,
    find_checkpoint_path,
    load_checkpoint_by_path,
    load_proteina_batch,
    model_num_layers,
)
from contact import run_contact_probe
from cath import run_cath_probe


GEARNET_CKPT = os.environ.get(
    "GEARNET_CKPT_PATH",
    "/rds/user/sr2173/hpc-work/proteina/data/metric_factory/model_weights/gearnet_ca.pth",
)

# Special sentinel layer used for frozen encoder rows; distinguishes them from
# real layer indices (0..nlayers-1) in the results table.
ENCODER_LAYER_SENTINEL = -1


# --------------------------------------------------------------------------- #
# Resume logic
# --------------------------------------------------------------------------- #


def _load_done_set(jsonl_path: Path) -> Tuple[Set[Tuple[str, int, int]], List[Dict]]:
    """Read existing JSONL; return set of completed (run, step, layer) keys + all rows."""
    if not jsonl_path.exists():
        return set(), []
    done: Set[Tuple[str, int, int]] = set()
    rows: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r and all(k in r for k in ("run", "step", "layer")):
                done.add((r["run"], int(r["step"]), int(r["layer"])))
            rows.append(r)
    return done, rows


def _append_row(jsonl_path: Path, row: Dict) -> None:
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


# --------------------------------------------------------------------------- #
# Probe helper
# --------------------------------------------------------------------------- #


def _probe_one(reps: torch.Tensor, batch, raw, cath_level) -> Dict:
    contact = run_contact_probe(reps, batch)
    cath = run_cath_probe(reps, batch["mask"], raw, preferred_level=cath_level)
    return {
        "dim": int(reps.shape[-1]),
        "contact": asdict(contact),
        "cath": asdict(cath),
    }


def _load_ckpt_with_retry(ckpt_path, is_repa, device, retries=3, backoff=10):
    last_err = None
    for attempt in range(retries):
        try:
            return load_checkpoint_by_path(ckpt_path, is_repa=is_repa, device=device)
        except Exception as e:
            last_err = e
            print(f"    load attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(backoff)
    raise last_err


def _load_gearnet():
    from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder

    enc = GearNetPerResidueEncoder(ckpt_path=GEARNET_CKPT)
    enc.eval()
    return enc.to(_default_device())


# --------------------------------------------------------------------------- #
# Consolidation (CSV + MD from JSONL)
# --------------------------------------------------------------------------- #


def consolidate(outdir: Path) -> None:
    """Rebuild sweep_results.{csv,md,json} from the append-only JSONL."""
    jsonl = outdir / "sweep_results.jsonl"
    rows: List[Dict] = []
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # JSON (consolidated list)
    (outdir / "sweep_results.json").write_text(json.dumps(rows, indent=2, default=str))

    # CSV
    with open(outdir / "sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "step",
                "layer",
                "dim",
                "p_at_L",
                "p_at_L_2",
                "p_at_L_5",
                "contact_n_test",
                "cath_level",
                "cath_acc",
                "cath_f1",
                "cath_classes",
                "ckpt_path",
                "error",
            ]
        )
        for r in rows:
            if "error" in r:
                w.writerow(
                    [
                        r.get("run", ""),
                        r.get("step", ""),
                        r.get("layer", ""),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        r.get("ckpt_path", ""),
                        r["error"],
                    ]
                )
                continue
            c, ca = r["contact"], r["cath"]
            w.writerow(
                [
                    r["run"],
                    r["step"],
                    r["layer"],
                    r["dim"],
                    f"{c['p_at_L']:.4f}",
                    f"{c['p_at_L_2']:.4f}",
                    f"{c['p_at_L_5']:.4f}",
                    c["n_proteins_test"],
                    ca["level"],
                    f"{ca['accuracy']:.4f}",
                    f"{ca['macro_f1']:.4f}",
                    ca["n_classes"],
                    r.get("ckpt_path", ""),
                    "",
                ]
            )

    # MD summary: one table grouped by (run, step), showing the best-layer peak.
    lines = ["# Proteina Probe Sweep — peak-layer summary\n"]
    lines.append("| run | step | best_layer | P@L/5 | CATH-acc | CATH-classes |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    # Group by (run, step), take max P@L/5 across layers
    buckets: Dict[Tuple[str, int], List[Dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r["run"], int(r["step"]))
        buckets.setdefault(key, []).append(r)
    for (run, step), group in sorted(buckets.items()):
        best = max(group, key=lambda r: r["contact"]["p_at_L_5"])
        lines.append(
            f"| {run} | {step} | {best['layer']} | "
            f"{best['contact']['p_at_L_5']:.3f} | {best['cath']['accuracy']:.3f} | "
            f"{best['cath']['n_classes']} |"
        )
    (outdir / "sweep_results.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_proteins", type=int, default=200)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument("--cath_level", type=str, default="T", choices=["C", "A", "T"])
    ap.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated subset of runs to probe (e.g. baseline,repa_l4).",
    )
    ap.add_argument("--skip_gearnet", action="store_true")
    ap.add_argument("--output_dir", type=str, default=str(HERE))
    ap.add_argument(
        "--consolidate_only",
        action="store_true",
        help="Just rebuild CSV/MD from existing JSONL without probing anything.",
    )
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "sweep_results.jsonl"

    if args.consolidate_only:
        consolidate(outdir)
        print(f"Consolidated to {outdir}/sweep_results.{{csv,md,json}}")
        return

    done, _ = _load_done_set(jsonl_path)
    if done:
        print(f"Resuming: {len(done)} probes already cached in {jsonl_path.name}")

    device = _default_device()

    # --- Shared batch: loaded once, reused everywhere ---
    print(f"Loading {args.n_proteins} proteins (≤ {args.max_size} residues)...")
    batch, raw = load_proteina_batch(
        n=args.n_proteins, max_size=args.max_size, device=device
    )
    print(f"  Loaded {len(raw)} proteins, mask sum = {int(batch['mask'].sum().item())}")

    # One-shot diagnostic: what does cath_code look like? CATH probe needs ≥2 classes.
    from collections import Counter

    cath_types: Counter = Counter()
    cath_examples = []
    for g in raw:
        cc = getattr(g, "cath_code", None)
        if cc is None:
            cath_types["None"] += 1
        else:
            cath_types[type(cc).__name__] += 1
            if len(cath_examples) < 3:
                cath_examples.append(cc)
    print(f"  cath_code type distribution: {dict(cath_types)}")
    print(f"  cath_code sample values: {cath_examples}")

    # --- Gearnet (flat reference, single "layer" = -1 sentinel) ---
    gearnet_key = ("gearnet", 0, ENCODER_LAYER_SENTINEL)
    if not args.skip_gearnet and gearnet_key not in done:
        print("\n=== gearnet (frozen reference) ===")
        try:
            enc = _load_gearnet()
            reps = extract_gearnet_embeddings(enc, batch)
            out = _probe_one(reps, batch, raw, args.cath_level)
            out.update(
                {
                    "run": "gearnet",
                    "step": 0,
                    "layer": ENCODER_LAYER_SENTINEL,
                    "ckpt_path": None,
                }
            )
            _append_row(jsonl_path, out)
            done.add(gearnet_key)
            print(
                f"  gearnet: P@L/5={out['contact']['p_at_L_5']:.3f}  "
                f"CATH-acc={out['cath']['accuracy']:.3f}"
            )
            del enc
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  ❌ gearnet failed: {e}")
            _append_row(
                jsonl_path,
                {
                    "run": "gearnet",
                    "step": 0,
                    "layer": ENCODER_LAYER_SENTINEL,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif gearnet_key in done:
        print("gearnet already cached, skip")

    # --- Iterate (run, step) grid — probe all layers per checkpoint ---
    chosen_runs = set(args.runs.split(",")) if args.runs else set(RUN_SCHEDULES)
    for run_name in RUN_SCHEDULES:
        if run_name not in chosen_runs:
            continue
        run_dir, is_repa, _, steps = RUN_SCHEDULES[run_name]
        print(f"\n=== {run_name} ({run_dir}, {len(steps)} steps) ===")

        for step in steps:
            ckpt = find_checkpoint_path(run_dir, step, prefer_ema=True)
            if ckpt is None:
                print(f"  step {step}: NO EMA CKPT — skip")
                continue

            # Infer layer list from an already-loaded result if we have one for this run.
            # Otherwise peek at the model once (below).
            print(f"\n  --- step {step} @ {ckpt.name} ---")

            try:
                # Load model once; probe all layers from one forward pass.
                model = _load_ckpt_with_retry(ckpt, is_repa=is_repa, device=device)
                n_layers = model_num_layers(model)
                all_layers = list(range(n_layers))

                # Skip layers already done; run the remaining set.
                todo_layers = [
                    L for L in all_layers if (run_name, int(step), L) not in done
                ]
                if not todo_layers:
                    print(f"    all {n_layers} layers already cached, skip")
                    del model
                    continue
                if len(todo_layers) < n_layers:
                    print(
                        f"    resuming: {len(all_layers) - len(todo_layers)}/{n_layers} "
                        f"layers already cached, probing {len(todo_layers)} more"
                    )

                t0 = time.time()
                reps_by_layer = extract_model_hidden_states_multilayer(
                    model, batch, todo_layers
                )
                t_extract = time.time() - t0
                print(f"    extracted {len(todo_layers)} layers in {t_extract:.1f}s")

                for L in todo_layers:
                    try:
                        out = _probe_one(reps_by_layer[L], batch, raw, args.cath_level)
                        out.update(
                            {
                                "run": run_name,
                                "step": int(step),
                                "layer": int(L),
                                "ckpt_path": str(ckpt),
                            }
                        )
                        _append_row(jsonl_path, out)
                        done.add((run_name, int(step), L))
                        print(
                            f"      L{L:2d}: P@L/5={out['contact']['p_at_L_5']:.3f}  "
                            f"CATH-{out['cath']['level']}-acc={out['cath']['accuracy']:.3f}"
                        )
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        _append_row(
                            jsonl_path,
                            {
                                "run": run_name,
                                "step": int(step),
                                "layer": int(L),
                                "ckpt_path": str(ckpt),
                                "error": f"{type(e).__name__}: {e}",
                            },
                        )
                del model, reps_by_layer

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"    ❌ {run_name}@{step} failed at ckpt load / extract: {e}")
                # Record a single failure row so we don't retry this checkpoint forever.
                _append_row(
                    jsonl_path,
                    {
                        "run": run_name,
                        "step": int(step),
                        "layer": -999,
                        "ckpt_path": str(ckpt),
                        "error": f"{type(e).__name__}: {e}",
                    },
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --- Consolidate into CSV / MD ---
    consolidate(outdir)
    print(f"\nFinal outputs in {outdir}:")
    for p in (
        "sweep_results.jsonl",
        "sweep_results.csv",
        "sweep_results.json",
        "sweep_results.md",
    ):
        print(f"  {outdir / p}")


if __name__ == "__main__":
    main()
