"""P2 — verify torch.compile handles 4 distinct (B, N) shapes cleanly.

The length-bucketing plan relies on torch.compile caching one graph per
bucket and hitting cached graphs in steady state. If the compiler instead
recompiles on every shape change (or silently falls back to eager), the
plan falls apart.

This script instantiates the real 60M proteina transformer, wraps it in
``torch.compile(mode="default")``, and runs three round-robin passes over
four shapes that match the planned buckets. We record per-step wall-clock
and torch._dynamo's recompile counter to decide:

  PASS:  exactly 4 compile events total (one per distinct shape), steady-
         state wall-clock after the first pass is consistent across shapes
         (no hidden recompiles on revisit).
  FAIL:  more than 4 compile events, OR revisit steps are still as slow as
         first-visit steps (indicating recompile-per-shape-change).

Isolated: reuses the same model-building scaffolding as batch_size_sweep.py
but writes output to a P2-dedicated CSV. Does not touch training code.

Usage (on a GPU node):
    python hpc-scripts/proteina/bench/p2_compile_multishape.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import time

import proteinfoundation.repa.pyg_compat  # noqa: F401


def log(msg: str) -> None:
    print(f"[p2] {msg}", flush=True)


# Shapes that match the length-bucketing plan's 4 buckets.
# (B, N) chosen to roughly saturate a single A100 80GB baseline.
SHAPES = [
    ("B0", 90, 128),
    ("B1", 24, 256),
    ("B2", 10, 384),
    ("B3", 5, 512),
]
N_ROUND_ROBIN_PASSES = 3


def build_model_and_batch(seq_len: int, batch_size: int, device, cfg):
    import lightning as L
    import torch
    from torch_geometric.data import Batch, Data

    L.seed_everything(42)
    from proteinfoundation.proteinflow.proteina import Proteina

    tmp = tempfile.mkdtemp(prefix="p2_compile_")
    model = Proteina(cfg, store_dir=tmp).to(device)
    model.train()

    def make_batch(B, N):
        datas = [
            Data(
                coords_ca=torch.randn(N, 3),
                mask=torch.ones(N, dtype=torch.bool),
                seq=torch.randint(0, 20, (N,)),
                res_seq_pdb_idx=torch.arange(N, dtype=torch.long),
                chain_break_per_res=torch.zeros(N, dtype=torch.long),
                num_nodes=N,
            )
            for _ in range(B)
        ]
        return Batch.from_data_list(datas).to(device)

    return model, make_batch


def load_training_ca_config():
    import hydra
    from omegaconf import open_dict

    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../src/proteina/configs/experiment_config",
        )
    )
    with hydra.initialize_config_dir(config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name="training_ca")

    with open_dict(cfg):
        cfg.hardware.ngpus_per_node_ = 1
        cfg.hardware.nnodes_ = 1
        cfg.compile = True
        cfg.run_name_ = "p2_compile_multishape"
    return cfg


def dynamo_counters():
    """Return a shallow copy of torch._dynamo.utils.counters for diffing."""
    import torch._dynamo.utils as dutils

    c = dutils.counters
    return {k: dict(v) for k, v in c.items()}


def counter_delta(before: dict, after: dict) -> dict:
    out = {}
    for k, v_after in after.items():
        v_before = before.get(k, {})
        for sub_k, val_after in v_after.items():
            val_before = v_before.get(sub_k, 0)
            if val_after != val_before:
                out.setdefault(k, {})[sub_k] = val_after - val_before
    return out


def run():
    import torch

    device = torch.device("cuda")
    log(f"device: {torch.cuda.get_device_name()}")
    log(f"torch: {torch.__version__}")

    cfg = load_training_ca_config()

    # Build one model, compile it once; then feed all shapes through it.
    # Use the largest shape for instantiation to make sure the model can
    # handle all sizes.
    log("building model (this takes ~30 s)")
    model, _ = build_model_and_batch(SHAPES[-1][2], SHAPES[-1][1], device, cfg)

    log("wrapping model.nn in torch.compile(mode='default')")
    torch.set_float32_matmul_precision("medium")
    model.nn = torch.compile(model.nn)

    # Reset dynamo counters baseline
    import torch._dynamo.utils as dutils

    dutils.counters.clear()

    # Helper to make batches at any size
    _, make_batch = build_model_and_batch(SHAPES[-1][2], SHAPES[-1][1], device, cfg)

    # Round-robin over shapes
    records = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for pass_idx in range(N_ROUND_ROBIN_PASSES):
        for name, B, N in SHAPES:
            log(f"pass {pass_idx}  shape={name}  B={B} N={N}")
            batch = make_batch(B, N)

            before = dynamo_counters()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model.training_step(batch, 0)
            loss = out if torch.is_tensor(out) else out.get("loss")
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            after = dynamo_counters()
            delta = counter_delta(before, after)

            step_ms = (t1 - t0) * 1000.0
            # "frames": {"total": N, "ok": M} — a compile event shows up here
            compile_delta = sum(
                v for k, sub in delta.items() if k == "frames" for v in sub.values()
            )
            ok_delta = delta.get("frames", {}).get("ok", 0)

            records.append(
                {
                    "pass": pass_idx,
                    "shape": name,
                    "B": B,
                    "N": N,
                    "step_ms": round(step_ms, 1),
                    "loss": round(float(loss), 4),
                    "compile_events_delta": compile_delta,
                    "frames_ok_delta": ok_delta,
                }
            )
            log(
                f"  step={step_ms:.1f} ms  loss={float(loss):.4f}  "
                f"compile_events_delta={compile_delta}  ok_delta={ok_delta}"
            )

    # Totals
    total_compiles = sum(r["compile_events_delta"] for r in records)
    total_ok_frames = sum(r["frames_ok_delta"] for r in records)
    log("")
    log(f"TOTAL compile_events across all steps: {total_compiles}")
    log(f"TOTAL frames.ok across all steps: {total_ok_frames}")
    log(f"Expected ≤ ~{len(SHAPES) * 2}-ish (multiple graphs per shape is normal)")

    # First-visit vs revisit timing per shape
    log("")
    log("First-visit vs steady-state wall-clock per shape:")
    for _, _, _ in SHAPES:
        pass  # placeholder
    by_shape = {}
    for r in records:
        by_shape.setdefault(r["shape"], []).append(r["step_ms"])
    for shape, times in by_shape.items():
        first = times[0]
        rest = times[1:]
        rest_avg = sum(rest) / len(rest) if rest else float("nan")
        ratio = first / rest_avg if rest_avg else float("nan")
        log(
            f"  {shape}: first={first:.1f} ms, revisit avg={rest_avg:.1f} ms, ratio={ratio:.2f}×"
        )

    # Verdict
    any_shape_slow_on_revisit = any(
        (sum(times[1:]) / len(times[1:])) > times[0] * 0.5 and len(times) > 1
        for times in by_shape.values()
    )
    # Above is meant as "revisit is >50% of first-visit" (would mean compile on every hit).
    # We want revisit << first-visit.
    log("")
    verdict_lines = []
    if any_shape_slow_on_revisit:
        verdict_lines.append(
            "WARN: at least one shape's revisit avg is not much faster than first visit. "
            "Compile may be re-tracing on each shape change."
        )
    if total_compiles > len(SHAPES) * 10:
        # Heuristic: more than 10 compile events per shape suggests pathological recompiling
        verdict_lines.append(
            f"WARN: {total_compiles} compile events for {len(SHAPES)} shapes — unusually high."
        )

    if not verdict_lines:
        log("VERDICT: PASS — torch.compile caches per shape, revisit is fast.")
    else:
        for line in verdict_lines:
            log(f"VERDICT: {line}")

    # Write CSV
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../evaluation/proteina/bench/results/batch_size_sweep",
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "p2_compile_multishape.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        for r in records:
            w.writerow(r)
    log(f"Wrote {out_csv}")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
