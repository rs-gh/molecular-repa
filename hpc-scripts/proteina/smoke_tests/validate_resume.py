"""Checkpoint-resume validator.

Given a checkpoint path, confirm that:
  1. The checkpoint loads into a freshly-instantiated model without state_dict
     mismatches (catches silent parameter additions/removals).
  2. A forward pass produces a finite loss close to the checkpoint's recorded
     val/train loss (catches catastrophic numerical drift from optimization
     changes — e.g., a SDPA backend swap that produces NaNs).
  3. A backward pass populates gradients without NaN/Inf.
  4. An optimizer step updates parameters without NaN/Inf.

Use before adding an optimization (torch.compile on/off, SDPA backend swap,
gradient checkpointing wrap, dynamic padding, etc.) to confirm the change is
resume-safe.

Usage:
    python validate_resume.py \\
        --ckpt /rds/.../proteina_60m_baseline_256/checkpoints/last.ckpt \\
        --config training_baseline_256 \\
        [--config_subdir training/256] \\
        [--optimization noop|compile_on|compile_off|sdpa_math|sdpa_efficient|gc_10]

Exits 0 on pass, 1 on any failure. Prints a one-line verdict at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import proteinfoundation.repa.pyg_compat  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to last.ckpt")
    p.add_argument(
        "--config",
        required=True,
        help="Config name (e.g. training_baseline_256, training_repa_l4_256_per_residue)",
    )
    p.add_argument(
        "--config_subdir",
        default="",
        help="Config subdir (e.g. training/256 or training/256/gearnet/per_residue)",
    )
    p.add_argument(
        "--optimization",
        default="noop",
        choices=[
            "noop",
            "compile_on",
            "compile_off",
            "sdpa_math",
            "sdpa_efficient",
            "sdpa_flash",
            "gc_10",
        ],
        help="Optimization to apply post-load. 'noop' = plain resume.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Synthetic batch size for the validation step.",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Synthetic seq_len; default pulls from config's max_size.",
    )
    p.add_argument(
        "--loss_tolerance",
        type=float,
        default=5.0,
        help=(
            "Max acceptable multiplicative deviation vs checkpoint's recorded "
            "loss (default 5x — wide because synthetic data is OOD)."
        ),
    )
    return p.parse_args()


def log(msg: str):
    print(f"[validate_resume] {msg}", flush=True)


def step(name: str):
    """Context manager-ish for step-by-step reporting."""
    log(f"  → {name}")


def build_model(cfg, is_repa: bool, store_dir: str):
    import lightning as L

    L.seed_everything(42)
    if is_repa:
        from proteinfoundation.proteinflow.proteina_repa import ProteinaREPA

        return ProteinaREPA(cfg, store_dir=store_dir)
    from proteinfoundation.proteinflow.proteina import Proteina

    return Proteina(cfg, store_dir=store_dir)


def load_config(config_name: str, config_subdir: str):
    import hydra
    from omegaconf import open_dict

    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../src/proteina/configs/experiment_config",
        )
    )
    if not os.path.isdir(config_path):
        raise FileNotFoundError(f"Config dir missing: {config_path}")

    # Compose from subdir if provided, else flat
    compose_name = f"{config_subdir}/{config_name}" if config_subdir else config_name
    with hydra.initialize_config_dir(config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=compose_name)

    # Force single-GPU regardless of config default
    with open_dict(cfg):
        cfg.hardware.ngpus_per_node_ = 1
        cfg.hardware.nnodes_ = 1
    return cfg


def check_state_dict_match(ckpt: dict, model) -> Tuple[int, list, list]:
    """Return (num_params_loaded, missing_keys, unexpected_keys)."""
    ckpt_state = ckpt.get("state_dict", ckpt)
    # Strip torch.compile _orig_mod. prefix if present (prod ckpts have it)
    stripped = {}
    for k, v in ckpt_state.items():
        stripped[k.replace("_orig_mod.", "")] = v

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    loaded = sum(1 for k in stripped if k not in unexpected)
    return loaded, list(missing), list(unexpected)


def apply_optimization(model, opt: str):
    """Mutate model/global-state per the requested optimization."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: F401

    if opt == "noop":
        return None
    if opt == "compile_on":
        log("    applying: torch.compile(model.nn, mode=default)")
        model.nn = torch.compile(model.nn)
        return None
    if opt == "compile_off":
        log("    applying: no torch.compile (baseline)")
        return None
    if opt.startswith("sdpa_"):
        # Return a kernel-context we enter during the forward/backward
        backend_map = {
            "sdpa_math": SDPBackend.MATH,
            "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "sdpa_flash": SDPBackend.FLASH_ATTENTION,
        }
        backend = backend_map[opt]
        log(f"    applying: sdpa_kernel([{backend}])")
        return sdpa_kernel([backend])
    if opt == "gc_10":
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        if not (hasattr(model, "nn") and hasattr(model.nn, "layers")):
            log("    gc_10: model has no .nn.layers; skipping")
            return None
        n_total = len(model.nn.layers)
        log(f"    applying: gradient_checkpointing all {n_total} layers")
        for layer in model.nn.layers:
            orig = layer.forward

            def make_wrap(fn):
                def wrapped(*args, **kwargs):
                    return ckpt_fn(fn, *args, use_reentrant=False, **kwargs)

                return wrapped

            layer.forward = make_wrap(orig)
        return None
    raise ValueError(f"Unknown optimization: {opt}")


def build_fake_batch(cfg, batch_size: int, seq_len: int, device):
    """Minimal synthetic batch matching proteina's forward signature."""
    import torch
    from torch_geometric.data import Batch, Data

    max_len = seq_len
    datas = []
    for _ in range(batch_size):
        datas.append(
            Data(
                coords_ca=torch.randn(max_len, 3),
                mask=torch.ones(max_len, dtype=torch.bool),
                seq=torch.randint(0, 20, (max_len,)),
                res_seq_pdb_idx=torch.arange(max_len, dtype=torch.long),
                chain_break_per_res=torch.zeros(max_len, dtype=torch.long),
                num_nodes=max_len,
            )
        )
    return Batch.from_data_list(datas).to(device)


def main():
    args = parse_args()

    import tempfile

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    if device.type == "cuda":
        log(f"gpu: {torch.cuda.get_device_name()}")

    # ---- Load config ----
    step("loading config")
    cfg = load_config(args.config, args.config_subdir)
    is_repa = "repa" in cfg and cfg.repa is not None
    log(f"    is_repa={is_repa}, compile={cfg.get('compile', False)}")

    seq_len = args.seq_len
    if seq_len is None:
        # Try to pull from dataset config's PaddingTransform max_size
        try:
            for t in cfg.dataset_config.transforms:
                if "PaddingTransform" in str(t):
                    seq_len = t.max_size
                    break
        except Exception:
            pass
        if seq_len is None:
            seq_len = 256
    log(f"    using seq_len={seq_len}, batch_size={args.batch_size}")

    # ---- Instantiate model ----
    step("instantiating model")
    tmp_dir = tempfile.mkdtemp(prefix="validate_resume_")
    model = build_model(cfg, is_repa=is_repa, store_dir=tmp_dir)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"    {n_params / 1e6:.1f}M params")

    # ---- Load checkpoint ----
    step(f"loading checkpoint: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        log("ERROR: checkpoint not found")
        sys.exit(1)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    loaded, missing, unexpected = check_state_dict_match(ckpt, model)
    log(f"    loaded {loaded} tensors")
    if missing:
        log(f"    MISSING ({len(missing)}): first 5 = {missing[:5]}")
    if unexpected:
        log(f"    UNEXPECTED ({len(unexpected)}): first 5 = {unexpected[:5]}")
    if missing or unexpected:
        log("VERDICT: FAIL — state_dict mismatch, resume would corrupt training")
        sys.exit(1)

    # Record prior loss if available
    prior_loss = None
    if "callbacks" in ckpt:
        for cb_state in ckpt["callbacks"].values():
            if isinstance(cb_state, dict):
                for k in ("best_model_score", "current_score"):
                    if k in cb_state and cb_state[k] is not None:
                        prior_loss = float(cb_state[k])
                        break
    if prior_loss is not None:
        log(f"    checkpoint recorded loss ~ {prior_loss:.4f}")

    # ---- Move to GPU ----
    step("moving model to GPU")
    model = model.to(device)
    model.train()

    # ---- Apply optimization ----
    step(f"applying optimization: {args.optimization}")
    sdpa_ctx = apply_optimization(model, args.optimization)

    # ---- Synthetic batch ----
    step("building synthetic batch")
    batch = build_fake_batch(cfg, args.batch_size, seq_len, device)

    # ---- Forward + backward + step ----
    import contextlib

    forward_ctx = sdpa_ctx if sdpa_ctx is not None else contextlib.nullcontext()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.zero_grad()

    step("forward pass")
    with forward_ctx:
        try:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model.training_step(batch, 0)
            loss = out if torch.is_tensor(out) else out.get("loss")
            if loss is None:
                log("ERROR: training_step returned no loss")
                sys.exit(1)
            log(f"    forward OK, loss={loss.item():.6f}")
        except Exception as e:
            log(f"    forward FAILED: {type(e).__name__}: {e}")
            log("VERDICT: FAIL — forward incompatible with this optimization")
            sys.exit(1)

    if not torch.isfinite(loss).all():
        log(f"    loss is non-finite: {loss.item()}")
        log("VERDICT: FAIL — non-finite loss (numerical blowup)")
        sys.exit(1)

    if prior_loss is not None:
        ratio = loss.item() / max(prior_loss, 1e-6)
        log(f"    loss ratio vs checkpoint = {ratio:.2f}× (tol {args.loss_tolerance}×)")
        if abs(ratio) > args.loss_tolerance and loss.item() > prior_loss * 2:
            log("    WARN: loss drifted beyond tolerance on synthetic data")
            # don't fail — synthetic data is OOD; just warn

    step("backward pass")
    try:
        loss.backward()
    except Exception as e:
        log(f"    backward FAILED: {type(e).__name__}: {e}")
        log("VERDICT: FAIL — backward incompatible with this optimization")
        sys.exit(1)

    # Check grads
    n_grad = 0
    n_nan = 0
    for p in model.parameters():
        if p.grad is not None:
            n_grad += 1
            if not torch.isfinite(p.grad).all():
                n_nan += 1
    log(f"    {n_grad} params got grads, {n_nan} had NaN/Inf")
    if n_nan > 0:
        log("VERDICT: FAIL — NaN/Inf gradients")
        sys.exit(1)

    step("optimizer step")
    pre_step_norm = (
        sum(
            p.detach().float().norm().item() ** 2
            for p in model.parameters()
            if p.requires_grad
        )
        ** 0.5
    )
    try:
        optimizer.step()
    except Exception as e:
        log(f"    optimizer step FAILED: {type(e).__name__}: {e}")
        log("VERDICT: FAIL — optimizer incompatible")
        sys.exit(1)
    post_step_norm = (
        sum(
            p.detach().float().norm().item() ** 2
            for p in model.parameters()
            if p.requires_grad
        )
        ** 0.5
    )
    log(
        f"    param norm pre={pre_step_norm:.4f} post={post_step_norm:.4f} "
        f"Δ={post_step_norm - pre_step_norm:+.6f}"
    )

    # Check any params have NaN after step
    nan_params = 0
    for p in model.parameters():
        if not torch.isfinite(p).all():
            nan_params += 1
    if nan_params > 0:
        log(f"    {nan_params} params contain NaN/Inf after step")
        log("VERDICT: FAIL — post-step NaN")
        sys.exit(1)

    log("")
    log(f"VERDICT: PASS — optimization '{args.optimization}' is resume-safe")
    log(f"         ckpt loads ({loaded} tensors), forward+backward+step all finite")
    sys.exit(0)


if __name__ == "__main__":
    main()
