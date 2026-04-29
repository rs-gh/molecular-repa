"""Mode-aware model loader for the generation-speedup benchmark.

Four modes:
  eager_fp32   - current eval path (baseline)
  compile_fp32 - torch.compile(nn_ca) in fp32
  eager_bf16   - bf16 autocast, no compile
  compile_bf16 - compile + bf16 (candidate fast path)

Usage:
    from playground.proteina.generation_speedup._speedup_utils import prepare_model
    model, autocast_ctx = prepare_model(ckpt_path, is_repa=False, mode="compile_bf16")
    with autocast_ctx():
        x = model.generate(...)
"""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path("/home/sr2173/git/molecular-repa")
PROTEINA_ROOT = REPO_ROOT / "src" / "proteina"

if str(PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTEINA_ROOT))
if str(PROTEINA_ROOT / "proteinfoundation") not in sys.path:
    sys.path.insert(0, str(PROTEINA_ROOT / "proteinfoundation"))


MODES = ("eager_fp32", "compile_fp32", "eager_bf16", "compile_bf16")


def _patch_torch_load_once():
    """Install the state-dict patch used by our SLURM eval wrappers.

    Strips `_orig_mod.` prefixes (left over from torch.compile-trained ckpts) and
    filters out `repa_loss.*` keys so REPA checkpoints load as plain Proteina.
    Idempotent - safe to call multiple times.
    """
    if getattr(torch.load, "_speedup_patched", False):
        return
    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs["weights_only"] = False
        result = _orig(*args, **kwargs)
        if isinstance(result, dict) and "state_dict" in result:
            sd = result["state_dict"]
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            result["state_dict"] = {
                k: v for k, v in sd.items() if not k.startswith("repa_loss.")
            }
        return result

    _patched._speedup_patched = True
    torch.load = _patched


def _load_base_model(ckpt_path: Path):
    """Load the Proteina LightningModule on GPU, eval mode."""
    # pyg_compat needs to be imported before proteinfoundation model imports
    import proteinfoundation.repa.pyg_compat  # noqa: F401
    from proteinfoundation.proteinflow.proteina import Proteina

    _patch_torch_load_once()
    model = Proteina.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    model.cuda()
    return model


def prepare_model(ckpt_path: Path, is_repa: bool, mode: str, inference_cfg=None):
    """Load a Proteina model and return (model, autocast_ctx_factory).

    Args:
        ckpt_path: Path to the .ckpt file.
        is_repa: Unused at inference (REPA keys already stripped by torch.load patch),
                 kept for API symmetry with callers.
        mode: One of MODES.
        inference_cfg: Optional OmegaConf / DictConfig passed to
                       `model.configure_inference(cfg, nn_ag=None)`. Required for
                       generation; can be None if only probing.

    Returns:
        (model, autocast_ctx_factory) where autocast_ctx_factory is a no-arg
        callable producing a context manager. Use it as
            with autocast_ctx():
                x = model.generate(...)
        For fp32 modes this is a nullcontext.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode={mode!r}, expected one of {MODES}")

    model = _load_base_model(ckpt_path)

    if inference_cfg is not None:
        model.configure_inference(inference_cfg, nn_ag=None)

    if mode.startswith("compile"):
        # Static-shape compile. bs + nres are fixed per bench cell so this is safe.
        # reduce-overhead mode caches CUDAgraphs; incurs a one-time compile cost
        # on the first forward which the bench treats as warmup.
        model.nn = torch.compile(
            model.nn, mode="reduce-overhead", dynamic=False, fullgraph=False
        )

    if mode.endswith("bf16"):

        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:

        def autocast_ctx():
            return nullcontext()

    return model, autocast_ctx


@contextmanager
def cuda_timer():
    """Measure wall time of a GPU block with torch.cuda.Event (synchronized)."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = {}
    try:
        yield result
    finally:
        end.record()
        torch.cuda.synchronize()
        result["seconds"] = start.elapsed_time(end) / 1000.0
