"""Centralized ``torch.load`` monkey-patch used by the proteina eval suites.

Two distinct flavours are needed across the repo:

- **Generation** loads a Proteina (non-REPA) class from a REPA-trained
  checkpoint, so ``repa_loss.*`` keys must be stripped before the strict
  Lightning load. ``apply(strip_repa=True)``.
- **Representation** keeps ``repa_loss.*`` (it loads ProteinaREPA) but still
  needs the ``_orig_mod.`` prefix-strip and the ``weights_only=False``
  override. ``apply(strip_repa=False)``.

Both flavours also force ``weights_only=False`` because Lightning checkpoints
contain non-tensor objects (Hydra configs, optimizer state). PyTorch 2.4+
defaults to ``weights_only=True``, which would raise on those.

Call ``apply()`` exactly once per Python process, before any
``Proteina.load_from_checkpoint`` / ``ProteinaREPA.load_from_checkpoint``
call. Subsequent calls are idempotent.

Other call-sites (eval_fid.sh, representation/run_sweep.sh,
run_pretrained_probe.sh, playground/.../_speedup_utils.py,
tabasco/generation/scripts/evaluate.py) still inline the same patch — they
can be migrated to this utility in a follow-up cleanup.
"""

from __future__ import annotations

import torch

_ORIGINAL_TORCH_LOAD = None


def apply(strip_repa: bool = True) -> None:
    """Install the patched ``torch.load`` for proteina checkpoint loading.

    Args:
        strip_repa: If True, drop ``repa_loss.*`` keys from the loaded
            ``state_dict`` (generation flavour). If False, keep them
            (representation flavour, where the ProteinaREPA class needs them).
    """
    global _ORIGINAL_TORCH_LOAD
    if _ORIGINAL_TORCH_LOAD is None:
        _ORIGINAL_TORCH_LOAD = torch.load

    original_load = _ORIGINAL_TORCH_LOAD

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        result = original_load(*args, **kwargs)
        if isinstance(result, dict) and "state_dict" in result:
            sd = result["state_dict"]
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            if strip_repa:
                sd = {k: v for k, v in sd.items() if not k.startswith("repa_loss.")}
            result["state_dict"] = sd
        return result

    torch.load = _patched_load
