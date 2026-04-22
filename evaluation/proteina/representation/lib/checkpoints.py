# Re-export from the shared evaluation/proteina/lib/checkpoints.py so that
# existing imports (from lib.checkpoints import ...) continue to work unchanged.
import importlib.util
import sys as _sys
from pathlib import Path

_shared_path = Path(__file__).resolve().parents[2] / "lib" / "checkpoints.py"
_spec = importlib.util.spec_from_file_location(
    "_proteina_shared_checkpoints", _shared_path
)
_mod = importlib.util.module_from_spec(_spec)
# Register before exec so any internal imports don't re-trigger loading.
_sys.modules["_proteina_shared_checkpoints"] = _mod
_spec.loader.exec_module(_mod)

# Populate this module's namespace with everything from the shared module.
from _proteina_shared_checkpoints import *  # noqa: E402,F401,F403
from _proteina_shared_checkpoints import (  # noqa: E402,F401 -- explicit for type checkers
    BASELINE_STEPS,
    CHECKPOINT_REGISTRY,
    GEN_RUN_CONFIGS,
    LMDB_PATH,
    PRETRAINED_CHECKPOINTS,
    REPA_L0_STEPS,
    REPA_L9_STEPS,
    REPA_STEPS,
    RUN_SCHEDULES,
    STORE_ROOT,
    find_checkpoint_path,
    load_checkpoint,
    load_checkpoint_by_path,
    resolve_step,
)
