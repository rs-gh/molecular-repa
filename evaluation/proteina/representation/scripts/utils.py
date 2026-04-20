# ruff: noqa: F401
"""Compatibility shim — forwards to ``lib``.

Historical entry point. New code should import from ``lib`` directly:

    from lib import load_proteina_batch, extract_model_hidden_states_multilayer

This shim exists so ``run_sweep.py``, ``run_all.py``, ``patch_cath.py``,
``pareto.py``, and any external callers keep working unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``lib`` lives in the parent dir (.../representation/lib); make sure it's
# importable even when utils.py is imported before sys.path is set up.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib import (  # noqa: E402
    BASELINE_STEPS,
    CATHResult,
    CHECKPOINT_REGISTRY,
    ContactResult,
    LMDB_PATH,
    PROTEINA_ROOT,
    REPA_L0_STEPS,
    REPA_L9_STEPS,
    REPA_STEPS,
    RUN_SCHEDULES,
    STORE_ROOT,
    cath_labels_from_raw,
    contact_labels,
    enable_hidden_states,
    extract_gearnet_embeddings,
    extract_model_hidden_states,
    extract_model_hidden_states_multilayer,
    find_checkpoint_path,
    linear_probe_contacts,
    load_checkpoint,
    load_checkpoint_by_path,
    load_proteina_batch,
    mean_pool_by_mask,
    model_num_layers,
    run_cath_probe,
    run_contact_probe,
)
from lib.data import _default_device  # noqa: E402, F401 — private but used by run_sweep

__all__ = [
    # constants
    "PROTEINA_ROOT",
    "LMDB_PATH",
    "STORE_ROOT",
    # schedules / registries
    "BASELINE_STEPS",
    "REPA_STEPS",
    "REPA_L0_STEPS",
    "REPA_L9_STEPS",
    "RUN_SCHEDULES",
    "CHECKPOINT_REGISTRY",
    # data
    "load_proteina_batch",
    "_default_device",
    # checkpoints
    "find_checkpoint_path",
    "load_checkpoint_by_path",
    "load_checkpoint",
    # extraction
    "enable_hidden_states",
    "extract_model_hidden_states",
    "extract_model_hidden_states_multilayer",
    "extract_gearnet_embeddings",
    "model_num_layers",
    # labels
    "contact_labels",
    "cath_labels_from_raw",
    "mean_pool_by_mask",
    # probes
    "ContactResult",
    "linear_probe_contacts",
    "run_contact_probe",
    "CATHResult",
    "run_cath_probe",
]
