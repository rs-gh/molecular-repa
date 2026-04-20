"""Probelib — modular internals for the proteina representation-quality probes.

Split out from the original monolithic ``utils.py``:

    data.py         LMDB loading, padding, sampling. Produces the shared batch
                    dict that every rep source is scored against.
    checkpoints.py  Run schedules, checkpoint path resolution, LightningModule
                    loading for Proteina / ProteinaREPA.
    extract.py      Hidden-state extraction from Proteina trunks + frozen
                    GearNet encoder.
    labels.py       Ground-truth constructors (contact map, CATH, mean-pool).
    probes.contact  Linear/MLP contact probe (P@L/k).
    probes.cath     CATH fold classification probe (accuracy + macro-F1).

Shape-annotation convention:
    reps:    [B, N, D]        per-residue hidden states (CPU unless stated)
    mask:    [B, N]           bool, True = real residue
    coords:  [B, N, 37, 3]    Å, atom-37 heavy-atom layout; CA at index 1
    labels:  [B, N, N]        binary contacts, 0/1
    feats:   [P, 3D]          pair features, h_i ‖ h_j ‖ |h_i - h_j|

The public names below are re-exported here so callers can write
``from lib import load_proteina_batch, run_contact_probe`` in new code.
The legacy ``utils.py`` / ``contact.py`` / ``cath.py`` are thin shims around
these exports — don't add new code there.
"""

from lib.checkpoints import (
    BASELINE_STEPS,
    CHECKPOINT_REGISTRY,
    LMDB_PATH,
    PROTEINA_ROOT,
    REPA_L0_STEPS,
    REPA_L9_STEPS,
    REPA_STEPS,
    RUN_SCHEDULES,
    STORE_ROOT,
    find_checkpoint_path,
    load_checkpoint,
    load_checkpoint_by_path,
)
from lib.data import _default_device, load_proteina_batch
from lib.extract import (
    enable_hidden_states,
    extract_gearnet_embeddings,
    extract_model_hidden_states,
    extract_model_hidden_states_multilayer,
    model_num_layers,
)
from lib.labels import (
    cath_labels_from_raw,
    contact_labels,
    mean_pool_by_mask,
)
from lib.manifest import (
    build_or_load_manifest,
    load_proteina_batch_from_manifest,
    sample_manifest,
)
from lib.probes.cath import CATHResult, run_cath_probe
from lib.probes.contact import (
    ContactResult,
    MultiSeedResult,
    evaluate_contact_from_scores,
    linear_probe_contacts,
    linear_probe_contacts_multi,
    run_contact_probe,
    run_contact_probe_full,
)
from lib.sources import (
    LAYER_DISTANCE_ONLY,
    LAYER_GEARNET,
    LAYER_RANDOM_GAUSS,
    LAYER_RANDOM_RANK,
    LAYER_SEQ_ONEHOT,
    AnalyticScorer,
    CheckpointRepSource,
    DistanceOnlyScorer,
    GearnetRepSource,
    RandomGaussianRepSource,
    RandomRankScorer,
    RepSource,
    SeqOnehotRepSource,
    UntrainedProteinaRepSource,
)

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
    # manifest
    "sample_manifest",
    "build_or_load_manifest",
    "load_proteina_batch_from_manifest",
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
    "MultiSeedResult",
    "linear_probe_contacts",
    "linear_probe_contacts_multi",
    "run_contact_probe",
    "run_contact_probe_full",
    "evaluate_contact_from_scores",
    "CATHResult",
    "run_cath_probe",
    # sources
    "RepSource",
    "AnalyticScorer",
    "CheckpointRepSource",
    "UntrainedProteinaRepSource",
    "GearnetRepSource",
    "RandomGaussianRepSource",
    "SeqOnehotRepSource",
    "RandomRankScorer",
    "DistanceOnlyScorer",
    "LAYER_GEARNET",
    "LAYER_RANDOM_GAUSS",
    "LAYER_SEQ_ONEHOT",
    "LAYER_RANDOM_RANK",
    "LAYER_DISTANCE_ONLY",
]
