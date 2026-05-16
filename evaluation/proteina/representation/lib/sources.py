"""Representation sources — a unified interface for every "thing that emits
per-residue features" so the sweep iterates over trained checkpoints, untrained
transformers, random features, sequence one-hot, encoders, and analytic
baselines through a single loop.

Two protocols:

    RepSource
        Produces [B, N, D] hidden states for one or more layers. The sweep
        feeds those reps through the standard contact-probe MLP + CATH probe.
        Used by: CheckpointRepSource, UntrainedProteinaRepSource,
                 GearnetRepSource, RandomGaussianRepSource, SeqOnehotRepSource.

    AnalyticScorer
        Skips the MLP entirely — emits a [B, N, N] pair-score tensor directly.
        The sweep evaluates P@L/k by ranking those scores against the same
        long-range contact labels. Used by: RandomRankScorer, DistanceOnlyScorer.

Layer-sentinel convention (negative ints for non-checkpoint sources, so they
never collide with real transformer layer indices 0..9 in plots):

    -1    gearnet (existing, unchanged)
    -2    random_gauss
    -3    seq_onehot
    -10..-19  untrained_proteina (one row per trunk layer L=0..9, sentinel = -(L + 10))
    -20..-29  trained_noise      (one row per trunk layer L=0..9, sentinel = -(L + 20))
    -100  random_rank   (analytic)
    -101  distance_only (analytic)
    -200  knn_dist      (8 nearest-CA distances; inverse-folding baseline)
    -201  local_frame   (5-atom local-frame coords; dihedral baseline)
    -202  seqsep_pair   (|i-j| only; distance baseline; head bypasses reps)

Untrained-proteina uses layer = -(L + 1) so L=0 → -1 would collide with
gearnet. We offset to -(L + 10) instead: layer 0 → -10, ..., layer 9 → -19.
Trained-noise uses the same scheme one block lower (offset 20) so the two
"shadow transformer" baselines occupy disjoint sentinel ranges and can be
plotted independently. Rationale: keep both visibly "transformer-shaped" in
plots (10 rows each) while still sentinel-negative and non-colliding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import torch
import torch.nn.functional as F

from lib.extract import (
    extract_gearnet_embeddings,
    extract_model_hidden_states_multilayer,
)

GEARNET_CKPT_PATH_ENV = "GEARNET_CKPT_PATH"

# Sentinel layer indices for non-checkpoint sources. Referenced by
# run_sweep.py and plot.py to route records appropriately.
LAYER_GEARNET = -1
LAYER_RANDOM_GAUSS = -2
LAYER_SEQ_ONEHOT = -3
LAYER_RANDOM_RANK = -100
LAYER_DISTANCE_ONLY = -101
# Trivial-geometric baselines for the new probes (P3-P5).
# These emit per-residue features that are deterministic functions of coords:
# the head's score on these establishes a hard floor that any informative
# learned representation must clear by a wide margin.
LAYER_KNN_DIST = -200  # 8 nearest-CA distances per residue (inverse folding)
LAYER_LOCAL_FRAME = -201  # 5-atom local-frame coords per residue (dihedral)
LAYER_SEQSEP_PAIR = -202  # |i-j| only (distance baseline; head bypasses reps entirely)


def _untrained_layer_sentinel(transformer_layer: int) -> int:
    """Map trunk layer k (0..9) to sentinel -(k + 10). Keeps untrained rows
    visually grouped as a 10-row "shadow" transformer without colliding with
    gearnet (-1) / random_gauss (-2) / seq_onehot (-3).
    """
    return -(transformer_layer + 10)


def _trained_noise_layer_sentinel(transformer_layer: int) -> int:
    """Map trunk layer k (0..9) to sentinel -(k + 20). Disjoint from
    untrained_proteina's -(k + 10) range so the two shadow transformers can
    appear on the same plot without colliding.
    """
    return -(transformer_layer + 20)


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


class RepSource(Protocol):
    """A source of per-residue hidden states consumed by the standard probes."""

    name: str
    step: int  # for the results schema; encoder-like sources use 0
    emits_layers: List[int]  # layer ids (real 0..9 for ckpts, negatives for baselines)

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Return {layer: [B, N, D]} on CPU for the requested layer subset."""
        ...


class AnalyticScorer(Protocol):
    """A source that emits pair scores directly, bypassing the MLP probe."""

    name: str
    step: int
    emits_layers: List[int]

    def score_pairs(self, batch: Dict) -> torch.Tensor:
        """Return [B, N, N] float scores. Higher = more likely contact."""
        ...


# --------------------------------------------------------------------------- #
# Checkpoint-backed sources (trained + untrained)
# --------------------------------------------------------------------------- #


@dataclass
class CheckpointRepSource:
    """Wraps a trained Proteina / ProteinaREPA checkpoint.

    Ports the existing per-checkpoint hidden-state extraction into the
    RepSource protocol — run_sweep.py's main loop becomes a single iteration
    over ``for source in sources``.

    Flow:
        lazy-load model on first get_reps()
        extract_model_hidden_states_multilayer(model, batch, layers)
        return {L: [B, N, D]}
    """

    name: str  # e.g. "baseline", "repa_l4"
    step: int  # checkpoint step
    ckpt_path: str  # full path to .ckpt
    is_repa: bool
    emits_layers: List[int]  # trunk layers to capture
    _model: Optional[object] = None

    def _load(self, device: str):
        from lib.checkpoints import load_checkpoint_by_path

        self._model = load_checkpoint_by_path(
            self.ckpt_path, is_repa=self.is_repa, device=device
        )

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        assert set(layers).issubset(
            self.emits_layers
        ), f"requested layers {layers} not in {self.emits_layers}"
        if self._model is None:
            device = (
                batch["coords"].device.type
                if isinstance(batch["coords"], torch.Tensor)
                else "cuda"
            )
            self._load(device)
        return extract_model_hidden_states_multilayer(self._model, batch, layers)

    def unload(self):
        """Drop the model from memory — call between checkpoints to free GPU."""
        import gc

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class UntrainedProteinaRepSource:
    """Freshly-initialized Proteina — same architecture as baseline, random weights.

    The "random transformer features" baseline. Shows what a 60M-parameter
    wide transformer gives you before any training. If trained models barely
    exceed this, either training isn't helping OR the probe is too easy.

    Instantiation uses Hydra to compose the baseline_256 config (the
    architecture is shared across all sequence lengths). No checkpoint load.

    Layer sentinels: _untrained_layer_sentinel(L) = -(L + 10) so the 10
    trunk layers appear as rows with layer ∈ {-10, -11, ..., -19}.
    """

    def __init__(
        self,
        config_name: str = "training_baseline_256",
        layers: Optional[List[int]] = None,
    ):
        self.name = "untrained_proteina"
        self.step = 0
        self.config_name = config_name
        # Default: all 10 trunk layers of the 60M model. Override for quick smokes.
        self._trunk_layers = layers if layers is not None else list(range(10))
        self.emits_layers = [_untrained_layer_sentinel(L) for L in self._trunk_layers]
        self._model = None
        # Inverse mapping for get_reps: sentinel -> trunk layer.
        self._sentinel_to_trunk = {
            _untrained_layer_sentinel(L): L for L in self._trunk_layers
        }

    def _load(self, device: str):
        import hydra
        from proteinfoundation.proteinflow.proteina import Proteina

        # Hydra must be re-initialized cleanly each call; use the context mgr.
        # config_path is relative to this file:
        # lib/sources.py -> ../../../../src/proteina/configs/experiment_config
        # (4 levels up: lib/, representation/, proteina/, evaluation/).
        rel_cfg_dir = "../../../../src/proteina/configs/experiment_config"
        with hydra.initialize(config_path=rel_cfg_dir, version_base="1.3"):
            cfg = hydra.compose(config_name=self.config_name)
        self._model = Proteina(cfg, store_dir="/tmp")
        self._model.eval()
        self._model.to(device)

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        if self._model is None:
            device = (
                batch["coords"].device.type
                if isinstance(batch["coords"], torch.Tensor)
                else "cuda"
            )
            self._load(device)
        # Translate sentinels -> real trunk layers for extraction.
        trunk_layers = [self._sentinel_to_trunk[s] for s in layers]
        reps_by_trunk = extract_model_hidden_states_multilayer(
            self._model, batch, trunk_layers
        )
        return {_untrained_layer_sentinel(L): r for L, r in reps_by_trunk.items()}

    def unload(self):
        import gc

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class NoiseInputCheckpointRepSource:
    """Trained Proteina checkpoint with inputs replaced by pure flow-matching
    reference noise at t=0.0.

    The "good model + uninformative coords" baseline. Loads a trained
    checkpoint and runs ``extract_model_hidden_states_multilayer`` with
    ``t_value=0.0``, which sets x_t = x_0 (a sample from the model's own
    reference distribution — the harmonic chain prior the model was trained
    to denoise). Real ``mask`` and ``residue_type`` from the batch are kept
    unchanged so the per-protein length distribution and sequence statistics
    match the trained-checkpoint rows; only the coords are replaced. This
    measures how much CATH-relevant structure is encoded in the trained
    weights independent of the input being a real protein.

    Counterpart to UntrainedProteinaRepSource: the latter is "real input,
    random weights"; this one is "noise input, trained weights".

    Layer sentinels: trunk layer L -> -(L + 20), so L=0 -> -20, ..., L=9 -> -29.
    """

    def __init__(
        self,
        ckpt_path: str,
        is_repa: bool,
        run_name: str = "trained_noise",
        layers: Optional[List[int]] = None,
    ):
        self.name = run_name
        self.step = 0
        self.ckpt_path = ckpt_path
        self.is_repa = is_repa
        self._trunk_layers = layers if layers is not None else list(range(10))
        self.emits_layers = [
            _trained_noise_layer_sentinel(L) for L in self._trunk_layers
        ]
        self._model = None
        self._sentinel_to_trunk = {
            _trained_noise_layer_sentinel(L): L for L in self._trunk_layers
        }

    def _load(self, device: str):
        from lib.checkpoints import load_checkpoint_by_path

        self._model = load_checkpoint_by_path(
            self.ckpt_path, is_repa=self.is_repa, device=device
        )

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        if self._model is None:
            device = (
                batch["coords"].device.type
                if isinstance(batch["coords"], torch.Tensor)
                else "cuda"
            )
            self._load(device)
        trunk_layers = [self._sentinel_to_trunk[s] for s in layers]
        # t_value=0.0 -> the extractor draws x_t from the model's reference
        # distribution and feeds that in place of real coords.
        reps_by_trunk = extract_model_hidden_states_multilayer(
            self._model, batch, trunk_layers, t_value=0.0
        )
        return {_trained_noise_layer_sentinel(L): r for L, r in reps_by_trunk.items()}

    def unload(self):
        import gc

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Encoder source (gearnet, ported from the existing path in run_sweep.py)
# --------------------------------------------------------------------------- #


class GearnetRepSource:
    """Frozen GearNet CA-only encoder — the REPA alignment target.

    Ported from run_sweep.py's existing gearnet branch. Single-layer output
    (emits_layers=[LAYER_GEARNET=-1]).
    """

    def __init__(self, ckpt_path: Optional[str] = None):
        self.name = "gearnet"
        self.step = 0
        self.emits_layers = [LAYER_GEARNET]
        self.ckpt_path = ckpt_path or os.environ.get(
            GEARNET_CKPT_PATH_ENV,
            "/rds/user/sr2173/hpc-work/proteina/data/metric_factory/model_weights/gearnet_ca.pth",
        )
        self._encoder = None

    def _load(self, device: str):
        from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder

        enc = GearNetPerResidueEncoder(ckpt_path=self.ckpt_path)
        enc.eval()
        self._encoder = enc.to(device)

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        if self._encoder is None:
            device = (
                batch["coords"].device.type
                if isinstance(batch["coords"], torch.Tensor)
                else "cuda"
            )
            self._load(device)
        reps = extract_gearnet_embeddings(self._encoder, batch)  # [B, N, 512]
        return {LAYER_GEARNET: reps}

    def unload(self):
        import gc

        self._encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Synthetic rep sources (no model involved)
# --------------------------------------------------------------------------- #


class RandomGaussianRepSource:
    """Random Gaussian features per residue.

    Probes the MLP's capacity to overfit pure noise — sets the "memorization
    floor" for the contact probe. If the trained model barely beats this,
    the MLP is doing most of the work.

    Flow:
        reps = randn(B, N, D) * mask[..., None]
        return {LAYER_RANDOM_GAUSS: reps}
    """

    def __init__(self, dim: int = 512, seed: int = 42):
        self.name = "random_gauss"
        self.step = 0
        self.emits_layers = [LAYER_RANDOM_GAUSS]
        self.dim = dim
        self.seed = seed

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed)
        B, N = batch["mask"].shape
        reps = torch.randn(B, N, self.dim, generator=g)  # [B, N, D]
        mask = batch["mask"].cpu().float().unsqueeze(-1)  # [B, N, 1]
        reps = reps * mask  # zero out padding positions
        return {LAYER_RANDOM_GAUSS: reps}

    def unload(self):
        pass


class KnnDistanceRepSource:
    """Sorted distances to the k nearest non-self real CAs (inverse-folding baseline).

    Tests how much of the AA-classification signal is in raw local geometry,
    independent of any learned representation. A linear/MLP head on these 8
    distances should noticeably beat random_gauss (geometry IS predictive of
    AA — Gly likes loose backbones, Pro forces kinks, etc.) but should still
    fall well short of a trained model that also encodes chemistry.
    """

    def __init__(self, k: int = 8):
        self.name = "knn_dist"
        self.step = 0
        self.emits_layers = [LAYER_KNN_DIST]
        self.k = k

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        from lib.labels import knn_distance_features

        ca = batch["coords"][:, :, 1, :].cpu()  # [B, N, 3] Å
        mask = batch["mask"].cpu()
        feats, _valid = knn_distance_features(ca, mask, k=self.k)  # [B, N, k]
        return {LAYER_KNN_DIST: feats}

    def unload(self):
        pass


class LocalFrameRepSource:
    """Backbone (N, CA, C of i-1, i, i+1) in a local frame at CA_i.

    Analytical lower bound for the dihedral probe: the 5 backbone atoms
    spanning φ_i and ψ_i are encoded directly, so a sufficiently expressive
    head can recover the angles essentially exactly. If our learned rep
    doesn't approach this MAE, the rep has lost information that's literally
    in the input coords.
    """

    def __init__(self):
        self.name = "local_frame"
        self.step = 0
        self.emits_layers = [LAYER_LOCAL_FRAME]

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        from lib.labels import local_frame_features

        coords = batch["coords"].cpu()  # [B, N, 37, 3]
        mask = batch["mask"].cpu()
        feats, _valid = local_frame_features(coords, mask)  # [B, N, 15]
        return {LAYER_LOCAL_FRAME: feats}

    def unload(self):
        pass


class SeqOnehotRepSource:
    """Sequence-only one-hot features (dim = 20 AA).

    Probes how much of the contact signal is carried by residue identity
    alone, before any structural context. The floor for any model that does
    more than just read the sequence.

    Flow:
        reps = F.one_hot(residue_type, 20).float() * mask[..., None]
        return {LAYER_SEQ_ONEHOT: reps}   # shape [B, N, 20]
    """

    def __init__(self, n_classes: int = 20):
        self.name = "seq_onehot"
        self.step = 0
        self.emits_layers = [LAYER_SEQ_ONEHOT]
        self.n_classes = n_classes

    def get_reps(self, batch: Dict, layers: List[int]) -> Dict[int, torch.Tensor]:
        rt = batch["residue_type"].cpu().long()  # [B, N]
        # Clamp to valid range; padded positions may have garbage indices.
        rt = rt.clamp(0, self.n_classes - 1)
        reps = F.one_hot(rt, num_classes=self.n_classes).float()  # [B, N, 20]
        mask = batch["mask"].cpu().float().unsqueeze(-1)  # [B, N, 1]
        reps = reps * mask
        return {LAYER_SEQ_ONEHOT: reps}

    def unload(self):
        pass


# --------------------------------------------------------------------------- #
# Analytic scorers (no MLP)
# --------------------------------------------------------------------------- #


class RandomRankScorer:
    """Uniform random per-pair scores, averaged across ``seeds`` draws.

    Establishes the analytical chance floor for P@L/k. Expected value equals
    the natural positive rate among long-range pairs (typically 1-3%).
    Seeds average so one unlucky permutation doesn't noise the baseline.
    """

    def __init__(self, seeds: Optional[List[int]] = None):
        self.name = "random_rank"
        self.step = 0
        self.emits_layers = [LAYER_RANDOM_RANK]
        self.seeds = list(seeds) if seeds is not None else [42, 43, 44, 45, 46]

    def score_pairs_per_seed(self, batch: Dict):
        """Yield (seed, [B, N, N] scores) for each of ``self.seeds``."""
        B, N = batch["mask"].shape
        for seed in self.seeds:
            g = torch.Generator().manual_seed(seed)
            s = torch.rand(B, N, N, generator=g)  # [B, N, N]
            yield seed, s

    def unload(self):
        pass


class DistanceOnlyScorer:
    """Score = |i - j| — rank pairs by sequence separation.

    A tiny nontrivial baseline that uses zero representation. In most proteins
    long-range contacts are roughly uniformly distributed across |i-j|, so
    this should score weakly (≈ 0.05-0.10) — strictly above random, strictly
    below anything that actually looks at structure.
    """

    def __init__(self):
        self.name = "distance_only"
        self.step = 0
        self.emits_layers = [LAYER_DISTANCE_ONLY]

    def score_pairs(self, batch: Dict) -> torch.Tensor:
        B, N = batch["mask"].shape
        idx = torch.arange(N)
        seqsep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().float()  # [N, N]
        return seqsep.unsqueeze(0).expand(B, N, N).contiguous()  # [B, N, N]

    def unload(self):
        pass
