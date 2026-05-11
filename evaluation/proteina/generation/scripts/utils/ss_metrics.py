"""Secondary-structure metrics for proteina generation eval.

Wraps biotite's pure-Python P-SEA implementation (`annotate_sse`, Cα-only)
to compute per-sample (helix, sheet, coil) fractions and a Jensen–Shannon
divergence vs a reference distribution. Sibling of `novelty_foldseek.py`,
called from `evaluate.compute_ss_metrics`.

Metric columns emitted (prefix `_res_`):
    ss_n            number of generated PDBs scored
    ss_n_designable number scored after the designable filter
    ss_frac_H/E/C   mean per-sample fraction across all generated PDBs
    ss_frac_H/E/C_designable   same, restricted to scRMSD<2.0Å subset
    ss_jsd_pdb / _afdb                JSD on (H,E,C) vs ref, all samples
    ss_jsd_pdb_designable / _afdb_…   same, designable subset
    ss_jsd_pdb_2d / _afdb_2d          JSD on 2D histogram of (f_H, f_E)
                                      with a 5×5 grid; captures shape not
                                      just centroid (see compute_ss_jsd_2d)
    ss_jsd_pdb_designable_2d / _afdb_designable_2d  same, designable subset

Reference files (.pt) are produced by `scripts/precompute_ss_reference.py`
and contain a `[N_ref, 3]` float tensor of per-structure fractions.

Caching: per-PDB fractions are written to a `.npz` next to the eval-run
output dir so repeated `--skip_generation` runs don't redo annotate_sse.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from loguru import logger

from ._metric_args import METRIC_PREFIX

SS_LABELS = ("a", "b", "c")  # biotite annotate_sse codes: helix / sheet / coil
SS_COLS = ("H", "E", "C")  # standard 3-state names used in column suffixes


def _annotate_one(pdb_path: str) -> Optional[np.ndarray]:
    """Return a length-3 (f_H, f_E, f_C) array for a single PDB, or None on failure.

    biotite returns one SSE label per residue: 'a'=helix, 'b'=sheet, 'c'=coil,
    ''=non-AA. We drop empties before normalising so the fractions sum to 1
    over actual residues.
    """
    from biotite.structure import annotate_sse
    from biotite.structure.io.pdb import PDBFile

    try:
        atoms = PDBFile.read(pdb_path).get_structure(model=1)
        sse = annotate_sse(atoms)
    except Exception as exc:
        logger.warning(
            f"annotate_sse failed on {pdb_path}: {type(exc).__name__}: {exc}"
        )
        return None
    if sse.size == 0:
        return None
    sse = sse[sse != ""]
    if sse.size == 0:
        return None
    counts = np.array([(sse == lbl).sum() for lbl in SS_LABELS], dtype=float)
    return counts / counts.sum()


def compute_per_sample_ss_fractions(
    pdb_paths: Iterable[str],
    cache_path: Optional[Path] = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute (f_H, f_E, f_C) per generated PDB.

    Args:
        pdb_paths: iterable of PDB file paths.
        cache_path: optional .npz path. If it exists with the expected schema
            (matching paths), reused; otherwise computed and written.

    Returns:
        (fractions [M, 3], paths_kept) — M ≤ len(pdb_paths) since failures are
        dropped. `paths_kept` is aligned with rows of `fractions`.
    """
    pdb_paths = list(pdb_paths)
    if cache_path is not None and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_paths = cached["paths"].tolist()
            if cached_paths == pdb_paths:
                logger.info(
                    f"SS: loaded cached fractions for {len(pdb_paths)} PDBs from {cache_path}"
                )
                return cached["fracs"].astype(float), cached_paths
            logger.info(f"SS: cache at {cache_path} stale (paths differ); recomputing")
        except Exception as exc:
            logger.warning(f"SS: cache read failed ({exc}); recomputing")

    fracs: list[np.ndarray] = []
    kept: list[str] = []
    for i, p in enumerate(pdb_paths):
        f = _annotate_one(p)
        if f is None:
            continue
        fracs.append(f)
        kept.append(p)
        if (i + 1) % 200 == 0:
            logger.info(f"SS: annotated {i + 1}/{len(pdb_paths)}")

    arr = np.stack(fracs, axis=0) if fracs else np.zeros((0, 3), dtype=float)

    if cache_path is not None and len(kept) == len(pdb_paths):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, fracs=arr, paths=np.array(pdb_paths))
        logger.info(f"SS: cached fractions to {cache_path}")

    return arr, kept


def compute_ss_jsd(gen_fracs: np.ndarray, ref_fracs: np.ndarray) -> float:
    """JSD between mean (H,E,C) distributions of generated vs reference.

    Aggregates each side to a 3-bin discrete distribution by averaging
    per-sample fractions, then computes JSD (base-2, ∈ [0,1]).

    Returns NaN when either side is empty.
    """
    from scipy.spatial.distance import jensenshannon

    if gen_fracs.size == 0 or ref_fracs.size == 0:
        return float("nan")
    p_gen = gen_fracs.mean(axis=0)
    p_ref = ref_fracs.mean(axis=0)
    # jensenshannon returns sqrt(JSD); square to get the divergence in [0,1].
    js_dist = jensenshannon(p_gen, p_ref, base=2)
    return float(js_dist**2)


SS_JSD_2D_N_BINS = 5


def compute_ss_jsd_2d(
    gen_fracs: np.ndarray, ref_fracs: np.ndarray, n_bins: int = SS_JSD_2D_N_BINS
) -> float:
    """JSD between 2D histograms of (f_H, f_E) over generated vs reference.

    Captures point-cloud shape rather than just the centroid. Uniform bins
    over [0, 1]^2; cells above the simplex diagonal (f_H + f_E > 1) are
    valid grid cells that receive zero mass on both sides and contribute 0.
    """
    from scipy.spatial.distance import jensenshannon

    if gen_fracs.size == 0 or ref_fracs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    hp, _, _ = np.histogram2d(gen_fracs[:, 0], gen_fracs[:, 1], bins=(edges, edges))
    hq, _, _ = np.histogram2d(ref_fracs[:, 0], ref_fracs[:, 1], bins=(edges, edges))
    p = hp.flatten() / hp.sum()
    q = hq.flatten() / hq.sum()
    d = jensenshannon(p, q, base=2)
    return float(d**2)


def _load_ref(path: Optional[str], label: str) -> Optional[np.ndarray]:
    """Load a [N, 3] reference fraction tensor; None if missing."""
    if path is None or not os.path.exists(path):
        if path is not None:
            logger.warning(
                f"SS: ref {label} not found at {path}, skipping JSD vs {label}"
            )
        return None
    import torch

    arr = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        logger.warning(
            f"SS: ref {label} at {path} has unexpected shape {arr.shape}; skipping"
        )
        return None
    logger.info(
        f"SS: loaded {label} reference distribution from {path} ({arr.shape[0]} structures)"
    )
    return arr


def compute_ss_metrics(
    list_of_pdbs: list[str],
    designable_pdbs: Optional[list[str]] = None,
    ss_reference_pdb_path: Optional[str] = None,
    ss_reference_afdb_path: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> dict:
    """Compute SS metrics for one eval run.

    Returns a flat dict of `_res_ss_*` columns suitable for appending to the
    sweep results row. Empty dict when there are no PDBs to score.
    """
    if not list_of_pdbs:
        return {}

    cache_path = (cache_dir / "ss_fractions.npz") if cache_dir is not None else None
    fracs_all, paths_all = compute_per_sample_ss_fractions(
        list_of_pdbs, cache_path=cache_path
    )
    if fracs_all.size == 0:
        logger.warning("SS: no usable structures, skipping")
        return {}

    out: dict = {
        f"{METRIC_PREFIX}ss_n": int(fracs_all.shape[0]),
    }
    mean_all = fracs_all.mean(axis=0)
    for col, val in zip(SS_COLS, mean_all):
        out[f"{METRIC_PREFIX}ss_frac_{col}"] = float(val)

    ref_pdb = _load_ref(ss_reference_pdb_path, "PDB")
    ref_afdb = _load_ref(ss_reference_afdb_path, "AFDB")
    if ref_pdb is not None:
        out[f"{METRIC_PREFIX}ss_jsd_pdb"] = compute_ss_jsd(fracs_all, ref_pdb)
        out[f"{METRIC_PREFIX}ss_jsd_pdb_2d"] = compute_ss_jsd_2d(fracs_all, ref_pdb)
    if ref_afdb is not None:
        out[f"{METRIC_PREFIX}ss_jsd_afdb"] = compute_ss_jsd(fracs_all, ref_afdb)
        out[f"{METRIC_PREFIX}ss_jsd_afdb_2d"] = compute_ss_jsd_2d(fracs_all, ref_afdb)

    if designable_pdbs:
        designable_set = set(designable_pdbs)
        mask = np.array([p in designable_set for p in paths_all], dtype=bool)
        fracs_des = fracs_all[mask]
        out[f"{METRIC_PREFIX}ss_n_designable"] = int(fracs_des.shape[0])
        if fracs_des.size > 0:
            mean_des = fracs_des.mean(axis=0)
            for col, val in zip(SS_COLS, mean_des):
                out[f"{METRIC_PREFIX}ss_frac_{col}_designable"] = float(val)
            if ref_pdb is not None:
                out[f"{METRIC_PREFIX}ss_jsd_pdb_designable"] = compute_ss_jsd(
                    fracs_des, ref_pdb
                )
                out[f"{METRIC_PREFIX}ss_jsd_pdb_designable_2d"] = compute_ss_jsd_2d(
                    fracs_des, ref_pdb
                )
            if ref_afdb is not None:
                out[f"{METRIC_PREFIX}ss_jsd_afdb_designable"] = compute_ss_jsd(
                    fracs_des, ref_afdb
                )
                out[f"{METRIC_PREFIX}ss_jsd_afdb_designable_2d"] = compute_ss_jsd_2d(
                    fracs_des, ref_afdb
                )

    return out
