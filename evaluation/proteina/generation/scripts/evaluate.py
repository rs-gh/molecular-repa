"""Generate samples from a Proteina checkpoint and score them with a suite of metrics.

Codeflow (one invocation = one checkpoint):

1. Parse args + resolve defaults; decide which metrics to run from
   ``--metrics`` / ``--skip_fid`` / ``--designability_subset_per_length`` /
   ``--cath_subset``.
2. Compose the hydra config under ``src/proteina/configs/experiment_config``
   and apply CLI overrides (``ckpt_name``, ``ckpt_path``, ``seed``).
3. Set up persistent output dirs under
   ``./eval_output/<config_slug>[_<suffix>]/`` — ``samples_fid/`` (PDBs) and
   ``tensors/`` (atom37 batch tensors used for resume + per-length grouping).
4. Generation phase (skipped with ``--skip_generation``):
     - Load the Proteina checkpoint, optionally ``torch.compile`` + bf16
       autocast under ``--fast_inference``.
     - Build a (length, nsamples) schedule from ``cfg.nres_lens`` /
       ``cfg.min_len/max_len/step_len`` and ``cfg.generation_batch_size``.
     - For each batch: if its tensor sidecar already exists, skip (resume);
       otherwise call ``model.generate`` + ``samples_to_atom37``, save the
       atom37 tensor and write per-sample PDBs.
5. Metric phase (collated into one CSV row, ``_res_*`` columns):
     - FID / fJSD / fS via ``compute_fid_metrics`` (one GearNet factory per
       reference distribution in ``cfg.metric_factory``).
     - Designability via ``compute_designability_metrics`` (per-length
       subsample when ``--designability_lengths`` set, else global cap);
       returns the designable-PDB set used to filter downstream metrics
       and writes ``designability_index.csv``.
     - SS fractions + JSD vs PDB/AFDB references via ``compute_ss_metrics``.
     - Diversity (cluster count + mean pairwise TM) via
       ``compute_diversity_metrics``, restricted to the designable subset
       per paper App. F.
     - Novelty against a centroid ``.pt`` file via
       ``compute_novelty_metrics``, and/or against Foldseek DBs via
       ``compute_novelty_foldseek``.
     - CATH-head distribution metrics via ``compute_cath_metrics``.
6. Write the row to ``results_<config_slug>[_<suffix>]_fid.csv``. Partial
   reruns (subset of metrics) merge ``_res_*`` columns into an existing CSV
   instead of overwriting. Return the row dict so ``run_sweep.py`` can
   consume it directly without re-reading the CSV.
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from contextlib import nullcontext

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import OmegaConf

# proteina imports (pyg_compat and torch.load patching done in hpc-scripts/proteina/evaluation/generation/eval_fid.sh wrapper)
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.metrics.metric_factory import (
    GenerationMetricFactory,
    generation_metric_from_list,
)
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

from utils._metric_args import METRIC_PREFIX, add_metric_args, resolve_evaluate_defaults
from utils.novelty_foldseek import compute_novelty_foldseek, parse_target_dbs_arg


def _subsample(items, n, seed):
    """Deterministically sample up to n items without replacement, sorted by index.

    Returns [] when n<=0 or items is empty. Used by designability, centroid
    novelty, and CATH metrics — three call-sites that previously inlined the
    same RandomState.choice pattern.
    """
    if n <= 0 or len(items) == 0:
        return []
    n_eval = min(n, len(items))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(items), size=n_eval, replace=False)
    return [items[i] for i in sorted(idx)]


def _write_designability_index(
    output_dir,
    list_of_pdbs,
    attempted_paths,
    evaluated_paths,
    scrmsds,
    tm_scores,
    plddts,
    length_to_pdbs=None,
    designable_threshold=2.0,
):
    """Persist per-PDB designability outcomes to <output_dir>/designability_index.csv.

    Covers the entire generated pool (`list_of_pdbs`), not just the evaluated
    subset, so callers can later answer "what's the design rate at length L
    given the K out of N PDBs we actually scored?". Un-evaluated PDBs get
    `evaluated=False` and NaN scRMSD; ESMFold/MPNN failures get `evaluated=True`
    with NaN values (matches the paper convention of counting failures as
    non-designable). Designable rows are scRMSD < `designable_threshold` (2.0 Å).

    `attempted_paths` is the subset that designability was actually run on
    (the subsample fed to batch_designability). `evaluated_paths` ⊆
    `attempted_paths` is the success subset returned in `pdb_paths_evaluated`.
    The set difference (attempted but not in evaluated) is the failure set.
    """
    eval_lookup = dict(zip(evaluated_paths, zip(scrmsds, tm_scores, plddts)))
    attempted_set = set(attempted_paths)
    pdb_to_length: dict = {}
    if length_to_pdbs is not None:
        for nres, pdbs in length_to_pdbs.items():
            for p in pdbs:
                pdb_to_length[p] = int(nres)

    rows = []
    for p in list_of_pdbs:
        rel = os.path.relpath(p, output_dir)
        length = pdb_to_length.get(p, "")
        if p in eval_lookup:
            scr, tm, pl = eval_lookup[p]
            row = {
                "pdb_path": rel,
                "length": length,
                "evaluated": True,
                "scRMSD": scr,
                "tm_score": tm,
                "plddt": pl,
                "designable": bool(scr < designable_threshold),
            }
        elif p in attempted_set:
            # Pipeline ran but raised (MPNN or ESMFold). Counts as evaluated
            # for accounting purposes but yields no metric values.
            row = {
                "pdb_path": rel,
                "length": length,
                "evaluated": True,
                "scRMSD": float("nan"),
                "tm_score": float("nan"),
                "plddt": float("nan"),
                "designable": False,
            }
        else:
            row = {
                "pdb_path": rel,
                "length": length,
                "evaluated": False,
                "scRMSD": float("nan"),
                "tm_score": float("nan"),
                "plddt": float("nan"),
                "designable": False,
            }
        rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "designability_index.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    n_eval = sum(1 for r in rows if r["evaluated"])
    n_des = sum(1 for r in rows if r["designable"])
    logger.info(
        f"Wrote designability index: {csv_path} "
        f"(N={len(rows)}, evaluated={n_eval}, designable={n_des})"
    )
    return csv_path


def _parse_lengths_csv(s):
    """Parse a comma-separated lengths string (e.g. '50,100,150') to a list of ints.

    Returns None for empty / None input so callers can use truthiness to decide
    whether the length filter is active.
    """
    if s is None or not str(s).strip():
        return None
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out or None


def parse_args():
    parser = argparse.ArgumentParser()
    # ── evaluate-only flags (not shared with run_sweep.py) ───────────────── #
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument(
        "--config_subdir",
        type=str,
        default=None,
        help="Subdirectory under experiment_config/ (e.g. 'inference').",
    )
    parser.add_argument(
        "--novelty_tm_threshold",
        type=float,
        default=0.5,
        help="TM-score threshold for novelty (default 0.5)",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation, only compute metrics on existing PDBs",
    )
    parser.add_argument(
        "--ckpt_name_override",
        type=str,
        default=None,
        help="Override ckpt_name from config (for sweeping checkpoints)",
    )
    parser.add_argument(
        "--ckpt_path_override",
        type=str,
        default=None,
        help="Override cfg.ckpt_path (the directory) from config. Lets a "
        "single shared protocol config be reused across many checkpoints.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Suffix appended to output directory (e.g. step_100000)",
    )
    # ── shared metric flags (also registered by run_sweep.py) ────────────── #
    add_metric_args(parser)
    return parser.parse_args()


_VALID_METRICS = {
    "fid",
    "designability",
    "diversity",
    "novelty_centroid",
    "novelty_foldseek",
    "cath",
    "ss",
}


def _parse_metrics(s):
    """Parse --metrics CSV into a set, or None for 'run everything'."""
    if s is None or not s.strip():
        return None
    names = {n.strip() for n in s.split(",") if n.strip()}
    bad = names - _VALID_METRICS
    if bad:
        raise ValueError(
            f"Unknown metric(s) {sorted(bad)}. Valid: {sorted(_VALID_METRICS)}"
        )
    return names


def _should_run(name, selected):
    """True if metric should run: either no selector set, or name is selected."""
    return selected is None or name in selected


def build_length_to_pdbs(tensors_dir, samples_dir):
    """Map protein length -> list of (pdb_path, tensor_path) by parsing tensor filenames.

    Tensor files are named batch_XXXX_nLEN_xNS.pt. We reconstruct which PDB
    indices came from each batch to group PDBs by generation length.

    Returns:
        dict mapping length (int) -> list of PDB file paths (sorted by index).
    """
    tensor_files = sorted(glob.glob(os.path.join(tensors_dir, "batch_*.pt")))
    length_to_pdbs = defaultdict(list)
    pdb_idx = 0
    for tf in tensor_files:
        basename = os.path.basename(tf)
        # Parse batch_0003_n120_x8.pt -> nres=120
        match = re.search(r"_n(\d+)_x(\d+)\.pt$", basename)
        if not match:
            continue
        nres = int(match.group(1))
        saved = torch.load(tf, map_location="cpu", weights_only=False)
        n_in_batch = saved.shape[0]
        for _ in range(n_in_batch):
            pdb_path = os.path.join(samples_dir, f"{pdb_idx}_fid.pdb")
            if os.path.exists(pdb_path):
                length_to_pdbs[nres].append(pdb_path)
            pdb_idx += 1
    return dict(length_to_pdbs)


def compute_fid_metrics(list_of_pdbs, metric_factory_cfgs):
    """Run GearNet FID / fJSD / fS metrics against each reference set.

    Each entry in `metric_factory_cfgs` is one reference distribution (typically
    PDB and AFDB); its metrics are prefixed via `cfg_mf.prefix` inside the
    factory. Loads the GearNet factory on GPU per entry and frees it before
    returning so downstream metrics (ESMFold etc.) have headroom.

    Returns:
        dict with FID/fJSD/fS metric columns (e.g. _res_PDB_FID, _res_fS_C).
    """
    results = {}
    for cfg_mf in metric_factory_cfgs:
        assert cfg_mf.ca_only, "Please turn on ca_only for CAFlow model"
        metric_factory = GenerationMetricFactory(**cfg_mf).cuda()
        metrics = generation_metric_from_list(list_of_pdbs, metric_factory)
        for k, v in metrics.items():
            results[METRIC_PREFIX + k] = v.cpu().item()
        logger.info(f"Metrics ({cfg_mf.get('prefix', '')}) computed")
        #! Free GearNet before next factory / downstream ESMFold
        del metric_factory
        torch.cuda.empty_cache()
    return results


def compute_designability_metrics(
    list_of_pdbs,
    subset_size,
    tmp_root,
    seed=42,
    length_filter=None,
    tensors_dir=None,
    samples_dir=None,
    output_dir=None,
):
    """Run designability on a random subset of generated PDBs.

    Loads ESMFold on-demand and offloads after completion.

    Args:
        list_of_pdbs: full list of generated PDB paths.
        subset_size: cap on PDBs to evaluate. If `length_filter` is set this
            is interpreted as PER-LENGTH; otherwise as a global cap.
        tmp_root: workspace dir for ProteinMPNN+ESMFold artefacts.
        seed: RNG for subsampling.
        length_filter: optional iterable of integer lengths. When provided,
            evaluation is restricted to those lengths and `subset_size` PDBs
            are sampled per length (paper App. F: 50 per length × 5 lengths).
            None or empty -> use entire list with a global subset_size cap.
        tensors_dir, samples_dir: required if `length_filter` is set, so we
            can rebuild the per-length PDB grouping. (The same `tensors_dir`
            already exists in the eval_output layout.) Also used (when present)
            to populate the `length` column of the designability index.
        output_dir: if set, writes `<output_dir>/designability_index.csv` with
            per-PDB scRMSD / tm_score / plddt / designable rows for the full
            pool. None disables index writing (back-compat).

    Returns:
        Tuple of (metrics_dict, designable_pdbs):
            metrics_dict: dict with designability metric columns.
            designable_pdbs: list of PDB paths from `list_of_pdbs` whose
                scRMSD < 2.0 Å. Empty list when designability is skipped or
                no PDBs were evaluated successfully. Used downstream to
                filter the diversity metric to designable backbones, matching
                the Proteina paper's protocol (App. F).
    """
    from proteinfoundation.metrics.designability import batch_designability

    if length_filter:
        if tensors_dir is None or samples_dir is None:
            raise ValueError(
                "length_filter requires tensors_dir and samples_dir to "
                "rebuild the per-length PDB grouping."
            )
        wanted = set(int(L) for L in length_filter)
        length_to_pdbs = build_length_to_pdbs(tensors_dir, samples_dir)
        subset_pdbs: list = []
        per_length_seen: list = []
        for nres in sorted(wanted):
            pool = length_to_pdbs.get(nres, [])
            if not pool:
                logger.warning(
                    f"Designability: length_filter requested L={nres} but "
                    f"no PDBs were generated at that length; skipping."
                )
                continue
            sampled = _subsample(pool, subset_size, seed + nres)
            subset_pdbs.extend(sampled)
            per_length_seen.append((nres, len(sampled), len(pool)))
        if not subset_pdbs:
            return {}, []
        logger.info(
            "Designability per-length subsamples: "
            + ", ".join(f"L={n}:{k}/{N}" for n, k, N in per_length_seen)
        )
    else:
        subset_pdbs = _subsample(list_of_pdbs, subset_size, seed)
        if not subset_pdbs:
            return {}, []

    n_eval = len(subset_pdbs)

    logger.info(f"Computing designability on {n_eval}/{len(list_of_pdbs)} PDBs...")
    results = batch_designability(subset_pdbs, tmp_root=tmp_root)

    # Build designable filter from per-PDB scRMSD list. `pdb_paths_evaluated`
    # is aligned with `scRMSD_list`; PDBs whose pipeline raised are absent
    # from both. Failures count as non-designable, matching the paper convention.
    paths = results.get("pdb_paths_evaluated", [])
    scrmsds = results.get("scRMSD_list", [])
    tm_scores = results.get("tm_score_list", [])
    plddts = results.get("plddt_list", [])
    designable_pdbs = [p for p, r in zip(paths, scrmsds) if r < 2.0]

    if output_dir is not None:
        # Lengths are best-effort: only available when the tensor sidecar dir
        # exists (it always does in the standard eval_output layout).
        length_to_pdbs = None
        if (
            tensors_dir is not None
            and samples_dir is not None
            and os.path.isdir(tensors_dir)
        ):
            try:
                length_to_pdbs = build_length_to_pdbs(tensors_dir, samples_dir)
            except Exception as exc:
                logger.warning(
                    f"Designability index: length lookup failed ({exc}); lengths blank"
                )
        _write_designability_index(
            output_dir=output_dir,
            list_of_pdbs=list_of_pdbs,
            attempted_paths=subset_pdbs,
            evaluated_paths=paths,
            scrmsds=scrmsds,
            tm_scores=tm_scores,
            plddts=plddts,
            length_to_pdbs=length_to_pdbs,
        )

    return {
        f"{METRIC_PREFIX}designability_n": n_eval,
        f"{METRIC_PREFIX}scRMSD_mean": results["scRMSD_mean"],
        f"{METRIC_PREFIX}scRMSD_median": results["scRMSD_median"],
        f"{METRIC_PREFIX}designability_rate": results["designability_rate"],
        f"{METRIC_PREFIX}tm_score_self_mean": results["tm_score_mean"],
        f"{METRIC_PREFIX}plddt_mean": results["plddt_mean"],
        f"{METRIC_PREFIX}plddt_median": results["plddt_median"],
    }, designable_pdbs


def compute_diversity_metrics(
    tensors_dir,
    samples_dir,
    tm_threshold=0.5,
    seed=42,
    designable_pdbs=None,
    length_filter=None,
):
    """Compute structural diversity per length bin via TM-score clustering.

    Runs on every designable PDB per length bin (no further subsampling) —
    pool size is already bounded upstream by designability_subset_per_length.

    Filters to the designable subset when `designable_pdbs` is provided,
    matching Proteina App. F (both diversity metrics are reported on
    designable samples only). Without that filter, low-quality near-duplicates
    can drag both metrics in either direction.

    Emits two complementary diversity columns per call:
        _res_diversity_clusters_mean      higher = more diverse (Yim 2023b style)
        _res_diversity_pairwise_tm_mean   lower  = more diverse (Bose 2024 style)

    Args:
        tensors_dir, samples_dir: where generated PDBs and atom37 tensors live.
        tm_threshold: cluster membership cutoff (default 0.5 = same fold).
        seed: unused, kept for API symmetry with designability/novelty.
        designable_pdbs: iterable of PDB paths classified as designable; when
            non-empty, length bins are restricted to these. None or empty -> no
            filter (legacy behaviour, kept so this function still runs when
            designability was skipped).
        length_filter: optional iterable of integer lengths. When provided,
            only those length bins are considered (paper-protocol DES uses a
            specific subset of lengths). None or empty -> all lengths.

    Returns:
        dict with diversity metric columns.
    """
    from proteinfoundation.metrics.tm_score import compute_diversity

    length_to_pdbs = build_length_to_pdbs(tensors_dir, samples_dir)
    if not length_to_pdbs:
        logger.warning("No length bins found for diversity computation")
        return {}

    if length_filter:
        wanted = set(int(L) for L in length_filter)
        kept = {nres: pdbs for nres, pdbs in length_to_pdbs.items() if nres in wanted}
        missing = wanted - set(kept.keys())
        if missing:
            logger.warning(
                f"Diversity: length_filter requested {sorted(wanted)} but "
                f"no PDBs at lengths {sorted(missing)} were generated; skipping those."
            )
        length_to_pdbs = kept
        if not length_to_pdbs:
            logger.warning(
                "No length bins matched the length_filter, skipping diversity"
            )
            return {}

    if designable_pdbs:
        designable_set = set(designable_pdbs)
        length_to_pdbs = {
            nres: [p for p in pdbs if p in designable_set]
            for nres, pdbs in length_to_pdbs.items()
        }
        n_total = sum(len(p) for p in length_to_pdbs.values())
        logger.info(
            f"Diversity: filtered to {n_total} designable PDBs "
            f"across {sum(1 for p in length_to_pdbs.values() if len(p) >= 2)} bins"
        )
        if n_total == 0:
            logger.warning("No designable PDBs after filtering, skipping diversity")
            return {}

    from proteinfoundation.metrics.designability import load_pdb

    bin_clusters = []
    bin_pairwise_tm = []
    bin_n = []
    bin_lengths = []

    for nres in sorted(length_to_pdbs.keys()):
        pdbs = length_to_pdbs[nres]
        if len(pdbs) < 2:
            continue

        logger.info(f"Diversity: length={nres}, n_structures={len(pdbs)}")

        coords_list = []
        for pdb_path in pdbs:
            prot = load_pdb(pdb_path)
            coords_list.append(np.array(prot.atom_positions))

        div = compute_diversity(coords_list, tm_threshold=tm_threshold)
        bin_clusters.append(div["n_clusters"])
        bin_pairwise_tm.append(div["mean_pairwise_tm"])
        bin_n.append(len(coords_list))
        bin_lengths.append(nres)
        logger.info(
            f"  -> {div['n_clusters']} clusters, "
            f"mean pairwise TM={div['mean_pairwise_tm']:.3f}"
        )

    if not bin_clusters:
        return {}

    clusters_arr = np.array(bin_clusters)
    pairwise_arr = np.array(bin_pairwise_tm)
    return {
        f"{METRIC_PREFIX}diversity_n_bins": len(bin_clusters),
        f"{METRIC_PREFIX}diversity_n": int(sum(bin_n)),
        f"{METRIC_PREFIX}diversity_designable_filtered": bool(designable_pdbs),
        f"{METRIC_PREFIX}diversity_clusters_mean": float(clusters_arr.mean()),
        f"{METRIC_PREFIX}diversity_clusters_median": float(np.median(clusters_arr)),
        f"{METRIC_PREFIX}diversity_clusters_total": int(clusters_arr.sum()),
        f"{METRIC_PREFIX}diversity_pairwise_tm_mean": float(np.nanmean(pairwise_arr)),
        f"{METRIC_PREFIX}diversity_pairwise_tm_median": float(
            np.nanmedian(pairwise_arr)
        ),
    }


def compute_novelty_metrics(
    list_of_pdbs,
    centroid_path,
    tm_threshold=0.5,
    seed=42,
    designable_pdbs=None,
):
    """Compute novelty as fraction of generated structures dissimilar to training centroids.

    Runs on every candidate PDB (no further subsampling) — pool size is
    already bounded upstream by designability_subset_per_length feeding into
    the designable filter.

    When ``designable_pdbs`` is non-empty the candidate pool is restricted to
    that intersection, matching the Proteina paper App. F protocol (novelty
    reported on the designable subset only). Pass ``None`` or an empty list
    to keep the legacy "all generated PDBs" behaviour.

    Args:
        list_of_pdbs: All generated PDB paths.
        centroid_path: Path to .pt file containing list of [n_i, 37, 3] centroid arrays.
        tm_threshold: A structure is novel if max TM-score to any centroid < threshold.
        seed: unused, kept for API symmetry.
        designable_pdbs: Optional iterable of PDB paths classified as designable.
            When non-empty, restricts the candidate pool.

    Returns:
        dict with novelty metric columns. Includes a
        ``_res_novelty_designable_filtered`` boolean for traceability,
        analogous to ``_res_diversity_designable_filtered``.
    """
    from proteinfoundation.metrics.tm_score import compute_tm_score
    from proteinfoundation.metrics.designability import load_pdb

    if centroid_path is None or not os.path.exists(centroid_path):
        if centroid_path is not None:
            logger.warning(
                f"Centroid file not found: {centroid_path}, skipping novelty"
            )
        return {}

    candidate_pdbs = list_of_pdbs
    filtered = bool(designable_pdbs)
    if filtered:
        designable_set = set(designable_pdbs)
        candidate_pdbs = [p for p in list_of_pdbs if p in designable_set]
        logger.info(
            f"Novelty (centroid): filtered to {len(candidate_pdbs)} "
            f"designable PDBs (paper App. F protocol)"
        )
        if not candidate_pdbs:
            logger.warning(
                "No designable PDBs after filtering, skipping centroid novelty"
            )
            return {}

    logger.info(f"Loading training set centroids from {centroid_path}")
    centroids = torch.load(centroid_path, map_location="cpu", weights_only=False)
    # centroids: list of numpy arrays, each [n_i, 37, 3]
    if isinstance(centroids, torch.Tensor):
        centroids = [centroids[i].numpy() for i in range(len(centroids))]
    logger.info(f"Loaded {len(centroids)} training set cluster centroids")

    eval_pdbs = candidate_pdbs
    n_eval = len(eval_pdbs)

    logger.info(
        f"Computing novelty on {n_eval} PDBs against {len(centroids)} centroids..."
    )

    max_tm_scores = []
    for i, pdb_path in enumerate(eval_pdbs):
        prot = load_pdb(pdb_path)
        gen_coords = np.array(prot.atom_positions)

        best_tm = 0.0
        for centroid_coords in centroids:
            tm = compute_tm_score(gen_coords, centroid_coords)
            best_tm = max(best_tm, tm)
            if best_tm >= tm_threshold:
                break  # Already non-novel, skip remaining centroids

        max_tm_scores.append(best_tm)

        if (i + 1) % 50 == 0:
            logger.info(f"  Novelty progress: {i + 1}/{n_eval}")

    max_tm_arr = np.array(max_tm_scores)
    novelty_rate = float((max_tm_arr < tm_threshold).mean())

    return {
        f"{METRIC_PREFIX}novelty_n": n_eval,
        f"{METRIC_PREFIX}novelty_designable_filtered": filtered,
        f"{METRIC_PREFIX}novelty_rate": novelty_rate,
        f"{METRIC_PREFIX}novelty_max_tm_mean": float(max_tm_arr.mean()),
        f"{METRIC_PREFIX}novelty_max_tm_median": float(np.median(max_tm_arr)),
    }


def compute_cath_metrics(
    list_of_pdbs,
    head_path,
    max_eval=100,
    batch_size=16,
    gearnet_ckpt=None,
    seed=42,
):
    """Score generated PDBs against a frozen GearNet + CATH head.

    Loads the artifact written by ``build_cath_classifier.py``, embeds each
    generated PDB through the same GearNet checkpoint (so train and inference
    embeddings live in the same space), and reports four columns:

        _res_cath_topclass_conf_mean   mean top-class softmax prob - recognizability
        _res_cath_n_unique_T            # distinct classes the head assigns
        _res_cath_distribution_kl       KL(p_gen || p_train) over the head's vocab
        _res_cath_entropy               Shannon entropy of p_gen
        _res_cath_n_pdbs                # PDBs scored (subset cap)

    The KL is over the head's vocab only (Laplace-smoothed on classes that the
    train side has but generation didn't produce, and vice versa). Generated
    samples that the head assigns to a class outside its vocab can't happen
    by construction - the head only outputs vocab classes.

    If ``head_path`` is missing, returns ``{}`` and logs a warning so the
    sweep keeps making progress without this column.
    """
    if head_path is None or not os.path.exists(head_path):
        if head_path is not None:
            logger.warning(
                f"CATH classifier artifact not found: {head_path}, skipping CATH metrics"
            )
        return {}

    import pickle as _pickle

    from proteinfoundation.metrics.gearnet_utils import NoTrainCAGearNet
    from proteinfoundation.metrics.metric_factory import DatasetWrapper
    from torch_geometric.loader import DataLoader as _PyGLoader

    logger.info(f"Loading CATH classifier from {head_path}")
    with open(head_path, "rb") as f:
        bundle = _pickle.load(f)
    head = bundle["head"]
    train_meta = bundle["train_meta"]
    train_dist = bundle.get("train_dist", {})
    ckpt = gearnet_ckpt or bundle.get("gearnet_ckpt")
    if ckpt is None or not os.path.exists(ckpt):
        logger.warning(
            f"GearNet ckpt for CATH metric not found ({ckpt}), skipping CATH metrics"
        )
        return {}

    eval_pdbs = _subsample(list_of_pdbs, max_eval, seed)
    n_eval = len(eval_pdbs)
    if n_eval == 0:
        return {}
    logger.info(
        f"Computing CATH metrics on {n_eval} PDBs (level={train_meta['level']})"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gearnet = NoTrainCAGearNet(ckpt).to(device).eval()

    # Reuse the FID code path's PDB->PyG converter so generated samples are
    # processed exactly the way the FID metric processes them.
    dataset = DatasetWrapper(eval_pdbs)
    loader = _PyGLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = gearnet(batch)
            feats.append(out["protein_feature"].detach().cpu())
    feats = torch.cat(feats, dim=0).numpy()

    del gearnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Predict in the head's class space. ``head.classes_`` indexes into
    # ``train_meta['vocab']`` (set up by train_cath_probe).
    proba = head.predict_proba(feats)  # [n_eval, K]
    pred_idx = head.classes_[proba.argmax(axis=1)]  # [n_eval]
    top1 = proba.max(axis=1)  # [n_eval]
    vocab = train_meta["vocab"]

    # Empirical generated distribution over the head's vocab.
    gen_counts = np.bincount(pred_idx, minlength=len(vocab)).astype(float)
    gen_counts_total = float(gen_counts.sum())
    p_gen = gen_counts / max(1.0, gen_counts_total)

    # Train distribution over the same vocab. Vocab indices match the head's,
    # which were derived from train.lmdb at the same level by build_cath_classifier.
    p_train = np.array([train_dist.get(c, 0.0) for c in vocab], dtype=float)
    if p_train.sum() == 0.0:
        # Defensive: if the artifact wasn't shipped with train_dist, fall back
        # to uniform over vocab so KL is at least defined and finite.
        p_train = np.full(len(vocab), 1.0 / max(1, len(vocab)))
    else:
        p_train = p_train / p_train.sum()

    # Laplace-smoothed KL(p_gen || p_train). eps absorbs any class with zero
    # support on either side.
    eps = 1e-8
    pg = p_gen + eps
    pt = p_train + eps
    pg = pg / pg.sum()
    pt = pt / pt.sum()
    kl = float(np.sum(pg * np.log(pg / pt)))
    entropy = float(-np.sum(pg * np.log(pg + 1e-12)))
    n_unique = int((gen_counts > 0).sum())

    return {
        f"{METRIC_PREFIX}cath_n_pdbs": int(n_eval),
        f"{METRIC_PREFIX}cath_level": str(train_meta["level"]),
        f"{METRIC_PREFIX}cath_topclass_conf_mean": float(top1.mean()),
        f"{METRIC_PREFIX}cath_n_unique": n_unique,
        f"{METRIC_PREFIX}cath_distribution_kl": kl,
        f"{METRIC_PREFIX}cath_entropy": entropy,
    }


def split_nlens(nlens_dict, max_nsamples=16):
    """Split lengths into (length, nsample) pairs. Copied from inference.py.

    WARNING: if `cnt` is not a multiple of `max_nsamples`, the final partial
    batch is rounded UP to a full batch, so actual samples generated > cnt.
    Example: cnt=200, max_nsamples=80 -> batches [80, 80, 40] -> forced to
    [80, 80, 80] -> 240 samples. This is intentional: torch.compile with
    dynamic=False recompiles per unique (batch_size, nres) shape (~60-90s),
    so forcing a uniform batch avoids a tail recompile.

    Takeaway: pick `generation_batch_size` that divides `nsamples_per_len`
    (i.e. nsamples_per_len % generation_batch_size == 0) to get exactly the
    requested count. Mismatches inflate FID/fS statistics but don't bias
    them; subset-based metrics (designability) re-sample so are unaffected.
    """
    lengths_range = nlens_dict["length_ranges"].tolist()
    length_distribution = nlens_dict["length_distribution"].tolist()
    lens_sample, nsamples = [], []
    for length, cnt in zip(lengths_range, length_distribution):
        for i in range(0, cnt, max_nsamples):
            lens_sample.append(length)
            if i + max_nsamples <= cnt:
                nsamples.append(max_nsamples)
            else:
                nsamples.append(cnt - i)
    max_ns = max(nsamples)
    for i in range(len(nsamples)):
        nsamples[i] = max_ns
    return lens_sample, nsamples


_logger_handler_id = None


def _setup_logger():
    """Add the stdout sink exactly once per process.

    Loguru's ``logger.add`` is not idempotent — calling it on every
    ``main()`` invocation duplicates log lines. We track the handler id at
    module level and reuse it across re-entries.
    """
    global _logger_handler_id
    if _logger_handler_id is None:
        _logger_handler_id = logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )


def main(args=None):
    """Run generation + metrics for one checkpoint.

    Args:
        args: Parsed argparse Namespace with the same shape as ``parse_args()``
            returns. When None, parses ``sys.argv`` and applies the standalone
            defaults via ``resolve_evaluate_defaults``. ``run_sweep.py`` builds
            this Namespace directly to avoid argv reconstruction.

    Returns:
        dict[str, Any] of column → value, one row's worth of the per-checkpoint
        CSV. ``run_sweep.py`` consumes the ``_res_*`` subset and writes it to
        the sweep JSONL.
    """
    if args is None:
        args = parse_args()
    resolve_evaluate_defaults(args)

    selected_metrics = _parse_metrics(args.metrics)
    run_fid = (not args.skip_fid) and _should_run("fid", selected_metrics)
    run_designability = args.designability_subset_per_length > 0 and _should_run(
        "designability", selected_metrics
    )
    run_diversity = _should_run("diversity", selected_metrics)
    run_novelty_centroid = _should_run("novelty_centroid", selected_metrics)
    run_novelty_foldseek = _should_run("novelty_foldseek", selected_metrics)
    run_cath = args.cath_subset > 0 and _should_run("cath", selected_metrics)
    run_ss = _should_run("ss", selected_metrics)

    needs_gpu = not (args.skip_generation and not run_fid and not run_designability)
    if needs_gpu:
        assert torch.cuda.is_available(), "CUDA not available (use --skip_generation --skip_fid --designability_subset_per_length 0 for CPU-only mode)"
        # TF32 matmuls for any residual fp32 ops left outside the bf16 autocast
        # context (e.g. normalisation, ODE-integrator arithmetic in
        # full_simulation). bf16 autocast handles the heavy matmuls but a few
        # ops fall back to fp32; enabling TF32 makes those cheap. Training sets
        # this in train_repa.py:153; mirror it here.
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _setup_logger()
    # Defensive: clear any GlobalHydra state from a previous main() that may
    # have crashed mid-`with hydra.initialize()`. Clean exit handles cleanup
    # via the context manager; this covers the abrupt-exit case so re-entry
    # from run_sweep.py doesn't need importlib.reload().
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    try:
        return _main_body(
            args,
            run_fid,
            run_designability,
            run_diversity,
            run_novelty_centroid,
            run_novelty_foldseek,
            run_cath,
            run_ss,
            selected_metrics,
        )
    finally:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()


def _main_body(
    args,
    run_fid,
    run_designability,
    run_diversity,
    run_novelty_centroid,
    run_novelty_foldseek,
    run_cath,
    run_ss,
    selected_metrics,
):
    """Body of main(); split out so the try/finally above stays readable."""

    # -- Load config --
    config_name = (
        f"{args.config_subdir}/{args.config_name}"
        if args.config_subdir
        else args.config_name
    )
    with hydra.initialize(
        config_path="../../../../src/proteina/configs/experiment_config",
        version_base=hydra.__version__,
    ):
        cfg = hydra.compose(config_name=config_name)

    # Apply CLI overrides
    if args.ckpt_name_override:
        cfg = OmegaConf.merge(cfg, {"ckpt_name": args.ckpt_name_override})
    if getattr(args, "ckpt_path_override", None):
        cfg = OmegaConf.merge(cfg, {"ckpt_path": args.ckpt_path_override})
    if args.seed is not None:
        cfg = OmegaConf.merge(cfg, {"seed": args.seed})
    logger.info(f"Config: {args.config_name}, seed: {cfg.seed}")

    # -- Paths (persist across restarts) --
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    config_slug = args.config_name.replace("/", "_")
    root_path = f"./eval_output/{config_slug}{suffix}"
    samples_dir = os.path.join(root_path, "samples_fid")
    tensors_dir = os.path.join(root_path, "tensors")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    # -- Generation phase --
    if not args.skip_generation:
        # -- Load model --
        ckpt_file = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
        logger.info(f"Loading checkpoint: {ckpt_file}")
        model = Proteina.load_from_checkpoint(ckpt_file)
        model.configure_inference(cfg, nn_ag=None)
        model.eval()
        model.cuda()

        # Fast inference: validated at L=256 as ~4.6x speedup with matching
        # geometry distributions (bond, Rg, clash, angles) - see
        # playground/proteina/generation_speedup/. Static-shape compile so
        # each unique (batch_size, nres) triggers a one-time ~60-90s
        # warmup; the n512_convergence lite sweep (12 unique lengths) runs
        # cleanly under this path, so multi-length is fine in itself. We
        # tried dynamic=True briefly to skip per-shape recompiles, but
        # inductor under dynamic shapes elided the .contiguous() guard in
        # _attn_sdpa (see TestContiguousBiasFix in tests/proteina/
        # test_sdpa_attention.py), causing SDPA's "(*bias): last dimension
        # must be contiguous" runtime error mid-sampling. Reverted.
        if args.fast_inference:
            logger.info(
                "Fast inference ON: torch.compile(reduce-overhead) + bf16 autocast"
            )
            model.nn = torch.compile(
                model.nn, mode="reduce-overhead", dynamic=False, fullgraph=False
            )

            def autocast_ctx():
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            logger.info("Fast inference OFF: using eager fp32 path")
            autocast_ctx = nullcontext

        L.seed_everything(cfg.seed)

        # -- Build batch schedule --
        if cfg.nres_lens:
            lens_list = list(cfg.nres_lens)
        else:
            lens_list = [
                int(v) for v in np.arange(cfg.min_len, cfg.max_len + 1, cfg.step_len)
            ]

        nlens_dict = {
            "length_ranges": torch.as_tensor(lens_list),
            "length_distribution": torch.as_tensor(
                [cfg.nsamples_per_len] * len(lens_list)
            ),
        }
        lens_sample, nsamples = split_nlens(
            nlens_dict, max_nsamples=cfg.generation_batch_size
        )
        total_batches = len(lens_sample)
        logger.info(
            f"Total batches: {total_batches} ({len(lens_list)} lengths x ~{cfg.nsamples_per_len}/{cfg.generation_batch_size} batches/len)"
        )

        # -- Generate with checkpointing --
        n_skipped = 0
        n_generated = 0

        # Count existing PDBs to resume numbering
        existing_pdbs = sorted(glob.glob(os.path.join(samples_dir, "*_fid.pdb")))
        pdb_idx = len(existing_pdbs)
        logger.info(
            f"Found {pdb_idx} existing PDBs, resuming from batch index search..."
        )

        for batch_idx in range(total_batches):
            nres = lens_sample[batch_idx]
            ns = nsamples[batch_idx]

            # Check if this batch is already done (tensor file exists)
            tensor_path = os.path.join(
                tensors_dir, f"batch_{batch_idx:04d}_n{nres}_x{ns}.pt"
            )
            if os.path.exists(tensor_path):
                # Count the PDBs from this batch
                saved = torch.load(tensor_path, map_location="cpu", weights_only=False)
                pdb_idx += saved.shape[0]
                n_skipped += 1
                continue

            # Generate using model.generate() directly (avoids Lightning trainer overhead)
            sampling_args = cfg.sampling_caflow
            with torch.inference_mode(), autocast_ctx():
                x = model.generate(
                    nsamples=ns,
                    n=nres,
                    dt=float(cfg.dt),
                    self_cond=cfg.self_cond,
                    cath_code=None,
                    guidance_weight=cfg.get("guidance_weight", 1.0),
                    autoguidance_ratio=cfg.get("autoguidance_ratio", 0.0),
                    dtype=torch.float32,
                    schedule_mode=cfg.schedule.schedule_mode,
                    schedule_p=cfg.schedule.schedule_p,
                    sampling_mode=sampling_args["sampling_mode"],
                    sc_scale_noise=sampling_args["sc_scale_noise"],
                    sc_scale_score=sampling_args["sc_scale_score"],
                    gt_mode=sampling_args["gt_mode"],
                    gt_p=sampling_args["gt_p"],
                    gt_clamp_val=sampling_args["gt_clamp_val"],
                )
                coors_atom37 = model.samples_to_atom37(x).cpu()  # [ns, nres, 37, 3]

            # Save tensor checkpoint
            torch.save(coors_atom37, tensor_path)

            # Save PDBs
            for i in range(coors_atom37.shape[0]):
                pdb_path = os.path.join(samples_dir, f"{pdb_idx}_fid.pdb")
                write_prot_to_pdb(
                    coors_atom37[i].numpy(),
                    pdb_path,
                    overwrite=True,
                    no_indexing=True,
                )
                pdb_idx += 1

            n_generated += 1
            if n_generated % 10 == 0 or batch_idx == total_batches - 1:
                logger.info(
                    f"Batch {batch_idx + 1}/{total_batches} done (len={nres}, n={ns}). "
                    f"Generated: {n_generated}, Skipped: {n_skipped}, Total PDBs: {pdb_idx}"
                )
                sys.stdout.flush()

        logger.info(
            f"Generation complete. {pdb_idx} PDBs total ({n_generated} new, {n_skipped} cached)"
        )

        # Free generation model before metric computation
        del model
        torch.cuda.empty_cache()
    else:
        logger.info("Skipping generation (--skip_generation), using existing PDBs")

    # -- Compute metrics --
    list_of_pdbs = sorted(
        glob.glob(os.path.join(samples_dir, "*_fid.pdb")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    logger.info(f"Computing metrics on {len(list_of_pdbs)} PDBs...")

    if run_fid:
        flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
        flat_dict = {k: str(v) for k, v in flat_dict.items()}
    else:
        flat_dict = {"config_name": args.config_name}
    columns = list(flat_dict.keys())
    res_row = list(flat_dict.values())

    # -- FID / fJSD / fS metrics --
    if run_fid:
        fid_results = compute_fid_metrics(list_of_pdbs, cfg.metric_factory)
        for k, v in fid_results.items():
            columns.append(k)
            res_row.append(v)
    else:
        logger.info(
            "Skipping GearNet FID/fJSD/fS metrics (--skip_fid or --metrics filter)"
        )

    # -- Designability metrics --
    # Returns the list of designable PDB paths so diversity / foldseek below can
    # filter to designable backbones (paper App. F protocol). When designability
    # is skipped, downstream filters fall back to all generated PDBs.
    # designability_lengths (CSV) restricts evaluation to specific lengths and
    # subsamples designability_subset_per_length PDBs per length (paper-protocol DES).
    designability_lengths = _parse_lengths_csv(args.designability_lengths)
    desig_subset = args.designability_subset_per_length if run_designability else 0
    desig_results, designable_pdbs = compute_designability_metrics(
        list_of_pdbs,
        subset_size=desig_subset,
        tmp_root=os.path.join(root_path, "tmp_designability"),
        seed=cfg.seed,
        length_filter=designability_lengths,
        tensors_dir=tensors_dir,
        samples_dir=samples_dir,
        output_dir=root_path,
    )
    for k, v in desig_results.items():
        columns.append(k)
        res_row.append(v)
    if desig_results:
        logger.info(
            f"Designability: rate={desig_results.get('_res_designability_rate', 'N/A'):.3f}, "
            f"scRMSD_mean={desig_results.get('_res_scRMSD_mean', 'N/A'):.3f}, "
            f"n_designable={len(designable_pdbs)}"
        )

    # -- Secondary-structure metrics --
    # Cα-only P-SEA assignment (biotite annotate_sse) on every generated PDB.
    # Cheap (~ms/structure) so always runs on the full pool. Slots here so
    # `designable_pdbs` is in scope for the designable-subset variant.
    if run_ss:
        from utils.ss_metrics import compute_ss_metrics

        from pathlib import Path as _Path

        ss_results = compute_ss_metrics(
            list_of_pdbs=list_of_pdbs,
            designable_pdbs=designable_pdbs,
            ss_reference_pdb_path=args.ss_reference_pdb_path,
            ss_reference_afdb_path=args.ss_reference_afdb_path,
            cache_dir=_Path(root_path) / "ss_cache",
        )
        for k, v in ss_results.items():
            columns.append(k)
            res_row.append(v)
        if ss_results:
            logger.info(
                f"SS: H={ss_results.get('_res_ss_frac_H', 'N/A'):.3f}, "
                f"E={ss_results.get('_res_ss_frac_E', 'N/A'):.3f}, "
                f"C={ss_results.get('_res_ss_frac_C', 'N/A'):.3f}, "
                f"jsd_pdb={ss_results.get('_res_ss_jsd_pdb', float('nan')):.4f}, "
                f"jsd_afdb={ss_results.get('_res_ss_jsd_afdb', float('nan')):.4f}"
            )
    else:
        logger.info("Skipping SS metrics (--metrics filter)")

    # -- Diversity metrics --
    # Diversity uses the same length filter as designability (per paper App. F
    # both metrics are reported on the same designable subset of length bins).
    # Runs on every designable PDB per length bin — at the per-length subset
    # cap (≤ designability_subset_per_length) the pairwise TM cost is seconds.
    div_results = (
        compute_diversity_metrics(
            tensors_dir,
            samples_dir,
            seed=cfg.seed,
            designable_pdbs=designable_pdbs,
            length_filter=designability_lengths,
        )
        if run_diversity
        else {}
    )
    for k, v in div_results.items():
        columns.append(k)
        res_row.append(v)
    if div_results:
        logger.info(
            f"Diversity: mean_clusters={div_results.get('_res_diversity_clusters_mean', 'N/A'):.1f}, "
            f"mean_pairwise_tm={div_results.get('_res_diversity_pairwise_tm_mean', 'N/A'):.3f}, "
            f"n_bins={div_results.get('_res_diversity_n_bins', 'N/A')}, "
            f"n={div_results.get('_res_diversity_n', 'N/A')}, "
            f"designable_filtered={div_results.get('_res_diversity_designable_filtered', False)}"
        )

    # -- Novelty metrics (centroid-file path) --
    # Filter to designable subset by default (paper App. F protocol). Earlier
    # sweeps had no such filter; the boolean column emitted below records
    # which protocol produced a given row.
    if run_novelty_centroid:
        centroid_designable = (
            designable_pdbs if args.centroid_filter_designable else None
        )
        nov_results = compute_novelty_metrics(
            list_of_pdbs,
            centroid_path=args.centroid_path,
            tm_threshold=args.novelty_tm_threshold,
            seed=cfg.seed,
            designable_pdbs=centroid_designable,
        )
        for k, v in nov_results.items():
            columns.append(k)
            res_row.append(v)
        if nov_results:
            logger.info(
                f"Novelty (centroid): rate={nov_results.get('_res_novelty_rate', 'N/A'):.3f}, "
                f"max_tm_mean={nov_results.get('_res_novelty_max_tm_mean', 'N/A'):.3f}"
            )
    else:
        logger.info("Skipping centroid-novelty metrics (--metrics filter)")

    # -- Novelty metrics (Foldseek path) --
    # Runs against prebuilt Foldseek DBs (PDB / AFDB-SwissProt). Internally
    # consistent across sweep checkpoints when flag values are pinned.
    target_dbs = (
        parse_target_dbs_arg(args.foldseek_target_dbs) if run_novelty_foldseek else []
    )
    if target_dbs:
        # Default to the designable subset (paper convention). Falls back to
        # all generated PDBs if designability was skipped or returned nothing.
        fs_queries = (
            designable_pdbs
            if (args.foldseek_filter_designable and designable_pdbs)
            else list_of_pdbs
        )
        for db_label, db_path in target_dbs:
            try:
                fs_results = compute_novelty_foldseek(
                    fs_queries,
                    target_db=db_path,
                    db_label=db_label,
                    alignment_type=args.foldseek_alignment_type,
                    max_seqs=args.foldseek_max_seqs,
                    sensitivity=args.foldseek_sensitivity,
                    threads=args.foldseek_threads,
                    tm_threshold=args.novelty_tm_threshold,
                )
            except Exception as exc:
                logger.warning(
                    f"Foldseek novelty failed for db={db_label}: "
                    f"{type(exc).__name__}: {exc}"
                )
                fs_results = {}
            for k, v in fs_results.items():
                columns.append(k)
                res_row.append(v)
            if fs_results:
                rate = fs_results.get(f"_res_novelty_foldseek_{db_label}_rate", "N/A")
                mean = fs_results.get(
                    f"_res_novelty_foldseek_{db_label}_max_tm_mean", "N/A"
                )
                logger.info(
                    f"Novelty (foldseek/{db_label}): rate={rate:.3f}, "
                    f"max_tm_mean={mean:.3f}, n={fs_results.get(f'_res_novelty_foldseek_{db_label}_n', 'N/A')}"
                )
    else:
        if not run_novelty_foldseek:
            logger.info("Skipping foldseek-novelty metrics (--metrics filter)")
        else:
            logger.info(
                "Foldseek novelty skipped (no --foldseek_target_dbs configured)"
            )

    # -- CATH metrics --
    if run_cath and list_of_pdbs:
        cath_results = compute_cath_metrics(
            list_of_pdbs,
            head_path=args.cath_head_path,
            max_eval=args.cath_subset,
            seed=cfg.seed,
        )
        for k, v in cath_results.items():
            columns.append(k)
            res_row.append(v)
        if cath_results:
            logger.info(
                f"CATH: top1_conf={cath_results['_res_cath_topclass_conf_mean']:.3f}, "
                f"n_unique={cath_results['_res_cath_n_unique']}, "
                f"KL={cath_results['_res_cath_distribution_kl']:.3f}, "
                f"entropy={cath_results['_res_cath_entropy']:.3f}"
            )
    else:
        logger.info("Skipping CATH metrics (--cath_subset 0 or --metrics filter)")

    # -- Save results --
    df = pd.DataFrame([res_row], columns=columns)
    if "metric_factory" in df.columns:
        df = df.drop("metric_factory", axis=1)

    results_csv = os.path.join(root_path, f"results_{config_slug}{suffix}_fid.csv")

    # If we ran fewer than all metrics and an existing CSV exists, merge new
    # columns in rather than overwriting (so a single-metric rerun does not
    # clobber FID/designability/etc. columns from a prior full eval). Also
    # overwrites any _res_ columns already present - allows a retry to fix
    # stale/corrupted values written by a previously timed-out run.
    partial_run = args.skip_fid or selected_metrics is not None
    if partial_run and os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        update_cols = [c for c in df.columns if c.startswith(METRIC_PREFIX)]
        if update_cols:
            for c in update_cols:
                existing_df[c] = df[c].iloc[0]
            existing_df.to_csv(results_csv, index=False)
            logger.info(
                f"Merged/updated {len(update_cols)} columns in {results_csv}: {update_cols}"
            )
        else:
            logger.info(f"No {METRIC_PREFIX} columns to merge into {results_csv}")
    else:
        df.to_csv(results_csv, index=False)
        logger.info(f"Results saved to {results_csv}")

    # Return the row dict so run_sweep.py can append directly to the JSONL
    # without round-tripping through the per-checkpoint CSV.
    return dict(zip(columns, res_row))


if __name__ == "__main__":
    main()
