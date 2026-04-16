"""Checkpointed FID evaluation for proteina models.

Replaces inference.py's monolithic predict-then-evaluate flow with:
1. Generate proteins one batch at a time, saving PDBs + atom37 tensors to disk
2. Skip batches whose output files already exist (resume support)
3. Compute metrics only after all generation is done
4. Compute designability, diversity, and novelty metrics

Usage (called by eval_fid.sh):
    python evaluate.py --config_name inference_fid_60m_baseline
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf

# proteina imports (pyg_compat and torch.load patching done in eval_fid.sh wrapper)
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.metrics.metric_factory import (
    GenerationMetricFactory,
    generation_metric_from_list,
)
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument(
        "--designability_subset",
        type=int,
        default=500,
        help="Number of PDBs to evaluate for designability (0 to skip)",
    )
    parser.add_argument(
        "--diversity_subset_per_bin",
        type=int,
        default=100,
        help="Max PDBs per length bin for diversity (0 to skip)",
    )
    parser.add_argument(
        "--centroid_path",
        type=str,
        default=None,
        help="Path to .pt file with precomputed training set centroid "
        "coords (list of [n_i, 37, 3] arrays). Required for novelty.",
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
        "--skip_fid",
        action="store_true",
        help="Skip GearNet FID/fJSD/fS metrics (no GPU required)",
    )
    return parser.parse_args()


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


def compute_designability_metrics(list_of_pdbs, subset_size, tmp_root):
    """Run designability on a random subset of generated PDBs.

    Loads ESMFold on-demand and offloads after completion.

    Returns:
        dict with designability metric columns.
    """
    from proteinfoundation.metrics.designability import batch_designability

    if subset_size <= 0 or len(list_of_pdbs) == 0:
        return {}

    n_eval = min(subset_size, len(list_of_pdbs))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(list_of_pdbs), size=n_eval, replace=False)
    subset_pdbs = [list_of_pdbs[i] for i in sorted(indices)]

    logger.info(f"Computing designability on {n_eval}/{len(list_of_pdbs)} PDBs...")
    results = batch_designability(subset_pdbs, tmp_root=tmp_root)

    return {
        "_res_designability_n": n_eval,
        "_res_scRMSD_mean": results["scRMSD_mean"],
        "_res_scRMSD_median": results["scRMSD_median"],
        "_res_designability_rate": results["designability_rate"],
        "_res_tm_score_self_mean": results["tm_score_mean"],
    }


def compute_diversity_metrics(
    tensors_dir, samples_dir, subset_per_bin, tm_threshold=0.5
):
    """Compute structural diversity per length bin via TM-score clustering.

    Returns:
        dict with diversity metric columns.
    """
    from proteinfoundation.metrics.tm_score import compute_diversity

    if subset_per_bin <= 0:
        return {}

    length_to_pdbs = build_length_to_pdbs(tensors_dir, samples_dir)
    if not length_to_pdbs:
        logger.warning("No length bins found for diversity computation")
        return {}

    from proteinfoundation.metrics.designability import load_pdb

    bin_clusters = []
    bin_lengths = []
    rng = np.random.RandomState(42)

    for nres in sorted(length_to_pdbs.keys()):
        pdbs = length_to_pdbs[nres]
        if len(pdbs) < 2:
            continue

        # Subsample if needed (pairwise TM is O(n^2))
        if len(pdbs) > subset_per_bin:
            idx = rng.choice(len(pdbs), size=subset_per_bin, replace=False)
            pdbs = [pdbs[i] for i in sorted(idx)]

        logger.info(f"Diversity: length={nres}, n_structures={len(pdbs)}")

        coords_list = []
        for pdb_path in pdbs:
            prot = load_pdb(pdb_path)
            coords_list.append(np.array(prot.atom_positions))

        n_clusters = compute_diversity(coords_list, tm_threshold=tm_threshold)
        bin_clusters.append(n_clusters)
        bin_lengths.append(nres)
        logger.info(f"  -> {n_clusters} clusters")

    if not bin_clusters:
        return {}

    clusters_arr = np.array(bin_clusters)
    return {
        "_res_diversity_n_bins": len(bin_clusters),
        "_res_diversity_clusters_mean": float(clusters_arr.mean()),
        "_res_diversity_clusters_median": float(np.median(clusters_arr)),
        "_res_diversity_clusters_total": int(clusters_arr.sum()),
    }


def compute_novelty_metrics(
    list_of_pdbs, centroid_path, tm_threshold=0.5, max_eval=500
):
    """Compute novelty as fraction of generated structures dissimilar to training centroids.

    Args:
        list_of_pdbs: All generated PDB paths.
        centroid_path: Path to .pt file containing list of [n_i, 37, 3] centroid arrays.
        tm_threshold: A structure is novel if max TM-score to any centroid < threshold.
        max_eval: Max number of generated PDBs to evaluate (random subset).

    Returns:
        dict with novelty metric columns.
    """
    from proteinfoundation.metrics.tm_score import compute_tm_score
    from proteinfoundation.metrics.designability import load_pdb

    if centroid_path is None or not os.path.exists(centroid_path):
        if centroid_path is not None:
            logger.warning(
                f"Centroid file not found: {centroid_path}, skipping novelty"
            )
        return {}

    logger.info(f"Loading training set centroids from {centroid_path}")
    centroids = torch.load(centroid_path, map_location="cpu", weights_only=False)
    # centroids: list of numpy arrays, each [n_i, 37, 3]
    if isinstance(centroids, torch.Tensor):
        centroids = [centroids[i].numpy() for i in range(len(centroids))]
    logger.info(f"Loaded {len(centroids)} training set cluster centroids")

    # Subsample generated PDBs
    n_eval = min(max_eval, len(list_of_pdbs))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(list_of_pdbs), size=n_eval, replace=False)
    eval_pdbs = [list_of_pdbs[i] for i in sorted(indices)]

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
        "_res_novelty_n": n_eval,
        "_res_novelty_rate": novelty_rate,
        "_res_novelty_max_tm_mean": float(max_tm_arr.mean()),
        "_res_novelty_max_tm_median": float(np.median(max_tm_arr)),
    }


def split_nlens(nlens_dict, max_nsamples=16):
    """Split lengths into (length, nsample) pairs. Copied from inference.py."""
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


def main():
    args = parse_args()

    needs_gpu = not (
        args.skip_generation and args.skip_fid and args.designability_subset == 0
    )
    if needs_gpu:
        assert torch.cuda.is_available(), "CUDA not available (use --skip_generation --skip_fid --designability_subset 0 for CPU-only mode)"

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    # ── Load config ──
    with hydra.initialize(
        config_path="../../src/proteina/configs/experiment_config",
        version_base=hydra.__version__,
    ):
        cfg = hydra.compose(config_name=args.config_name)
    logger.info(f"Config: {args.config_name}")

    # ── Paths (persist across restarts) ──
    root_path = f"./eval_output/{args.config_name}"
    samples_dir = os.path.join(root_path, "samples_fid")
    tensors_dir = os.path.join(root_path, "tensors")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    # ── Generation phase ──
    if not args.skip_generation:
        # ── Load model ──
        ckpt_file = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
        logger.info(f"Loading checkpoint: {ckpt_file}")
        model = Proteina.load_from_checkpoint(ckpt_file)
        model.configure_inference(cfg, nn_ag=None)
        model.eval()
        model.cuda()

        L.seed_everything(cfg.seed)

        # ── Build batch schedule ──
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
        lens_sample, nsamples = split_nlens(nlens_dict, max_nsamples=cfg.max_nsamples)
        total_batches = len(lens_sample)
        logger.info(
            f"Total batches: {total_batches} ({len(lens_list)} lengths x ~{cfg.nsamples_per_len}/{cfg.max_nsamples} batches/len)"
        )

        # ── Generate with checkpointing ──
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
            with torch.no_grad():
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

    # ── Compute metrics ──
    list_of_pdbs = sorted(
        glob.glob(os.path.join(samples_dir, "*_fid.pdb")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    logger.info(f"Computing metrics on {len(list_of_pdbs)} PDBs...")

    if not args.skip_fid:
        flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
        flat_dict = {k: str(v) for k, v in flat_dict.items()}
    else:
        flat_dict = {"config_name": args.config_name}
    columns = list(flat_dict.keys())
    res_row = list(flat_dict.values())

    if not args.skip_fid:
        for cfg_mf in cfg.metric_factory:
            assert cfg_mf.ca_only, "Please turn on ca_only for CAFlow model"
            metric_factory = GenerationMetricFactory(**cfg_mf).cuda()
            metrics = generation_metric_from_list(list_of_pdbs, metric_factory)
            for k, v in metrics.items():
                columns += ["_res_" + k]
                res_row += [v.cpu().item()]
            logger.info(f"Metrics ({cfg_mf.get('prefix', '')}) computed")

        # Free GearNet from GPU before loading ESMFold
        del metric_factory
        torch.cuda.empty_cache()
    else:
        logger.info("Skipping GearNet FID/fJSD/fS metrics (--skip_fid)")

    # ── Designability metrics ──
    desig_results = compute_designability_metrics(
        list_of_pdbs,
        subset_size=args.designability_subset,
        tmp_root=os.path.join(root_path, "tmp_designability"),
    )
    for k, v in desig_results.items():
        columns.append(k)
        res_row.append(v)
    if desig_results:
        logger.info(
            f"Designability: rate={desig_results.get('_res_designability_rate', 'N/A'):.3f}, "
            f"scRMSD_mean={desig_results.get('_res_scRMSD_mean', 'N/A'):.3f}"
        )

    # ── Diversity metrics ──
    div_results = compute_diversity_metrics(
        tensors_dir,
        samples_dir,
        subset_per_bin=args.diversity_subset_per_bin,
    )
    for k, v in div_results.items():
        columns.append(k)
        res_row.append(v)
    if div_results:
        logger.info(
            f"Diversity: mean_clusters={div_results.get('_res_diversity_clusters_mean', 'N/A'):.1f}, "
            f"n_bins={div_results.get('_res_diversity_n_bins', 'N/A')}"
        )

    # ── Novelty metrics ──
    nov_results = compute_novelty_metrics(
        list_of_pdbs,
        centroid_path=args.centroid_path,
        tm_threshold=args.novelty_tm_threshold,
    )
    for k, v in nov_results.items():
        columns.append(k)
        res_row.append(v)
    if nov_results:
        logger.info(
            f"Novelty: rate={nov_results.get('_res_novelty_rate', 'N/A'):.3f}, "
            f"max_tm_mean={nov_results.get('_res_novelty_max_tm_mean', 'N/A'):.3f}"
        )

    # ── Save results ──
    df = pd.DataFrame([res_row], columns=columns)
    if "metric_factory" in df.columns:
        df = df.drop("metric_factory", axis=1)

    results_csv = os.path.join(root_path, f"results_{args.config_name}_fid.csv")

    # If skipping FID and an existing CSV has FID columns, merge new columns in
    if args.skip_fid and os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        new_cols = [
            c
            for c in df.columns
            if c.startswith("_res_") and c not in existing_df.columns
        ]
        if new_cols:
            for c in new_cols:
                existing_df[c] = df[c].iloc[0]
            existing_df.to_csv(results_csv, index=False)
            logger.info(
                f"Merged {len(new_cols)} new columns into {results_csv}: {new_cols}"
            )
        else:
            logger.info(f"No new columns to add to {results_csv}")
    else:
        df.to_csv(results_csv, index=False)
        logger.info(f"Results saved to {results_csv}")


if __name__ == "__main__":
    main()
