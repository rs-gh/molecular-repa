"""Convert raw protein structure files directly to LMDB format.

Usage:
    # From raw CIF/PDB files (no intermediate .pt needed):
    python convert_to_lmdb.py --config_name pdb_train_compile --config_subdir pdb

    # Incremental — re-run to add newly downloaded structures:
    python convert_to_lmdb.py --config_name pdb_train_compile --config_subdir pdb

Reads the dataset config to find the data directory, loads the CSV and
split assignments, then processes raw files directly into LMDB (one per split).
"""

import argparse
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../src/proteina/proteinfoundation")
    )
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/proteina"))
)

import proteinfoundation.repa.pyg_compat  # noqa: F401

import hydra
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.datasets.lmdb_utils import process_raw_to_lmdb


def main():
    parser = argparse.ArgumentParser(description="Process raw structures → LMDB")
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Dataset config name (e.g., pdb_train_compile)",
    )
    parser.add_argument(
        "--config_subdir",
        type=str,
        default="pdb",
        help="Config subdirectory (pdb or afdb)",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../src/proteina/configs/datasets_config",
            args.config_subdir,
        )
    )

    with hydra.initialize_config_dir(
        config_dir=config_path, version_base=hydra.__version__
    ):
        cfg_data = hydra.compose(config_name=args.config_name)

    cfg_resolved = OmegaConf.to_container(cfg_data.datamodule, resolve=True)
    data_dir = os.path.expandvars(cfg_resolved["data_dir"])
    file_format = cfg_resolved.get("format", "cif")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"File format: {file_format}")

    # Find the CSV
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if len(csv_files) == 0:
        logger.error(f"No CSV found in {data_dir}. Run prepare_data first.")
        sys.exit(1)

    csv_name = csv_files[0]
    if len(csv_files) > 1:
        logger.warning(f"Multiple CSVs found, using {csv_name}")

    logger.info(f"Loading CSV: {csv_name}")
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    logger.info(f"Total entries: {len(df)}")

    # Split using the datasplitter config
    splitter = hydra.utils.instantiate(cfg_resolved["datasplitter"])
    file_identifier = csv_name.replace(".csv", "")
    dfs_splits, _ = splitter.split_data(df, file_identifier)

    # Output and raw directories
    lmdb_dir = os.path.join(data_dir, "lmdb")
    raw_dir = os.path.join(data_dir, "raw")

    for split_name, df_split in dfs_splits.items():
        pdb_codes = df_split["pdb"].tolist()
        chains = df_split["chain"].tolist() if "chain" in df_split.columns else None

        lmdb_path = os.path.join(lmdb_dir, f"{split_name}.lmdb")
        logger.info(f"Processing {split_name}: {len(pdb_codes)} entries → {lmdb_path}")

        n_written = process_raw_to_lmdb(
            raw_dir=raw_dir,
            output_path=lmdb_path,
            pdb_codes=pdb_codes,
            chains=chains,
            file_format=file_format,
            apply_coord_reorder=True,
        )
        logger.info(f"{split_name}: {n_written} new samples written")

    logger.info(f"\nLMDB files in {lmdb_dir}:")
    for f in sorted(os.listdir(lmdb_dir)):
        size_mb = os.path.getsize(os.path.join(lmdb_dir, f)) / 1e6
        logger.info(f"  {f}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
