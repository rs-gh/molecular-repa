"""Run registry for the proteina ablation blocks.

Mirrors `docs/research/proteina_ablation_checkpoints.md` 1:1, one entry per
wandb run. `run_id` is the wandb run id, which equals the `run_name_` config
field (train_repa.py L250 `id=run_name`). Display names carry a leading
`MMDD-HHMM-` prefix; the id does not.

Each ablation maps to an ordered list of (run_id, label, color, linestyle).
Baseline first within each block.
"""

from __future__ import annotations

ENTITY_PROJECT = "sr2173-university-of-cambridge/proteina-repa"

# Per-run-id metadata: (display_label, color, linestyle, is_baseline)
# Colors are reused across ablation blocks where the same run appears.
_BASE = {
    # n=128 PDB
    "proteina_60m_baseline_128": ("baseline_128 (bs24)", "black", "-", True),
    "proteina_60m_baseline_128_bs80": ("baseline_128_bs80", "dimgray", "-", True),
    "proteina_60m_repa_l4_128_per_residue_bs12": (
        "repa L4 per_res bs12",
        "tab:red",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_bs24": (
        "repa L4 per_res bs24",
        "tab:orange",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_bs80": (
        "repa L4 per_res bs80 (CA-gn)",
        "tab:blue",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_bs80_lambda025": (
        "repa L4 λ=0.25",
        "tab:green",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_bs80_lambda1": (
        "repa L4 λ=1.0",
        "tab:purple",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_bs80_lambda2": (
        "repa L4 λ=2.0",
        "tab:brown",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_bs80_wd1e-2": (
        "repa L4 wd=1e-2",
        "tab:cyan",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_bs80_lr3x": (
        "repa L4 lr=3x",
        "tab:pink",
        "-",
        False,
    ),
    "proteina_60m_repa_l0_128_per_residue_bs80": (
        "repa L0 per_res bs80",
        "tab:olive",
        "-",
        False,
    ),
    "proteina_60m_repa_l9_128_per_residue_bs80": (
        "repa L9 per_res bs80",
        "tab:cyan",
        "-",
        False,
    ),
    "proteina_60m_repa_l0_128_per_sample": (
        "repa L0 per_sample",
        "tab:olive",
        "--",
        False,
    ),
    "proteina_60m_repa_l4_128_per_sample": (
        "repa L4 per_sample",
        "tab:blue",
        "--",
        False,
    ),
    "proteina_60m_repa_l9_128_per_sample": (
        "repa L9 per_sample",
        "tab:cyan",
        "--",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_random": (
        "repa L4 random-init gn",
        "tab:green",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l4_128_per_residue_bs80": (
        "repa L4 MPNN",
        "tab:purple",
        "-",
        False,
    ),
    "proteina_60m_repa_esm_l4_t30_128_per_residue": (
        "repa L4 ESM-t30",
        "tab:gray",
        "--",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_pw_structure": (
        "repa L4 pw-structure",
        "darkgreen",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_128_per_residue_pw_torsional": (
        "repa L4 pw-torsional",
        "magenta",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu": (
        "repa L9 MPNN bs80 2gpu",
        "tab:purple",
        "-",
        False,
    ),
    # n=128 AFDB
    "proteina_60m_baseline_afdb_128_bs80_2gpu": (
        "baseline AFDB bs80 2gpu",
        "black",
        "-",
        True,
    ),
    "proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu": (
        "repa L4 AFDB (CA-gn)",
        "tab:blue",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu": (
        "repa L4 AFDB MPNN",
        "tab:purple",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu": (
        "repa L9 AFDB MPNN",
        "tab:orange",
        "-",
        False,
    ),
    # n=256 PDB
    "proteina_60m_baseline_256_bs24_2gpu": (
        "baseline 256 bs24 2gpu",
        "black",
        "-",
        True,
    ),
    "proteina_60m_repa_l4_256_per_residue_bs24_2gpu": (
        "repa L4 256 bs24 (CA-gn)",
        "tab:blue",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_256_per_residue_bs80_2gpu": (
        "repa L4 256 bs80",
        "tab:cyan",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu": (
        "repa L4 256 random gn",
        "tab:green",
        "-",
        False,
    ),
    "proteina_60m_repa_l9_256_per_residue_bs24_2gpu": (
        "repa L9 256 (CA-gn)",
        "tab:olive",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l4_256_per_residue": (
        "repa L4 256 MPNN",
        "tab:purple",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l9_256_per_residue": (
        "repa L9 256 MPNN",
        "tab:brown",
        "-",
        False,
    ),
    "proteina_60m_repa_esm_l9_t30_256_per_residue": (
        "repa L9 256 ESM-t30",
        "tab:gray",
        "--",
        False,
    ),
    "proteina_60m_repa_l4_256_per_residue_lambda025": (
        "repa L4 256 λ=0.25",
        "tab:green",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_256_per_residue_lambda1": (
        "repa L4 256 λ=1.0",
        "tab:purple",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_256_per_residue_lambda2": (
        "repa L4 256 λ=2.0",
        "tab:brown",
        "-",
        False,
    ),
    # n=256 AFDB
    "proteina_60m_baseline_afdb_swissprot_256": (
        "baseline AFDB 256",
        "black",
        "-",
        True,
    ),
    "proteina_60m_repa_l4_256_afdb_per_residue": (
        "repa L4 256 AFDB (CA-gn)",
        "tab:blue",
        "-",
        False,
    ),
    "proteina_60m_repa_l9_256_afdb_per_residue": (
        "repa L9 256 AFDB (CA-gn)",
        "tab:olive",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l4_256_afdb_per_residue": (
        "repa L4 256 AFDB MPNN",
        "tab:purple",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l4_256_afdb_per_residue_bs80_2gpu": (
        "repa L4 256 AFDB MPNN bs80",
        "tab:pink",
        "-",
        False,
    ),
    "proteina_60m_repa_mpnn_l9_256_afdb_per_residue": (
        "repa L9 256 AFDB MPNN",
        "tab:brown",
        "-",
        False,
    ),
    # n=256 averaging — split runs. The 2026-04-17 rename forked each logical
    # model into two wandb runs: pre-rename id (part1) and canonical id (part2).
    # Checkpoint continuity is preserved through a physical store-dir rename.
    "proteina_60m_repa_l0_256_perres": (
        "repa L0 per_res (part1: pre-rename)",
        "tab:olive",
        "-",
        False,
    ),
    "proteina_60m_repa_l0_256_per_residue": (
        "repa L0 per_res (part2: post-rename)",
        "tab:olive",
        "--",
        False,
    ),
    "proteina_60m_repa_l4_256": (
        "repa L4 per_res (part1: pre-rename)",
        "tab:blue",
        "-",
        False,
    ),
    "proteina_60m_repa_l4_256_per_residue": (
        "repa L4 per_res (part2: post-rename)",
        "tab:blue",
        "--",
        False,
    ),
    "proteina_60m_repa_l9_256_perres": (
        "repa L9 per_res (part1: pre-rename)",
        "tab:cyan",
        "-",
        False,
    ),
    "proteina_60m_repa_l9_256_per_residue": (
        "repa L9 per_res (part2: post-rename)",
        "tab:cyan",
        "--",
        False,
    ),
    "proteina_60m_repa_l0_256_per_sample": (
        "repa L0 per_sample",
        "tab:olive",
        ":",
        False,
    ),
    "proteina_60m_repa_l4_256_persamp": (
        "repa L4 per_sample (part1: pre-rename)",
        "tab:blue",
        "-.",
        False,
    ),
    "proteina_60m_repa_l4_256_per_sample": (
        "repa L4 per_sample (part2: post-rename)",
        "tab:blue",
        ":",
        False,
    ),
    "proteina_60m_repa_l9_256_per_sample": (
        "repa L9 per_sample",
        "tab:cyan",
        ":",
        False,
    ),
}


def run_meta(run_id: str):
    return _BASE.get(run_id)


# Ablation blocks. Each is a list of run_ids in display order (baseline first).
# Names match the ablation doc section headers.
ABLATIONS: dict[str, list[str]] = {
    "n128_pdb_bs": [
        "proteina_60m_baseline_128",
        "proteina_60m_baseline_128_bs80",
        "proteina_60m_repa_l4_128_per_residue_bs12",
        "proteina_60m_repa_l4_128_per_residue_bs24",
        "proteina_60m_repa_l4_128_per_residue_bs80",
    ],
    "n128_pdb_l4_wd_lambda_lr": [
        "proteina_60m_baseline_128_bs80",
        "proteina_60m_repa_l4_128_per_residue_bs80",
        "proteina_60m_repa_l4_128_per_residue_bs80_lambda025",
        "proteina_60m_repa_l4_128_per_residue_bs80_lambda1",
        "proteina_60m_repa_l4_128_per_residue_bs80_lambda2",
        "proteina_60m_repa_l4_128_per_residue_bs80_wd1e-2",
        "proteina_60m_repa_l4_128_per_residue_bs80_lr3x",
    ],
    "n128_pdb_averaging": [
        "proteina_60m_baseline_128_bs80",
        "proteina_60m_repa_l0_128_per_residue_bs80",
        "proteina_60m_repa_l4_128_per_residue_bs80",
        "proteina_60m_repa_l9_128_per_residue_bs80",
        "proteina_60m_repa_l0_128_per_sample",
        "proteina_60m_repa_l4_128_per_sample",
        "proteina_60m_repa_l9_128_per_sample",
    ],
    "n128_pdb_l4_encoder": [
        "proteina_60m_baseline_128_bs80",
        "proteina_60m_repa_l4_128_per_residue_bs80",
        "proteina_60m_repa_l4_128_per_residue_random",
        "proteina_60m_repa_mpnn_l4_128_per_residue_bs80",
        "proteina_60m_repa_esm_l4_t30_128_per_residue",
        "proteina_60m_repa_l4_128_per_residue_pw_structure",
        "proteina_60m_repa_l4_128_per_residue_pw_torsional",
    ],
    "n128_pdb_l9_encoder": [
        "proteina_60m_baseline_128_bs80",
        "proteina_60m_repa_l9_128_per_residue_bs80",
        "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu",
    ],
    "n128_pdb_layer": [
        "proteina_60m_baseline_128_bs80",
        "proteina_60m_repa_l0_128_per_residue_bs80",
        "proteina_60m_repa_l4_128_per_residue_bs80",
        "proteina_60m_repa_l9_128_per_residue_bs80",
    ],
    "n128_afdb_encoder": [
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        "proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu",
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        "proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu",
    ],
    "n256_pdb_encoder_layer_bs": [
        "proteina_60m_baseline_256_bs24_2gpu",
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        "proteina_60m_repa_l4_256_per_residue_bs80_2gpu",
        "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu",
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        "proteina_60m_repa_esm_l9_t30_256_per_residue",
    ],
    "n256_pdb_l4_wd_lambda_bs": [
        "proteina_60m_baseline_256_bs24_2gpu",
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        "proteina_60m_repa_l4_256_per_residue_bs80_2gpu",
        "proteina_60m_repa_l4_256_per_residue_lambda025",
        "proteina_60m_repa_l4_256_per_residue_lambda1",
        "proteina_60m_repa_l4_256_per_residue_lambda2",
        "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu",
    ],
    "n256_afdb_encoder_layer_bs": [
        "proteina_60m_baseline_afdb_swissprot_256",
        "proteina_60m_repa_l4_256_afdb_per_residue",
        "proteina_60m_repa_l9_256_afdb_per_residue",
        "proteina_60m_repa_mpnn_l4_256_afdb_per_residue",
        "proteina_60m_repa_mpnn_l4_256_afdb_per_residue_bs80_2gpu",
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
    ],
    "n256_pdb_averaging": [
        # No baseline launched for this sweep — see ablation doc.
        # Pre-rename (part1) and canonical (part2) ids per the 2026-04-17 fork.
        "proteina_60m_repa_l0_256_perres",
        "proteina_60m_repa_l0_256_per_residue",
        "proteina_60m_repa_l4_256",
        "proteina_60m_repa_l4_256_per_residue",
        "proteina_60m_repa_l9_256_perres",
        "proteina_60m_repa_l9_256_per_residue",
        "proteina_60m_repa_l0_256_per_sample",
        "proteina_60m_repa_l4_256_persamp",
        "proteina_60m_repa_l4_256_per_sample",
        "proteina_60m_repa_l9_256_per_sample",
    ],
}


def all_run_ids() -> list[str]:
    seen, out = set(), []
    for rs in ABLATIONS.values():
        for r in rs:
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out
