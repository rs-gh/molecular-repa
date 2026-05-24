"""n=256 head-to-head multi-seed convergence plots.

Sibling of ``plot_h2h_n128.py`` — same rendering machinery (in
``_h2h_common.py``), different data sources / output dirs / run prefixes.
Comparisons cover both axes of the REPA ablation (alignment layer × encoder)
plus the original pairwise baselines.

Outputs to ``figures/paper/n256_convergence/h2h/<dataset>/<name>/``:
  * ``h2h_des.png``     — designability, diversity, novelty, SS
  * ``h2h_fid.png``     — FID-family distributional metrics
  * ``h2h_quality.png`` — pLDDT, scRMSD, continuous max-TM novelty, TM-self
"""

from __future__ import annotations

from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[5]))

from evaluation.proteina.generation.scripts.paper._h2h_common import run_all  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"
FIG_BASE = ROOT / "figures/paper/n256_convergence/h2h"


# Run prefixes mirror those in ``plot_convergence_fid_multi_seed.py`` / single-seed scripts.
_PDB_BASELINE = ("baseline_256_bs24_2gpu", "Baseline (PDB)", "tab:blue", "-", "s")
_PDB_GEAR_L4 = (
    "repa_l4_256_per_residue_bs24_2gpu",
    "REPA L4 GearNet (PDB)",
    "tab:red",
    "-",
    "o",
)
_PDB_GEAR_L9 = (
    "repa_l9_256_per_residue_bs24_2gpu",
    "REPA L9 GearNet (PDB)",
    "tab:green",
    "-",
    "o",
)
_PDB_MPNN_L4 = (
    "repa_mpnn_l4_256_per_residue",
    "REPA L4 MPNN (PDB)",
    "tab:red",
    "-",
    "^",
)
_PDB_MPNN_L9 = (
    "repa_mpnn_l9_256_per_residue",
    "REPA L9 MPNN (PDB)",
    "tab:green",
    "-",
    "^",
)

_AFDB_BASELINE = ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "-", "s")
_AFDB_GEAR_L4 = ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "-", "o")
_AFDB_GEAR_L9 = (
    "repa_l9_afdb_256",
    "REPA L9 GearNet (AFDB, partial)",
    "tab:green",
    "-",
    "o",
)
_AFDB_MPNN_L4 = ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "-", "^")
_AFDB_MPNN_L9 = ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:green", "-", "^")

# n=256 reps are sparser than n=128 (paper compute budget). PDB families
# generally have seeds 42/1042/2042; AFDB is often single-seed.
_PDB_REPS_NOTE = "3 reps where available (seeds 42/1042/2042)"
_AFDB_REPS_NOTE = "1–2 reps (seed 42 + legacy unseeded rows)"


COMPARISONS = [
    # ── original pairwise baselines ───────────────────────────────────────
    {
        "name": "baseline_vs_mpnn_l9",
        "title": "Baseline vs REPA L9 MPNN",
        "dataset": "PDB",
        "results_subdir": "n256_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_MPNN_L9],
    },
    {
        "name": "baseline_vs_mpnn_l4",  # requested addition for n256 PDB
        "title": "Baseline vs REPA L4 MPNN",
        "dataset": "PDB",
        "results_subdir": "n256_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_MPNN_L4],
    },
    {
        "name": "baseline_vs_gearnet_l4",
        "title": "Baseline vs REPA L4 GearNet",
        "dataset": "AFDB",
        "results_subdir": "n256_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_GEAR_L4],
    },
    {
        "name": "baseline_vs_mpnn_l9",
        "title": "Baseline vs REPA L9 MPNN",
        "dataset": "AFDB",
        "results_subdir": "n256_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_MPNN_L9],
    },
    # ── layer-based: fix the encoder, vary the alignment layer ───────────
    {
        "name": "layers/gearnet",
        "title": "Layer sweep — GearNet (L4 vs L9)",
        "dataset": "PDB",
        "results_subdir": "n256_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_GEAR_L4, _PDB_GEAR_L9],
    },
    {
        "name": "layers/mpnn",
        "title": "Layer sweep — MPNN (L4 vs L9)",
        "dataset": "PDB",
        "results_subdir": "n256_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_MPNN_L4, _PDB_MPNN_L9],
    },
    {
        "name": "layers/gearnet",
        "title": "Layer sweep — GearNet (L4 vs L9, partial)",
        "dataset": "AFDB",
        "results_subdir": "n256_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_GEAR_L4, _AFDB_GEAR_L9],
    },
    {
        "name": "layers/mpnn",
        "title": "Layer sweep — MPNN (L4 vs L9)",
        "dataset": "AFDB",
        "results_subdir": "n256_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_MPNN_L4, _AFDB_MPNN_L9],
    },
    # ── encoder-based: fix the alignment layer, vary the encoder ─────────
    {
        "name": "encoders/L4",
        "title": "Encoder sweep at L4 — GearNet vs MPNN",
        "dataset": "PDB",
        "results_subdir": "n256_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_GEAR_L4, _PDB_MPNN_L4],
    },
    {
        "name": "encoders/L9",
        "title": "Encoder sweep at L9 — GearNet vs MPNN",
        "dataset": "PDB",
        "results_subdir": "n256_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_GEAR_L9, _PDB_MPNN_L9],
    },
    {
        "name": "encoders/L4",
        "title": "Encoder sweep at L4 — GearNet vs MPNN",
        "dataset": "AFDB",
        "results_subdir": "n256_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_GEAR_L4, _AFDB_MPNN_L4],
    },
    {
        "name": "encoders/L9",
        "title": "Encoder sweep at L9 — GearNet (partial) vs MPNN",
        "dataset": "AFDB",
        "results_subdir": "n256_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_GEAR_L9, _AFDB_MPNN_L9],
    },
]


def main() -> None:
    run_all(
        comparisons=COMPARISONS,
        results_root=RESULTS,
        fig_base=FIG_BASE,
        n_label="n=256",
        pretrained_key="n256",
    )


if __name__ == "__main__":
    main()
