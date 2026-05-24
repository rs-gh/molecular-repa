"""n=128 head-to-head multi-seed convergence plots.

Parameterised over (dataset, h2h comparison). For each comparison emits three
panels into ``figures/paper/n128_convergence/h2h/<dataset>/<name>/``:

  * ``h2h_des.png``     — designability, diversity, novelty, SS
  * ``h2h_fid.png``     — FID-family distributional metrics
  * ``h2h_quality.png`` — pLDDT, scRMSD, continuous max-TM novelty, TM-self

Same band semantics as ``plot_convergence_*_multi_seed.py``:
mean across reps + min/max envelope; single-rep legacy rows plot as bare markers.

Rendering machinery lives in ``_h2h_common.py`` — only ``COMPARISONS`` and the
output paths differ between n=128 and n=256. See ``plot_h2h_n256.py``.

Adding a new comparison: append a dict to ``COMPARISONS`` with the dataset's
results subdir, FID suffix, and the family list (baseline first).
"""

from __future__ import annotations

from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[5]))

from evaluation.proteina.generation.scripts.paper._h2h_common import run_all  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"
FIG_BASE = ROOT / "figures/paper/n128_convergence/h2h"


_PDB_BASELINE = ("baseline_128_bs80", "Baseline (PDB)", "tab:blue", "-", "s")
_PDB_GEAR_L4 = ("repa_l4_128_bs80", "REPA L4 GearNet (PDB)", "tab:red", "-", "o")
_PDB_GEAR_L9 = ("repa_l9_128_bs80", "REPA L9 GearNet (PDB)", "tab:green", "-", "o")
_PDB_MPNN_L4 = ("repa_mpnn_l4_128_bs80", "REPA L4 MPNN (PDB)", "tab:red", "-", "^")
_PDB_MPNN_L9 = (
    "repa_mpnn_l9_128_bs80_2gpu",
    "REPA L9 MPNN (PDB)",
    "tab:green",
    "-",
    "^",
)

_AFDB_BASELINE = ("baseline_afdb_128_bs80", "Baseline (AFDB)", "tab:blue", "-", "s")
_AFDB_GEAR_L4 = ("repa_l4_afdb_128_bs80", "REPA L4 GearNet (AFDB)", "tab:red", "-", "o")
_AFDB_MPNN_L4 = (
    "repa_mpnn_l4_afdb_128_bs80",
    "REPA L4 MPNN (AFDB)",
    "tab:red",
    "-",
    "^",
)
_AFDB_MPNN_L9 = (
    "repa_mpnn_l9_afdb_128_bs80_2gpu",
    "REPA L9 MPNN (AFDB)",
    "tab:green",
    "-",
    "^",
)

_PDB_REPS_NOTE = "5 reps (seeds 42/1042/2042/3042/4042)"
_AFDB_REPS_NOTE = "3 reps where available (seeds 42/1042/2042)"


COMPARISONS = [
    # ── original pairwise baselines ───────────────────────────────────────
    {
        "name": "baseline_vs_mpnn_l9",
        "title": "Baseline vs REPA L9 MPNN",
        "dataset": "PDB",
        "results_subdir": "n128_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_MPNN_L9],
    },
    {
        "name": "baseline_vs_gearnet_l4",
        "title": "Baseline vs REPA L4 GearNet",
        "dataset": "AFDB",
        "results_subdir": "n128_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_GEAR_L4],
    },
    # ── layer-based: fix the encoder, vary the alignment layer ───────────
    {
        "name": "layers/gearnet",
        "title": "Layer sweep — GearNet (L4 vs L9)",
        "dataset": "PDB",
        "results_subdir": "n128_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_GEAR_L4, _PDB_GEAR_L9],
    },
    {
        "name": "layers/mpnn",
        "title": "Layer sweep — MPNN (L4 vs L9)",
        "dataset": "PDB",
        "results_subdir": "n128_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_MPNN_L4, _PDB_MPNN_L9],
    },
    {
        "name": "layers/gearnet",
        "title": "Layer sweep — GearNet (L4 only; no L9 run on AFDB)",
        "dataset": "AFDB",
        "results_subdir": "n128_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_GEAR_L4],
    },
    {
        "name": "layers/mpnn",
        "title": "Layer sweep — MPNN (L4 vs L9)",
        "dataset": "AFDB",
        "results_subdir": "n128_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_MPNN_L4, _AFDB_MPNN_L9],
    },
    # ── encoder-based: fix the alignment layer, vary the encoder ─────────
    {
        "name": "encoders/L4",
        "title": "Encoder sweep at L4 — GearNet vs MPNN",
        "dataset": "PDB",
        "results_subdir": "n128_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_GEAR_L4, _PDB_MPNN_L4],
    },
    {
        "name": "encoders/L9",
        "title": "Encoder sweep at L9 — GearNet vs MPNN",
        "dataset": "PDB",
        "results_subdir": "n128_convergence_pdb",
        "fid_suffix": "PDB",
        "reps_note": _PDB_REPS_NOTE,
        "families": [_PDB_BASELINE, _PDB_GEAR_L9, _PDB_MPNN_L9],
    },
    {
        "name": "encoders/L4",
        "title": "Encoder sweep at L4 — GearNet vs MPNN",
        "dataset": "AFDB",
        "results_subdir": "n128_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_GEAR_L4, _AFDB_MPNN_L4],
    },
    {
        "name": "encoders/L9",
        "title": "Encoder sweep at L9 — MPNN only (no GearNet L9 run on AFDB)",
        "dataset": "AFDB",
        "results_subdir": "n128_convergence_afdb",
        "fid_suffix": "AFDB",
        "reps_note": _AFDB_REPS_NOTE,
        "families": [_AFDB_BASELINE, _AFDB_MPNN_L9],
    },
]


def main() -> None:
    # n128 historically calls ``pretrained_overlay.load_gen()`` with no arg —
    # preserve that by passing pretrained_key=None.
    run_all(
        comparisons=COMPARISONS,
        results_root=RESULTS,
        fig_base=FIG_BASE,
        n_label="n=128",
        pretrained_key=None,
    )


if __name__ == "__main__":
    main()
