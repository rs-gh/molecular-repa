"""Pick a checkpoint + mode + length, generate a few samples, view them.

Edit the config block near the top, then run:
    python playground/proteina/generation_speedup/explore.py

Outputs:
    samples/explore/sample_XX.pdb  (atom37 PDBs)
    samples/explore/view.html      (py3Dmol viewer, open in browser)
    samples/explore/trace_XX.png   (matplotlib CA-trace fallback)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _speedup_utils import prepare_model, MODES  # noqa: E402
from _geometry import per_sample_summary  # noqa: E402
from checkpoints import resolve_active, CHECKPOINTS  # noqa: E402

# proteina PDB writer
sys.path.insert(0, str(HERE.parent.parent.parent / "src" / "proteina"))
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb  # noqa: E402

# ===========================================================================
# Config - edit these
# ===========================================================================

CKPT_KEY = "baseline_512_sm"  # must be uncommented in checkpoints.py:CHECKPOINTS
MODE = "compile_bf16"  # one of MODES
LENGTH = 256
N_SAMPLES = 4
SEED = 0

# ===========================================================================

GENERATE_KWARGS = dict(
    dt=0.0025,
    self_cond=True,
    cath_code=None,
    guidance_weight=1.0,
    autoguidance_ratio=1.0,
    schedule_mode="log",
    schedule_p=2.0,
    sampling_mode="sc",
    sc_scale_noise=0.45,
    sc_scale_score=1.0,
    gt_mode="1/t",
    gt_p=1.0,
    gt_clamp_val=None,
)


def build_html_viewer(pdb_paths: list[Path], out_html: Path, title: str):
    """Build a standalone HTML with side-by-side py3Dmol views."""
    pdb_strs = [p.read_text() for p in pdb_paths]
    # py3Dmol doesn't produce a multi-panel layout natively; make each viewer
    # its own div and tile them with simple flex CSS.
    divs = []
    inits = []
    for i, (path, pdb) in enumerate(zip(pdb_paths, pdb_strs)):
        div_id = f"view_{i}"
        divs.append(
            f'<div class="tile"><div class="cap">{path.name}</div>'
            f'<div id="{div_id}" class="mol"></div></div>'
        )
        inits.append(
            f"""
            var v{i} = $3Dmol.createViewer("{div_id}", {{backgroundColor:"white"}});
            v{i}.addModel(`{pdb}`, "pdb");
            v{i}.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
            v{i}.zoomTo();
            v{i}.render();
            """
        )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
 body {{ font-family: sans-serif; margin: 16px; }}
 h1 {{ font-size: 14px; margin: 4px 0; }}
 .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
 .tile {{ border: 1px solid #ccc; padding: 6px; }}
 .cap {{ font-size: 11px; margin-bottom: 4px; }}
 .mol {{ width: 360px; height: 360px; position: relative; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="grid">
{"".join(divs)}
</div>
<script>
{"".join(inits)}
</script>
</body>
</html>
"""
    out_html.write_text(html)


def plot_trace_2d(coords_atom37: np.ndarray, out_path: Path, title: str):
    """Save a 2-panel XY/YZ CA-trace PNG colored by residue index."""
    ca = coords_atom37[:, 1, :]  # CA_IDX = 1
    n = ca.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    colors = np.arange(n)
    for ax, (i_ax, j_ax, label_i, label_j) in [
        (ax1, (0, 1, "X", "Y")),
        (ax2, (1, 2, "Y", "Z")),
    ]:
        ax.plot(ca[:, i_ax], ca[:, j_ax], "-", color="lightgray", lw=0.8, zorder=1)
        sc = ax.scatter(
            ca[:, i_ax], ca[:, j_ax], c=colors, cmap="viridis", s=6, zorder=2
        )
        ax.set_xlabel(f"{label_i} (A)")
        ax.set_ylabel(f"{label_j} (A)")
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax2, label="residue idx", fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    if CKPT_KEY not in CHECKPOINTS:
        raise KeyError(
            f"CKPT_KEY={CKPT_KEY!r} not in CHECKPOINTS. Valid keys: {list(CHECKPOINTS)}"
        )
    assert MODE in MODES, f"MODE={MODE!r} not in {MODES}"

    active = {rc.name: rc for rc in resolve_active()}
    if CKPT_KEY not in active:
        raise KeyError(
            f"CKPT_KEY={CKPT_KEY!r} is commented-out in checkpoints.py. "
            f"Uncomment it first. Currently active: {list(active)}"
        )
    ckpt = active[CKPT_KEY]

    print(f"Loading {ckpt.name} in mode={MODE}")
    model, autocast_ctx = prepare_model(ckpt.ckpt_path, ckpt.is_repa, MODE)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"Generating {N_SAMPLES} samples @ length={LENGTH}")
    with torch.no_grad(), autocast_ctx():
        x = model.generate(
            nsamples=N_SAMPLES, n=LENGTH, dtype=torch.float32, **GENERATE_KWARGS
        )
        coords = model.samples_to_atom37(x).cpu().numpy()  # [N, L, 37, 3]

    out_dir = HERE / "samples" / "explore"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale
    for f in out_dir.glob("sample_*.pdb"):
        f.unlink()
    for f in out_dir.glob("trace_*.png"):
        f.unlink()

    pdb_paths = []
    for i in range(coords.shape[0]):
        pdb_path = out_dir / f"sample_{i:02d}.pdb"
        write_prot_to_pdb(coords[i], str(pdb_path), overwrite=True, no_indexing=True)
        pdb_paths.append(pdb_path)
        plot_trace_2d(
            coords[i],
            out_dir / f"trace_{i:02d}.png",
            title=f"{ckpt.name} | {MODE} | L={LENGTH} | sample {i}",
        )
        summary = per_sample_summary(coords[i])
        print(
            f"  sample {i}: Rg={summary['rg']:.2f}A  bond_mean={summary['bond_mean']:.3f}A "
            f"clash={summary['clash_rate']:.3%}"
        )

    title = f"{ckpt.name} | mode={MODE} | L={LENGTH} | seed={SEED}"
    view_html = out_dir / "view.html"
    build_html_viewer(pdb_paths, view_html, title)
    print(f"\nOutputs: {out_dir}")
    print(f"  open {view_html} in a browser for 3D view")


if __name__ == "__main__":
    main()
