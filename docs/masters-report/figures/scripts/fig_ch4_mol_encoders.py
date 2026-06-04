#!/usr/bin/env python
"""Ch4 molecule-encoder contrast figure (CheMeleon vs MACE).

Two panels, each contrasting BOTH encoders:
  (A) embedding value distribution -> CheMeleon is 93.8% exact zeros (ReLU,
      sparse); MACE is dense and ~symmetric. The Q3 conditioning contrast.
  (B) cross-conformer cosine -> CheMeleon is identical across conformers
      (cos = 1.000, 2D by construction); MACE shifts slightly (cos ~ 0.998).
      The Q1 3D-awareness contrast.

Reuses the CheMeleon/MACE encoders and the GEOM loader from the profiling
code. CPU-only. Output: figures/fig_ch4_mol_encoders.{png,pdf}.

Usage:
    source .venv/bin/activate
    export PROJECT_ROOT=$(pwd)/src/tabasco
    python docs/masters-report/figures/scripts/fig_ch4_mol_encoders.py
"""

import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src" / "tabasco" / "src"))
sys.path.insert(0, str(REPO / "encoder_profiling" / "tabasco" / "chemeleon"))
sys.path.insert(0, str(REPO / "docs" / "masters-report" / "figures" / "scripts"))

from rdkit import RDLogger  # noqa: E402

RDLogger.logger().setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore")

import investigate as inv  # noqa: E402  (CheMeleon encoder + GEOM loader)
from tabasco.models.components.encoders import MACEEncoder  # noqa: E402
from style import use_report_style  # noqa: E402

OUT = REPO / "docs" / "masters-report" / "figures" / "fig_ch4_mol_encoders"

# Locked greys/colours -- keep CheMeleon and MACE visually distinct but neutral.
C_CHEM = "#c0392b"  # CheMeleon (2D) -- warm
C_MACE = "#2c6fbb"  # MACE (3D)      -- cool

N_PANEL_A = 150  # molecules for the value-distribution panel
N_CONF_MOLS = 40  # multi-conformer molecules for panel B
MAX_CONF = 5  # conformers per molecule


def atom_embeddings(encoder, items, smiles, batch_size=32, pass_smiles=True):
    """Stack real-atom embeddings [n_atoms_total, dim] for a list of items."""
    out = []
    for s in range(0, len(items), batch_size):
        bi = items[s : s + batch_size]
        bs = smiles[s : s + batch_size]
        coords = torch.stack([it["coords"] for it in bi])
        atomics = torch.stack([it["atomics"] for it in bi])
        masks = torch.stack([it["padding_mask"] for it in bi])
        with torch.no_grad():
            emb = encoder(coords, atomics, masks, smiles=bs if pass_smiles else None)
        for i in range(emb.shape[0]):
            n_real = int((~masks[i]).sum().item())
            out.append(emb[i, :n_real].float())
    return torch.cat(out, dim=0).numpy()


def cross_conformer_cosine(encoder, ds, groups, pass_smiles=True):
    """Mean within-molecule cross-conformer cosine, one value per molecule."""
    vals = []
    for smi, idxs in groups:
        items = [ds[i] for i in idxs[:MAX_CONF]]
        coords = torch.stack([it["coords"] for it in items])
        atomics = torch.stack([it["atomics"] for it in items])
        masks = torch.stack([it["padding_mask"] for it in items])
        with torch.no_grad():
            emb = encoder(
                coords,
                atomics,
                masks,
                smiles=[smi] * len(items) if pass_smiles else None,
            )
        n_real = int((~masks[0]).sum().item())
        per = emb[:, :n_real].float()  # [n_conf, n_real, dim]
        cs = []
        for i in range(per.shape[0]):
            for j in range(i + 1, per.shape[0]):
                cs.append(F.cosine_similarity(per[i], per[j], dim=-1).mean().item())
        if cs:
            vals.append(float(np.mean(cs)))
    return np.array(vals)


def compute():
    """Compute the four arrays (slow: loads both encoders + embeds GEOM)."""
    print("Loading GEOM ...")
    ds = inv.load_dataset("geom")
    print("Loading CheMeleon ...")
    chem = inv.get_encoder()
    print("Loading MACE (sets float64 at init; encoder restores) ...")
    mace = MACEEncoder(model_name="small")
    mace.eval()
    for p in mace.parameters():
        p.requires_grad = False

    # ---- sample molecules for panel A ----
    smiles_a, items_a = inv.sample_smiles_from_dataset(ds, n=N_PANEL_A)
    print(f"Panel A: {len(items_a)} molecules")
    print("  CheMeleon embeddings ...")
    chem_vals = atom_embeddings(chem, items_a, smiles_a, pass_smiles=True)
    print("  MACE embeddings ...")
    mace_vals = atom_embeddings(mace, items_a, smiles_a, pass_smiles=False)

    # ---- multi-conformer groups for panel B ----
    print("Panel B: scanning GEOM for conformer groups ...")
    s2i = defaultdict(list)
    for idx in range(min(10000, len(ds))):
        try:
            s2i[ds[idx].get_non_tensor("smiles")].append(idx)
        except (KeyError, AttributeError):
            continue
    groups = [(s, ix) for s, ix in s2i.items() if len(ix) >= 3][:N_CONF_MOLS]
    print(f"  {len(groups)} multi-conformer molecules")
    print("  CheMeleon cross-conformer cosine ...")
    chem_conf = cross_conformer_cosine(chem, ds, groups, pass_smiles=True)
    print("  MACE cross-conformer cosine ...")
    mace_conf = cross_conformer_cosine(mace, ds, groups, pass_smiles=False)
    return chem_vals, mace_vals, chem_conf, mace_conf


def main():
    use_report_style()
    cache = REPO / "docs" / "masters-report" / "figures" / ".fig_ch4_cache.npz"
    if cache.exists():
        print(f"Loading cached arrays from {cache.name}")
        z = np.load(cache)
        chem_vals, mace_vals = z["chem_vals"], z["mace_vals"]
        chem_conf, mace_conf = z["chem_conf"], z["mace_conf"]
    else:
        chem_vals, mace_vals, chem_conf, mace_conf = compute()
        np.savez(
            cache,
            chem_vals=chem_vals,
            mace_vals=mace_vals,
            chem_conf=chem_conf,
            mace_conf=mace_conf,
        )
        print(f"Cached arrays to {cache.name}")

    chem_zero = (chem_vals == 0).mean()
    mace_zero = (mace_vals == 0).mean()
    print(f"  sparsity: CheMeleon {chem_zero:.3f}  MACE {mace_zero:.3f}")
    print(f"  mean cos: CheMeleon {chem_conf.mean():.5f}  MACE {mace_conf.mean():.5f}")

    # ================= plot =================
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # --- Panel A: value distribution (density, log-y) ---
    lo, hi = -2.5, 3.5
    bins = np.linspace(lo, hi, 90)
    axA.hist(
        np.clip(mace_vals.ravel(), lo, hi),
        bins=bins,
        density=True,
        color=C_MACE,
        alpha=0.55,
        label=f"MACE ({mace_zero*100:.0f}% zeros)",
    )
    axA.hist(
        np.clip(chem_vals.ravel(), lo, hi),
        bins=bins,
        density=True,
        color=C_CHEM,
        alpha=0.55,
        label=f"CheMeleon ({chem_zero*100:.0f}% zeros)",
    )
    axA.set_yscale("log")
    axA.set_xlabel("embedding value")
    axA.set_ylabel("density (log)")
    axA.set_title("(a) Conditioning: sparsity", fontsize=10)
    axA.legend(fontsize=7, frameon=False, loc="upper right")

    # --- Panel B: cross-conformer cosine ---
    rng = np.random.RandomState(0)
    yj_m = 1 + (rng.rand(len(mace_conf)) - 0.5) * 0.5
    yj_c = 0 + (rng.rand(len(chem_conf)) - 0.5) * 0.5
    axB.scatter(mace_conf, yj_m, s=14, color=C_MACE, alpha=0.7, edgecolor="none")
    axB.scatter(chem_conf, yj_c, s=14, color=C_CHEM, alpha=0.7, edgecolor="none")
    axB.axvline(1.0, color="grey", lw=0.8, ls="--", alpha=0.7)
    axB.set_yticks([0, 1])
    axB.set_yticklabels(["CheMeleon", "MACE"])
    axB.set_ylim(-0.6, 1.6)
    axB.set_xlim(0.990, 1.0008)
    axB.set_xlabel("cross-conformer cosine")
    axB.set_title("(b) 3D-awareness: conformers", fontsize=10)
    axB.annotate(
        f"mean {chem_conf.mean():.3f}",
        (chem_conf.mean(), -0.45),
        ha="center",
        fontsize=7,
        color=C_CHEM,
    )
    axB.annotate(
        f"mean {mace_conf.mean():.3f}",
        (mace_conf.mean(), 1.45),
        ha="center",
        fontsize=7,
        color=C_MACE,
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}.png / .pdf")
    # stash the headline numbers for the caption
    print(
        "CAPTION_NUMS",
        f"chem_zero={chem_zero:.3f} mace_zero={mace_zero:.3f}",
        f"chem_conf={chem_conf.mean():.4f} mace_conf={mace_conf.mean():.4f}",
    )


if __name__ == "__main__":
    main()
