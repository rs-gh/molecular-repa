#!/usr/bin/env python3
"""Ch2 figure: the two molecular domains we study, side by side, to make the
multi-scale / multi-factorial point concrete. Left: a small molecule (3D ball-and-
stick, ~tens of atoms). Right: a protein backbone (cartoon, hundreds of residues).
Both 3D, shared cartoon styling. Output: figures/fig_ch2_domains.png
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import use_report_style  # noqa: E402
from cartoon_util import draw_cartoon, ca_sse  # noqa: E402

use_report_style()
OUTDIR = os.path.join(HERE, "..")
PROTEIN_PDB = os.path.join(HERE, "1ubq.pdb")  # ubiquitin (alpha+beta), from RCSB


def protein_atom_count(pdb_path):
    """Heavy (non-H) protein atom count, for the scale comparison in the title."""
    import biotite.structure.io.pdb as pdbio
    import biotite.structure as struc

    arr = pdbio.PDBFile.read(pdb_path).get_structure(model=1)
    prot = arr[struc.filter_amino_acids(arr)]
    return int((prot.element != "H").sum())


# CPK-ish element colours / radii (heavy atoms)
CPK = {
    "C": ("#404040", 90),
    "N": ("#3050f8", 100),
    "O": ("#ff2d2d", 100),
    "F": ("#74d24a", 95),
    "S": ("#e6c533", 130),
    "Cl": ("#46c246", 130),
}


def draw_molecule(ax, smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE)
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    conf = m.GetConformer()
    pos = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(m.GetNumAtoms())
        ]
    )
    pos -= pos.mean(0)
    # PCA-orient so the molecular plane faces the viewer (ring reads as a ring)
    _, _, Vt = np.linalg.svd(pos, full_matrices=False)
    pos = pos @ Vt.T
    # bonds first (under atoms)
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        xs, ys, zs = zip(pos[i], pos[j])
        ax.plot(xs, ys, zs, color="#606060", lw=3, zorder=1, solid_capstyle="round")
    for i, at in enumerate(m.GetAtoms()):
        col, sz = CPK.get(at.GetSymbol(), ("#b0b0b0", 90))
        ax.scatter(
            *pos[i],
            s=sz,
            c=col,
            edgecolors="black",
            linewidths=0.4,
            depthshade=False,
            zorder=2,
        )
    rng = (pos.max(0) - pos.min(0)).max() / 2 + 0.3
    for L, mm in zip("xyz", pos.mean(0)):
        getattr(ax, f"set_{L}lim")(mm - rng, mm + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-88)
    return m.GetNumAtoms()


fig = plt.figure(figsize=(7.2, 3.6))
# left: small molecule (aspirin)
axm = fig.add_subplot(1, 2, 1, projection="3d")
natoms = draw_molecule(axm, "CC(=O)Oc1ccccc1C(=O)O")
axm.set_title(f"(a) Small molecule: aspirin\n{natoms} heavy atoms", fontsize=11, y=0.92)
# right: protein backbone (ubiquitin)
axp = fig.add_subplot(1, 2, 2, projection="3d")
co, ss = ca_sse(PROTEIN_PDB)
draw_cartoon(axp, co, ss, elev=12, azim=-70)
n_heavy = protein_atom_count(PROTEIN_PDB)
axp.set_title(
    f"(b) Protein backbone: ubiquitin\n{len(co)} residues, {n_heavy} heavy atoms",
    fontsize=11,
    y=0.92,
)
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0)
out = os.path.join(OUTDIR, "fig_ch2_domains.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
print("wrote", out, "| mol atoms", natoms, "| protein residues", len(co))
