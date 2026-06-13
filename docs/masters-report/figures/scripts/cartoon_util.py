"""Shared Ca-trace cartoon renderer for the report's structure figures.

Headless, in-venv (biotite P-SEA + scipy spline + matplotlib). Works from Ca-only
coordinates via a parallel-transport frame: helix/sheet drawn as flat ribbons
(sheets taper to an arrowhead), coil as a thin tube; coloured by secondary structure.
Keeps one styling so Ch2 and Ch3 share a visual grammar. CM-serif fonts via style.py.
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev
import biotite.structure.io.pdb as pdbio
import biotite.structure as struc

SS_COLOR = {"a": "#e23b3b", "b": "#e8b317", "c": "#9aa0a6"}  # helix / sheet / coil
SS_WIDTH = {"a": 1.6, "b": 1.8, "c": 0.35}  # ribbon width (Angstrom)


def ca_sse(path):
    """Return (Ca coords [n,3], per-residue SSE chars 'a'/'b'/'c') for a PDB file."""
    arr = pdbio.PDBFile.read(path).get_structure(model=1)
    aa = arr[struc.filter_amino_acids(arr)]
    ca = aa[aa.atom_name == "CA"]
    sse = struc.annotate_sse(aa)
    n = min(len(ca), len(sse))
    return ca.coord[:n], sse[:n]


def pca_orient(P):
    """Rotate so principal axes -> (x,y,z); presents the 'flattest' view to +z."""
    c = P - P.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt.T


def _ptf(P):
    """Parallel-transport frame (tangent, in-plane normal) along a polyline."""
    n = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    N = np.zeros_like(P)
    a = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    N[0] = np.cross(T[0], a)
    N[0] /= np.linalg.norm(N[0]) + 1e-9
    for i in range(1, n):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        if s < 1e-6:
            N[i] = N[i - 1]
        else:
            v /= s
            ang = np.arccos(np.clip(np.dot(T[i - 1], T[i]), -1, 1))
            N[i] = (
                N[i - 1] * np.cos(ang)
                + np.cross(v, N[i - 1]) * np.sin(ang)
                + v * np.dot(v, N[i - 1]) * (1 - np.cos(ang))
            )
        N[i] -= np.dot(N[i], T[i]) * T[i]
        N[i] /= np.linalg.norm(N[i]) + 1e-9
    return T, N


def draw_cartoon(ax, coords, sse, orient=True, dens=6, elev=20, azim=-60):
    """Render a Ca cartoon into a 3D axis. coords [n,3], sse per-residue chars."""
    if orient:
        coords = pca_orient(coords)
    n = len(coords)
    k = 3 if n > 3 else max(1, n - 1)
    tck, _ = splprep(coords.T, s=0, k=k)
    uu = np.linspace(0, 1, n * dens)
    P = np.array(splev(uu, tck)).T
    ss = sse[np.clip((uu * (n - 1)).round().astype(int), 0, n - 1)]
    _, N = _ptf(P)
    W = np.array([SS_WIDTH[s] for s in ss])
    # taper each beta strand to an arrowhead at its C-terminal end
    i = 0
    while i < n:
        if sse[i] == "b":
            j = i
            while j + 1 < n and sse[j + 1] == "b":
                j += 1
            hi = min((j + 1) * dens, len(P) - 1)
            base = max(i * dens, hi - dens * 2)
            W[base : hi + 1] = np.linspace(2.8, 0.05, hi + 1 - base)
            i = j + 1
        else:
            i += 1
    eL = P + (W[:, None] / 2) * N
    eR = P - (W[:, None] / 2) * N
    polys = [[eL[m], eR[m], eR[m + 1], eL[m + 1]] for m in range(len(P) - 1)]
    cols = [SS_COLOR[s] for s in ss[:-1]]
    ax.add_collection3d(
        Poly3DCollection(polys, facecolors=cols, edgecolors="black", linewidths=0.15)
    )
    rng = (coords.max(0) - coords.min(0)).max() / 2
    mid = (coords.max(0) + coords.min(0)) / 2
    for L, m in zip("xyz", mid):
        getattr(ax, f"set_{L}lim")(m - rng, m + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
