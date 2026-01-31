"""Visualize generated molecules from test_sample.py output."""

import pickle
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import py3Dmol


def visualize_2d_grid(mols, mols_per_row=3, img_size=(300, 300)):
    """Create a 2D grid of molecules.

    Args:
        mols: List of RDKit molecules
        mols_per_row: Number of molecules per row
        img_size: Size of each molecule image

    Returns:
        PIL Image that can be displayed or saved
    """
    # Filter out None molecules and sanitize
    valid_mols = []
    for m in mols:
        if m is not None:
            try:
                Chem.SanitizeMol(m)
                valid_mols.append(m)
            except:  # noqa: E722
                pass

    # Add labels with properties
    legends = []
    for i, mol in enumerate(valid_mols):
        mw = Descriptors.MolWt(mol)
        n_atoms = mol.GetNumAtoms()
        legends.append(f"Mol {i+1}\nMW: {mw:.1f}\nAtoms: {n_atoms}")

    img = Draw.MolsToGridImage(
        valid_mols,
        molsPerRow=mols_per_row,
        subImgSize=img_size,
        legends=legends,
        returnPNG=False,
    )

    return img


def visualize_3d_inline(mol, width=400, height=400, style="stick"):
    """Visualize a single molecule in 3D (for Jupyter notebooks).

    Args:
        mol: RDKit molecule with 3D coordinates
        width: Viewer width in pixels
        height: Viewer height in pixels
        style: Visualization style ('stick', 'sphere', 'line', 'cartoon')

    Returns:
        py3Dmol viewer object
    """
    # Convert to PDB format for py3Dmol
    pdb_block = Chem.MolToPDBBlock(mol)

    # Create viewer
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(pdb_block, "pdb")
    viewer.setStyle({style: {}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()

    return viewer


def save_molecules_as_sdf(mols, output_path):
    """Save molecules to SDF file for viewing in external tools.

    Args:
        mols: List of RDKit molecules
        output_path: Path to save SDF file
    """
    valid_mols = [m for m in mols if m is not None]

    writer = Chem.SDWriter(output_path)
    for i, mol in enumerate(valid_mols):
        mol.SetProp("_Name", f"Generated_Molecule_{i+1}")
        mol.SetProp("MolecularWeight", f"{Descriptors.MolWt(mol):.2f}")
        mol.SetProp("NumAtoms", str(mol.GetNumAtoms()))
        writer.write(mol)
    writer.close()

    print(f"✓ Saved {len(valid_mols)} molecules to {output_path}")


def print_molecule_info(mols):
    """Print detailed information about each molecule.

    Args:
        mols: List of RDKit molecules
    """
    from rdkit.Chem import Lipinski

    print("=" * 70)
    print("Generated Molecules Details")
    print("=" * 70)

    for i, mol in enumerate(mols):
        if mol is None:
            print(f"\nMolecule {i+1}: INVALID")
            continue

        # Sanitize molecule to ensure ring info is computed
        try:
            Chem.SanitizeMol(mol)
        except:  # noqa: E722
            pass

        smiles = Chem.MolToSmiles(mol)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)

        print(f"\nMolecule {i+1}:")
        print(f"  SMILES: {smiles}")
        print(f"  Molecular Weight: {mw:.2f}")
        print(f"  LogP: {logp:.2f}")
        print(f"  H-Bond Donors: {hbd}")
        print(f"  H-Bond Acceptors: {hba}")
        print(f"  Rotatable Bonds: {rotatable}")
        print(f"  Atoms: {mol.GetNumAtoms()}")
        print(f"  Bonds: {mol.GetNumBonds()}")


def main():
    # Load molecules
    print("Loading molecules from test_molecules.pkl...")
    with open("test_molecules.pkl", "rb") as f:
        mols = pickle.load(f)

    print(f"Loaded {len(mols)} molecules ({sum(m is not None for m in mols)} valid)\n")

    # Print detailed info
    print_molecule_info(mols)

    # Save 2D grid image
    print("\n" + "=" * 70)
    print("Creating 2D visualization...")
    img = visualize_2d_grid(mols, mols_per_row=3, img_size=(400, 400))
    img.save("molecules_2d.png")
    print("✓ Saved 2D grid to molecules_2d.png")

    # Save to SDF for external viewers (like ChimeraX, PyMOL, VMD)
    print("\nSaving SDF file...")
    save_molecules_as_sdf(mols, "molecules_3d.sdf")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print("\nFiles created:")
    print("  • molecules_2d.png - 2D grid view (open with any image viewer)")
    print("  • molecules_3d.sdf - 3D structures (open with ChimeraX, PyMOL, etc.)")
    print("\nFor interactive 3D visualization in Jupyter:")
    print("  from visualize_molecules import visualize_3d_inline")
    print("  viewer = visualize_3d_inline(mols[0])")
    print("  viewer.show()")


if __name__ == "__main__":
    main()
