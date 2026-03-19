"""Create a tiny dataset for testing REPA training on M3 Mac.

This script creates a small dataset of simple molecules for quick testing.
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tabasco.chem.convert import MoleculeConverter


def create_tiny_dataset(output_path: str = "tiny_dataset.pt", num_molecules: int = 20):
    """Create a tiny dataset of simple molecules.

    Args:
        output_path: Where to save the dataset
        num_molecules: Number of molecules to generate
    """

    # Simple SMILES strings - small molecules for fast testing
    smiles_list = [
        "C",  # Methane
        "CC",  # Ethane
        "CCC",  # Propane
        "CCCC",  # Butane
        "CO",  # Methanol
        "CCO",  # Ethanol
        "CC(C)C",  # Isobutane
        "C=C",  # Ethene
        "C#C",  # Ethyne
        "c1ccccc1",  # Benzene
        "CC(=O)C",  # Acetone
        "CC(=O)O",  # Acetic acid
        "CCN",  # Ethylamine
        "C1CC1",  # Cyclopropane
        "C1CCC1",  # Cyclobutane
        "CCl",  # Chloroethane
        "CF",  # Fluoromethane
        "C(C)N",  # Ethylamine
        "COC",  # Dimethyl ether
        "C(C)O",  # Ethanol variant
    ]

    # Repeat to get desired number of molecules
    smiles_list = (smiles_list * (num_molecules // len(smiles_list) + 1))[
        :num_molecules
    ]

    molecules = []
    failed = []

    print(f"Creating dataset with {num_molecules} molecules...")

    for i, smi in enumerate(smiles_list):
        try:
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failed.append(smi)
                continue

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            result = AllChem.EmbedMolecule(mol, randomSeed=42 + i)
            if result != 0:
                failed.append(smi)
                continue

            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol)

            molecules.append(mol)

        except Exception as e:
            print(f"Failed to process {smi}: {e}")
            failed.append(smi)

    print(f"Successfully created {len(molecules)} molecules")
    if failed:
        print(f"Failed: {len(failed)} molecules")

    # Convert to TABASCO format (TensorDict batches)
    print("\nConverting to TABASCO format...")
    converter = MoleculeConverter()
    batches = []
    for mol in molecules:
        try:
            batch = converter.to_tensor(
                mol, normalize_coords=True, remove_hydrogens=True
            )
            batches.append(batch)
        except Exception as e:
            print(f"Failed to convert molecule: {e}")

    print(f"Converted {len(batches)} molecules to batches")

    # Save dataset
    torch.save(batches, output_path)
    print(f"\n✓ Saved dataset to {output_path}")
    print(f"  Total molecules: {len(batches)}")

    # Print statistics
    num_atoms = [batch["coords"].shape[0] for batch in batches]
    print(
        f"  Atoms per molecule: min={min(num_atoms)}, max={max(num_atoms)}, avg={sum(num_atoms)/len(num_atoms):.1f}"
    )

    return batches


if __name__ == "__main__":
    batches = create_tiny_dataset("tiny_dataset.pt", num_molecules=20)
    print("\n✓ Dataset ready for training!")
