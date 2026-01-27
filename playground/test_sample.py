"""Simplified sampling script that avoids posebusters dependency issues."""

import torch
import pickle
import lightning as L

# Avoid importing from tabasco.callbacks which imports posebusters
from tabasco.models.lightning_tabasco import LightningTabasco
from tabasco.chem.convert import MoleculeConverter
from tensordict import TensorDict

torch.set_float32_matmul_precision("high")
L.seed_everything(42)

def main():
    checkpoint_path = "checkpoints/tabasco-mild/tabasco-geom-mild.ckpt"
    num_mols = 5
    num_steps = 50
    output_path = "test_molecules.pkl"

    print(f"Loading checkpoint from {checkpoint_path}...")
    # PyTorch 2.6+ requires weights_only=False for checkpoints with config objects
    lightning_module = LightningTabasco.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",  # Load to CPU first, then move to device
        weights_only=False,  # Required for checkpoints with OmegaConf configs
    )
    lightning_module.model.net.eval()

    # Use MPS (Metal) if available on M3, otherwise CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")

    lightning_module.to(device)

    print(f"\nGenerating {num_mols} molecules with {num_steps} steps...")
    with torch.no_grad():
        out_batch = lightning_module.sample(
            batch_size=num_mols,
            num_steps=num_steps
        )

    # Convert to RDKit molecules
    mol_converter = MoleculeConverter()
    generated_mols = mol_converter.from_batch(out_batch, sanitize=False)

    # Count valid molecules
    valid_count = sum(m is not None for m in generated_mols)
    print(f"\nResults:")
    print(f"  Total molecules: {len(generated_mols)}")
    print(f"  Valid molecules: {valid_count}")
    print(f"  Invalid molecules: {len(generated_mols) - valid_count}")

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(generated_mols, f)
    print("✓ Done!")

if __name__ == "__main__":
    main()
