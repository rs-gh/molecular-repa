"""Strip training checkpoints to inference-only format.

Extracts EMA weights for model.net, data_stats, and hyper_parameters,
producing ~15MB files instead of 4.2GB originals.

Usage:
    python playground/tabasco/model_evaluation/strip_checkpoint.py \
        --input /rds/.../checkpoints/last.ckpt \
        --output ./checkpoints/baseline.ckpt

    # Or strip all known GEOM checkpoints at once:
    python playground/tabasco/model_evaluation/strip_checkpoint.py --all
"""

import argparse
import torch
from pathlib import Path


OUTPUTS_ROOT = Path("/rds/user/sr2173/hpc-work/tabasco/outputs")

CHECKPOINT_MAP = {
    "baseline": "geom_mild/checkpoints/last.ckpt",
    "additive_fused": "geom_chemprop_additive_v2/checkpoints/last.ckpt",
    "additive_same": "geom_chemprop_additive/checkpoints/last.ckpt",
    "tradeoff_fused": "geom_chemprop_tradeoff_v2/checkpoints/last.ckpt",
    "tradeoff_same": "geom_chemprop_tradeoff/checkpoints/last.ckpt",
}


def _extract_net_config(hyper_parameters):
    """Extract minimal TransformerModule config from the stored model object.

    hyper_parameters["model"] is the actual FlowMatchingModel nn.Module —
    for REPA variants this includes the 4GB CheMeleon encoder. We extract
    only what's needed to reconstruct model.net (TransformerModule).
    """
    if hyper_parameters is None:
        return {}

    model = hyper_parameters.get("model")
    if model is None:
        return {}

    net = model.net
    # Unwrap torch.compile's OptimizedModule wrapper
    if hasattr(net, "_orig_mod"):
        net = net._orig_mod

    config = {
        "hidden_dim": net.hidden_dim,
        "num_layers": len(net.transformer.layers),
        "num_heads": net.transformer.layers[0].attn_block.attention.mha.num_heads,
        "cross_attention": hasattr(net, "coord_cross_attention"),
    }
    return config


def strip_checkpoint(input_path: str, output_path: str):
    """Load a full training checkpoint, extract EMA weights for model.net only."""
    print(f"Loading {input_path} ...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    state_dict = ckpt["state_dict"]

    # --- Swap in EMA weights ---
    # EMA weights are stored inside optimizer_states[0]["ema"] as a tuple of
    # tensors that correspond 1:1 to the optimizer param groups.
    # During training, the EMA callback swaps them into the model for
    # validation/checkpointing.  In the saved checkpoint the model state_dict
    # already contains EMA weights (the callback swaps before saving).
    # Verify by checking if optimizer_states has an EMA key:
    has_ema = (
        "optimizer_states" in ckpt
        and len(ckpt["optimizer_states"]) > 0
        and isinstance(ckpt["optimizer_states"][0], dict)
        and "ema" in ckpt["optimizer_states"][0]
    )
    if has_ema:
        print(
            "  EMA weights found in optimizer_states — checkpoint state_dict "
            "already contains EMA weights (swapped before save)."
        )
    else:
        print("  No EMA in optimizer_states — using state_dict weights as-is.")

    # --- Extract only model.net.* keys ---
    # Strip _orig_mod. prefix injected by torch.compile's OptimizedModule wrapper
    net_state = {}
    for k, v in state_dict.items():
        if k.startswith("model.net."):
            clean_key = k.replace("._orig_mod", "")
            net_state[clean_key] = v

    print(
        f"  Extracted {len(net_state)} model.net keys "
        f"({sum(v.nelement() * v.element_size() for v in net_state.values()) / 1024 / 1024:.1f} MB)"
    )

    # --- Extract minimal architecture config from hyper_parameters ---
    # hyper_parameters stores the actual instantiated FlowMatchingModel nn.Module,
    # which for REPA variants includes the 4GB frozen CheMeleon encoder.
    # We extract only the TransformerModule config needed to reconstruct model.net.
    model_config = _extract_net_config(ckpt.get("hyper_parameters"))
    print(f"  Extracted model config: {model_config}")

    # --- Strip all_smiles from data_stats (57MB for GEOM) ---
    # Training SMILES are extracted separately as a text file for FCD/novelty.
    data_stats = ckpt.get("data_stats")
    if data_stats is not None:
        data_stats = {k: v for k, v in data_stats.items() if k != "all_smiles"}

    # --- Build minimal checkpoint ---
    stripped = {
        "state_dict": net_state,
        "data_stats": data_stats,
        "model_config": model_config,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "pytorch-lightning_version": ckpt.get("pytorch-lightning_version"),
        "stripped": True,  # marker so evaluate.py can detect format
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, output_path)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  Saved stripped checkpoint: {output_path} ({size_mb:.1f} MB)")
    return stripped


def main():
    parser = argparse.ArgumentParser(description="Strip checkpoints for inference")
    parser.add_argument("--input", type=str, help="Path to full training checkpoint")
    parser.add_argument("--output", type=str, help="Path for stripped checkpoint")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Strip all known GEOM checkpoints to ./evaluation_checkpoints/tabasco/",
    )
    args = parser.parse_args()

    if args.all:
        out_dir = Path("evaluation_checkpoints/tabasco")
        for name, rel_path in CHECKPOINT_MAP.items():
            input_path = OUTPUTS_ROOT / rel_path
            if not input_path.exists():
                print(f"SKIP {name}: {input_path} not found")
                continue
            output_path = out_dir / f"{name}.ckpt"
            strip_checkpoint(str(input_path), str(output_path))
        print(f"\nAll done. Stripped checkpoints in {out_dir}/")
    else:
        if not args.input or not args.output:
            parser.error("Provide --input and --output, or use --all")
        strip_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
