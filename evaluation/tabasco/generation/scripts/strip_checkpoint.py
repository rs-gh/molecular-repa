"""Strip training checkpoints to inference-only format.

Extracts EMA weights for model.net, data_stats, and hyper_parameters,
producing ~15MB files instead of 4.2GB originals.

Usage:
    python evaluation/tabasco/generation/scripts/strip_checkpoint.py \
        --input /rds/.../checkpoints/last.ckpt \
        --output ./checkpoints/baseline.ckpt

    # Or strip all known GEOM checkpoints at once:
    python evaluation/tabasco/generation/scripts/strip_checkpoint.py --all
"""

import argparse
import torch
from pathlib import Path


OUTPUTS_ROOT = Path("/rds/user/sr2173/hpc-work/tabasco/outputs")

CHECKPOINT_MAP = {
    "baseline": "geom_mild/checkpoints/last.ckpt",
    "chemeleon_additive_fused": "geom_chemprop_additive_v2/checkpoints/last.ckpt",
    "chemeleon_additive_same": "geom_chemprop_additive/checkpoints/last.ckpt",
    "chemeleon_tradeoff_fused": "geom_chemprop_tradeoff_v2/checkpoints/last.ckpt",
    "chemeleon_tradeoff_same": "geom_chemprop_tradeoff/checkpoints/last.ckpt",
    "mace_additive": "geom_mace_cached_additive_v2/checkpoints/last.ckpt",
    "mace_tradeoff": "geom_mace_cached_tradeoff_v2/checkpoints/last.ckpt",
}


def _extract_net_config(hyper_parameters, state_dict=None):
    """Extract minimal TransformerModule config from the checkpoint.

    Tries two sources:
    1. hyper_parameters["model"] — the actual nn.Module (available when
       save_hyperparameters includes 'model', i.e. pre-MACE checkpoints).
    2. state_dict weight shapes — always available, infers architecture
       from tensor dimensions.
    """
    # --- Try from hyper_parameters first ---
    if hyper_parameters is not None:
        model = hyper_parameters.get("model")
        if model is not None:
            net = model.net
            if hasattr(net, "_orig_mod"):
                net = net._orig_mod
            return {
                "hidden_dim": net.hidden_dim,
                "num_layers": len(net.transformer.layers),
                "num_heads": net.transformer.layers[
                    0
                ].attn_block.attention.mha.num_heads,
                "cross_attention": hasattr(net, "coord_cross_attention"),
            }

    # --- Fallback: infer from state_dict weight shapes ---
    if state_dict is None:
        return {}

    # Normalize keys: strip common prefixes so patterns match uniformly
    keys = {}
    for k, v in state_dict.items():
        if "model.net" not in k:
            continue
        clean = (
            k.replace("model.net.", "")
            .replace("._orig_mod.", ".")
            .replace("._orig_mod", "")
        )
        # Strip leading dot if present
        clean = clean.lstrip(".")
        keys[clean] = v

    if not keys:
        return {}

    config = {}

    # hidden_dim: from in_proj_weight [3*hidden_dim, hidden_dim]
    for k, v in keys.items():
        if "mha.in_proj_weight" in k and v.dim() == 2:
            config["hidden_dim"] = v.shape[1]
            break

    # num_layers: count transformer layer indices
    layer_indices = set()
    for k in keys:
        if "transformer.layers." in k:
            for part in k.split("."):
                if part.isdigit():
                    layer_indices.add(int(part))
                    break
    if layer_indices:
        config["num_layers"] = max(layer_indices) + 1

    # num_heads: hidden_dim / head_dim (head_dim=16 for all current models)
    if "hidden_dim" in config:
        config["num_heads"] = config["hidden_dim"] // 16

    # cross_attention: check for cross attention keys
    config["cross_attention"] = any("cross_attention" in k for k in keys)

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
    model_config = _extract_net_config(ckpt.get("hyper_parameters"), state_dict)
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
        help="Strip all known GEOM checkpoints to evaluation/tabasco/checkpoints/geom/",
    )
    args = parser.parse_args()

    if args.all:
        out_dir = Path("evaluation/tabasco/checkpoints/geom")
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
