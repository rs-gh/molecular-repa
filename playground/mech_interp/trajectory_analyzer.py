"""Core trajectory analysis framework for tabasco.

Captures per-layer, per-timestep information during flow-matching generation
and computes metrics analogous to proteina's crystallization analysis.

Four analysis lenses:
1. Structure lens: apply decoders to intermediate hidden states
2. Attention analysis: entropy, bond precision, distance correlation
3. REPA alignment landscape: cosine similarity to frozen encoder
4. Input ablation: zero out embedding components to test causal roles
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from playground.mech_interp.metrics import (
    attention_entropy,
    bond_precision,
    distance_correlation,
    structure_lens_rmsd,
    atom_type_accuracy,
    alignment_cosine_sim,
)


@dataclass
class AblationConfig:
    """Controls which input embeddings to zero out and when during generation.

    This is the tabasco analog of proteina's BiasAblationConfig. Since tabasco
    has no pair bias B, we instead ablate the four summed input embeddings:
    coords, atoms, posenc, time.
    """

    ablate_coords: bool = False
    ablate_atoms: bool = False
    ablate_posenc: bool = False
    ablate_time: bool = False
    t_min: float = 0.0
    t_max: float = 1.0

    def should_ablate(self, t: float) -> bool:
        """Check if ablation should be active at timestep t."""
        return self.t_min <= t <= self.t_max

    @property
    def label(self) -> str:
        """Human-readable label for this ablation condition."""
        parts = []
        if self.ablate_coords:
            parts.append("coords")
        if self.ablate_atoms:
            parts.append("atoms")
        if self.ablate_posenc:
            parts.append("posenc")
        if self.ablate_time:
            parts.append("time")
        if not parts:
            return "baseline"
        ablated = "+".join(parts)
        if self.t_min == 0.0 and self.t_max == 1.0:
            return f"no_{ablated}"
        elif self.t_min == 0.0:
            return f"no_{ablated}_early(t<={self.t_max})"
        elif self.t_max == 1.0:
            return f"no_{ablated}_late(t>={self.t_min})"
        else:
            return f"no_{ablated}_t[{self.t_min},{self.t_max}]"


@dataclass
class TrajectoryMetrics:
    """Stores computed metrics across the generation trajectory.

    Arrays indexed by [T, L] (timestep, layer) or [T, L, H] (with heads).
    """

    timesteps: np.ndarray  # [T]
    timestep_indices: np.ndarray  # [T]
    num_layers: int
    num_heads: int

    # Structure lens
    coord_rmsd_to_final: np.ndarray  # [T, L]
    coord_rmsd_to_gt: Optional[np.ndarray]  # [T, L] or None
    atom_accuracy: np.ndarray  # [T, L]

    # Attention
    attn_entropy: np.ndarray  # [T, L, H]
    bond_precision_vals: Optional[np.ndarray]  # [T, L, H] or None
    dist_correlation: Optional[np.ndarray]  # [T, L, H] or None

    # REPA alignment landscape
    chemprop_cosine_sim: Optional[np.ndarray]  # [T, L] or None

    def save(self, path: str):
        """Save metrics to npz file."""
        d = {
            "timesteps": self.timesteps,
            "timestep_indices": self.timestep_indices,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "coord_rmsd_to_final": self.coord_rmsd_to_final,
            "atom_accuracy": self.atom_accuracy,
            "attn_entropy": self.attn_entropy,
        }
        for key in [
            "coord_rmsd_to_gt",
            "bond_precision_vals",
            "dist_correlation",
            "chemprop_cosine_sim",
        ]:
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        np.savez(path, **d)

    @classmethod
    def load(cls, path: str) -> "TrajectoryMetrics":
        """Load metrics from npz file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            timesteps=data["timesteps"],
            timestep_indices=data["timestep_indices"],
            num_layers=int(data["num_layers"]),
            num_heads=int(data["num_heads"]),
            coord_rmsd_to_final=data["coord_rmsd_to_final"],
            coord_rmsd_to_gt=data.get("coord_rmsd_to_gt"),
            atom_accuracy=data["atom_accuracy"],
            attn_entropy=data["attn_entropy"],
            bond_precision_vals=data.get("bond_precision_vals"),
            dist_correlation=data.get("dist_correlation"),
            chemprop_cosine_sim=data.get("chemprop_cosine_sim"),
        )


class TrajectoryAnalyzer:
    """Captures and analyses per-layer, per-timestep information during generation.

    Usage:
        analyzer = TrajectoryAnalyzer(model)
        metrics = analyzer.analyze_generation(num_molecules=10, num_steps=100)
        metrics.save("output.npz")
    """

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: FlowMatchingModel instance with trained weights.
            device: Device to run analysis on.
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def analyze_generation(
        self,
        num_molecules: int = 10,
        num_steps: int = 100,
        capture_every_n: int = 5,
        gt_batch=None,
        encoder: Optional[nn.Module] = None,
        projector: Optional[nn.Module] = None,
        bond_matrices: Optional[Tensor] = None,
        dist_matrices: Optional[Tensor] = None,
        precomputed_x_final=None,
        precomputed_captures=None,
    ) -> TrajectoryMetrics:
        """Run generation with full trajectory capture and compute all metrics.

        Args:
            num_molecules: Number of molecules to generate.
            num_steps: Euler integration steps.
            capture_every_n: Capture every N timesteps.
            gt_batch: Optional ground-truth batch for GT-RMSD comparison.
            encoder: Optional frozen encoder for REPA landscape.
            projector: Optional projector for REPA landscape.
            bond_matrices: Optional [B, N, N] bond adjacency for bond precision.
            dist_matrices: Optional [B, N, N] distance matrices for spatial alignment.

        Returns:
            TrajectoryMetrics with all computed metrics.
        """
        net = self.model.net

        # Generate with analysis capture (or use pre-computed results)
        if precomputed_x_final is not None and precomputed_captures is not None:
            x_final = precomputed_x_final
            captures = precomputed_captures
        else:
            x_final, captures = self.model.sample_with_analysis(
                batch_size=num_molecules,
                num_steps=num_steps,
                capture_every_n=capture_every_n,
            )

        # Get final-layer prediction for structure lens reference
        final_coords = x_final["coords"].cpu()
        final_atomics = x_final["atomics"].cpu()
        final_mask = ~x_final["padding_mask"].cpu()  # True = real atom

        # Ground truth if available
        gt_coords = gt_batch["coords"].cpu() if gt_batch is not None else None
        _ = (
            gt_batch["atomics"].cpu() if gt_batch is not None else None
        )  # reserved for future use

        num_layers = net.num_layers
        num_heads = (
            net.transformer.layers[0].attn_block.attention.mha.num_heads
            if hasattr(net, "transformer") and hasattr(net.transformer, "layers")
            else 8
        )

        T = len(captures)
        L = num_layers
        H = num_heads

        # Pre-allocate metric arrays
        timesteps = np.array([c["timestep"] for c in captures])
        timestep_indices = np.array([c["timestep_idx"] for c in captures])
        coord_rmsd_to_final = np.full((T, L), np.nan)
        coord_rmsd_to_gt = np.full((T, L), np.nan) if gt_coords is not None else None
        atom_acc = np.full((T, L), np.nan)
        entropy = np.full((T, L, H), np.nan)
        bp = np.full((T, L, H), np.nan) if bond_matrices is not None else None
        dc = np.full((T, L, H), np.nan) if dist_matrices is not None else None
        repa_sim = (
            np.full((T, L), np.nan)
            if (encoder is not None and projector is not None)
            else None
        )

        # Compute encoder targets once if needed
        if encoder is not None and projector is not None:
            with torch.no_grad():
                # Use final generated coords as "clean" input to encoder
                target_repr = encoder(
                    final_coords.to(self.device),
                    final_atomics.to(self.device),
                    (~final_mask).to(self.device),  # padding_mask
                ).cpu()

        for ti, capture in enumerate(captures):
            analysis = capture["analysis_dict"]
            intermediates = analysis.get("intermediates", [])

            for li, h_layer in enumerate(intermediates):
                # h_layer: [B, N, D] — hidden state after transformer layer li
                h = h_layer.to(self.device)

                # Structure lens: apply decoders to intermediate hidden states
                with torch.no_grad():
                    decoded_coords = net.out_coord_linear(h).cpu()
                    decoded_logits = net.out_atom_type_linear(h).cpu()

                coord_rmsd_to_final[ti, li] = structure_lens_rmsd(
                    decoded_coords,
                    final_coords,
                    final_mask,
                )

                if gt_coords is not None:
                    coord_rmsd_to_gt[ti, li] = structure_lens_rmsd(
                        decoded_coords,
                        gt_coords,
                        final_mask,
                    )

                atom_acc[ti, li] = atom_type_accuracy(
                    decoded_logits,
                    final_atomics,
                    final_mask,
                )

                # REPA alignment landscape
                if repa_sim is not None:
                    repa_sim[ti, li] = alignment_cosine_sim(
                        h,
                        target_repr.to(self.device),
                        projector.to(self.device),
                        final_mask.to(self.device),
                    )

            # Attention metrics
            attn_weights_list = analysis.get("attn_weights", [])
            for li, attn_w in enumerate(attn_weights_list):
                # attn_w: [B, H, N, N] or [B, N, N]
                attn_w = attn_w.to(self.device)
                if attn_w.dim() == 3:
                    attn_w = attn_w.unsqueeze(1)

                ent = attention_entropy(attn_w, mask=final_mask.to(self.device))
                ent_mean = ent.mean(dim=0).cpu().numpy()  # [H]
                n_heads = min(H, ent_mean.shape[0])
                entropy[ti, li, :n_heads] = ent_mean[:n_heads]

                if bond_matrices is not None:
                    bp_val = bond_precision(
                        attn_w,
                        bond_matrices.to(self.device),
                    )
                    bp_mean = bp_val.mean(dim=0).cpu().numpy()
                    bp[ti, li, :n_heads] = bp_mean[:n_heads]

                if dist_matrices is not None:
                    dc_val = distance_correlation(
                        attn_w,
                        dist_matrices.to(self.device),
                        mask=(final_mask.unsqueeze(-1) & final_mask.unsqueeze(-2)).to(
                            self.device
                        )
                        if final_mask is not None
                        else None,
                    )
                    dc_mean = dc_val.mean(dim=0).cpu().numpy()
                    dc[ti, li, :n_heads] = dc_mean[:n_heads]

        return TrajectoryMetrics(
            timesteps=timesteps,
            timestep_indices=timestep_indices,
            num_layers=L,
            num_heads=H,
            coord_rmsd_to_final=coord_rmsd_to_final,
            coord_rmsd_to_gt=coord_rmsd_to_gt,
            atom_accuracy=atom_acc,
            attn_entropy=entropy,
            bond_precision_vals=bp,
            dist_correlation=dc,
            chemprop_cosine_sim=repa_sim,
        )

    def analyze_ablation(
        self,
        ablation_config: AblationConfig,
        num_molecules: int = 10,
        num_steps: int = 100,
    ) -> Dict:
        """Run generation with input embedding ablation.

        Temporarily patches TransformerModule.forward to zero out specified
        embedding components during generation, then evaluates the output.

        Args:
            ablation_config: Specifies what to ablate and when.
            num_molecules: Number of molecules to generate.
            num_steps: Euler steps.

        Returns:
            Dict with "config", "molecules" (TensorDict), and quality metrics.
        """
        net = self.model.net

        # Store original method
        original_compute = net._compute_input_embedding

        def patched_compute(coords, atomics, padding_mask, t):
            h_in, real_mask, embed_components = original_compute(
                coords,
                atomics,
                padding_mask,
                t,
            )

            # Check if we should ablate at this timestep
            t_val = t.mean().item() if isinstance(t, Tensor) else t
            if ablation_config.should_ablate(t_val):
                # Re-sum embeddings, zeroing out ablated components
                ec = embed_components
                parts = []
                if not ablation_config.ablate_coords:
                    parts.append(ec["coords"])
                if not ablation_config.ablate_atoms:
                    parts.append(ec["atoms"])
                if not ablation_config.ablate_posenc:
                    parts.append(ec["posenc"])
                if not ablation_config.ablate_time:
                    parts.append(ec["time"])

                if parts:
                    h_in = sum(parts)
                else:
                    h_in = torch.zeros_like(h_in)
                h_in = h_in * real_mask.unsqueeze(-1)

            return h_in, real_mask, embed_components

        # Monkey-patch
        net._compute_input_embedding = patched_compute

        try:
            with torch.no_grad():
                x_final = self.model.sample(
                    batch_size=num_molecules,
                    num_steps=num_steps,
                )
        finally:
            # Restore original
            net._compute_input_embedding = original_compute

        return {
            "config": ablation_config,
            "label": ablation_config.label,
            "molecules": x_final.cpu(),
        }


def get_standard_ablation_configs() -> List[AblationConfig]:
    """Return the standard set of ablation experiments.

    Mirrors proteina's temporal sweep: for each input component,
    test full ablation, early-only, and late-only.
    """
    configs = [
        # Baseline (no ablation)
        AblationConfig(),
        # Full coord ablation
        AblationConfig(ablate_coords=True),
        # Coords early-only (ablate coords for t > 0.5)
        AblationConfig(ablate_coords=True, t_min=0.5, t_max=1.0),
        # Coords late-only (ablate coords for t <= 0.5)
        AblationConfig(ablate_coords=True, t_min=0.0, t_max=0.5),
        # Full atom ablation
        AblationConfig(ablate_atoms=True),
        # Atoms early-only
        AblationConfig(ablate_atoms=True, t_min=0.5, t_max=1.0),
        # Atoms late-only
        AblationConfig(ablate_atoms=True, t_min=0.0, t_max=0.5),
        # Full posenc ablation
        AblationConfig(ablate_posenc=True),
        # Full time ablation
        AblationConfig(ablate_time=True),
    ]
    return configs
