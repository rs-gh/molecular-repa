"""Shared encoder-probe pipeline.

Each encoder driver in encoder_profiling/proteina/{esm,gearnet,mc_gearnet,...}/explore_*.py
imports `EncoderProbe` and `run_pipeline` from `lib`, supplies an `embed_fn` that
wraps its encoder's forward pass, and delegates the standard battery of analyses
to the lib. Encoder-specific bits (layerwise hooks, ckpt loading) stay local.
"""

from .lib import (
    EncoderProbe,
    analyze_dimensionality,
    analyze_distribution,
    analyze_norms,
    analyze_projector_saturation,
    analyze_protein_similarity,
    analyze_residue_discrimination,
    analyze_residue_shuffle,
    analyze_rotation_invariance,
    analyze_perturbation,
    analyze_sequence_context,
    analyze_structural_context,
    collect_all_embeddings,
    graph_to_inputs,
    load_proteins,
    make_embed_fn,
    run_pipeline,
)

__all__ = [
    "EncoderProbe",
    "analyze_dimensionality",
    "analyze_distribution",
    "analyze_norms",
    "analyze_projector_saturation",
    "analyze_protein_similarity",
    "analyze_residue_discrimination",
    "analyze_residue_shuffle",
    "analyze_rotation_invariance",
    "analyze_perturbation",
    "analyze_sequence_context",
    "analyze_structural_context",
    "collect_all_embeddings",
    "graph_to_inputs",
    "load_proteins",
    "make_embed_fn",
    "run_pipeline",
]
