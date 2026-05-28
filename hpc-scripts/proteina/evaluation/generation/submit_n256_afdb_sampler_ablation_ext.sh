#!/bin/bash
#!
#! One-off submitter for n256_afdb_sampler_ablation_ext (2026-05-27).
#!
#! Extends the existing results/variance/n256_afdb_sampler_ablation jsonl with:
#!   - baseline + repa_l4 at the missing 400/1000/1300k anchors
#!   - full repa_l9_GearNet trajectory (100/400/700/900k)
#!   - full repa_mpnn_l9 trajectory (100/400/700/1000/1300k)
#! Five samplers per ckpt at rep_idx=0; default sde_n0.45 is already covered
#! by the n256_convergence_afdb sweep (multi-seed) so we skip it here.
#!
#! Total: 15 ckpts × 5 samplers × 1 rep = 75 tasks (5 array jobs of 15).
#!
#! Usage:
#!   bash hpc-scripts/proteina/evaluation/generation/submit_n256_afdb_sampler_ablation_ext.sh [--dry_run]
#!

set -euo pipefail

DRY=0
[[ "${1:-}" == "--dry_run" ]] && DRY=1

REPO_DIR="/home/sr2173/git/molecular-repa"
RUN_SH="$REPO_DIR/hpc-scripts/proteina/evaluation/generation/run_sweep.sh"
PROFILE="n256_afdb_sampler_ablation_ext"
OUTPUT_DIR="$REPO_DIR/evaluation/proteina/generation/results/variance/n256_afdb_sampler_ablation"
ARRAY_RANGE="0-14"  # 15 (run, step) tuples — matches profile dry-run

# (sampling_mode, sc_scale_noise) pairs. sde_n0.45 omitted on purpose.
SAMPLERS=("vf:" "sc:0.0" "sc:0.35" "sc:0.5" "sc:1.0")

mkdir -p "$OUTPUT_DIR"

echo "=== n256_afdb_sampler_ablation_ext submitter ==="
echo "  profile:    $PROFILE"
echo "  output_dir: $OUTPUT_DIR"
echo "  array:      $ARRAY_RANGE"
echo "  samplers:   ${SAMPLERS[*]}"
echo "  rep_idx:    0 (single seed; sde_n0.45 already multi-seed elsewhere)"
echo

for sampler in "${SAMPLERS[@]}"; do
    mode="${sampler%%:*}"
    noise="${sampler##*:}"
    cmd=(
        sbatch
        --array="$ARRAY_RANGE"
        "$RUN_SH"
        --config "$PROFILE"
        --output_dir "$OUTPUT_DIR"
        --sampling_mode "$mode"
        --rep_idx 0
    )
    [[ -n "$noise" ]] && cmd+=( --sc_scale_noise "$noise" )

    # seed 42 for rep 0 (matches submit_variance_sweep.sh convention).
    echo "  EVAL_SEED=42 ${cmd[*]}"
    if [[ "$DRY" -eq 0 ]]; then
        env EVAL_SEED=42 "${cmd[@]}"
    fi
done

echo
[[ "$DRY" -eq 1 ]] && echo "DRY RUN — no jobs submitted."
