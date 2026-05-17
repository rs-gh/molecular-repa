#!/bin/bash
#!
#! Submitter for the variance + sampler-regime sweep. Wraps `run_sweep.sh`
#! and dispatches the (sampler-config × replicate) matrix as a series of
#! independent array jobs, each writing to a shared JSONL keyed by
#! (run, step, sampler_tag, rep_idx).
#!
#! Output dirs and JSONL rows are namespaced so existing single-rep results
#! (those without a sampler_tag / rep_idx) are never overwritten. See
#! evaluation/proteina/generation/scripts/run_sweep.py:make_sampler_tag.
#!
#! Usage:
#!   # Axis A — variance reps on the current sde_n0.45 sampler.
#!   hpc-scripts/proteina/evaluation/generation/submit_variance_sweep.sh \
#!     --profile n128 --axis A --reps 3
#!
#!   # Axis B — sampler ablation on a narrowed run list.
#!   hpc-scripts/proteina/evaluation/generation/submit_variance_sweep.sh \
#!     --profile n128 --axis B --reps 3 --runs baseline_128,repa_l4_128
#!
#!   # Dry: print sbatch commands without submitting.
#!   ... --dry_run
#!
#! Replicate seeds: 42 + 1000·rep_idx (avoids collision with ProteinMPNN's
#! per-length seed = seed + nres derivation in evaluate.py:_subsample).
#!
#! Axis B sampler grid (sde_n0.45 omitted — reuse Axis A reps for it):
#!   - sampling_mode=vf                       -> tag "ode"
#!   - sampling_mode=sc, sc_scale_noise=0.0   -> tag "sde_n0.0"
#!   - sampling_mode=sc, sc_scale_noise=1.0   -> tag "sde_n1.0"

set -euo pipefail

PROFILE=""
AXIS=""
REPS=3
RUNS=""
DRY=0
EXTRA_ARGS=()
OUTPUT_DIR=""

usage() {
    sed -n '2,30p' "$0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --axis)    AXIS="$2"; shift 2 ;;
        --reps)    REPS="$2"; shift 2 ;;
        --runs)    RUNS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry_run) DRY=1; shift ;;
        -h|--help) usage ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

[[ -z "$PROFILE" ]] && { echo "ERROR: --profile required (e.g. n128, n256)"; exit 1; }
[[ -z "$AXIS"    ]] && { echo "ERROR: --axis required (A=variance, B=sampler)"; exit 1; }
[[ "$AXIS" != "A" && "$AXIS" != "B" ]] && { echo "ERROR: --axis must be A or B"; exit 1; }

REPO_DIR="/home/sr2173/git/molecular-repa"
RUN_SH="$REPO_DIR/hpc-scripts/proteina/evaluation/generation/run_sweep.sh"

# Profile size determines the array range. Sample-matched n128 / n256 each
# have 4 tasks; if a profile differs, callers can override via --runs (which
# bypasses the profile's full run list).
case "$PROFILE" in
    n128|n256) ARRAY_RANGE="0-3" ;;
    *)         ARRAY_RANGE="0-3" ;;  # default; rely on dry_run to confirm
esac
if [[ -n "$RUNS" ]]; then
    N_RUNS=$(echo "$RUNS" | tr ',' '\n' | grep -c .)
    ARRAY_RANGE="0-$((N_RUNS - 1))"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_DIR/evaluation/proteina/generation/results/variance/${PROFILE}"
fi
mkdir -p "$OUTPUT_DIR"

# Axis A: only the current sampler (sde_n0.45), N reps.
# Axis B: the other 3 sampler configs, N reps. sde_n0.45 reuses Axis A.
if [[ "$AXIS" == "A" ]]; then
    SAMPLERS=("sc:0.45")
else
    SAMPLERS=("vf:" "sc:0.0" "sc:1.0")
fi

echo "=== submit_variance_sweep ==="
echo "  profile:    $PROFILE"
echo "  axis:       $AXIS"
echo "  reps:       $REPS (seeds 42, 1042, 2042, ...)"
echo "  runs:       ${RUNS:-<from profile>}"
echo "  output_dir: $OUTPUT_DIR"
echo "  samplers:   ${SAMPLERS[*]}"
echo "  array:      $ARRAY_RANGE"
echo

for sampler in "${SAMPLERS[@]}"; do
    mode="${sampler%%:*}"
    noise="${sampler##*:}"
    for ((rep = 0; rep < REPS; rep++)); do
        seed=$((42 + 1000 * rep))
        cmd=(
            sbatch
            --array="$ARRAY_RANGE"
            "$RUN_SH"
            --config "$PROFILE"
            --output_dir "$OUTPUT_DIR"
            --sampling_mode "$mode"
            --rep_idx "$rep"
        )
        [[ -n "$noise" ]] && cmd+=( --sc_scale_noise "$noise" )
        [[ -n "$RUNS"  ]] && cmd+=( --runs "$RUNS" )
        cmd+=( "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" )

        # Inject the rep-specific seed via env.
        env_prefix=( EVAL_SEED="$seed" )

        echo "  ${env_prefix[*]} ${cmd[*]}"
        if [[ "$DRY" -eq 0 ]]; then
            env "${env_prefix[@]}" "${cmd[@]}"
        fi
    done
done

echo
if [[ "$DRY" -eq 1 ]]; then
    echo "DRY RUN — no jobs submitted."
fi
