#!/bin/bash
#
# Bundle files needed for lite FID evaluation on a remote server.
#
# Usage:
#   # 1. Create the bundle (on HPC):
#   bash hpc-scripts/proteina/evaluation/transfer_lite_eval.sh bundle [run_type] [step]
#   bash hpc-scripts/proteina/evaluation/transfer_lite_eval.sh bundle baseline 250000
#   bash hpc-scripts/proteina/evaluation/transfer_lite_eval.sh bundle_all   # all 47 checkpoints
#
#   # 2. Transfer to remote:
#   rsync -avP lite_eval_bundle/ remote_server:~/lite_eval_bundle/
#
#   # 3. Run on remote (see eval_lite_remote.sh generated in the bundle):
#   cd molecular-repa && bash lite_eval_bundle/eval_lite_remote.sh baseline 250000
#

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STORE_BASE="/rds/user/sr2173/hpc-work/proteina/store"
DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
BUNDLE_DIR="$REPO_DIR/lite_eval_bundle"

# Checkpoint steps (same as eval_fid_lite_sweep.sh)
BASELINE_STEPS=(10000 20000 40000 80000 150000 250000 350000 450000 550000 650000 740000)
REPA_STEPS=(10000 20000 40000 80000 150000 250000 350000 450000 550000 650000 750000 840000)
REPA_L0_STEPS=(10000 20000 40000 80000 150000 250000 350000 450000 550000 650000 750000 830000)

declare -A RUN_STORES=(
    [baseline]="proteina_60m_baseline_v2"
    [repa]="proteina_60m_repa_v2"
    [repa_layer0]="proteina_60m_repa_layer0_v2"
    [repa_layer9]="proteina_60m_repa_layer9_v2"
)

declare -A RUN_CONFIGS=(
    [baseline]="inference_fid_60m_baseline_lite"
    [repa]="inference_fid_60m_repa_lite"
    [repa_layer0]="inference_fid_60m_repa_layer0_lite"
    [repa_layer9]="inference_fid_60m_repa_layer9_lite"
)

get_steps_for_run() {
    case "$1" in
        baseline)    echo "${BASELINE_STEPS[@]}" ;;
        repa)        echo "${REPA_STEPS[@]}" ;;
        repa_layer0) echo "${REPA_L0_STEPS[@]}" ;;
        repa_layer9) echo "${REPA_STEPS[@]}" ;;
    esac
}

find_ckpt_name() {
    local store="$1" step="$2"
    local padded=$(printf "%012d" "$step")
    ls "$STORE_BASE/$store/checkpoints/" | grep "step=${padded}-EMA.ckpt$" | head -1
}

copy_shared_data() {
    echo "=== Copying shared metric data ==="
    local data_dir="$BUNDLE_DIR/data/metric_factory"
    mkdir -p "$data_dir/model_weights" "$data_dir/features"

    cp -v "$DATA_PATH/metric_factory/model_weights/gearnet_ca.pth" "$data_dir/model_weights/"
    cp -v "$DATA_PATH/metric_factory/features/pdb_eval_ca_features.pth" "$data_dir/features/"
    cp -v "$DATA_PATH/metric_factory/features/D_FS_eval_ca_features.pth" "$data_dir/features/"
    cp -v "$DATA_PATH/metric_factory/features/D_FS_afdb_cath_codes.pth" "$data_dir/features/"

    echo "Shared data: $(du -sh "$data_dir" | cut -f1)"
}

copy_checkpoint() {
    local run_type="$1" step="$2"
    local store="${RUN_STORES[$run_type]}"
    local ckpt_name=$(find_ckpt_name "$store" "$step")

    if [ -z "$ckpt_name" ]; then
        echo "ERROR: No checkpoint for $run_type step $step"
        return 1
    fi

    local ckpt_dir="$BUNDLE_DIR/checkpoints/$run_type"
    mkdir -p "$ckpt_dir"
    local src="$STORE_BASE/$store/checkpoints/$ckpt_name"
    local dst="$ckpt_dir/$ckpt_name"

    if [ -f "$dst" ]; then
        echo "  Already bundled: $run_type/$ckpt_name"
    else
        echo "  Copying: $ckpt_name ($(du -h "$src" | cut -f1))"
        cp "$src" "$dst"
    fi
}

generate_remote_script() {
    cat > "$BUNDLE_DIR/eval_lite_remote.sh" << 'REMOTE_EOF'
#!/bin/bash
#
# Run lite FID evaluation on a remote server.
#
# Prerequisites:
#   - git clone + pull the molecular-repa repo
#   - Copy this lite_eval_bundle/ into the repo root
#   - Install Python deps (pip install -e src/proteina)
#
# Usage:
#   bash lite_eval_bundle/eval_lite_remote.sh baseline 250000
#   bash lite_eval_bundle/eval_lite_remote.sh repa 420000
#   bash lite_eval_bundle/eval_lite_remote.sh all   # run all bundled checkpoints
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_DIR="$REPO_DIR/lite_eval_bundle"

export DATA_PATH="$BUNDLE_DIR/data"

declare -A RUN_STORES=(
    [baseline]="proteina_60m_baseline_v2"
    [repa]="proteina_60m_repa_v2"
    [repa_layer0]="proteina_60m_repa_layer0_v2"
    [repa_layer9]="proteina_60m_repa_layer9_v2"
)

declare -A RUN_CONFIGS=(
    [baseline]="inference_fid_60m_baseline_lite"
    [repa]="inference_fid_60m_repa_lite"
    [repa_layer0]="inference_fid_60m_repa_layer0_lite"
    [repa_layer9]="inference_fid_60m_repa_layer9_lite"
)

run_one() {
    local run_type="$1" step="$2"
    local config="${RUN_CONFIGS[$run_type]}"
    local ckpt_dir="$BUNDLE_DIR/checkpoints/$run_type"
    local padded=$(printf "%012d" "$step")
    local ckpt_name=$(ls "$ckpt_dir" | grep "step=${padded}-EMA.ckpt$" | head -1)

    if [ -z "$ckpt_name" ]; then
        echo "ERROR: No checkpoint for $run_type step $step in $ckpt_dir"
        return 1
    fi

    echo ""
    echo "=== Evaluating $run_type step $step ==="
    echo "=== Checkpoint: $ckpt_name ==="
    echo "=== Time: $(date) ==="

    cd "$REPO_DIR"

    python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')

import proteinfoundation.repa.pyg_compat

import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    result = _orig_torch_load(*args, **kwargs)
    if isinstance(result, dict) and 'state_dict' in result:
        sd = result['state_dict']
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        result['state_dict'] = {k: v for k, v in sd.items() if not k.startswith('repa_loss.')}
    return result
torch.load = _patched_load

sys.argv = ['evaluate.py',
    '--config_name', '$config',
    '--ckpt_name_override', '$ckpt_name',
    '--output_suffix', 'step_${step}',
    '--designability_subset', '0',
    '--diversity_subset_per_bin', '0',
]
import runpy
runpy.run_path('evaluation/proteina/generation/scripts/evaluate.py', run_name='__main__')
"
    echo "=== Done: $run_type step $step ($(date)) ==="
}

run_all() {
    for run_type in baseline repa repa_layer0 repa_layer9; do
        local ckpt_dir="$BUNDLE_DIR/checkpoints/$run_type"
        [ -d "$ckpt_dir" ] || continue
        for ckpt_file in $(ls "$ckpt_dir"/*.ckpt 2>/dev/null | sort); do
            local step=$(echo "$ckpt_file" | grep -oP 'step=0*\K[0-9]+(?=-EMA)')
            run_one "$run_type" "$step"
        done
    done
}

if [ "${1:-}" = "all" ]; then
    run_all
else
    RUN_TYPE="${1:?Usage: eval_lite_remote.sh <run_type|all> [step]}"
    STEP="${2:?Usage: eval_lite_remote.sh $RUN_TYPE <step>}"
    run_one "$RUN_TYPE" "$STEP"
fi
REMOTE_EOF
    chmod +x "$BUNDLE_DIR/eval_lite_remote.sh"
    echo "Generated: $BUNDLE_DIR/eval_lite_remote.sh"
}

# --- Main ---

CMD="${1:?Usage: $0 <bundle|bundle_all> [run_type] [step]}"

case "$CMD" in
    bundle)
        RUN_TYPE="${2:?Usage: $0 bundle <run_type> <step>}"
        STEP="${3:?Usage: $0 bundle $RUN_TYPE <step>}"
        mkdir -p "$BUNDLE_DIR"
        [ -f "$BUNDLE_DIR/data/metric_factory/model_weights/gearnet_ca.pth" ] || copy_shared_data
        copy_checkpoint "$RUN_TYPE" "$STEP"
        generate_remote_script
        echo ""
        echo "Bundle size: $(du -sh "$BUNDLE_DIR" | cut -f1)"
        echo "Transfer with: rsync -avP $BUNDLE_DIR/ remote:~/molecular-repa/lite_eval_bundle/"
        ;;

    bundle_all)
        mkdir -p "$BUNDLE_DIR"
        copy_shared_data
        for run_type in baseline repa repa_layer0 repa_layer9; do
            echo ""
            echo "=== Bundling $run_type ==="
            for step in $(get_steps_for_run "$run_type"); do
                copy_checkpoint "$run_type" "$step" || true
            done
        done
        generate_remote_script
        echo ""
        echo "Total bundle size: $(du -sh "$BUNDLE_DIR" | cut -f1)"
        echo "Transfer with: rsync -avP $BUNDLE_DIR/ remote:~/molecular-repa/lite_eval_bundle/"
        ;;

    *)
        echo "Usage: $0 <bundle|bundle_all> [run_type] [step]"
        echo ""
        echo "  bundle <run_type> <step>  — bundle one checkpoint"
        echo "  bundle_all                — bundle all 47 checkpoints"
        echo ""
        echo "Run types: baseline, repa, repa_layer0, repa_layer9"
        exit 1
        ;;
esac
