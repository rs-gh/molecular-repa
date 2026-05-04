#!/usr/bin/env bash
# Re-run every proteina encoder probe against the same 200 PDB train proteins,
# then collate. Run from repo root with the venv active.
#
# Smoke first:  ./encoder_profiling/proteina/run_all.sh smoke
# Full run:     ./encoder_profiling/proteina/run_all.sh
#
# Per-encoder failures don't abort the sweep — we collect failures and exit
# nonzero at the end if any encoder failed. Collate still runs so partial
# results are usable.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
HERE="encoder_profiling/proteina"
DATA_PATH="${DATA_PATH:-/rds/user/sr2173/hpc-work/proteina/data}"
PW_TORSIONAL="$DATA_PATH/metric_factory/model_weights/pw_gearnet_torsional_denoising_ca_angles.ckpt"
PW_STRUCTURE="$DATA_PATH/metric_factory/model_weights/pw_gearnet_structure_denoising_ca_angles.ckpt"

MODE="${1:-full}"
if [[ "$MODE" == "smoke" ]]; then
  N=10
  EXTRA_ESM="--quick"
else
  N=200
  EXTRA_ESM=""
fi
SEED=0
echo "Mode=$MODE  N=$N  seed=$SEED"

FAILED=()
run_encoder() {
  local label="$1"; shift
  echo ""
  echo "=== $label ==="
  if "$@"; then
    echo "=== $label OK ==="
  else
    local rc=$?
    echo "=== $label FAILED rc=$rc ==="
    FAILED+=("$label(rc=$rc)")
  fi
}

run_encoder "ca-gearnet" \
  python "$HERE/gearnet/explore_gearnet.py" --n-proteins "$N" --random-seed "$SEED"

run_encoder "ca-gearnet-random" \
  python "$HERE/gearnet_random/explore_gearnet_random.py" --n-proteins "$N" --random-seed "$SEED" \
    --init-seeds 0 1 2

run_encoder "esm2-650m" \
  python "$HERE/esm/explore_esm.py" --n-proteins "$N" --random-seed "$SEED" $EXTRA_ESM

run_encoder "mc-gearnet-edge" \
  python "$HERE/mc_gearnet/explore_mc_gearnet.py" --n-proteins "$N" --random-seed "$SEED"

run_encoder "pw-gearnet-torsional" \
  python "$HERE/pw_gearnet/explore_pw_gearnet.py" --n-proteins "$N" --random-seed "$SEED" \
    --ckpt "$PW_TORSIONAL" --variant torsional

# Collate (best-effort: if all encoders failed, this will produce empty rows)
python "$HERE/collate.py" || echo "[warn] collate.py exited nonzero"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo ""
  echo "FAILED encoders: ${FAILED[*]}"
  echo "See $HERE/comparison.csv and $HERE/figures/ for whatever succeeded."
  exit 1
fi
echo "Done. See $HERE/comparison.csv and $HERE/figures/."
