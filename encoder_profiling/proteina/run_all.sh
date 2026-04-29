#!/usr/bin/env bash
# Re-run every proteina encoder probe against the same 200 PDB train proteins,
# then collate. Run from repo root with the venv active.
#
# Smoke first:  ./encoder_profiling/proteina/run_all.sh smoke
# Full run:     ./encoder_profiling/proteina/run_all.sh
set -euo pipefail

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

# CA-GearNet (trained)
python "$HERE/gearnet/explore_gearnet.py" --n-proteins "$N" --random-seed "$SEED"

# CA-GearNet (random init x3 seeds)
python "$HERE/gearnet_random/explore_gearnet_random.py" --n-proteins "$N" --random-seed "$SEED" \
  --init-seeds 0 1 2

# ESM2 650M
python "$HERE/esm/explore_esm.py" --n-proteins "$N" --random-seed "$SEED" $EXTRA_ESM

# MC-GearNet-Edge
python "$HERE/mc_gearnet/explore_mc_gearnet.py" --n-proteins "$N" --random-seed "$SEED"

# PW-GearNet (torsional - the recommended variant)
python "$HERE/pw_gearnet/explore_pw_gearnet.py" --n-proteins "$N" --random-seed "$SEED" \
  --ckpt "$PW_TORSIONAL" --variant torsional

# Collate
python "$HERE/collate.py"
echo "Done. See $HERE/comparison.csv and $HERE/figures/."
