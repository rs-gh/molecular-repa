#!/bin/bash
#!
#! Benchmark: MC-GearNet-Edge REPA encoder, torch.compile ON (n=128, bs=80).
#! Runs 150 steps and reports per-step wall times for steps 50-150.
#! MC-GearNet eager is ~4s/step so allow 4h for Dynamo warmup + 100 measured.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/bench_mc_repa_compiled.sh

#SBATCH -J bench-mc-compiled
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --qos=gpu1
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/bench-mc-compiled-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/bench-mc-compiled-%j.err
#SBATCH -p ampere
#SBATCH --exclude=gpu-q-43

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="${DATA_PATH:-/rds/user/sr2173/hpc-work/proteina/data}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_MODE=disabled
export WANDB_INIT_TIMEOUT=120

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo ""

# Stage LMDB to local NVMe
LMDB_SRC="$DATA_PATH/pdb_train/lmdb"
LMDB_LOCAL="/tmp/proteina_pdb_lmdb_$$"
mkdir -p "$LMDB_LOCAL"
trap 'rm -rf "$LMDB_LOCAL"' EXIT

LMDB_SIZE_KB=$(du -sk "$LMDB_SRC" 2>/dev/null | cut -f1)
TMP_FREE_KB=$(df /tmp --output=avail 2>/dev/null | tail -1 | tr -d ' ')
if [ "${TMP_FREE_KB:-0}" -gt "$((LMDB_SIZE_KB + 1048576))" ]; then
    echo "=== Staging LMDB to local NVMe ==="
    for split in val train; do
        src="$LMDB_SRC/${split}.lmdb"
        [ -f "$src" ] && cp "$src" "$LMDB_LOCAL/${split}.lmdb" && echo "  Staged ${split}.lmdb"
    done
    export LMDB_DIR="$LMDB_LOCAL"
else
    echo "=== Using Lustre LMDB ==="
    export LMDB_DIR="$LMDB_SRC"
fi

rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo ""
echo "=== Benchmark: MC-GearNet-Edge REPA, compile=ON (150 steps, bs=40) ==="
echo ""

cd "$PROTEINA_DIR"
python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys, runpy, hydra, time
import lightning as L
import lightning.pytorch as lpl

class StepTimer(lpl.Callback):
    WARMUP = 50
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t0 = time.perf_counter()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.perf_counter() - self._t0
        step = trainer.global_step
        if step >= self.WARMUP:
            print(f'[STEP_TIME] step={step} elapsed={elapsed:.4f}s', flush=True)

_orig_fit = L.Trainer.fit
def _patched_fit(self, model, *a, **kw):
    self.callbacks.append(StepTimer())
    return _orig_fit(self, model, *a, **kw)
L.Trainer.fit = _patched_fit
lpl.Trainer.fit = _patched_fit

_orig_compose = hydra.compose
def _patched_compose(*a, **kw):
    cfg = _orig_compose(*a, **kw)
    try:
        cfg.run_name_ = 'bench_mc_compiled_${SLURM_JOB_ID:-0}'
    except Exception:
        pass
    return cfg
hydra.compose = _patched_compose

sys.argv = [
    'train_repa.py',
    '--config_name', 'training_repa_l0_128_per_residue_mc_edge',
    '--config_subdir', 'training/128/gearnet_rep/per_residue',
    '--max_steps', '150',
    '--batch_size', '40',
    '--nolog',
]
runpy.run_path('train_repa.py', run_name='__main__')
"
EXIT=$?

echo ""
python -u -c "
import re, statistics
lines = open('/rds/user/sr2173/hpc-work/proteina/logs/bench-mc-compiled-${SLURM_JOB_ID}.out').readlines()
times = [float(m.group(1)) for l in lines if (m := re.search(r'elapsed=([0-9.]+)s', l))]
if times:
    print(f'=== STEADY-STATE STEP TIMES (steps 50-150) ===')
    print(f'  n={len(times)}, mean={statistics.mean(times):.3f}s, median={statistics.median(times):.3f}s')
    print(f'  min={min(times):.3f}s, max={max(times):.3f}s')
    print(f'  throughput: {1/statistics.mean(times):.2f} steps/s')
else:
    print('No STEP_TIME lines found.')
" 2>/dev/null || true

echo ""
echo "=== BENCH COMPLETE (exit $EXIT) - $(date) ==="
exit $EXIT
