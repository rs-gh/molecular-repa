#!/bin/bash
#SBATCH --job-name=pw_gearnet_smoke
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home/sr2173/git/molecular-repa/pw_gearnet_smoke_%j.log
#SBATCH --qos=intr
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=gpu-q-39

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

cd /home/sr2173/git/molecular-repa
source .venv/bin/activate
export PROJECT_ROOT=$(pwd)/src/proteina
export PYTHONUNBUFFERED=1

echo "=== NODE: $(hostname) ==="
echo "=== SLURM GPUS: job=${SLURM_JOB_GPUS:-unset} step=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi
echo ""

# Quick CUDA sanity check
python -c "
import torch
print(f'torch.cuda.is_available: {torch.cuda.is_available()}')
print(f'torch.cuda.device_count: {torch.cuda.device_count()}')
x = torch.randn(4, 4, device='cuda')
print(f'Basic CUDA tensor: {x.device}, sum={x.sum().item():.3f}')
print('CUDA SANITY OK')
"
echo ""

CKPT=/rds/user/sr2173/hpc-work/proteina/data/metric_factory/model_weights/pw_gearnet_torsional_denoising_ca_angles.ckpt

python -c "
import torch, sys
ckpt_path = '$CKPT'

# 1. Inspect checkpoint shapes
raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = {k[len('encoder.'):]: v for k, v in raw['state_dict'].items() if k.startswith('encoder.')}
print(f'Keys: {len(state)}')
print(f'layers.0.self_loop.weight:   {list(state[\"layers.0.self_loop.weight\"].shape)}')
print(f'layers.0.edge_linear.weight: {list(state[\"layers.0.edge_linear.weight\"].shape)}')
print(f'edge_layers.0.linear.weight: {list(state[\"edge_layers.0.linear.weight\"].shape)}')
print()

# 2. Load model (patch torch_scatter/sparse/cluster first)
import proteinfoundation.repa.pyg_compat  # noqa: F401
from proteinfoundation.metrics.gearnet_utils import NoTrainPWGearNetEdge
enc = NoTrainPWGearNetEdge(ckpt_path=ckpt_path)
enc = enc.cuda()
print(f'Loaded. Params: {sum(p.numel() for p in enc.parameters()):,}')

# 3. Forward pass (GPU)
torch.manual_seed(42)
n = 80
coords   = torch.randn(n, 3, device='cuda') * 5.0
restypes = torch.randint(0, 20, (n,), device='cuda')
atom2b   = torch.cat([torch.full((50,), 0, dtype=torch.long, device='cuda'),
                      torch.full((30,), 1, dtype=torch.long, device='cuda')])
with torch.no_grad():
    out = enc(coords, restypes, atom2b)
print(f'Forward OK. shape={out.shape}  mean={out.mean():.4f}  std={out.std():.4f}  finite={out.isfinite().all().item()}')
assert out.shape == (80, 3072)

# 4. Wrapper
from proteinfoundation.repa.gearnet_encoder import PWGearNetEdgePerResidueEncoder
wrapper = PWGearNetEdgePerResidueEncoder(ckpt_path=ckpt_path).cuda()
ca_nm   = torch.randn(4, 64, 3, device='cuda') * 0.5
mask    = torch.ones(4, 64, dtype=torch.bool, device='cuda')
restype = torch.randint(0, 20, (4, 64), device='cuda')
with torch.no_grad():
    out2 = wrapper(ca_nm, mask, restype)
print(f'Wrapper OK. shape={out2.shape}  mean={out2.mean():.4f}  std={out2.std():.4f}')
assert out2.shape == (4, 64, 3072)

# 5. Masked positions are zero
mask[0, 40:] = False
with torch.no_grad():
    out3 = wrapper(ca_nm, mask, restype)
assert (out3[0, 40:] == 0).all()
print('Masking: OK')
print()
print('ALL CHECKS PASSED')
"
