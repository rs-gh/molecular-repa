#!/bin/bash
#!
#! Smoke test: verify the NGC proteina_v1.3_DFS_60M_notri.ckpt loads via
#! Proteina.load_from_checkpoint and expose its trunk depth / hidden dim.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_pretrained_load.sh
#!

#SBATCH -J smoke-pre
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --qos=intr
#SBATCH --output=/home/sr2173/git/molecular-repa/.local_ckpts/smoke-pre-%j.out
#SBATCH --error=/home/sr2173/git/molecular-repa/.local_ckpts/smoke-pre-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# NGC ckpt construction uses ${oc.env:DATA_PATH}/pdb_raw/cath_label_mapping.pt
# in FoldEmbeddingSeqFeat; point at the real data dir on /rds.
export DATA_PATH="${DATA_PATH:-/rds/user/sr2173/hpc-work/proteina/data}"

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

python -u - <<'PY'
import sys
sys.path.insert(0, "src/proteina")
import torch
import proteinfoundation.repa.pyg_compat  # noqa: F401 — patch torch_scatter/cluster
from proteinfoundation.proteinflow.proteina import Proteina

# NGC ckpts contain omegaconf DictConfig in the pickle; PyTorch 2.6 weights_only
# default rejects it. The ckpt comes from a trusted NVIDIA source.
from omegaconf import DictConfig, ListConfig  # noqa: E402
from omegaconf.base import ContainerMetadata  # noqa: E402
import collections  # noqa: E402
torch.serialization.add_safe_globals([
    DictConfig, ListConfig, ContainerMetadata,
    dict, list, str, int, float, bool, type(None),
    collections.defaultdict,
])

ckpt = "/home/sr2173/git/molecular-repa/.local_ckpts/proteina_v1.3_DFS_60M_notri.ckpt"
print(f"Loading {ckpt}")
try:
    m = Proteina.load_from_checkpoint(ckpt, map_location="cpu")
except Exception as e:
    print(f"safe-globals path failed ({type(e).__name__}: {e}); falling back to weights_only=False")
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    m = Proteina.load_from_checkpoint(ckpt, map_location="cpu")
    torch.load = _orig_load
m.eval()

nn = m.nn
print(f"nn type: {type(nn).__name__}")
for a in ("blocks", "layers", "trunk", "transformer", "encoder", "trunk_blocks"):
    if hasattr(nn, a):
        v = getattr(nn, a)
        try: n = len(v)
        except TypeError: n = None
        print(f"  nn.{a}: type={type(v).__name__}, len={n}")

total = sum(p.numel() for p in m.parameters())
print(f"total params: {total/1e6:.2f}M")

# Try to mimic probelib's extract_model_hidden_states_multilayer discovery:
from proteinfoundation.nn import protein_transformer as pt_mod
print(f"ProteinTransformerAF3 module: {pt_mod.__file__}")

# Find trunk block ModuleList heuristically.
def find_blocks(mod, depth=0):
    import torch.nn as tnn
    for name, child in mod.named_children():
        if isinstance(child, tnn.ModuleList) and len(child) >= 4:
            yield (name, child, depth)
        yield from find_blocks(child, depth + 1)

hits = list(find_blocks(nn))
print(f"ModuleList hits (name, len, depth):")
for name, ml, d in hits:
    print(f"  {name}: len={len(ml)}, depth={d}")

print("OK")
PY
