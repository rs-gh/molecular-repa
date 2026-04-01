# Overnight Job Monitoring Log — 2026-04-01

## Jobs Submitted

| Job ID | Name | Purpose |
|--------|------|---------|
| 26678723 | prot-lmd | LMDB conversion: 579k PDB CIF files to LMDB |
| 26690247 | prot-bas | Baseline training (60M, no REPA, torch.compile) — resuming from checkpoint |
| 26690409 | prot-rep | REPA training (60M + GearNet encoder) — fresh start |

---

## Issues Encountered & Fixes Applied

### 1. `_FakeTorchCluster.__getattr__` crash (pyg_compat.py)
**Symptom**: `AttributeError: 'function' object has no attribute 'endswith'` on torch import.
**Root cause**: Our fake `torch_cluster` module's `__getattr__` returned stub functions for ALL attributes, including `__file__`. When Python's `inspect.getsourcefile()` tried `filename.endswith()`, it got a function, not a string.
**Fix**: Added guard to raise `AttributeError` for dunder attributes (`__file__`, `__path__`, etc.), letting Python handle them normally.
**Commit**: `1bbb6d8` (proteina submodule)

### 2. Validation DataLoader segfault (base_data.py)
**Symptom**: `RuntimeError: DataLoader worker (pid) exited unexpectedly` at epoch ~120.
**Root cause**: `num_workers=2` in the validation DataLoader causes segfaults on this cluster (fork+CUDA interaction). Training DataLoader with `num_workers=2` works because it uses `spawn` start method, but validation doesn't.
**Fix**: Override `val_dataloader()` to force `num_workers=0`. Val set is only 11 samples, so no performance impact.
**Commit**: `1bbb6d8` (proteina submodule)

### 3. Checkpoint resume `weights_only` (train_repa.py, train.py, ema_callback.py)
**Symptom**: `_pickle.UnpicklingError: Weights only load failed ... omegaconf.dictconfig.DictConfig`
**Root cause**: PyTorch 2.9 changed `torch.load` default to `weights_only=True`. Lightning checkpoints contain serialized omegaconf objects. Three separate load paths needed fixing:
  - `trainer.fit()` — Lightning's main checkpoint loader (fixed via `weights_only=False` parameter)
  - `ema_callback.py` line 157 — EMA state loading (fixed via explicit `weights_only=False`)
  - `train_repa.py` / `train.py` — pretrain checkpoint loading (fixed via explicit `weights_only=False`)
**Fix**: Pass `weights_only=False` at all three call sites. These are our own trusted checkpoints.
**Commits**: `2fa93e6`, `c4e17e4` (proteina submodule)

### 4. `radius_graph` crash on single-atom batch elements (pyg_compat.py)
**Symptom**: `RuntimeError: dimensions must larger than 1` in `fill_diagonal_()`, then `IndexError: Dimension out of range` in `dists_masked.size(1)`.
**Root cause**: Our native `_radius_graph_native` shim didn't handle batch elements with 0 or 1 atoms. GearNet's `construct_graph` can produce these for short proteins with padding.
**Fix**: Added `if n <= 1: continue` guard at the top of the per-batch loop. Also simplified the `max_num_neighbors` check to `n > max_num_neighbors`.
**Tests**: Added 24 unit tests (`tests/proteina/test_pyg_compat.py`) covering scatter ops, radius_graph edge cases (single atom, empty, batch boundaries, max_num_neighbors, GearNet patterns), and fake module inspect compatibility. All passing.
**Commit**: `c4e17e4` (proteina submodule)

### 5. NVMe full on compute node (gpu-q-2)
**Symptom**: `OSError: [Errno 28] No space left on device` when copying LMDB to local NVMe.
**Root cause**: The local `/tmp` on gpu-q-2 was full (possibly from other users' jobs). The SLURM script tries to copy LMDB to local NVMe to avoid Lustre mmap thrashing.
**Fix**: No code fix — resubmitted to get a different node. The SLURM script already has a fallback (`WARNING: LMDB not found ... using Lustre directly`) but the NVMe was full before torch could even write temp files.
**Action**: Resubmitted.

### 6. Baseline log buffering (NOT a hang)
**Symptom**: After checkpoint resume, no log output for 20+ minutes. Appeared to be torch.compile hanging.
**Root cause**: NOT a hang. GPU utilization logs show 93-96% GPU usage and 67.7GB/80GB memory. The model is training. Lustre filesystem buffers SLURM stdout/stderr in 4MB blocks, so output appears in bursts.
**Status**: RESOLVED — baseline is training normally. Logs will flush eventually.

### 7. `radius_graph` crash on 1D positional input (pyg_compat.py)
**Symptom**: Same `fill_diagonal_` crash (`RuntimeError: dimensions must larger than 1`) even after the single-atom fix.
**Root cause**: GearNet's sequential graph passes `atom_seq_pos` as a 1D tensor `[N]` to `radius_graph`, not 2D `[N, 3]`. When `x_b` is 1D, `cdist` misinterprets the dimensions: `[1, k]` is treated as 1 point in k-dimensional space, producing a `[1, 1]` distance matrix, which becomes 1D after squeeze.
**Fix**: Added `if x_b.dim() == 1: x_b = x_b.unsqueeze(-1)` to reshape 1D inputs to `[N, 1]` before cdist.
**Tests**: Added `test_1d_input` test case. Updated `test_gearnet_sequential_graph_pattern` to use truly 1D tensor. Total 25 tests, all passing.
**Commit**: `ab11607` (proteina submodule), `50db96a` (outer repo)

### 8. Repeated NVMe full on gpu-q-2
**Symptom**: `OSError: [Errno 28] No space left on device` — REPA job landed on gpu-q-2 twice.
**Fix**: Resubmitted with `--exclude=gpu-q-2` to avoid the full node.

---

## LMDB Conversion Progress
- **Job 26678723**: Healthy throughout, no parse failures.
- At ~00:40 BST: 142k / 579k (24.5%), ~8.8 it/s.
- Batch commits every 5,000 entries, SIGTERM handler for graceful shutdown.
- File growing steadily: 72MB -> 670MB -> 9.9GB.

---

## Current Status (as of 00:50 BST)
- **prot-lmd** (26678723): RUNNING, ~25% done, healthy
- **prot-bas** (26690247): RUNNING, GPU at 96% utilization, training (logs buffered)
- **prot-rep** (26692496): PENDING/RUNNING, resubmitted excluding gpu-q-2, monitoring

## Commits Pushed
- `e08ce72` — Update proteina: fix checkpoint resume and radius_graph crash
- `15bf7b5` — Update proteina submodule, add pyg_compat tests
- `50db96a` — Fix radius_graph 1D input, add test, update proteina submodule
