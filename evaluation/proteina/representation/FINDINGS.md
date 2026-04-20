# Proteina Representation Quality Probes — Findings

**Status: code complete, execution blocked on Lustre.**

**Date:** 2026-04-18

## What this is

Probing-style evaluation for proteina, mirroring the approach in
[../../tabasco/probes/FINDINGS.md](../../tabasco/probes/FINDINGS.md). Two
probes, each targeting a distinct axis of representation quality:

- **P1 — long-range contact prediction (P@L/5)**
  MLP on pair features `[h_i ‖ h_j ‖ |h_i − h_j|]` → binary contact (CA-CA
  distance < 8 Å, `|i − j| ≥ 24`). Headline metric is top-L/5 precision per
  protein, averaged across the test set.

- **P2 — CATH fold classification**
  Linear classifier on mean-pooled per-residue reps → CATH topology
  (T-level). Falls back to A- or C-level if a subset has too few samples
  per class.

## Sources (when executed)

Each probe runs against five representation sources, held out the same way:

| Source | Type | Path |
|---|---|---|
| GearNet (frozen) | Encoder | `/rds/.../metric_factory/model_weights/gearnet_ca.pth` |
| baseline | Checkpoint | `store/proteina_60m_baseline_v2/checkpoints/last.ckpt` |
| repa_l0_per_residue | Checkpoint | `store/proteina_60m_repa_l0_256_per_residue/.../last.ckpt` |
| repa_l4_per_residue | Checkpoint | `store/proteina_60m_repa_l4_256_per_residue/.../last.ckpt` |
| repa_l9_per_residue | Checkpoint | `store/proteina_60m_repa_l9_256_per_residue/.../last.ckpt` |

All four checkpoints are present on `/rds/user/sr2173/hpc-work/proteina/store/`.
Hidden states are extracted at the matching layer for each REPA variant (0, 4, 9)
and at layer 4 for the no-REPA baseline.

## Current blocker

Lustre `/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb/` is returning
`Input/output error` / `Cannot send after transport endpoint shutdown` for
`val.lmdb`, `test.lmdb`, and `train.lmdb` at the time of this run. The probe
code is complete and smoke-tested (tabasco side runs end-to-end; proteina
model loading path verified against the `ProteinaREPA` codepath), but the full
proteina ablation cannot run from the login node while Lustre is in this state.

## How to run once Lustre recovers (or from a compute node)

A SLURM submission script lives at
[hpc-scripts/proteina/evaluation/run_probes.sh](../../../hpc-scripts/proteina/evaluation/run_probes.sh).
It copies `val.lmdb` (+ keys + length index) to `/tmp` on the compute node
before running, following the pattern in
`feedback_lmdb_local_nvme.md` — Lustre mmap thrashes on compute nodes, so
every probe read needs to hit local NVMe.

```bash
sbatch hpc-scripts/proteina/evaluation/run_probes.sh --n_proteins 200
```

Writes machine-readable results to `results.json` and a human-readable
table to `results.md` under this directory.

To debug the same code on the login node (if Lustre recovers):

```bash
source .venv/bin/activate
export PROJECT_ROOT=$(pwd)/src/proteina
export DATA_PATH=/rds/user/sr2173/hpc-work/proteina/data
export LMDB_DIR=/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb
python evaluation/proteina/representation/scripts/run_all.py --n_proteins 50 --only baseline
```

## Design notes

- **Clean-endpoint probing**: forward pass uses `x_t = x_1` (clean CA coords
  in nm) and `t = 1.0`. This matches the setting in which REPA itself is
  evaluated — you want to know what the student has learned about real
  structures, not about noise.
- **Hidden-state extraction for baseline**: the baseline uses plain
  `ProteinTransformerAF3`. We swap `model.nn.__class__` to
  `ProteinTransformerAF3WithHiddenStates` (a subclass) and set
  `model.nn.repa_layers = [L]`. The forward pass then captures layer-L
  output with zero weight changes.
- **CATH labels**: extracted from `graph.cath_code` (list of domain
  assignments per protein) and masked to T-level. If fewer than two classes
  have ≥5 samples, we fall back to A-level (class.arch), then C-level.
- **Contact head**: MLP rather than a single linear layer because pair
  features are high-dim (3D, where D = token_dim = 256 or 512) and a
  linear head struggles to separate long-range contacts from non-contacts
  at the top-L/5 threshold. Depth is 1 hidden layer (SiLU) — still in the
  "shallow probe" regime.

## Expected shape of the headline result

Based on the plan's hypotheses:

- **GearNet frozen** should set a clear P@L/5 upper bound on the structure
  side (GearNet is 3D-structure-aware, trained on pretext tasks) and a
  strong-but-not-saturating CATH accuracy (GearNet's per-residue reps encode
  topology but not fold explicitly).
- **Baseline vs REPA-trained** on P@L/5 is the clean REPA-paper-style
  ablation: if REPA transfers GearNet's structural signal, the REPA
  student should close some of the gap to the encoder.
- **Layer choice** (0/4/9) should show a U- or monotonic-shape depending on
  where the structural information is most useful — the per-layer
  comparison is the novel part of this eval beyond what the REPA paper
  did.
