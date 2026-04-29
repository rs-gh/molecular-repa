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

## Data funnel — what n_proteins actually means

Passing `--n_proteins 200` loads 200 proteins from the LMDB, but the
reported metrics come from a **held-out subset**, not all 200. The funnel
differs per probe:

```
                       n_proteins=200 loaded from LMDB
                                |
              ┌─────────────────┴─────────────────┐
              │ Contact probe (P1)                 │ CATH probe (P2)
              │ 80/20 train/test split             │ drop unlabelled + rare classes
              │   160 train  →  probe head fits    │ then 75/25 stratified split
              │    40 test   →  P@L/5 reported     │   ~120 train  →  LR fits
              │    (filter: length ≥ 50)           │    ~40 test   →  acc/F1 reported
              │    → n_proteins_test ~35–40        │    → n_test in CATHResult
              └────────────────────────────────────┘
```

**Why hold out at all?** The probe head must not train on the proteins it
is evaluated on. Without a held-out split, a sufficiently expressive head
could memorise training pairs and report inflated metrics regardless of
representation quality.

**Is this standard?** Yes. Probing studies (Tenney et al. 2019, the REPA
paper, ProteinWorkshop) all use exactly this pattern — small pools, in-place
random splits, report on the held-out fraction. The comparison is relative
(REPA vs baseline), so the absolute numbers are secondary to consistency
across runs.

**The 80/20 vs 75/25 discrepancy** between the two probes is historical and
not intentional. Neither choice materially changes the interpretation.

**Implications for n_proteins choice:**
- At n=200: ~35–40 test proteins for contacts, ~40 for CATH → noisy but fine for relative comparisons during experimentation.
- For final reported numbers: use n≥1000 and multi-seed (`--seeds 42 43 44`) to get stable estimates with error bars.

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

## NGC pretrained 60M vs our 60M — architecture mismatch (2026-04-20)

Probed NVIDIA's released `proteina_v1.3_DFS_60M_notri.ckpt` (`pretrained_dfs_60m`
entry in `PRETRAINED_CHECKPOINTS`) to get a REPA-paper-style layer-wise curve
on a frozen, well-trained reference.

- **NGC ckpt**: `ProteinTransformerAF3` with **`nlayers=12`**, 58.93M params.
- **Our in-house 60M runs** (`baseline`, `repa_l0/l4/l9`): **`nlayers=10`**,
  fixed by `configs/experiment_config/model/nn/ca_af3_60M_notri.yaml:5`.
- The `nlayers: 10` yaml was inherited verbatim from NVIDIA's initial release
  (commit `a5a2ae6 Proteina`); we've only edited it to add `use_sdpa: True`
  (commit `ddb747d`). The depth is *not* a choice we made.
- **Upstream inconsistency**: NVIDIA's shipped yaml (10 layers) does not
  reproduce their shipped ckpt (12 layers). Two different 60M architectures
  ship under the same "60M" label. Param totals happen to land ~59M in both,
  so presumably a width trade was involved.

**Implications for REPA layer-search**:
- Absolute layer index is *not* comparable between the two models. When
  plotting jointly, use normalized depth (layer_idx / nlayers) so L0 of
  one aligns with L0 of the other and L9-of-10 ≈ L11-of-12.
- Our `repa_l9` variant sits at the second-to-last trunk block (9/10); the
  NGC analogue for "second-to-last" is L10/12, not L9/12.

**Headline layer curve on NGC 60M** (n=200 proteins ≤ 256 residues, P1 only;
P2/CATH is NaN — orthogonal pre-existing bug, 231/231 historical rows also
NaN):

| Layer | P@L/5 | P@L/2 | P@L |
|---:|---:|---:|---:|
| 0 | **0.958** | 0.879 | 0.708 |
| 1 | 0.933 | 0.844 | 0.661 |
| 2 | 0.924 | 0.822 | 0.649 |
| 3 | 0.914 | 0.810 | 0.636 |
| 4 | 0.895 | 0.792 | 0.623 |
| 5 | 0.901 | 0.795 | 0.620 |
| 6 | 0.897 | 0.790 | 0.615 |
| 7 | 0.870 | 0.773 | 0.603 |
| 8 | 0.900 | 0.794 | 0.622 |
| 9 | 0.883 | 0.774 | 0.596 |
| 10 | 0.907 | 0.791 | 0.613 |
| 11 | 0.848 | 0.732 | 0.560 |

Monotonic-ish decrease with depth, peak at L0 — opposite of the
SiT→DINOv2 curve in the REPA paper (which peaked at layer 20/24). The
structural/contact signal is strongest at the earliest trunk block and the
later layers trade it away for flow-matching-velocity specifics, which is
consistent with REPA's hypothesis about "later layers focus on
high-frequency details". Suggests the promising REPA-alignment depths for
our 10-layer student are **L0-L2**; our `repa_l0` variant is the most
aligned candidate.

Sample size is small (single 200-protein manifest). If the L0 peak needs
confirmation, rerun with one or two additional `--manifest_version v2`
seeds.

## L0-peak confirmed on our own 60M — no normalization needed (2026-04-20)

The NGC 60M → our 60M extrapolation is brittle (different depth, width, and
training), so we checked: does our own 60M baseline also peak at L0? Yes.

Layer-wise P@L/5 at final step (same probe config, same manifest):

| Layer | baseline (ours, 10L, step=740k) | repa_l4 (ours, step=840k) | NGC pretrained (12L) |
|---:|---:|---:|---:|
| 0 | **0.943** | **0.943** | **0.958** |
| 1 | 0.937 | 0.938 | 0.933 |
| 2 | 0.930 | 0.920 | 0.924 |
| 3 | 0.922 | 0.930 | 0.914 |
| 4 | 0.921 | 0.913 | 0.895 |
| 5 | 0.914 | 0.913 | 0.901 |
| 6 | 0.926 | 0.891 | 0.897 |
| 7 | 0.921 | 0.899 | 0.870 |
| 8 | 0.912 | 0.878 | 0.900 |
| 9 | 0.892 | 0.892 | 0.883 |
| 10 | — | — | 0.907 |
| 11 | — | — | 0.848 |

Observations:
- **L0 is the peak for our 60M as well.** Same qualitative shape as NGC
  — monotonic-ish decrease with depth, small bump mid-stack. So the
  REPA-target-layer recommendation (L0, maybe L1) is grounded in
  direct measurement on our architecture, not normalized-depth
  extrapolation from NGC.
- **Our curve is flatter** (range 0.051 vs NGC's 0.110). Possible
  explanations: undertraining at 740k steps, the width-for-depth trade
  at 10 vs 12 layers spreading structural info more uniformly, or
  different training-data distribution.
- **REPA at L4 did *not* move the L0 peak.** `repa_l4` trained with
  alignment at layer 4 still has peak P@L/5 at L0 (0.943, identical
  to baseline). Aligning at L4 during training does not relocate the
  structural information — it just adds pressure at that specific
  injection point. Useful reminder: "where REPA aligns" ≠ "where the
  student's structural peak ends up."

Missing rows: `repa_l0` and `repa_l9` last-step probes are not in the
consolidated jsonl. Worth a re-run to fill them in before finalizing the
layer-recommendation.

**Methodological note on normalized depth**: empirically the two curves
have the same qualitative shape, so normalized depth works as a rough
first-pass ordering *for this specific pair*. But it's not a principled
tool — residual-stream arguments (a block's role depends on absolute
prior blocks, not relative position), capacity-per-block differences
from the width-depth trade, and training differences all mean a
different-depth model can legitimately behave differently. Since we
have the direct measurement on our architecture, we don't need the
heuristic and shouldn't rely on it.

## Pretrained-probe pipeline (2026-04-23)

In-place-split probe has two structural weaknesses: (1) only ~40 eval
proteins per row after the 80/20 split of 200 val.lmdb proteins,
(2) train and test both sampled from val.lmdb so the evaluation is
informally "seen" by the probe itself. The REPA paper's probe (DAE
protocol) trains on the full ImageNet training set and evaluates on
ImageNet val — structurally disjoint pools, much larger probe-training
set.

New pipeline mirrors that protocol for proteins:

- Probe head trains on features extracted from a sample of `train.lmdb`
  (the 425K-protein PDB training split).
- Probe head evaluates on features extracted from a fixed manifest drawn
  from `val.lmdb`. Same manifest used across every (run, step, layer) row.
- Probe is retrained per `(run, step, layer)` — feature distributions
  differ per backbone state, so one global head would not generalise.
- Per-checkpoint feature cache: features for all layers extracted in one
  backbone forward pass, saved as fp16, then deleted after all layer
  probes complete. Bounded at ~5 GB per checkpoint.

Code layout:

| File | Purpose |
|---|---|
| [lib/probes/contact_pretrained.py](lib/probes/contact_pretrained.py) | `train_contact_probe`, `eval_contact_probe`, `run_pretrained_contact_probe` |
| [lib/feature_cache.py](lib/feature_cache.py) | Extract+cache+purge helpers, per-checkpoint tmp layout |
| [scripts/sample_size_probe.py](scripts/sample_size_probe.py) | Phase 1 — learning curve at N_train ∈ {500, 1K, 2K, 5K, 10K} |
| [scripts/pretrain_probe_sweep.py](scripts/pretrain_probe_sweep.py) | Phase 2 — full RUN_SCHEDULES × layers sweep |
| [sweep_config.yaml](sweep_config.yaml) `pretrained_probe` | Canonical N_train, N_eval, probe hyperparams |
| [../../../hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh](../../../hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh) | SLURM wrapper; stages train.lmdb (51 GB) + val.lmdb to /dev/shm |

Results live in `results/pretrained_probe/` — separate from the in-place
sweep so both regimes remain queryable. Rows carry `train_manifest` and
`eval_manifest` tags for reproducibility.

**Scope of this change**: contact probe only, PDB train only, val.lmdb
evaluation only. CATH probe keeps its in-place split. AFDB pretraining is
a separate phase (different pool, different homology considerations).

**How to run**:

1. **Phase 1** (once): pick N_train by running the sample-size learning curve.
   ```
   sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh --sample_size
   ```
   Inspect `results/pretrained_probe/sample_size_curve.png`, pick the elbow,
   update `sweep_config.yaml` `pretrained_probe.n_train`.

2. **Phase 2** (per reporting cycle): full sweep.
   ```
   sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
       --config pretrained_probe
   ```
   Resumes JSONL on preempt. Writes `pretrained_sweep_results.{jsonl,csv,json}`.

**Cross-checking against the in-place sweep**: pretrained-probe P@L/5
should be higher than in-place (more train data → stronger probe →
tighter upper bound on what's decodable). The *ordering* of REPA vs
baseline should be preserved; any reordering means the in-place split
was introducing sample variance rather than measuring representation
quality.
