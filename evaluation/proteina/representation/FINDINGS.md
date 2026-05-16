# Proteina Representation Quality Probes — Findings

**Status:** execution complete; results live under `results/{lite,convergence,paper}/`.
This document captures methodology + interpretation; for invocation see
[README.md](README.md).

**First run:** 2026-04-18. **Findings document last updated:** 2026-05-11.

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
  (T-level by default; the paper-table sweeps also probe C and A levels).
  Falls back to A- or C-level if a subset has too few samples per class.

## Sources

Two driver scripts iterate over a shared registry of representation sources
defined in [lib/checkpoints.py](lib/checkpoints.py) (`RUN_SCHEDULES` for the
in-place sweep, `PRETRAINED_CHECKPOINTS` for the pretrained-probe sweep).

Categories present in the registry:
- **Frozen GearNet** (Encoder; reference ceiling on structural decodability).
- **Pretrained NVIDIA NGC 60M** (`proteina_v1.3_DFS_60M_notri.ckpt`; 12-layer
  reference for layer-curve shape).
- **Our in-house 60M baselines and REPA variants** (10-layer trunk), trained at
  n=128, 256, 512 with different target encoders (CA-GearNet, ESM2, ProteinMPNN,
  PW-Structure, PW-Torsional, random init) and different alignment layers
  (L0/L4/L9) — ~17 runs in total. See `RUN_SCHEDULES` for the live list; this
  document does not enumerate them since the registry changes faster than the
  doc would.

Hidden states are extracted at every trunk layer in a single forward pass
([lib/extract.py](lib/extract.py)); the probe head is fit per layer.

## How to run

Full invocation patterns + SLURM wrappers are in [README.md](README.md).
TL;DR:

```bash
# Pipeline A (in-place split, fast triage)
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh \
    --sweep --config n128

# Pipeline B (pretrained probe, paper-quality)
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
    --config pretrained_probe

# Local smoke (no SLURM, ~2 min on a GPU node)
python evaluation/proteina/representation/scripts/lite/run_sweep.py \
    --config n128 --runs baseline_128 --n_proteins 20
```

Both wrappers stage the relevant LMDB(s) to local NVMe before running —
Lustre mmap thrashes on compute nodes (see `feedback_lmdb_local_nvme.md`).

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
| [scripts/paper/sample_size_probe.py](scripts/paper/sample_size_probe.py) | Phase 1 — learning curve at N_train ∈ {500, 1K, 2K, 5K, 10K} |
| [scripts/paper/pretrain_probe_sweep.py](scripts/paper/pretrain_probe_sweep.py) | Phase 2 — full RUN_SCHEDULES × layers sweep |
| [sweep_config.yaml](sweep_config.yaml) `pretrained_probe` | Canonical N_train, N_eval, probe hyperparams |
| [../../../hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh](../../../hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh) | SLURM wrapper; stages train.lmdb (51 GB) + val.lmdb to /dev/shm |

Results live in `results/paper/contact_max256/` — separate from the in-place
sweep so both regimes remain queryable. Rows carry `train_manifest` and
`eval_manifest` tags for reproducibility.

**Scope of this change (2026-04-23)**: contact probe only, PDB train only,
val.lmdb evaluation only. AFDB pretraining is a separate phase (different
pool, different homology considerations).

**Update (2026-04-28)**: CATH probe was added to the pretrained-split
pipeline via the `paper_n128_cath` / `paper_n256_cath` profiles. Rows are
tagged `probe_kind="cath"` in the JSONL and carry `cath_accuracy` /
`cath_macro_f1` at the C/A/T levels. The original Pipeline-A in-place CATH
probe still exists (run via `--config n{128,256,512}`) but the paper
numbers come from the pretrained-split protocol.

**How to run**:

1. **Phase 1** (once): pick N_train by running the sample-size learning curve.
   ```
   sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh --sample_size
   ```
   Inspect `results/paper/contact_max256/sample_size_curve.png`, pick the elbow,
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

## Residue/pair-level structural probes added 2026-05-14

Two long-standing weaknesses in the existing pair (P1=contact, P2=CATH):

1. **Contact P@L/k** is a pair-ranking metric on a binary thresholding of
   distance at 8 Å. A sequence-position-only feature can do non-trivially
   well; the threshold throws away geometric detail.
2. **CATH-T (mean-pool)** predicts a per-chain label from a single pooled
   feature — a global descriptor like radius-of-gyration + length already
   gets non-trivial macro-F1, so it tells us little about whether `h_i`
   encodes residue-local structure.

Three new probes target *residue-* and *pair-level* structural quality
that the existing pair don't measure, each with a task-specific
trivial-geometric baseline that establishes a hard floor:

- **P3 — Inverse folding** ([lib/probes/inverse_folding.py](lib/probes/inverse_folding.py)).
  Per-residue 20-way amino-acid classification from `h_i ∈ R^D`. Direct
  REPA-style residue-level analogue of an ImageNet linear probe — if the
  hidden states encode the local chemical environment that determines the
  AA, a linear head should recover it. Reports top-1 accuracy and macro-F1.
  Trivial-geometric baseline: `knn_dist` (sorted distances to 8 nearest CAs;
  isolates "what does the local geometry alone tell you about AA identity").

- **P4 — Backbone dihedral regression** ([lib/probes/dihedral.py](lib/probes/dihedral.py)).
  Per-residue (φ, ψ) regression. The most direct test of whether the
  diffusion model's hidden states encode the local backbone geometry it is
  trained to denoise. Targets parameterised as (sin φ, cos φ, sin ψ, cos ψ)
  so MSE handles the angular wrap-around; reports mean *circular* angular
  error in degrees. Trivial-geometric baseline: `local_frame` (5 backbone
  atoms in a per-residue local frame — analytical lower bound, since a
  sufficiently expressive head can recover the angles essentially exactly
  from these atoms).

- **P5 — Pair distance regression** ([lib/probes/distance.py](lib/probes/distance.py)).
  Per-pair Cα-Cα distance regression in Å from `[h_i ‖ h_j ‖ |h_i - h_j|]`.
  Strict generalisation of the contact probe — contact is the same pair
  feature thresholded at 8 Å. Reports MAE bucketed by sequence separation
  (short < 6, medium 6-24, long ≥ 24). Trivial-geometric baseline:
  `seqsep_pair` (head fed only `|i-j|`; tests how much of the MAE is just
  chain-prior).

For each new probe we also run the existing generic baselines — `random_gauss`
(memorisation-floor for the head), `seq_onehot` (only-chemistry baseline),
`untrained_proteina` (architectural prior, no learning), `trained_noise`
(weights help but input is uninformative). The `BASELINE_PROBE_KINDS` map
in `pretrain_probe_sweep.py` gates which (baseline × probe) cells are run.

### Citations

These probes adapt established benchmark tasks; cite as:

- **Inverse folding**: Ingraham et al., *"Generative models for graph-based
  protein design,"* NeurIPS 2019 (formulation, CATH-topology splits);
  Jamasb et al., *"Evaluating Representation Learning on the Protein
  Structure Universe,"* ICLR 2024 (ProteinWorkshop §D.8 — adopts CATH-IF
  as a downstream node-level evaluation task).
- **Distance regression**: Zhang et al., *"Protein Representation Learning by
  Geometric Structure Pretraining"* (GearNet), ICLR 2023, §3.2 (proposes
  masked distance/angle/dihedral SSL); adopted in ProteinWorkshop §D.7.
- **Backbone dihedral regression**: Jamasb et al., ICLR 2024, §D.7
  (ProteinWorkshop adds per-residue backbone dihedral prediction beyond
  GearNet's quadruplet variant — no external attribution given).

### How to run

Two paper-style profiles in [sweep_config.yaml](sweep_config.yaml) mirror the
`paper_n{128,256}_cath` sample sizes (n_train=5000, n_eval≈val-clipped, t=1.0
clean only, linear head, 15 epochs) so the new probe rows are directly
comparable in scale to the existing CATH rows on the same checkpoints:

```bash
# n=128 paper-table sweep (22 ckpts)
python evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py \
    --config paper_n128_struct

# n=256 paper-table sweep (18 ckpts)
python evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py \
    --config paper_n256_struct
```

Both profiles auto-include the trivial-geometric baselines (`knn_dist`,
`local_frame`, `seqsep_pair`) and the generic baselines (`random_gauss`,
`seq_onehot`, `untrained_proteina`, `trained_noise`); `BASELINE_PROBE_KINDS`
in `pretrain_probe_sweep.py` gates which baseline applies to which probe.

Quick smoke (no SLURM, single ckpt, single layer, 200/100 sample sizes):

```bash
python evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py \
    --config paper_n256_struct \
    --runs baseline_256_ep21 --steps 200000 \
    --n_train 200 --n_eval 100 \
    --output_dir results/smoke/three_probes
```

Linear head only for v1 — sharper ranking signal between checkpoints. The
`_build_head` factory (`lib/probes/contact.py`) supports MLP via
`--head_type mlp` if linear underseparates.

## CATH baseline interpretation — what the floor actually is (2026-05-14)

The paper-CATH sweeps ship four rep-source baselines: `random_gauss`,
`seq_onehot`, `untrained_proteina`, and `trained_noise`. Their meaning is
easy to misread; in particular the "random" baselines do **not** score at
`1/K` chance. This section captures the empirically-measured floor and
what each baseline isolates.

### The right floor is the *intercept-only* floor, not 1/K

A linear LogReg with no informative features still has a per-class intercept
term. It learns the marginal `P(class)` from the train labels and predicts
the argmax of that prior — i.e. it always predicts the **train majority
class**. Its accuracy on eval is therefore the **eval prevalence of the
train-majority class**, not `1/K`.

CATH is heavily imbalanced (alpha-beta dominates C, Rossmann-like 3.40
dominates A, immunoglobulin-like 2.60.40 / 3.40.50 dominate T), so the
intercept floor is much higher than naive `1/K`.

Profiled directly from the manifests at `results/paper/n{128,256}_paper_cath/cath/`
(via `lmdb.open` + `pickle.loads` over the val.lmdb keys; see
`results/paper/n128_paper_cath/cath/batch_manifest_*.json`):

| level | n128 K_invocab | n128 floor | n128 maj class | n256 K_invocab | n256 floor | n256 maj class |
|-------|----------------|------------|----------------|----------------|------------|----------------|
| C     | 5              | **0.333**  | 2 (mostly-beta)| 5              | **0.484**  | 3 (alpha-beta) |
| A     | 21             | **0.169**  | 3.30           | 24             | **0.176**  | 3.40           |
| T     | 116            | **0.123**  | 2.60.40        | 128            | **0.112**  | 3.40.50        |

Notes on the funnel: of 1237 n128 eval proteins, 786 are *unlabelled*
(no `cath_code`) and drop out of both numerator and denominator entirely;
the in-vocab denominators above (451, 451, 405) reflect this. The
n256 split has 3190 evals → 1811 unlabelled, 1379/1371/1168 in-vocab.
`cath_min_per_class=3` removes T-classes with <3 train examples (46 / 211
OOV eval drops at n128 / n256).

### What each baseline actually measures

| baseline | features | what it isolates |
|----------|----------|------------------|
| `random_gauss` | `torch.randn(B,N,512)` masked, mean-pooled to `[B,512]` | The LogReg's response to features with zero signal. Mean-pool of i.i.d. noise leaks one bit: `‖μ‖² ∝ 1/L` (length signal through norm). |
| `seq_onehot`   | 20-dim per-residue one-hot of `residue_type`, mean-pooled → AA composition vector | How much CATH structure is in AA composition alone (Chou-Fasman floor). |
| `untrained_proteina` | freshly-initialised 60M Proteina, 10 trunk layers, real coords + seq | The architecture prior alone — what an SE(3) attention stack with random weights extracts from a real backbone. |
| `trained_noise` | trained baseline ckpt (n128: `baseline_128_bs80_step200k`, n256: `baseline_256_ep21`), `x_t` drawn from the model's reference distribution at `t=0.0`, real residue_type/mask | The model's learned structural priors independent of the input being a real protein — "good model + uninformative coords". |

### Measured numbers vs floor

| level | floor | random_gauss | seq_onehot | untrained_proteina (best L) |
|-------|-------|--------------|------------|-----------------------------|
| n128 C | 0.333 | 0.326 | 0.545 | 0.432 |
| n128 A | 0.169 | 0.129 | 0.293 | 0.282 |
| n128 T | 0.123 | 0.057 | 0.126 | 0.217 |
| n256 C | 0.484 | 0.410 | 0.563 | 0.524 |
| n256 A | 0.176 | 0.150 | 0.282 | 0.292 |
| n256 T | 0.112 | 0.098 | 0.165 | 0.226 |

Source: `figures/paper/n{128,256}_paper_cath/cath/table_baselines.csv`.
`trained_noise` numbers will be added when the in-flight backfill jobs
(`29348110` n128 + waiter-scheduled n256) finish.

### What this tells us

1. **`random_gauss` sits at or slightly *below* the intercept-only floor
   everywhere.** It dips below because the fitted LogReg learns small
   spurious weights on the noise that can't help at eval (fresh noise) but
   *can* perturb the per-sample argmax away from the majority class. So
   it's a legitimate "no information" sanity check — slightly worse than a
   pure intercept predictor would be.

2. **`seq_onehot` adds real signal on C and A, but ≈ at floor on T.**
   Composition captures alpha/beta/mixed structure cleanly (n128 C: +21pts
   over floor) but cannot disambiguate 116 fine-grained topologies that
   share AA-composition statistics.

3. **`untrained_proteina` adds modest signal on C, *substantial* signal on
   T.** On n128 T the best layer sits +9pts above floor with non-trivial
   macro-F1 (0.092 vs 0.007 for random_gauss). The SE(3) architecture
   prior alone is doing real structural work at the hardest level,
   independent of training.

### `cath_macro_f1` is the discriminating column

Accuracy can be inflated by always predicting the majority class. The ratio
`macro_f1 / accuracy` reveals whether a baseline is genuinely classifying
multiple classes or just collapsing to the prior:

| | T-accuracy | T-macro_f1 | ratio |
|---|---|---|---|
| n128 random_gauss | 0.057 | 0.007 | 0.12 — degenerate to majority |
| n128 untrained_proteina (best L) | 0.217 | 0.092 | 0.42 — classifying many |
| n128 seq_onehot | 0.126 | 0.004 | 0.03 — degenerate |
| n128 baseline_128_bs80_step200k (best L, real ckpt) | 0.449 | 0.269 | 0.60 — strong multi-class |

For interpretation in paper figures, watch macro-F1 to confirm "this row
is actually doing classification" rather than "this row is matching the
intercept floor."

### How to reproduce the floor measurement

```bash
.venv/bin/python -c "
import json, lmdb, pickle
from collections import Counter
# Load eval manifest keys + train manifest keys, open val.lmdb / train.lmdb
# at /rds/.../pdb_train/lmdb/ with subdir=False (single-file LMDB), pickle.loads
# each entry, extract g.cath_code, mask to C/A/T levels, tally Counter, intersect
# with cath_min_per_class=3 train-vocab filter, divide.
"
```

Full one-shot script lives in this session's transcript; collapse into
`scripts/paper/profile_cath_priors.py` if it gets re-run more than once.

### Caveats and follow-ups

- The intercept floor is the **train**-majority class evaluated on **eval**
  prevalence. If train/eval distributions diverge, this differs from
  "eval-majority class" prevalence. In practice for our manifests they
  match (same majority class, same prevalence to 3 decimals) because train
  and eval are i.i.d. samples from val.lmdb post-filtering.
- A `majority_only` analytic baseline that ignores features and always
  predicts the train majority would pin the floor exactly (no LogReg
  fitting noise). Adding it is ~15 lines in `lib/sources.py` and would
  let us *measure* the random_gauss dip-below-floor effect explicitly.
- A `length_only` baseline (1-dim `log(L)` feature) would quantify the
  length-leak in `random_gauss` directly. Composition-vs-length isolation.
