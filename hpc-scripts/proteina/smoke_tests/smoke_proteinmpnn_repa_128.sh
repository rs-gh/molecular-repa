#!/bin/bash
#!
#! Smoke test: ProteinMPNN CA-only REPA at n=128.
#! Thin wrapper around smoke_repa_128_param.sh — submits 200 steps eager
#! against the new MPNN config so we can verify checkpoint load + REPA
#! forward/backward + bs=80 fit + finite cos_sim before launching anything
#! longer.
#!
#! Usage: hpc-scripts/proteina/smoke_tests/smoke_proteinmpnn_repa_128.sh
#! (calls sbatch on smoke_repa_128_param.sh — do NOT sbatch this file.)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$HERE/smoke_repa_128_param.sh" \
  training_repa_l4_128_per_residue \
  training/128/mpnn
