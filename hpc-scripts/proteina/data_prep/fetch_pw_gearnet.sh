#!/usr/bin/env bash
# Download a ProteinWorkshop GearNet-Edge checkpoint from Zenodo 8287754.
# (Jamasb et al., ICLR 2024 - MIT license)
#
# Uses streaming tar extraction to avoid materialising the full 5+ GB tarball.
# Only the ca_angles variant is supported (input_dim=43, CA-only features).
# ca_bb (input_dim=49) requires phi/psi/omega dihedrals unavailable in proteina.
#
# Five pretraining tasks available (pass TASK= to select):
#   structure_denoising   (recommended: most 3D-sensitive)
#   torsional_denoising   (smallest tarball, ~5 GB; ca_angles ckpt ~350 MB in)
#   inverse_folding
#   sequence_denoising
#   plddt_prediction
#
# Usage:
#   TASK=torsional_denoising DATA_PATH=/path/to/data bash fetch_pw_gearnet.sh
set -euo pipefail

: "${DATA_PATH:?DATA_PATH must be set (e.g. /rds/user/.../proteina/data)}"
: "${TASK:=torsional_denoising}"

VALID_TASKS="structure_denoising torsional_denoising inverse_folding sequence_denoising plddt_prediction"
if ! echo "$VALID_TASKS" | grep -qw "$TASK"; then
    echo "ERROR: Unknown TASK=$TASK. Must be one of: $VALID_TASKS" >&2
    exit 1
fi

DEST="${DATA_PATH}/metric_factory/model_weights/pw_gearnet_${TASK}_ca_angles.ckpt"
URL="https://zenodo.org/records/8287754/files/${TASK}.tar.gz?download=1"
# Internal tarball path (GearNet-Edge, ca_angles feature config, last checkpoint)
TAR_PATH="${TASK}/gear_net_edge/ca_angles/last.ckpt"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
    echo "Already exists: $DEST - skipping download."
    exit 0
fi

echo "Streaming ${TASK}.tar.gz from Zenodo 8287754 and extracting ${TAR_PATH} ..."
echo "(Downloads ~400 MB of the tarball stream to reach the ca_angles checkpoint)"
echo ""

# Stream the tarball and extract only the target checkpoint to stdout.
# -xzO: decompress, output matching file to stdout (avoids materialising tarball).
TMP="${DEST}.tmp"
wget -qO- "$URL" | tar -xzO "$TAR_PATH" > "$TMP"
mv "$TMP" "$DEST"

echo ""
echo "Done: $DEST ($(du -h "$DEST" | cut -f1))"
echo ""
echo "To use in a training config:"
echo "  repa:"
echo "    encoder:"
echo "      type: pw_gearnet"
echo "      gearnet_ckpt_path: \${DATA_PATH}/metric_factory/model_weights/pw_gearnet_${TASK}_ca_angles.ckpt"
