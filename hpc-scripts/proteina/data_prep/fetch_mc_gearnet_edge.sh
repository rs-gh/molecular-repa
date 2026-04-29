#!/usr/bin/env bash
# Download mc_gearnet_edge.pth from Zenodo 7593637 (Zhang et al., ICLR 2023).
# License: CC-BY-4.0
#
# Usage: DATA_PATH=/path/to/data bash fetch_mc_gearnet_edge.sh
set -euo pipefail

: "${DATA_PATH:?DATA_PATH must be set (e.g. /rds/user/.../proteina/data)}"

DEST="${DATA_PATH}/metric_factory/model_weights/mc_gearnet_edge.pth"
URL="https://zenodo.org/records/7593637/files/mc_gearnet_edge.pth?download=1"
EXPECTED_MD5="c35402108f14e43f20feb475918f9c26"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
    echo "Already exists: $DEST - skipping download."
else
    echo "Downloading mc_gearnet_edge.pth (~81 MB) ..."
    wget -q --show-progress -O "$DEST" "$URL"
fi

echo "Verifying MD5 ..."
echo "${EXPECTED_MD5}  ${DEST}" | md5sum -c -
echo "Done: $DEST"
