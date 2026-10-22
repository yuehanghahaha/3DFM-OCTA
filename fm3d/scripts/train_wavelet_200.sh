#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash fm3d/scripts/train_wavelet_200.sh MANIFEST {3M|6M} OUTPUT [extra training args]" >&2
  exit 2
fi
manifest=$1
fov=$2
output=$3
shift 3
if [[ "$fov" != "3M" && "$fov" != "6M" ]]; then
  echo "FOV must be 3M or 6M" >&2
  exit 2
fi

# Call from the code root, so relative dataset/output paths have their usual meaning.
if [[ ! -f fm3d/train_wavelet.py ]]; then
  echo "Run from the directory containing fm3d/." >&2
  exit 2
fi
exec "${PYTHON:-python}" -u -m fm3d.train_wavelet \
  --manifest "$manifest" --fov "$fov" --output "$output" \
  --axis-order ZYX --intensity-scale 255 --depth 256 \
  --patch 256 64 64 --base-channels 16 --condition \
  --batch-size 1 --epochs 200 --steps-per-epoch 200 \
  --lr 2e-4 --val-every 5 --ode-steps 8 --overlap 0.5 \
  --seed 2026 --workers 0 --device cuda --amp --save-every 5 \
  "$@"
