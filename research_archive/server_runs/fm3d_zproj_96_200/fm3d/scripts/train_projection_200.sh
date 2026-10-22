#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash fm3d/scripts/train_projection_200.sh MANIFEST {3M|6M} OUTPUT [extra training args]" >&2
  exit 2
fi
manifest=$1
fov=$2
output=$3
shift 3
if [[ ! -f fm3d/train_projection.py ]]; then
  echo "Run from the directory containing fm3d/." >&2
  exit 2
fi
exec "${PYTHON:-python}" -u -m fm3d.train_projection \
  --manifest "$manifest" --fov "$fov" --output "$output" \
  --axis-order ZYX --intensity-scale 255 --depth 256 \
  --patch 256 96 96 --base-channels 16 --condition --representation voxel \
  --lambda-projection 0.1 --batch-size 1 --epochs 200 --steps-per-epoch 200 \
  --lr 2e-4 --val-every 1 --ode-steps 8 --overlap 0.5 \
  --seed 2026 --workers 0 --device cuda --amp --save-every 5 \
  "$@"
