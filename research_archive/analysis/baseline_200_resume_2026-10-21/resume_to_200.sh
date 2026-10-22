#!/usr/bin/env bash
set -uo pipefail

root=/docsystem/niuke/lvyuehang/fm3d_ubuntu
log="$root/logs/fm3d_3M_run01_resume_101_200_val_each_epoch_20261021.log"
cd "$root"

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  /docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python -m fm3d.train \
  --manifest /docsystem/niuke/lvyuehang/OCTA500_common_v1/split_3M_z256.csv \
  --fov 3M \
  --output "$root/runs/fm3d_3M_run01" \
  --axis-order ZYX \
  --intensity-scale 255 \
  --depth 256 \
  --patch 256 64 64 \
  --base-channels 16 \
  --batch-size 1 \
  --epochs 200 \
  --window-best-every 10 \
  --steps-per-epoch 200 \
  --lr 0.0002 \
  --val-every 1 \
  --ode-steps 8 \
  --overlap 0.5 \
  --seed 2026 \
  --workers 0 \
  --device cuda \
  --amp \
  --resume > "$log" 2>&1
rc=$?
printf 'TRAIN_EXIT_CODE=%s\n' "$rc" >> "$log"
exit "$rc"
