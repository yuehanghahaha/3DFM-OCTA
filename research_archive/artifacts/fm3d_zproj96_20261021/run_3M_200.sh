#!/usr/bin/env bash
set -euo pipefail
root=/docsystem/niuke/lvyuehang/fm3d_zproj_96_200
cd "$root"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHON=/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python
exec bash fm3d/scripts/train_projection_200.sh \
  /docsystem/niuke/lvyuehang/OCTA500_common_v1/split_3M_z256.csv 3M \
  "$root/runs/fm3d_zproj_3M_96_200" "$@"
