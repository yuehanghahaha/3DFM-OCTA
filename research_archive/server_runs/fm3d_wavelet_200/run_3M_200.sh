#!/usr/bin/env bash
set -euo pipefail
cd /docsystem/niuke/lvyuehang/fm3d_wavelet_200
export PYTHON=/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec bash fm3d/scripts/train_wavelet_200.sh   /docsystem/niuke/lvyuehang/OCTA500_common_v1/split_3M_z256.csv   3M /docsystem/niuke/lvyuehang/fm3d_wavelet_200/runs/fm3d_haar3d_3M_200   "$@"
