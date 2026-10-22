#!/usr/bin/env bash
set -uo pipefail
cd /docsystem/niuke/lvyuehang/fm3d_wavelet_200
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python -m fm3d.evaluate --checkpoint /docsystem/niuke/lvyuehang/fm3d_wavelet_200/runs/fm3d_haar3d_3M_200/best.pt --split test --output /docsystem/niuke/lvyuehang/fm3d_wavelet_200/runs/fm3d_haar3d_3M_200_test_best195_20261021 --ode-steps 8 > /docsystem/niuke/lvyuehang/fm3d_wavelet_200/logs/test_best195_20261021.log 2>&1
rc=$?
printf 'EVAL_EXIT_CODE=%s\n' "$rc" >> /docsystem/niuke/lvyuehang/fm3d_wavelet_200/logs/test_best195_20261021.log
exit "$rc"
