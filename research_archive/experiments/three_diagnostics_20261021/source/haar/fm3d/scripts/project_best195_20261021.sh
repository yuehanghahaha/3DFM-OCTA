#!/usr/bin/env bash
set -uo pipefail
cd /docsystem/niuke/lvyuehang/fm3d_ubuntu
PYTHONUNBUFFERED=1 /docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python -m fm3d.projection_report --input /docsystem/niuke/lvyuehang/fm3d_wavelet_200/runs/fm3d_haar3d_3M_200_test_best195_20261021 --assets /docsystem/niuke/lvyuehang/fm3d_ubuntu/runs/fm3d_projection_assets_test_20261020 --output /docsystem/niuke/lvyuehang/fm3d_wavelet_200/runs/fm3d_haar3d_3M_200_test_best195_20261021_projection_report > /docsystem/niuke/lvyuehang/fm3d_wavelet_200/logs/projection_best195_20261021.log 2>&1
rc=$?
printf 'PROJECTION_EXIT_CODE=%s\n' "$rc" >> /docsystem/niuke/lvyuehang/fm3d_wavelet_200/logs/projection_best195_20261021.log
exit "$rc"
