#!/usr/bin/env bash
set -euo pipefail
cd /docsystem/niuke/lvyuehang/fm3d_zproj_96_200
python_bin=/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$python_bin" -m zipfile -e fm3d_zproj96_source.zip .
bash -n fm3d/scripts/train_projection_200.sh
"$python_bin" -m unittest fm3d.test_core fm3d.test_wavelet fm3d.test_projection -v > logs/server_unit_tests.log 2>&1
tail -n 5 logs/server_unit_tests.log
"$python_bin" -m fm3d.benchmark --device cuda --patch 256 96 96 --lambda-projection 0.1 \
  --base-channels 16 --batch-size 1 --steps 3 --amp --output verification/server_3090_benchmark.json
"$python_bin" -m verification.verify_real_data
date --iso-8601=seconds > verification/preflight_passed.txt
