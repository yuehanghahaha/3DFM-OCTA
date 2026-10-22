#!/usr/bin/env bash
set -euo pipefail
cd /docsystem/niuke/lvyuehang/fm_start_control_20261022
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python -u run_control.py --phase run > run.log 2>&1
