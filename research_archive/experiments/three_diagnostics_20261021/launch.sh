#!/usr/bin/env bash
set -euo pipefail
cd /docsystem/niuke/lvyuehang/three_diagnostics_20261021
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python -u run_diagnostics.py --phase run > run.log 2>&1
