#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python long_controls.py --kind "$1" > "logs/long_${1}.log" 2>&1
