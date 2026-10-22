#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
PYTHON=/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python
"$PYTHON" test_flowoct_control.py > verification/server_flowoct_tests.txt 2>&1
"$PYTHON" flowoct_control.py --benchmark > 06_flowoct/benchmark.log 2>&1
"$PYTHON" flowoct_control.py --resume > logs/flowoct_control.log 2>&1
