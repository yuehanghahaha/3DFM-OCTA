#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python}"
mkdir -p logs verification
"$PYTHON" test_experiments.py > verification/server_tests.txt 2>&1
"$PYTHON" run.py --benchmark > verification/benchmark.log 2>&1
"$PYTHON" run.py --stage one_case --resume 2>&1 | tee logs/one_case.log
"$PYTHON" report.py
"$PYTHON" run.py --stage few_case --resume 2>&1 | tee logs/few_case.log
"$PYTHON" report.py
