#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
PY=/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python
"$PY" verify_memory.py > logs/memory_equivalence.log 2>&1
"$PY" run256.py --prepare /docsystem/niuke/lvyuehang/OCTA500_common_v1 > logs/prepare.log 2>&1
for arm in 01_fm3d 02_fm3d_wavelet 03_fm3d_zproj 04_fm3d_wavelet_zproj 05_fm3d_wavelet_zproj_reconstruction 06_flowoct; do
    "$PY" run256.py --arm "$arm" --benchmark > "logs/benchmark_${arm}.log" 2>&1
done
