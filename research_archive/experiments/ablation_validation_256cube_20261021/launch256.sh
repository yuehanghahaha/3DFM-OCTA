#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
PY=/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python
"$PY" -c 'import pathlib,json; p=pathlib.Path("verification"); rows=[json.loads(x.read_text()) for x in p.glob("benchmark_*.json")]; assert len(rows)==6 and all(x["status"]=="passed" and any(y["applied"] for y in x["updates"]) for x in rows), rows'
for arm in 02_fm3d_wavelet 04_fm3d_wavelet_zproj 05_fm3d_wavelet_zproj_reconstruction 06_flowoct 01_fm3d 03_fm3d_zproj; do
    "$PY" run256.py --arm "$arm" > "logs/train_${arm}.log" 2>&1
    "$PY" report256.py > logs/report.log 2>&1
done
"$PY" export256.py > logs/export.log 2>&1
