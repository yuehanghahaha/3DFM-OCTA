#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
PY=/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python
"$PY" -c 'import json,pathlib; r=pathlib.Path("verification"); names=["07_flowoct_fm_only","08_flowoct_wavelet_only"]; assert all(json.loads((r/("benchmark_"+n+".json")).read_text())["status"]=="passed" for n in names)'
"$PY" audit_flowoct_ablation.py --before > logs/flowoct_ablation_protect.log 2>&1
for arm in 08_flowoct_wavelet_only 07_flowoct_fm_only; do
    "$PY" run_flowoct_ablation.py --arm "$arm" > "logs/train_${arm}.log" 2>&1
done
"$PY" report_flowoct_ablation.py > logs/report_flowoct_ablation.log 2>&1
"$PY" report256.py > logs/report_with_flowoct_ablation.log 2>&1
"$PY" audit_flowoct_ablation.py > logs/audit_flowoct_ablation.log 2>&1
"$PY" package_flowoct_ablation.py > logs/package_flowoct_ablation.log 2>&1
