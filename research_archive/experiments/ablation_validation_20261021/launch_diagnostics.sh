#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
PYTHON=/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python
until "$PYTHON" -c 'import json,pathlib; p=pathlib.Path("06_flowoct/status.json"); raise SystemExit(0 if p.exists() and json.loads(p.read_text()).get("status")=="complete" else 1)'; do
  tmux has-session -t lyh_flowoct_validation_20261021
  sleep 10
done
"$PYTHON" diagnose_comparison.py > logs/flowoct_diagnostics.log 2>&1
