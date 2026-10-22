#!/usr/bin/env bash
set -uo pipefail

root=/docsystem/niuke/lvyuehang/fm3d_ubuntu
output="$root/runs/fm3d_3M_run01_test_best106_20261022"
log="$root/logs/fm3d_3M_run01_test_best106_20261022.log"

cd "$root" || exit 1
if [[ -e "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
  printf 'Output already contains files: %s\n' "$output" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  /docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python -m fm3d.evaluate \
  --checkpoint "$root/runs/fm3d_3M_run01/best.pt" \
  --split test \
  --output "$output" \
  --device cuda \
  > "$log" 2>&1
rc=$?
printf 'TEST_EXIT_CODE=%s\n' "$rc" >> "$log"
exit "$rc"
