#!/usr/bin/env bash
set -uo pipefail
root=/docsystem/niuke/lvyuehang/fm3d_zproj_96_200
cd "$root"
if [[ ! -f verification/preflight_passed.txt ]]; then
  echo "Preflight has not passed" >&2
  exit 2
fi
# Keep this shell after training ends so the exit status remains visible in tmux.
bash run_3M_200.sh "$@" 2>&1 | tee -a logs/train_3M_200.log
rc=${PIPESTATUS[0]}
printf 'TRAIN_EXIT_CODE=%s\n' "$rc" | tee -a logs/train_3M_200.log
printf '%s\n' "$rc" > verification/train_exit_code.txt
exec bash --noprofile --norc
