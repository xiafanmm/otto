#!/usr/bin/env bash

set -euo pipefail

# 按实验名清理产物目录，只删除 processed / logs / results / models / submissions。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

EXP_NAME="${1:-}"

if [[ -z "${EXP_NAME}" ]]; then
  echo "Usage: $0 <exp_name>" >&2
  exit 1
fi

TARGETS=(
  "${PROCESSED_DIR}/${EXP_NAME}"
  "${LOG_DIR}/${EXP_NAME}"
  "${RESULTS_DIR}/${EXP_NAME}"
  "${MODEL_DIR}/${EXP_NAME}"
  "${SUBMISSIONS_DIR}/${EXP_NAME}"
)

echo "Cleaning experiment: ${EXP_NAME}"

for target in "${TARGETS[@]}"; do
  if [[ -e "${target}" ]]; then
    echo "  remove ${target}"
    rm -rf "${target}"
  else
    echo "  skip   ${target} (not found)"
  fi
done

echo "Done."
