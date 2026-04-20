#!/usr/bin/env bash

set -euo pipefail

# 融合入口：把多路召回结果合并成后续 ranker / submit 可用的候选集。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

MODE="${1:-train}"

case "${MODE}" in
  train)
    # 训练/验证阶段读取 recall 目录，输出到 fusion 目录。
    ensure_dir "${EXP_FUSION_DIR}"
    run_python_step "fusion" "code/fusion/main.py" \
      --mode train \
      --input-dir "${EXP_RECALL_DIR}" \
      --output-dir "${EXP_FUSION_DIR}"
    ;;
  submit)
    # 提交阶段单独落到 fusion/submit，避免覆盖训练侧融合结果。
    ensure_dir "${EXP_FUSION_SUBMIT_DIR}"
    run_python_step "fusion" "code/fusion/main.py" \
      --mode submit \
      --input-dir "${EXP_RECALL_DIR}" \
      --output-dir "${EXP_FUSION_SUBMIT_DIR}"
    ;;
  *)
    echo "Usage: $0 [train|submit]" >&2
    exit 1
    ;;
esac
