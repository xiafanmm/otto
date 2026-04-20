#!/usr/bin/env bash

set -euo pipefail

# NN 召回入口：结构和共现召回保持一致，方便总控脚本串联。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

MODE="${1:-train}"

case "${MODE}" in
  train)
    # 训练/验证阶段输出到实验隔离目录，避免污染别的实验。
    ensure_dir "${EXP_RECALL_DIR}/nn"
    run_python_step "recall-nn" "code/recall/nn/main.py" \
      --mode train \
      --input-dir "${VALIDATION_DIR}" \
      --output-dir "${EXP_RECALL_DIR}/nn"
    ;;
  submit)
    # 提交阶段单独写入 nn_submit，便于和训练阶段产物区分。
    ensure_dir "${EXP_RECALL_DIR}/nn_submit"
    run_python_step "recall-nn" "code/recall/nn/main.py" \
      --mode submit \
      --input-dir "${PARQUET_DIR}" \
      --output-dir "${EXP_RECALL_DIR}/nn_submit"
    ;;
  *)
    echo "Usage: $0 [train|submit]" >&2
    exit 1
    ;;
esac
