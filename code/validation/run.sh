#!/usr/bin/env bash

set -euo pipefail

# 本地评测入口：读取 validation 预测和标签，写出 metrics.json。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

ensure_dir "${EXP_RESULTS_DIR}"

# 评测阶段只允许读取预测文件和 validation 标签，不参与训练逻辑。
run_python_step "validation-evaluator" "code/validation/main.py" \
  --predictions-file "${EXP_VALIDATION_PREDICTIONS_PATH}" \
  --labels-file "${VALIDATION_DIR}/test_labels.parquet" \
  --metrics-output "${EXP_VALIDATION_METRICS_PATH}" \
  --exp-name "${EXP_NAME}"
