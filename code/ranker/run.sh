#!/usr/bin/env bash

set -euo pipefail

# Ranker 入口：读取融合结果和验证标签，产出模型、日志和训练侧输出。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

ensure_dir "${EXP_RANKER_DIR}"
ensure_dir "${EXP_MODEL_DIR}"
ensure_dir "${EXP_LOG_DIR}"

# 这里默认是训练模式；如果后续需要预测模式，再单独扩参数。
run_python_step "ranker" "code/ranker/main.py" \
  --mode train \
  --input-dir "${EXP_FUSION_DIR}" \
  --labels "${VALIDATION_DIR}/test_labels.parquet" \
  --output-dir "${EXP_RANKER_DIR}" \
  --model-dir "${EXP_MODEL_DIR}" \
  --log-dir "${EXP_LOG_DIR}"
