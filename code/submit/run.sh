#!/usr/bin/env bash

set -euo pipefail

# 提交入口：既支持正式 submission，也支持本地 validation 预测导出。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

MODE="${1:-submit}"

case "${MODE}" in
  submit)
    # 正式提交模式：读取 submit 融合结果，写出 submission.csv。
    ensure_dir "${EXP_SUBMISSION_DIR}"
    SUBMISSION_NAME="${2:-${SUBMISSION_NAME}}"
    EXP_SUBMISSION_PATH="${EXP_SUBMISSION_DIR}/${SUBMISSION_NAME}"

    run_python_step "submit" "code/submit/main.py" \
      --mode submit \
      --input-dir "${EXP_FUSION_SUBMIT_DIR}" \
      --model-dir "${EXP_MODEL_DIR}" \
      --output-file "${EXP_SUBMISSION_PATH}"

    if [[ -f "${EXP_SUBMISSION_PATH}" ]]; then
      echo "Generated: ${EXP_SUBMISSION_PATH}"
    else
      echo "[warn] submit output was not created: ${EXP_SUBMISSION_PATH}"
    fi
    ;;
  validation)
    # 本地验证模式：输出一份可交给 evaluator 打分的预测文件。
    ensure_dir "${EXP_RESULTS_DIR}"

    run_python_step "submit-validation" "code/submit/main.py" \
      --mode validation \
      --input-dir "${EXP_FUSION_DIR}" \
      --model-dir "${EXP_MODEL_DIR}" \
      --output-file "${EXP_VALIDATION_PREDICTIONS_PATH}"

    if [[ -f "${EXP_VALIDATION_PREDICTIONS_PATH}" ]]; then
      echo "Generated: ${EXP_VALIDATION_PREDICTIONS_PATH}"
    else
      echo "[warn] validation predictions were not created: ${EXP_VALIDATION_PREDICTIONS_PATH}"
    fi
    ;;
  *)
    echo "Usage: $0 [submit [submission_name]|validation]" >&2
    exit 1
    ;;
esac
