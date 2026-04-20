#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

ensure_dir "${EXP_SUBMISSION_DIR}"

SUBMISSION_NAME="${1:-${SUBMISSION_NAME}}"
EXP_SUBMISSION_PATH="${EXP_SUBMISSION_DIR}/${SUBMISSION_NAME}"

run_python_step "submit" "code/submit/main.py" \
  --input-dir "${EXP_FUSION_SUBMIT_DIR}" \
  --model-dir "${EXP_MODEL_DIR}" \
  --output-file "${EXP_SUBMISSION_PATH}"

echo "Generated: ${EXP_SUBMISSION_PATH}"
