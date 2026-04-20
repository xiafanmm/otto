#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

load_experiment_config "$@"

ensure_dir "${EXP_RECALL_DIR}"
ensure_dir "${EXP_FUSION_SUBMIT_DIR}"
ensure_dir "${EXP_SUBMISSION_DIR}"

print_common_paths
echo "SUBMISSION_PATH=${EXP_SUBMISSION_PATH}"

if stage_enabled "${ENABLE_CONVISITATION}"; then
  run_shell_step "recall-convisitation" "code/recall/convisitation/run.sh" submit
else
  echo "[skip] recall-convisitation: disabled by config"
fi

if stage_enabled "${ENABLE_NN}"; then
  run_shell_step "recall-nn" "code/recall/nn/run.sh" submit
else
  echo "[skip] recall-nn: disabled by config"
fi

if stage_enabled "${ENABLE_FUSION}"; then
  run_shell_step "fusion" "code/fusion/run.sh" submit
else
  echo "[skip] fusion: disabled by config"
fi

run_shell_step "submit" "code/submit/run.sh" "${SUBMISSION_NAME}"

record_experiment_result "submit" "completed"

echo "Submission pipeline finished."
