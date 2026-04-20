#!/usr/bin/env bash

set -euo pipefail

# 提交总控：在测试数据上跑召回、融合和最终 submission 导出。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

load_experiment_config "$@"

ensure_dir "${EXP_RECALL_DIR}"
ensure_dir "${EXP_FUSION_SUBMIT_DIR}"
ensure_dir "${EXP_SUBMISSION_DIR}"

print_common_paths
echo "SUBMISSION_PATH=${EXP_SUBMISSION_PATH}"

# 提交阶段顺序固定为：召回 -> 融合 -> 生成 submission 文件。
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

run_shell_step "submit" "code/submit/run.sh" submit "${SUBMISSION_NAME}"

record_experiment_result "submit" "completed"

echo "Submission pipeline finished."
