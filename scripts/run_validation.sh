#!/usr/bin/env bash

set -euo pipefail

# 本地验证总控：先跑训练侧特征链路，再生成验证预测并计算离线指标。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

load_experiment_config "$@"

ensure_dir "${EXP_RECALL_DIR}"
ensure_dir "${EXP_FUSION_DIR}"
ensure_dir "${EXP_RANKER_DIR}"
ensure_dir "${EXP_MODEL_DIR}"
ensure_dir "${EXP_LOG_DIR}"
ensure_dir "${EXP_RESULTS_DIR}"

print_common_paths
echo "VALIDATION_PREDICTIONS=${EXP_VALIDATION_PREDICTIONS_PATH}"
echo "VALIDATION_METRICS=${EXP_VALIDATION_METRICS_PATH}"

# 验证链路沿用训练侧输入，最后额外补一个 evaluator。
if stage_enabled "${ENABLE_CONVISITATION}"; then
  run_shell_step "recall-convisitation" "code/recall/convisitation/run.sh" train
else
  echo "[skip] recall-convisitation: disabled by config"
fi

if stage_enabled "${ENABLE_NN}"; then
  run_shell_step "recall-nn" "code/recall/nn/run.sh" train
else
  echo "[skip] recall-nn: disabled by config"
fi

if stage_enabled "${ENABLE_FUSION}"; then
  run_shell_step "fusion" "code/fusion/run.sh" train
else
  echo "[skip] fusion: disabled by config"
fi

if stage_enabled "${ENABLE_RANKER}"; then
  run_shell_step "ranker" "code/ranker/run.sh"
else
  echo "[skip] ranker: disabled by config"
fi

run_shell_step "submit-validation" "code/submit/run.sh" validation

if [[ -f "${EXP_VALIDATION_PREDICTIONS_PATH}" ]]; then
  run_shell_step "validation-evaluator" "code/validation/run.sh"
else
  echo "[skip] validation-evaluator: missing predictions file ${EXP_VALIDATION_PREDICTIONS_PATH}"
  exit 0
fi

# 从 metrics.json 提取总分，写入实验汇总表。
VALIDATION_SCORE="$("${PROJECT_PYTHON}" - <<PY
from pathlib import Path
import json

metrics_path = Path("${EXP_VALIDATION_METRICS_PATH}")
if not metrics_path.exists():
    print("")
else:
    data = json.loads(metrics_path.read_text())
    print(data.get("overall_recall@20", ""))
PY
)"

record_experiment_result "validation" "completed" "${VALIDATION_SCORE}"

echo "Validation pipeline finished."
