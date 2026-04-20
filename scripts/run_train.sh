#!/usr/bin/env bash

set -euo pipefail

# 训练总控：按 config 决定依次跑召回、融合和 ranker。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

load_experiment_config "$@"

ensure_dir "${EXP_RECALL_DIR}"
ensure_dir "${EXP_FUSION_DIR}"
ensure_dir "${EXP_RANKER_DIR}"
ensure_dir "${EXP_MODEL_DIR}"
ensure_dir "${EXP_LOG_DIR}"

print_common_paths

# 训练阶段顺序固定为：共现召回 -> NN 召回 -> 融合 -> 排序模型。
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

record_experiment_result "train" "completed"

echo "Training pipeline finished."
