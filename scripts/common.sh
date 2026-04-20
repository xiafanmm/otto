#!/usr/bin/env bash

set -euo pipefail

# 公共脚本库：统一路径、实验配置解析和子步骤执行方式。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 项目级目录约定，供所有训练/验证/提交脚本复用。
DATA_DIR="${ROOT_DIR}/data"
DOWNLOAD_DIR="${DATA_DIR}/downloads"
RAW_DIR="${DATA_DIR}/raw"
VALIDATION_DIR="${RAW_DIR}/validation"
PARQUET_DIR="${RAW_DIR}/parquet_chunks"
PROCESSED_DIR="${DATA_DIR}/processed"
MODEL_DIR="${DATA_DIR}/models"
LOG_DIR="${DATA_DIR}/logs"
SUBMISSIONS_DIR="${DATA_DIR}/submissions"
RESULTS_DIR="${DATA_DIR}/results"
EXPERIMENTS_DIR="${ROOT_DIR}/experiments"

PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PROJECT_PYTHON}" ]]; then
  PROJECT_PYTHON="${PROJECT_PYTHON_FALLBACK:-python}"
fi

# 保证目录存在，避免各阶段自行重复 mkdir。
ensure_dir() {
  mkdir -p "$1"
}

# 把常见的 0/1、true/false 风格开关统一转成布尔判断。
to_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    0|false|FALSE|no|NO|off|OFF|"") return 1 ;;
    *)
      echo "Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}

stage_enabled() {
  to_bool "${1:-0}"
}

# 实验配置默认值：不传 config 时按 baseline 全流程运行。
set_experiment_defaults() {
  EXP_NAME="${EXP_NAME:-baseline}"
  ENABLE_CONVISITATION="${ENABLE_CONVISITATION:-1}"
  ENABLE_NN="${ENABLE_NN:-1}"
  ENABLE_FUSION="${ENABLE_FUSION:-1}"
  ENABLE_RANKER="${ENABLE_RANKER:-1}"
  RUN_NOTES="${RUN_NOTES:-}"
  SUBMISSION_NAME="${SUBMISSION_NAME:-submission.csv}"
  CONFIG_PATH="${CONFIG_PATH:-}"
}

# 基于实验名派生本次运行的所有输出目录，保证实验之间互不覆盖。
initialize_experiment_context() {
  set_experiment_defaults

  EXP_PROCESSED_DIR="${PROCESSED_DIR}/${EXP_NAME}"
  EXP_RECALL_DIR="${EXP_PROCESSED_DIR}/recall"
  EXP_FUSION_DIR="${EXP_PROCESSED_DIR}/fusion"
  EXP_FUSION_SUBMIT_DIR="${EXP_FUSION_DIR}/submit"
  EXP_RANKER_DIR="${EXP_PROCESSED_DIR}/ranker"
  EXP_MODEL_DIR="${MODEL_DIR}/${EXP_NAME}"
  EXP_LOG_DIR="${LOG_DIR}/${EXP_NAME}"
  EXP_SUBMISSION_DIR="${SUBMISSIONS_DIR}/${EXP_NAME}"
  EXP_SUBMISSION_PATH="${EXP_SUBMISSION_DIR}/${SUBMISSION_NAME}"
  EXP_RESULTS_DIR="${RESULTS_DIR}/${EXP_NAME}"
  EXP_VALIDATION_PREDICTIONS_PATH="${EXP_RESULTS_DIR}/validation_predictions.csv"
  EXP_VALIDATION_METRICS_PATH="${EXP_RESULTS_DIR}/validation_metrics.json"
  ABLATION_RESULTS_FILE="${RESULTS_DIR}/ablation.csv"

  export EXP_NAME
  export ENABLE_CONVISITATION
  export ENABLE_NN
  export ENABLE_FUSION
  export ENABLE_RANKER
  export RUN_NOTES
  export SUBMISSION_NAME
  export CONFIG_PATH
  export EXP_PROCESSED_DIR
  export EXP_RECALL_DIR
  export EXP_FUSION_DIR
  export EXP_FUSION_SUBMIT_DIR
  export EXP_RANKER_DIR
  export EXP_MODEL_DIR
  export EXP_LOG_DIR
  export EXP_SUBMISSION_DIR
  export EXP_SUBMISSION_PATH
  export EXP_RESULTS_DIR
  export EXP_VALIDATION_PREDICTIONS_PATH
  export EXP_VALIDATION_METRICS_PATH
  export ABLATION_RESULTS_FILE
}

# 读取命令行参数和 experiments/*.env，并初始化实验上下文。
load_experiment_config() {
  local argv=("$@")
  local config_provided=0
  local exp_name_provided=0

  set_experiment_defaults

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG_PATH="$2"
        config_provided=1
        shift 2
        ;;
      --exp-name)
        EXP_NAME="$2"
        exp_name_provided=1
        shift 2
        ;;
      --submission-name)
        SUBMISSION_NAME="$2"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 1
        ;;
    esac
  done

  if [[ -n "${CONFIG_PATH}" && "${CONFIG_PATH}" != /* ]]; then
    CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH}"
  fi

  if [[ -n "${CONFIG_PATH}" ]]; then
    if [[ ! -f "${CONFIG_PATH}" ]]; then
      echo "Missing config: ${CONFIG_PATH}" >&2
      exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    source "${CONFIG_PATH}"
    set +a
  fi

  set_experiment_defaults
  initialize_experiment_context

  if [[ "${config_provided}" -eq 0 && "${exp_name_provided}" -eq 0 ]]; then
    echo "[warn] no --config or --exp-name provided; using default experiment '${EXP_NAME}'"
  fi

  if [[ ${#argv[@]} -gt 0 ]]; then
    RUN_ARGS=("${argv[@]}")
  else
    RUN_ARGS=()
  fi
  export RUN_ARGS
}

# 结果汇总写 CSV，先简单清理换行和逗号，避免破坏表结构。
sanitize_csv_field() {
  local value="${1:-}"
  value="${value//$'\n'/ }"
  value="${value//,/;}"
  printf '%s' "${value}"
}

# 把 train / validation / submit 的运行结果记录到 ablation.csv。
record_experiment_result() {
  local pipeline="$1"
  local status="$2"
  local score="${3:-}"

  ensure_dir "${RESULTS_DIR}"

  if [[ ! -f "${ABLATION_RESULTS_FILE}" ]]; then
    cat > "${ABLATION_RESULTS_FILE}" <<EOF
timestamp,exp_name,pipeline,status,config_path,enable_convisitation,enable_nn,enable_fusion,enable_ranker,score,notes
EOF
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$(sanitize_csv_field "${EXP_NAME}")" \
    "$(sanitize_csv_field "${pipeline}")" \
    "$(sanitize_csv_field "${status}")" \
    "$(sanitize_csv_field "${CONFIG_PATH}")" \
    "$(sanitize_csv_field "${ENABLE_CONVISITATION}")" \
    "$(sanitize_csv_field "${ENABLE_NN}")" \
    "$(sanitize_csv_field "${ENABLE_FUSION}")" \
    "$(sanitize_csv_field "${ENABLE_RANKER}")" \
    "$(sanitize_csv_field "${score}")" \
    "$(sanitize_csv_field "${RUN_NOTES}")" \
    >> "${ABLATION_RESULTS_FILE}"
}

# 统一执行 Python 子步骤：自动补全项目根目录到 PYTHONPATH。
run_python_step() {
  local step_name="$1"
  local script_path="$2"
  shift 2

  local absolute_path="${ROOT_DIR}/${script_path}"
  local pythonpath="${ROOT_DIR}"
  if [[ ! -f "${absolute_path}" ]]; then
    echo "[skip] ${step_name}: missing ${script_path}"
    return 0
  fi

  if [[ -n "${PYTHONPATH:-}" ]]; then
    pythonpath="${ROOT_DIR}:${PYTHONPATH}"
  fi

  echo "[run] ${step_name}: ${script_path}"
  PYTHONPATH="${pythonpath}" "${PROJECT_PYTHON}" "${absolute_path}" "$@"
}

# 统一执行 shell 子步骤：缺文件时只提示 skip，方便渐进搭骨架。
run_shell_step() {
  local step_name="$1"
  local script_path="$2"
  shift 2

  local absolute_path="${ROOT_DIR}/${script_path}"
  if [[ ! -f "${absolute_path}" ]]; then
    echo "[skip] ${step_name}: missing ${script_path}"
    return 0
  fi

  echo "[run] ${step_name}: ${script_path}"
  bash "${absolute_path}" "$@"
}

# 打印关键路径和实验上下文，方便排查当前脚本到底在跑哪套实验。
print_common_paths() {
  cat <<EOF
ROOT_DIR=${ROOT_DIR}
VALIDATION_DIR=${VALIDATION_DIR}
PARQUET_DIR=${PARQUET_DIR}
PROCESSED_DIR=${PROCESSED_DIR}
MODEL_DIR=${MODEL_DIR}
LOG_DIR=${LOG_DIR}
SUBMISSIONS_DIR=${SUBMISSIONS_DIR}
RESULTS_DIR=${RESULTS_DIR}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR}
EXP_NAME=${EXP_NAME:-}
CONFIG_PATH=${CONFIG_PATH:-}
EXP_PROCESSED_DIR=${EXP_PROCESSED_DIR:-}
EXP_MODEL_DIR=${EXP_MODEL_DIR:-}
EXP_LOG_DIR=${EXP_LOG_DIR:-}
EXP_SUBMISSION_PATH=${EXP_SUBMISSION_PATH:-}
EXP_RESULTS_DIR=${EXP_RESULTS_DIR:-}
EXP_VALIDATION_PREDICTIONS_PATH=${EXP_VALIDATION_PREDICTIONS_PATH:-}
EXP_VALIDATION_METRICS_PATH=${EXP_VALIDATION_METRICS_PATH:-}
ENABLE_CONVISITATION=${ENABLE_CONVISITATION:-}
ENABLE_NN=${ENABLE_NN:-}
ENABLE_FUSION=${ENABLE_FUSION:-}
ENABLE_RANKER=${ENABLE_RANKER:-}
PYTHON=${PROJECT_PYTHON}
EOF
}
