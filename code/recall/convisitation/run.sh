#!/usr/bin/env bash

set -euo pipefail

# 共现召回入口：支持 train / submit 两种模式，并可切换权重版本。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

MODE="${1:-train}"
shift || true

WEIGHT_VERSION="${CONVIS_WEIGHT_VERSION:-base}"
CLICK_WEIGHT="${CONVIS_CLICK_WEIGHT:-}"
CART_WEIGHT="${CONVIS_CART_WEIGHT:-}"
ORDER_WEIGHT="${CONVIS_ORDER_WEIGHT:-}"

# 预设一组常用权重版本，便于做消融实验。
resolve_weight_preset() {
  case "$1" in
    base|default|v12|v16|v20)
      echo "1 3 6"
      ;;
    v13)
      echo "0 3 6"
      ;;
    v14)
      echo "1 9 1"
      ;;
    v15)
      echo "1 0 0"
      ;;
    v17)
      echo "1 9 6"
      ;;
    v18)
      echo "1 15 20"
      ;;
    v19)
      echo "1 9 1"
      ;;
    *)
      echo "Unknown convisitation weight preset: $1" >&2
      exit 1
      ;;
  esac
}

# 命令行参数优先级高于 env：可临时覆盖版本和单个权重值。
while [[ $# -gt 0 ]]; do
  case "$1" in
    --weight-version)
      WEIGHT_VERSION="$2"
      shift 2
      ;;
    --click-weight)
      CLICK_WEIGHT="$2"
      shift 2
      ;;
    --cart-weight)
      CART_WEIGHT="$2"
      shift 2
      ;;
    --order-weight)
      ORDER_WEIGHT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument for convisitation/run.sh: $1" >&2
      echo "Usage: $0 [train|submit] [--weight-version VERSION] [--click-weight N --cart-weight N --order-weight N]" >&2
      exit 1
      ;;
  esac
done

read -r PRESET_CLICK_WEIGHT PRESET_CART_WEIGHT PRESET_ORDER_WEIGHT <<< "$(resolve_weight_preset "${WEIGHT_VERSION}")"

CLICK_WEIGHT="${CLICK_WEIGHT:-${PRESET_CLICK_WEIGHT}}"
CART_WEIGHT="${CART_WEIGHT:-${PRESET_CART_WEIGHT}}"
ORDER_WEIGHT="${ORDER_WEIGHT:-${PRESET_ORDER_WEIGHT}}"

echo "[config] convisitation weights version=${WEIGHT_VERSION} clicks=${CLICK_WEIGHT} carts=${CART_WEIGHT} orders=${ORDER_WEIGHT}"

case "${MODE}" in
  train)
    # 训练/验证阶段读取 validation 数据，输出到当前实验的 recall 目录。
    ensure_dir "${EXP_RECALL_DIR}/convisitation"
    run_python_step "recall-convisitation" "code/recall/convisitation/main.py" \
      --mode train \
      --exp-name "${EXP_NAME}" \
      --input-dir "${VALIDATION_DIR}" \
      --output-dir "${EXP_RECALL_DIR}/convisitation" \
      --weight-version "${WEIGHT_VERSION}" \
      --click-weight "${CLICK_WEIGHT}" \
      --cart-weight "${CART_WEIGHT}" \
      --order-weight "${ORDER_WEIGHT}"
    ;;
  submit)
    # 提交阶段读取测试 parquet，输出到 submit 专用目录。
    ensure_dir "${EXP_RECALL_DIR}/convisitation_submit"
    run_python_step "recall-convisitation" "code/recall/convisitation/main.py" \
      --mode submit \
      --exp-name "${EXP_NAME}" \
      --input-dir "${PARQUET_DIR}" \
      --output-dir "${EXP_RECALL_DIR}/convisitation_submit" \
      --weight-version "${WEIGHT_VERSION}" \
      --click-weight "${CLICK_WEIGHT}" \
      --cart-weight "${CART_WEIGHT}" \
      --order-weight "${ORDER_WEIGHT}"
    ;;
  *)
    echo "Usage: $0 [train|submit] [--weight-version VERSION] [--click-weight N --cart-weight N --order-weight N]" >&2
    exit 1
    ;;
esac
