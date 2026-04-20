#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

MODE="${1:-train}"

case "${MODE}" in
  train)
    ensure_dir "${EXP_FUSION_DIR}"
    run_python_step "fusion" "code/fusion/main.py" \
      --mode train \
      --input-dir "${EXP_RECALL_DIR}" \
      --output-dir "${EXP_FUSION_DIR}"
    ;;
  submit)
    ensure_dir "${EXP_FUSION_SUBMIT_DIR}"
    run_python_step "fusion" "code/fusion/main.py" \
      --mode submit \
      --input-dir "${EXP_RECALL_DIR}" \
      --output-dir "${EXP_FUSION_SUBMIT_DIR}"
    ;;
  *)
    echo "Usage: $0 [train|submit]" >&2
    exit 1
    ;;
esac
