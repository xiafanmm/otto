#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"
initialize_experiment_context

MODE="${1:-train}"

case "${MODE}" in
  train)
    ensure_dir "${EXP_RECALL_DIR}/convisitation"
    run_python_step "recall-convisitation" "code/recall/convisitation/main.py" \
      --mode train \
      --input-dir "${VALIDATION_DIR}" \
      --output-dir "${EXP_RECALL_DIR}/convisitation"
    ;;
  submit)
    ensure_dir "${EXP_RECALL_DIR}/convisitation_submit"
    run_python_step "recall-convisitation" "code/recall/convisitation/main.py" \
      --mode submit \
      --input-dir "${PARQUET_DIR}" \
      --output-dir "${EXP_RECALL_DIR}/convisitation_submit"
    ;;
  *)
    echo "Usage: $0 [train|submit]" >&2
    exit 1
    ;;
esac
