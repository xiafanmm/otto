#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
DOWNLOAD_DIR="${DATA_DIR}/downloads"
RAW_DIR="${DATA_DIR}/raw"
VALIDATION_DIR="${RAW_DIR}/validation"
PARQUET_DIR="${RAW_DIR}/parquet_chunks"

VALIDATION_DATASET="cdeotte/otto-validation"
VALIDATION_ARCHIVE="otto-validation.zip"
PARQUET_DATASET="columbia2131/otto-chunk-data-inparquet-format"
PARQUET_ARCHIVE="otto-chunk-data-inparquet-format.zip"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

download_dataset() {
  local dataset="$1"
  local archive_name="$2"
  local destination="$3"

  mkdir -p "${DOWNLOAD_DIR}"
  rm -f "${DOWNLOAD_DIR}/${archive_name}"

  echo "Downloading ${dataset} ..."
  kaggle datasets download -d "${dataset}" -p "${DOWNLOAD_DIR}"

  rm -rf "${destination}"
  mkdir -p "${destination}"

  echo "Extracting ${archive_name} to ${destination} ..."
  unzip -q "${DOWNLOAD_DIR}/${archive_name}" -d "${destination}"
}

main() {
  require_command kaggle
  require_command unzip

  mkdir -p "${DATA_DIR}" "${RAW_DIR}"

  download_dataset "${VALIDATION_DATASET}" "${VALIDATION_ARCHIVE}" "${VALIDATION_DIR}"
  download_dataset "${PARQUET_DATASET}" "${PARQUET_ARCHIVE}" "${PARQUET_DIR}"

  echo "Data prepared under:"
  echo "  ${VALIDATION_DIR}"
  echo "  ${PARQUET_DIR}"
}

main "$@"
