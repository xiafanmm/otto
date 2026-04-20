#!/usr/bin/env bash

set -euo pipefail

# 数据准备脚本：优先解压本地已有 zip，没有再走 Kaggle 下载。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

VALIDATION_DATASET="cdeotte/otto-validation"
VALIDATION_ARCHIVE="otto-validation.zip"
PARQUET_DATASET="columbia2131/otto-chunk-data-inparquet-format"
PARQUET_ARCHIVE="otto-chunk-data-inparquet-format.zip"

# 检查外部命令是否可用，缺少时直接失败。
require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

# 按统一目录规范解压数据，解压前会清空旧目录避免脏数据。
extract_archive() {
  local archive_name="$1"
  local destination="$2"

  if [[ ! -f "${DOWNLOAD_DIR}/${archive_name}" ]]; then
    echo "Missing archive: ${DOWNLOAD_DIR}/${archive_name}" >&2
    exit 1
  fi

  rm -rf "${destination}"
  mkdir -p "${destination}"

  echo "Extracting ${archive_name} to ${destination} ..."
  unzip -q "${DOWNLOAD_DIR}/${archive_name}" -d "${destination}"
}

# 仅负责下载 zip，不负责解压。
download_dataset() {
  local dataset="$1"
  local archive_name="$2"

  mkdir -p "${DOWNLOAD_DIR}"
  rm -f "${DOWNLOAD_DIR}/${archive_name}"

  echo "Downloading ${dataset} ..."
  kaggle datasets download -d "${dataset}" -p "${DOWNLOAD_DIR}"
}

# 主流程：准备目录 -> 按需下载 -> 解压到 validation / parquet_chunks。
main() {
  require_command unzip

  mkdir -p "${DATA_DIR}" "${DOWNLOAD_DIR}" "${RAW_DIR}"

  if [[ ! -f "${DOWNLOAD_DIR}/${VALIDATION_ARCHIVE}" || ! -f "${DOWNLOAD_DIR}/${PARQUET_ARCHIVE}" ]]; then
    require_command kaggle
    [[ -f "${DOWNLOAD_DIR}/${VALIDATION_ARCHIVE}" ]] || download_dataset "${VALIDATION_DATASET}" "${VALIDATION_ARCHIVE}"
    [[ -f "${DOWNLOAD_DIR}/${PARQUET_ARCHIVE}" ]] || download_dataset "${PARQUET_DATASET}" "${PARQUET_ARCHIVE}"
  fi

  extract_archive "${VALIDATION_ARCHIVE}" "${VALIDATION_DIR}"
  extract_archive "${PARQUET_ARCHIVE}" "${PARQUET_DIR}"

  echo "Data prepared under:"
  echo "  ${VALIDATION_DIR}"
  echo "  ${PARQUET_DIR}"
}

main "$@"
