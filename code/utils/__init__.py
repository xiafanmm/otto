"""OTTO 项目共用工具：统一导出路径、日志和计时辅助函数。"""

from .logger import get_logger, setup_logger
from .paths import (
    DATA_DIR,
    DOWNLOAD_DIR,
    LOG_DIR,
    MODEL_DIR,
    PARQUET_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ROOT_DIR,
    SUBMISSIONS_DIR,
    VALIDATION_DIR,
    ensure_dir,
)
from .timer import StageTimer, log_duration

# 通过 `from code.utils import ...` 暴露公共接口，避免业务代码直接关心子模块布局。
__all__ = [
    "DATA_DIR",
    "DOWNLOAD_DIR",
    "LOG_DIR",
    "MODEL_DIR",
    "PARQUET_DIR",
    "PROCESSED_DIR",
    "RAW_DIR",
    "ROOT_DIR",
    "SUBMISSIONS_DIR",
    "VALIDATION_DIR",
    "StageTimer",
    "ensure_dir",
    "get_logger",
    "log_duration",
    "setup_logger",
]
