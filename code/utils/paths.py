from __future__ import annotations

from pathlib import Path

# 项目根目录。`utils/paths.py` 位于 `code/utils` 下，所以向上取两层。
ROOT_DIR = Path(__file__).resolve().parents[2]
# 常用数据目录约定，供各个训练/召回/验证脚本统一复用。
DATA_DIR = ROOT_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
RAW_DIR = DATA_DIR / "raw"
VALIDATION_DIR = RAW_DIR / "validation"
PARQUET_DIR = RAW_DIR / "parquet_chunks"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
SUBMISSIONS_DIR = DATA_DIR / "submissions"


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回标准化后的 Path 对象。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
