from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
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
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
