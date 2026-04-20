from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .paths import LOG_DIR, ensure_dir

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _normalize_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logger(
    name: str,
    *,
    log_dir: str | Path | None = None,
    run_name: str | None = None,
    level: int | str = logging.INFO,
    with_console: bool = True,
) -> logging.Logger:
    """Create a console+file logger for a pipeline stage.

    Example:
        logger = setup_logger("fusion", log_dir=LOG_DIR / "fusion", run_name="train")
        logger.info("starting fusion stage")
    """

    logger = logging.getLogger(name)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if with_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    target_dir = ensure_dir(log_dir or (LOG_DIR / name))
    log_stem = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = target_dir / f"{log_stem}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger initialized at %s", log_path)
    return logger
