from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .paths import LOG_DIR, ensure_dir

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _normalize_level(level: int | str) -> int:
    """把字符串或整数形式的日志级别统一转成 logging 可识别的整数值。"""
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """按名字获取 logger；如果外部已经配置过 handlers，会复用同一个 logger。"""
    return logging.getLogger(name)


def setup_logger(
    name: str,
    *,
    log_dir: str | Path | None = None,
    run_name: str | None = None,
    level: int | str = logging.INFO,
    with_console: bool = True,
) -> logging.Logger:
    """为某个 pipeline 阶段创建 logger，同时输出到控制台和日志文件。

    Example:
        logger = setup_logger("fusion", log_dir=LOG_DIR / "fusion", run_name="train")
        logger.info("starting fusion stage")
    """

    logger = logging.getLogger(name)
    logger.setLevel(_normalize_level(level))
    # 关闭向 root logger 继续冒泡，避免一条日志被重复打印。
    logger.propagate = False

    # 同名 logger 如果已经初始化过，直接复用，避免重复添加 handler。
    if logger.handlers:
        return logger

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if with_console:
        # 控制台输出，方便本地直接观察运行进度。
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # 文件日志默认写到 `data/logs/<name>`，也允许调用方显式覆盖。
    target_dir = ensure_dir(log_dir or (LOG_DIR / name))
    log_stem = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = target_dir / f"{log_stem}.log"

    # 文件日志会持久化保存，便于离线排查实验结果和耗时。
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger initialized at %s", log_path)
    return logger
