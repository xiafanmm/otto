import logging
from datetime import datetime
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(filename)-12s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
ROOT = Path(__file__).resolve().parents[2]


LOG_DIR = ROOT / "data/logs"

def _normalize_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(
    exp_name: str, 
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

    logger = logging.getLogger(exp_name)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if with_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    target_dir = ensure_dir(log_dir or (LOG_DIR / exp_name))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_stem = f"{run_name}_{timestamp}" if run_name else timestamp
    log_path = target_dir / f"{log_stem}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger initialized at %s", log_path)
    return logger

def main():
    print(ROOT)

    
if __name__ == "__main__":
    main()