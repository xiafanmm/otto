from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


class StageTimer:
    def __init__(self, label: str) -> None:
        self.label = label
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed = perf_counter() - self._start


@contextmanager
def log_duration(logger, label: str) -> Iterator[None]:
    start = perf_counter()
    logger.info("%s started", label)
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("%s finished in %.2fs", label, elapsed)
