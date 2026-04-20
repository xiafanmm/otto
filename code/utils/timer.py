from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


class StageTimer:
    """最轻量的计时器：包住一段代码后，把耗时存到 `elapsed`。"""

    def __init__(self, label: str) -> None:
        # label 主要用于外部业务代码标记阶段名，当前类本身不直接打印日志。
        self.label = label
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "StageTimer":
        # 进入 with 代码块时记录起始时间。
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # 无论代码块是否报错，退出时都更新本次耗时。
        self.elapsed = perf_counter() - self._start


@contextmanager
def log_duration(logger, label: str) -> Iterator[None]:
    """记录某个阶段的开始和结束耗时，适合直接配合 logger 使用。"""

    start = perf_counter()
    logger.info("%s started", label)
    try:
        # `yield` 把执行权交给 with 语句内部的业务代码。
        yield
    finally:
        # finally 保证即使中间抛异常，也会把结束耗时记下来。
        elapsed = perf_counter() - start
        logger.info("%s finished in %.2fs", label, elapsed)
