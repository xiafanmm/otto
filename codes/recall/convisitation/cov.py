import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = Path(__file__).stem
sys.path.insert(0, str(ROOT / "codes"))

from utils.logger import setup_logger  # noqa: E402
# TODO 提交时候需要改一下地址
TRAIN_DIR = ROOT / "data/raw/validation/train_parquet"
TEST_DIR = ROOT / "data/raw/validation/test_parquet"
OUT_DIR = ROOT / "data/recall/covisit"

MS_PER_DAY = 24 * 3600 * 1000
MS_PER_HOUR = 3600 * 1000
TIME_DECAY_ALPHA = 3.0  # Chris Deotte 方案经验值：最新事件权重放大到 1+α 倍


"""
训练 example:

uv run python codes/recall/convisitation/cov.py --mode train \
    --exp-name try \
    --click 1 --cart 6 --order 3 \
    --n-lookback 30 --hours 24 --days 14 --topk 50
"""


@dataclass
class Args:
    mode: str
    exp_name: str
    click: int
    cart: int
    order: int
    n_lookback: int
    hours: int
    days: int | None
    topk: int
    time_decay: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build covisitation matrix for OTTO.")
    p.add_argument("--mode", choices=["train", "submit"], required=True)
    p.add_argument("--exp-name", default="baseline")
    # 事件权重：给 pair 的 "B 侧" 事件类型赋分（Radek baseline 方向：click<order<cart）。
    p.add_argument("--click", type=int, required=True)
    p.add_argument("--cart", type=int, required=True)
    p.add_argument("--order", type=int, required=True)
    # 同一 session 内 pair 的位置差上限（事件个数）
    p.add_argument("--n-lookback", type=int, default=30)
    # 同一 session 内 pair 的时间差上限（小时）
    # 最大的一个session的时间差距是500多小时
    p.add_argument("--hours", type=int, default=24)
    # 只用最近 N 天数据（None 表示用全部）
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--topk", type=int, default=50)
    # 时间衰减：最新事件权重更大（w *= 1 + α·(ts-ts_min)/(ts_max-ts_min)）；关掉可做 A/B
    p.add_argument(
        "--time-decay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用时间衰减；用 --no-time-decay 关闭",
    )
    ns = p.parse_args()
    return Args(**vars(ns))


def load_events(mode: str, days: int | None, logger) -> pl.LazyFrame:
    """train 只用 train_parquet；submit 用 train+test 的全部事件。"""
    files = sorted(TRAIN_DIR.glob("*.parquet"))
    if mode == "submit":
        files += sorted(TEST_DIR.glob("*.parquet"))
    logger.info("loading %d parquet files (mode=%s)", len(files), mode)

    lf = pl.scan_parquet([str(f) for f in files]).select(
        pl.col("session").cast(pl.Int32),
        pl.col("aid").cast(pl.Int32),
        pl.col("ts").cast(pl.Int64),
        pl.col("type").cast(pl.Utf8),
    )

    if days is not None:
        max_ts = lf.select(pl.col("ts").max()).collect().item()
        min_ts = max_ts - days * MS_PER_DAY
        logger.info("filter ts >= %d (keep last %d days)", min_ts, days)
        lf = lf.filter(pl.col("ts") >= min_ts)

    return lf


def build_covisit(
    events: pl.LazyFrame,
    type_weights: dict[str, int],
    n_lookback: int,
    window_ms: int,
    topk: int,
    logger,
    ts_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
    """向量化 shift 版本：每步 shift 产生 N 行而非 self-join 的 L² 膨胀。

    做法：按 (session, ts) 排序后，对每个回看步长 i∈[1..K]，把
    (session, aid, ts, type) 整列 shift(i)，filter 掉跨 session / 超时间窗 / 同 aid
    的行，即得到"前 i 位 → 当前位"的 pair。正反两个方向各拼一份，语义与原先
    self-join + |Δpos|≤K 的对称 co-visit 对等。

    ts_range 非空时启用时间衰减：w = type_weight * (1 + α·(ts_b - ts_min)/(ts_max - ts_min))。
    """
    base = events.sort(["session", "ts"]).select("session", "aid", "ts", "type")

    parts = []
    for i in range(1, n_lookback + 1):
        shifted = base.with_columns(
            # session aid ts type
            # session aid ts type ----  sess_prev aid_prev ts_prev type_prev
            pl.col("session").shift(i).alias("sess_prev"),
            pl.col("aid").shift(i).alias("aid_prev"),
            pl.col("ts").shift(i).alias("ts_prev"),
            pl.col("type").shift(i).alias("type_prev"),
        ).filter(
            (pl.col("session") == pl.col("sess_prev"))
            & ((pl.col("ts") - pl.col("ts_prev")).abs() <= window_ms)
            & (pl.col("aid") != pl.col("aid_prev"))
        )
        forward = shifted.select(
            pl.col("aid_prev").alias("aid_a"),
            pl.col("aid").alias("aid_b"),
            pl.col("type").alias("type_b"),
            pl.col("ts").alias("ts_b"),
        )
        backward = shifted.select(
            pl.col("aid").alias("aid_a"),
            pl.col("aid_prev").alias("aid_b"),
            pl.col("type_prev").alias("type_b"),
            pl.col("ts_prev").alias("ts_b"),
        )
        parts.append(forward)
        parts.append(backward)

    w_type = pl.col("type_b").replace_strict(
        type_weights, default=0, return_dtype=pl.Int32
    )
    if ts_range is None:
        w_expr = w_type.alias("w")
    else:
        ts_min, ts_max = ts_range
        span = max(ts_max - ts_min, 1)
        w_expr = (
            w_type.cast(pl.Float32)
            * (1.0 + TIME_DECAY_ALPHA * (pl.col("ts_b") - ts_min) / span).cast(
                pl.Float32
            )
        ).alias("w")

    pairs = pl.concat(parts).with_columns(w_expr)

    scores = (
        pairs.group_by(["aid_a", "aid_b"])
        .agg(pl.col("w").sum().alias("score"))
        .sort(["aid_a", "score"], descending=[False, True])
        .group_by("aid_a", maintain_order=True)
        .head(topk)
    )

    logger.info("collecting co-visit pairs (shift, n_lookback=%d)", n_lookback)
    out = scores.collect(engine="streaming")
    logger.info("pairs=%d, unique aid_a=%d", out.height, out["aid_a"].n_unique())
    return out


def main():
    args = parse_args()
    logger = setup_logger(args.exp_name, run_name=SCRIPT_NAME)
    logger.info("args: %s", args)
    logger.info("=" * 100)

    type_weights = {"clicks": args.click, "carts": args.cart, "orders": args.order}
    window_ms = args.hours * MS_PER_HOUR

    events = load_events(args.mode, args.days, logger)

    ts_range = None
    if args.time_decay:
        ts_stats = events.select(
            pl.col("ts").min().alias("lo"), pl.col("ts").max().alias("hi")
        ).collect()
        ts_range = (int(ts_stats["lo"][0]), int(ts_stats["hi"][0]))
        logger.info(
            "time decay ON: ts_min=%d ts_max=%d alpha=%.1f",
            ts_range[0], ts_range[1], TIME_DECAY_ALPHA,
        )
    else:
        logger.info("time decay OFF")

    cov = build_covisit(
        events,
        type_weights=type_weights,
        n_lookback=args.n_lookback,
        window_ms=window_ms,
        topk=args.topk,
        logger=logger,
        ts_range=ts_range,
    )

    out_dir = OUT_DIR / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.mode}.parquet"
    cov.write_parquet(out_path)
    logger.info("saved covisit matrix to %s", out_path)


if __name__ == "__main__":
    main()
