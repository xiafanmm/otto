import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = Path(__file__).stem
sys.path.insert(0, str(ROOT / "codes"))

from utils.logger import setup_logger  # noqa: E402

COVISIT_DIR = ROOT / "data/recall/covisit"
NN_DIR = ROOT / "data/recall/nn"
TEST_DIR = ROOT / "data/raw/validation/test_parquet"
OUT_DIR = ROOT / "data/recall/fused"

EVENT_WEIGHTS = {"clicks": 1, "carts": 3, "orders": 6}
TYPE_TO_NN_COL = {
    "clicks": "nn_clicks_score",
    "carts": "nn_carts_score",
    "orders": "nn_orders_score",
}


@dataclass
class Args:
    mode: str
    covisit_exp: str
    nn_exp: str
    exp_name: str
    topk: int
    alpha: float
    beta: float
    gamma: float


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Fuse OTTO recall candidates.")
    parser.add_argument("--mode", choices=["train", "submit"], required=True)
    parser.add_argument("--covisit-exp", required=True)
    parser.add_argument("--nn-exp", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    namespace = parser.parse_args()

    if namespace.topk <= 0:
        raise ValueError("--topk must be positive")

    return Args(**vars(namespace))


def load_events(logger) -> pl.LazyFrame:
    files = sorted(TEST_DIR.glob("*.parquet"))
    logger.info("loading %d event parquet files", len(files))
    return pl.scan_parquet([str(path) for path in files]).select(
        pl.col("session").cast(pl.Int32),
        pl.col("aid").cast(pl.Int32),
        pl.col("type").cast(pl.Utf8),
    )


def build_covisit_scores(events: pl.LazyFrame, args: Args, logger) -> pl.LazyFrame:
    covisit_path = COVISIT_DIR / args.covisit_exp / f"{args.mode}.parquet"
    if not covisit_path.exists():
        raise FileNotFoundError(f"covisit parquet not found: {covisit_path}")

    logger.info("load covisit from: %s", covisit_path)
    covisit = pl.scan_parquet(str(covisit_path)).select(
        pl.col("aid_a").cast(pl.Int32),
        pl.col("aid_b").cast(pl.Int32),
        pl.col("score").cast(pl.Float32),
    )
    history = events.select("session", "aid").unique(subset=["session", "aid"])
    return (
        history.join(covisit, left_on="aid", right_on="aid_a", how="inner")
        .select(
            "session",
            pl.col("aid_b").alias("aid"),
            pl.col("score").cast(pl.Float32),
        )
        .group_by(["session", "aid"])
        .agg(pl.col("score").sum().cast(pl.Float32).alias("covisit_score"))
    )


def build_nn_scores(args: Args, logger) -> pl.LazyFrame:
    nn_path = NN_DIR / args.nn_exp / f"{args.mode}.parquet"
    if not nn_path.exists():
        raise FileNotFoundError(f"nn recall parquet not found: {nn_path}")

    schema = pl.scan_parquet(str(nn_path)).collect_schema()
    missing = {"type", "session", "candidates", "scores"} - set(schema.names())
    if missing:
        raise ValueError(f"nn recall parquet missing columns: {sorted(missing)}")

    logger.info("load nn recall from: %s", nn_path)
    base = pl.scan_parquet(str(nn_path)).select(
        pl.col("type").cast(pl.Utf8),
        pl.col("session").cast(pl.Int32),
        pl.col("candidates"),
        pl.col("scores"),
    )
    parts = []
    for event_type, out_col in TYPE_TO_NN_COL.items():
        parts.append(
            base.filter(pl.col("type") == event_type)
            .explode(["candidates", "scores"])
            .select(
                pl.col("session").cast(pl.Int32),
                pl.col("candidates").cast(pl.Int32).alias("aid"),
                pl.col("scores").cast(pl.Float32).alias(out_col),
            )
        )
    return pl.concat(parts, how="diagonal_relaxed").group_by(["session", "aid"]).agg(
        pl.col("nn_clicks_score").max().cast(pl.Float32).alias("nn_clicks_score"),
        pl.col("nn_carts_score").max().cast(pl.Float32).alias("nn_carts_score"),
        pl.col("nn_orders_score").max().cast(pl.Float32).alias("nn_orders_score"),
    )


def build_history_scores(events: pl.LazyFrame) -> pl.LazyFrame:
    return (
        events.with_columns(
            pl.col("type")
            .replace_strict(EVENT_WEIGHTS, default=0, return_dtype=pl.Int32)
            .cast(pl.Float32)
            .alias("weight")
        )
        .group_by(["session", "aid"])
        .agg(pl.col("weight").sum().cast(pl.Float32).alias("history_score"))
    )


def build_wide_scores(
    covisit_scores: pl.LazyFrame,
    nn_scores: pl.LazyFrame,
    history_scores: pl.LazyFrame,
    logger,
) -> pl.LazyFrame:
    logger.info("building wide scores via concat + group_by")
    return (
        pl.concat(
            [
                covisit_scores.select(
                    "session",
                    "aid",
                    pl.col("covisit_score").cast(pl.Float32),
                ),
                nn_scores.select(
                    "session",
                    "aid",
                    pl.col("nn_clicks_score").cast(pl.Float32),
                    pl.col("nn_carts_score").cast(pl.Float32),
                    pl.col("nn_orders_score").cast(pl.Float32),
                ),
                history_scores.select(
                    "session",
                    "aid",
                    pl.col("history_score").cast(pl.Float32),
                ),
            ],
            how="diagonal_relaxed",
        )
        .group_by(["session", "aid"])
        .agg(
            pl.col("covisit_score").max().alias("covisit_score"),
            pl.col("nn_clicks_score").max().alias("nn_clicks_score"),
            pl.col("nn_carts_score").max().alias("nn_carts_score"),
            pl.col("nn_orders_score").max().alias("nn_orders_score"),
            pl.col("history_score").max().alias("history_score"),
        )
        .with_columns(
            pl.col("covisit_score").fill_null(0.0).cast(pl.Float32),
            pl.col("nn_clicks_score").fill_null(0.0).cast(pl.Float32),
            pl.col("nn_carts_score").fill_null(0.0).cast(pl.Float32),
            pl.col("nn_orders_score").fill_null(0.0).cast(pl.Float32),
            pl.col("history_score").fill_null(0.0).cast(pl.Float32),
        )
        .select(
            pl.col("session").cast(pl.Int32),
            pl.col("aid").cast(pl.Int32),
            pl.col("covisit_score").cast(pl.Float32),
            pl.col("nn_clicks_score").cast(pl.Float32),
            pl.col("nn_carts_score").cast(pl.Float32),
            pl.col("nn_orders_score").cast(pl.Float32),
            pl.col("history_score").cast(pl.Float32),
        )
    )


def minmax_score_expr(score_col: str, alias: str) -> pl.Expr:
    max_val = pl.col(score_col).max().over("session").cast(pl.Float32)
    return (
        pl.when((pl.col(score_col) > 0) & (max_val > 0))
        .then((pl.col(score_col) / max_val).cast(pl.Float32))
        .otherwise(pl.lit(0.0).cast(pl.Float32))
        .alias(alias)
    )


def build_topk_table(wide_path: Path, args: Args, logger) -> pl.DataFrame:
    logger.info("building fused top-%d candidates per type", args.topk)
    alpha = pl.lit(args.alpha).cast(pl.Float32)
    beta = pl.lit(args.beta).cast(pl.Float32)
    gamma = pl.lit(args.gamma).cast(pl.Float32)
    tmp_dir = wide_path.parent
    tmp_paths: list[Path] = []

    try:
        for event_type, nn_col in TYPE_TO_NN_COL.items():
            logger.info("processing type=%s", event_type)
            tmp_path = tmp_dir / f"_topk_{event_type}.tmp.parquet"
            per_type = (
                pl.scan_parquet(str(wide_path))
                .select(
                    pl.lit(event_type, dtype=pl.Utf8).alias("type"),
                    pl.col("session").cast(pl.Int32),
                    pl.col("aid").cast(pl.Int32),
                    pl.col("covisit_score").cast(pl.Float32),
                    pl.col(nn_col).cast(pl.Float32).alias("nn_score"),
                    pl.col("history_score").cast(pl.Float32),
                )
                .with_columns(
                    minmax_score_expr("covisit_score", "covisit_rank"),
                    minmax_score_expr("nn_score", "nn_rank"),
                    minmax_score_expr("history_score", "history_rank"),
                )
                .with_columns(
                    (
                        alpha * pl.col("covisit_rank")
                        + beta * pl.col("nn_rank")
                        + gamma * pl.col("history_rank")
                    )
                    .cast(pl.Float32)
                    .alias("fused_score")
                )
                .filter(pl.col("fused_score") > 0)
                .group_by(["type", "session"])
                .agg(
                    pl.struct("aid", "fused_score")
                    .top_k_by("fused_score", args.topk)
                    .alias("pairs")
                )
                .explode("pairs")
                .unnest("pairs")
                .group_by(["type", "session"], maintain_order=True)
                .agg(
                    pl.col("aid").cast(pl.Int32).alias("candidates"),
                    pl.col("fused_score").cast(pl.Float32).alias("scores"),
                )
                .select("type", "session", "candidates", "scores")
            )
            per_type.sink_parquet(str(tmp_path))
            tmp_paths.append(tmp_path)

        logger.info("concatenating %d per-type topk files", len(tmp_paths))
        topk = pl.concat(
            [pl.scan_parquet(str(path)) for path in tmp_paths],
            how="vertical",
        ).collect(engine="streaming")
    finally:
        for path in tmp_paths:
            path.unlink(missing_ok=True)

    logger.info("topk rows=%d, unique sessions=%d", topk.height, topk["session"].n_unique())
    return topk


def main():
    args = parse_args()
    logger = setup_logger(args.exp_name, stage="fused", run_name=SCRIPT_NAME)
    logger.info("*" * 150)
    logger.info("args: %s", args)
    logger.info("load_test_data from: %s", TEST_DIR)
    logger.info("=" * 150)

    events = load_events(logger)
    covisit_scores = build_covisit_scores(events, args, logger)
    nn_scores = build_nn_scores(args, logger)
    history_scores = build_history_scores(events)

    out_dir = OUT_DIR / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_path = out_dir / f"{args.mode}.parquet"
    topk_path = out_dir / f"{args.mode}_topk.parquet"

    wide_lf = build_wide_scores(covisit_scores, nn_scores, history_scores, logger)
    wide_lf.sink_parquet(str(wide_path))
    logger.info("saved fused wide parquet to %s", wide_path)

    topk_df = build_topk_table(wide_path, args, logger)
    topk_df.write_parquet(topk_path)
    logger.info("saved fused topk parquet to %s", topk_path)


if __name__ == "__main__":
    main()
