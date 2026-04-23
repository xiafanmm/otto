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
    nn_long = (
        pl.scan_parquet(str(nn_path))
        .select(
            pl.col("type").cast(pl.Utf8),
            pl.col("session").cast(pl.Int32),
            pl.col("candidates").cast(pl.List(pl.Int32)),
            pl.col("scores").cast(pl.List(pl.Float32)),
        )
        .explode(["candidates", "scores"])
        .select(
            "session",
            "type",
            pl.col("candidates").alias("aid"),
            pl.col("scores").alias("nn_score").cast(pl.Float32),
        )
    )
    return nn_long.group_by(["session", "aid"]).agg(
        pl.col("nn_score")
        .filter(pl.col("type") == "clicks")
        .max()
        .cast(pl.Float32)
        .alias("nn_clicks_score"),
        pl.col("nn_score")
        .filter(pl.col("type") == "carts")
        .max()
        .cast(pl.Float32)
        .alias("nn_carts_score"),
        pl.col("nn_score")
        .filter(pl.col("type") == "orders")
        .max()
        .cast(pl.Float32)
        .alias("nn_orders_score"),
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
) -> pl.DataFrame:
    logger.info("joining covisit, nn, history scores into wide table")
    wide = (
        covisit_scores.join(
            nn_scores,
            on=["session", "aid"],
            how="full",
            coalesce=True,
        )
        .join(
            history_scores,
            on=["session", "aid"],
            how="full",
            coalesce=True,
        )
        .select(
            pl.col("session").cast(pl.Int32),
            pl.col("aid").cast(pl.Int32),
            pl.col("covisit_score").fill_null(0.0).cast(pl.Float32),
            pl.col("nn_clicks_score").fill_null(0.0).cast(pl.Float32),
            pl.col("nn_carts_score").fill_null(0.0).cast(pl.Float32),
            pl.col("nn_orders_score").fill_null(0.0).cast(pl.Float32),
            pl.col("history_score").fill_null(0.0).cast(pl.Float32),
        )
    )
    wide_df = wide.collect(engine="streaming")
    logger.info("wide rows=%d, unique sessions=%d", wide_df.height, wide_df["session"].n_unique())
    return wide_df


def rank_score_expr(score_col: str, alias: str) -> pl.Expr:
    positive_count = (
        pl.when(pl.col(score_col) > 0)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .sum()
        .over("session")
        .cast(pl.Float32)
    )
    denom = pl.max_horizontal(positive_count, pl.lit(1.0).cast(pl.Float32))
    rank = pl.col(score_col).rank("ordinal", descending=True).over("session").cast(pl.Float32)
    return (
        pl.when(pl.col(score_col) > 0)
        .then((pl.lit(1.0) - (rank - 1.0) / denom).cast(pl.Float32))
        .otherwise(pl.lit(0.0).cast(pl.Float32))
        .alias(alias)
    )


def build_type_topk(
    wide_scores: pl.LazyFrame,
    *,
    event_type: str,
    nn_col: str,
    args: Args,
) -> pl.LazyFrame:
    return (
        wide_scores.select(
            pl.lit(event_type).alias("type"),
            "session",
            "aid",
            "covisit_score",
            pl.col(nn_col).alias("nn_score"),
            "history_score",
        )
        .with_columns(
            rank_score_expr("covisit_score", "covisit_rank"),
            rank_score_expr("nn_score", "nn_rank"),
            rank_score_expr("history_score", "history_rank"),
        )
        .with_columns(
            (
                pl.lit(args.alpha).cast(pl.Float32) * pl.col("covisit_rank")
                + pl.lit(args.beta).cast(pl.Float32) * pl.col("nn_rank")
                + pl.lit(args.gamma).cast(pl.Float32) * pl.col("history_rank")
            )
            .cast(pl.Float32)
            .alias("fused_score")
        )
        .filter(pl.col("fused_score") > 0)
        .group_by(["type", "session"])
        .agg(
            pl.col("aid")
            .sort_by("fused_score", descending=True)
            .head(args.topk)
            .alias("candidates"),
            pl.col("fused_score")
            .sort_by("fused_score", descending=True)
            .head(args.topk)
            .alias("scores"),
        )
        .select("type", "session", "candidates", "scores")
    )


def build_topk_table(wide_path: Path, args: Args, logger) -> pl.DataFrame:
    logger.info("building fused top-%d candidates per type", args.topk)
    wide_scores = pl.scan_parquet(str(wide_path)).select(
        pl.col("session").cast(pl.Int32),
        pl.col("aid").cast(pl.Int32),
        pl.col("covisit_score").cast(pl.Float32),
        pl.col("nn_clicks_score").cast(pl.Float32),
        pl.col("nn_carts_score").cast(pl.Float32),
        pl.col("nn_orders_score").cast(pl.Float32),
        pl.col("history_score").cast(pl.Float32),
    )
    topk = pl.concat(
        [
            build_type_topk(wide_scores, event_type=event_type, nn_col=nn_col, args=args)
            for event_type, nn_col in TYPE_TO_NN_COL.items()
        ]
    ).collect(engine="streaming")
    logger.info("topk rows=%d, unique sessions=%d", topk.height, topk["session"].n_unique())
    return topk


def main():
    args = parse_args()
    logger = setup_logger(args.exp_name, run_name=SCRIPT_NAME)
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

    wide_df = build_wide_scores(covisit_scores, nn_scores, history_scores, logger)
    wide_df.write_parquet(wide_path)
    logger.info("saved fused wide parquet to %s", wide_path)

    topk_df = build_topk_table(wide_path, args, logger)
    topk_df.write_parquet(topk_path)
    logger.info("saved fused topk parquet to %s", topk_path)


if __name__ == "__main__":
    main()
