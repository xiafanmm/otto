import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = Path(__file__).stem
sys.path.insert(0, str(ROOT / "codes"))

from utils.logger import setup_logger  # noqa: E402

RECALL_DIR = ROOT / "data/recall/covisit"
TEST_DIR = ROOT / "data/raw/validation/test_parquet"
LABELS_PATH = ROOT / "data/raw/validation/test_labels.parquet"
EVAL_OUT_DIR = ROOT / "data/eval/covisit"
TYPE_WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}


@dataclass
class Args:
    exp_name: str
    k_list: list[int]
    baseline: str | None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate OTTO covisit recall.")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--k-list", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument(
        "--baseline",
        choices=["history"],
        default=None,
        help="若设为 history，则跳过 covisit，只用 session 历史 aid 当候选（用于 sanity check）",
    )
    namespace = parser.parse_args()

    k_list = sorted(set(namespace.k_list))
    if not k_list or any(k <= 0 for k in k_list):
        raise ValueError("--k-list must contain positive integers")

    namespace.k_list = k_list
    return Args(**vars(namespace))


def load_covisit(exp_name: str, logger) -> pl.LazyFrame:
    covisit_path = RECALL_DIR / exp_name / "train.parquet"
    if not covisit_path.exists():
        raise FileNotFoundError(f"covisit parquet not found: {covisit_path}")

    logger.info("load covisit from: %s", covisit_path)
    return pl.scan_parquet(str(covisit_path)).select(
        pl.col("aid_a").cast(pl.Int32),
        pl.col("aid_b").cast(pl.Int32),
        pl.col("score").cast(pl.Float32),
    )


def load_test_events(logger) -> pl.LazyFrame:
    files = sorted(TEST_DIR.glob("*.parquet"))
    logger.info("loading %d test parquet files", len(files))

    return pl.scan_parquet([str(path) for path in files]).select(
        pl.col("session").cast(pl.Int32),
        pl.col("aid").cast(pl.Int32),
    )


def load_labels(logger) -> pl.LazyFrame:
    logger.info("load labels from: %s", LABELS_PATH)
    return pl.scan_parquet(str(LABELS_PATH)).select("session", "type", "ground_truth")


def build_top_candidates(
    events: pl.LazyFrame,
    covisit: pl.LazyFrame | None,
    max_k: int,
    logger,
    baseline: str | None,
) -> pl.DataFrame:
    logger.info("building top-%d candidates per session", max_k)

    if baseline == "history":
        all_sessions = events.select("session").unique()
        scored = events.group_by(["session", "aid"]).agg(
            pl.len().cast(pl.Float32).alias("score")
        )
        top_candidates = scored.group_by("session").agg(
            pl.col("aid").sort_by("score", descending=True).head(max_k).alias("cand_list")
        )
    else:
        history = events.unique(subset=["session", "aid"])
        all_sessions = history.select("session").unique()
        candidate_scores = (
            history.join(covisit, left_on="aid", right_on="aid_a", how="inner")
            .select("session", "aid_b", "score")
            .group_by(["session", "aid_b"])
            .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
        )
        top_candidates = candidate_scores.group_by("session").agg(
            pl.col("aid_b")
            .sort_by("score", descending=True)
            .head(max_k)
            .alias("cand_list")
        )

    top_candidates_df = (
        all_sessions.join(top_candidates, on="session", how="left")
        .with_columns(
            pl.col("cand_list").fill_null(pl.lit([], dtype=pl.List(pl.Int32)))
        )
        .collect(engine="streaming")
    )
    logger.info("candidate sessions=%d", top_candidates_df.height)
    return top_candidates_df


def evaluate_type(
    top_candidates: pl.DataFrame,
    labels: pl.LazyFrame,
    event_type: str,
    k_list: list[int],
) -> tuple[int, dict[int, float]]:
    labels_t = labels.filter(pl.col("type") == event_type).select(
        pl.col("session").cast(pl.Int32),
        pl.col("ground_truth").cast(pl.List(pl.Int32)),
    )

    joined = top_candidates.lazy().join(labels_t, on="session", how="inner")
    denom = pl.min_horizontal(pl.col("ground_truth").list.len(), pl.lit(20)).cast(
        pl.Float32
    )

    exprs: list[pl.Expr] = [pl.len().alias("sessions")]
    for k in k_list:
        hits = (
            pl.col("cand_list")
            .list.head(k)
            .list.set_intersection(pl.col("ground_truth"))
            .list.len()
            .cast(pl.Float32)
        )
        exprs.append((hits.sum() / denom.sum()).cast(pl.Float32).alias(f"recall_{k}"))

    metrics = joined.select(exprs).collect(engine="streaming").row(0, named=True)
    sessions = int(metrics["sessions"])
    recall_at = {k: float(metrics[f"recall_{k}"]) for k in k_list}
    return sessions, recall_at


def round4(value: float) -> float:
    return round(float(value), 4)


def main():
    args = parse_args()
    logger = setup_logger(args.exp_name, run_name=SCRIPT_NAME)
    logger.info("*" * 150)
    logger.info("args: %s", args)
    if args.baseline == "history":
        logger.info("baseline mode: history (pure session-history, no covisit)")
    else:
        logger.info("baseline mode: None (pure covisit)")
    logger.info("load_test_data from: %s", TEST_DIR)
    logger.info("load_labels from: %s", LABELS_PATH)
    logger.info("=" * 150)

    max_k = max(args.k_list)
    if args.baseline == "history":
        covisit = None
    else:
        covisit = load_covisit(args.exp_name, logger)
    events = load_test_events(logger)
    labels = load_labels(logger)

    top_candidates = build_top_candidates(
        events,
        covisit,
        max_k=max_k,
        logger=logger,
        baseline=args.baseline,
    )

    sessions_evaluated: dict[str, int] = {}
    recall_by_type: dict[str, dict[int, float]] = {}
    for event_type in TYPE_WEIGHTS:
        sessions, recall_at = evaluate_type(top_candidates, labels, event_type, args.k_list)
        sessions_evaluated[event_type] = sessions
        recall_by_type[event_type] = recall_at

    logger.info(
        "sessions evaluated: clicks=%d, carts=%d, orders=%d",
        sessions_evaluated["clicks"],
        sessions_evaluated["carts"],
        sessions_evaluated["orders"],
    )

    recall_at_output: dict[str, dict[str, float]] = {}
    for k in args.k_list:
        clicks = recall_by_type["clicks"][k]
        carts = recall_by_type["carts"][k]
        orders = recall_by_type["orders"][k]
        total = (
            TYPE_WEIGHTS["clicks"] * clicks
            + TYPE_WEIGHTS["carts"] * carts
            + TYPE_WEIGHTS["orders"] * orders
        )

        logger.info(
            "Recall@%d:  clicks=%.4f  carts=%.4f  orders=%.4f  total=%.4f",
            k,
            clicks,
            carts,
            orders,
            total,
        )

        recall_at_output[str(k)] = {
            "clicks": round4(clicks),
            "carts": round4(carts),
            "orders": round4(orders),
            "total": round4(total),
        }

    summary = {
        "exp_name": args.exp_name,
        "baseline": args.baseline,
        "sessions_evaluated": sessions_evaluated,
        "recall_at": recall_at_output,
    }

    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_OUT_DIR / f"{args.exp_name}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    logger.info("saved evaluation summary to %s", out_path)


if __name__ == "__main__":
    main()
