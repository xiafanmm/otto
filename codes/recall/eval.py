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

COVISIT_DIR = ROOT / "data/recall/covisit"
NN_DIR = ROOT / "data/recall/nn"
FUSED_DIR = ROOT / "data/recall/fused"
TEST_DIR = ROOT / "data/raw/validation/test_parquet"
LABELS_PATH = ROOT / "data/raw/validation/test_labels.parquet"
EVAL_ROOT = ROOT / "data/eval"
TYPE_WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}


@dataclass
class Args:
    exp_name: str
    k_list: list[int]
    method: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate OTTO recall.")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--k-list", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument(
        "--method",
        choices=["covisit", "history", "nn", "fused"],
        default="covisit",
        help="recall source to evaluate",
    )
    namespace = parser.parse_args()

    k_list = sorted(set(namespace.k_list))
    if not k_list or any(k <= 0 for k in k_list):
        raise ValueError("--k-list must contain positive integers")

    namespace.k_list = k_list
    return Args(**vars(namespace))


def load_covisit(exp_name: str, logger) -> pl.LazyFrame:
    covisit_path = COVISIT_DIR / exp_name / "train.parquet"
    if not covisit_path.exists():
        raise FileNotFoundError(f"covisit parquet not found: {covisit_path}")

    logger.info("load covisit from: %s", covisit_path)
    return pl.scan_parquet(str(covisit_path)).select(
        pl.col("aid_a").cast(pl.Int32),
        pl.col("aid_b").cast(pl.Int32),
        pl.col("score").cast(pl.Float32),
    )


def load_typed_candidates(
    path: Path,
    *,
    max_k: int,
    logger,
    label: str,
) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} recall parquet not found: {path}")

    schema = pl.scan_parquet(str(path)).collect_schema()
    missing = {"type", "session", "candidates"} - set(schema.names())
    if missing:
        raise ValueError(f"{label} recall parquet missing columns: {sorted(missing)}")

    logger.info("load %s recall from: %s", label, path)
    candidates = (
        pl.scan_parquet(str(path))
        .select(
            pl.col("type").cast(pl.Utf8),
            pl.col("session").cast(pl.Int32),
            pl.col("candidates").list.head(max_k).alias("cand_list"),
        )
        .collect(engine="streaming")
    )
    logger.info(
        "candidate rows=%d, unique sessions=%d",
        candidates.height,
        candidates["session"].n_unique(),
    )
    return candidates


def load_nn_candidates(exp_name: str, max_k: int, logger) -> pl.DataFrame:
    return load_typed_candidates(
        NN_DIR / exp_name / "train.parquet",
        max_k=max_k,
        logger=logger,
        label="nn",
    )


def load_fused_candidates(exp_name: str, max_k: int, logger) -> pl.DataFrame:
    return load_typed_candidates(
        FUSED_DIR / exp_name / "train_topk.parquet",
        max_k=max_k,
        logger=logger,
        label="fused",
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
    method: str,
) -> pl.DataFrame:
    logger.info("building top-%d candidates per session", max_k)

    if method == "history":
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

    per_session_candidates = (
        all_sessions.join(top_candidates, on="session", how="left")
        .with_columns(
            pl.col("cand_list").fill_null(pl.lit([], dtype=pl.List(pl.Int32)))
        )
        .collect(engine="streaming")
    )
    typed_candidates = pl.concat(
        [
            per_session_candidates.with_columns(pl.lit(event_type).alias("type")).select(
                "type",
                "session",
                "cand_list",
            )
            for event_type in TYPE_WEIGHTS
        ]
    )
    logger.info(
        "candidate rows=%d, unique sessions=%d",
        typed_candidates.height,
        per_session_candidates.height,
    )
    return typed_candidates


def evaluate_type(
    candidates: pl.DataFrame,
    labels: pl.LazyFrame,
    event_type: str,
    k_list: list[int],
) -> tuple[int, dict[int, float]]:
    labels_t = labels.filter(pl.col("type") == event_type).select(
        pl.col("session").cast(pl.Int32),
        pl.col("ground_truth").cast(pl.List(pl.Int32)),
    )

    candidates_t = candidates.lazy().filter(pl.col("type") == event_type).select(
        "session",
        "cand_list",
    )
    joined = candidates_t.join(labels_t, on="session", how="inner")
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
    logger.info("method: %s", args.method)
    logger.info("load_test_data from: %s", TEST_DIR)
    logger.info("load_labels from: %s", LABELS_PATH)
    logger.info("=" * 150)

    max_k = max(args.k_list)
    if args.method == "nn":
        top_candidates = load_nn_candidates(args.exp_name, max_k, logger)
    elif args.method == "fused":
        top_candidates = load_fused_candidates(args.exp_name, max_k, logger)
    else:
        covisit = None if args.method == "history" else load_covisit(args.exp_name, logger)
        events = load_test_events(logger)
        top_candidates = build_top_candidates(
            events,
            covisit,
            max_k=max_k,
            logger=logger,
            method=args.method,
        )
    labels = load_labels(logger)

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
        "method": args.method,
        "sessions_evaluated": sessions_evaluated,
        "recall_at": recall_at_output,
    }

    out_dir = EVAL_ROOT / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.exp_name}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    logger.info("saved evaluation summary to %s", out_path)


if __name__ == "__main__":
    main()
