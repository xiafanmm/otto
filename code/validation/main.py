from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from code.utils import LOG_DIR, ensure_dir, log_duration, setup_logger

TYPE_WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet requires pyarrow or fastparquet. "
                "Install one of them before running local validation."
            ) from exc

    raise ValueError(f"Unsupported file type: {path.suffix}")


def _to_item_list(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, str):
        return [int(part) for part in value.split() if part]

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value if not pd.isna(item)]

    return [int(value)]


def _normalize_session_type(session: Any, event_type: str) -> str:
    return f"{int(session)}_{event_type}"


def normalize_predictions(df: pd.DataFrame) -> dict[str, list[int]]:
    if {"session_type", "labels"}.issubset(df.columns):
        return {
            str(row.session_type): _to_item_list(row.labels)[:20]
            for row in df.itertuples(index=False)
        }

    if {"session", "type", "labels"}.issubset(df.columns):
        return {
            _normalize_session_type(row.session, row.type): _to_item_list(row.labels)[:20]
            for row in df.itertuples(index=False)
        }

    raise ValueError(
        "Predictions must contain either ['session_type', 'labels'] "
        "or ['session', 'type', 'labels'] columns."
    )


def normalize_labels(df: pd.DataFrame) -> dict[str, list[int]]:
    if {"session_type", "ground_truth"}.issubset(df.columns):
        return {
            str(row.session_type): _to_item_list(row.ground_truth)
            for row in df.itertuples(index=False)
        }

    if {"session", "type", "ground_truth"}.issubset(df.columns):
        return {
            _normalize_session_type(row.session, row.type): _to_item_list(row.ground_truth)
            for row in df.itertuples(index=False)
        }

    raise ValueError(
        "Labels must contain either ['session_type', 'ground_truth'] "
        "or ['session', 'type', 'ground_truth'] columns."
    )


def compute_metrics(
    predictions: dict[str, list[int]], labels: dict[str, list[int]]
) -> dict[str, Any]:
    stats = {
        event_type: {"hits": 0, "targets": 0, "rows": 0}
        for event_type in TYPE_WEIGHTS
    }

    for session_type, ground_truth in labels.items():
        event_type = session_type.rsplit("_", 1)[-1]
        if event_type not in stats:
            continue

        predicted = predictions.get(session_type, [])[:20]
        hits = len(set(predicted).intersection(ground_truth))

        stats[event_type]["hits"] += hits
        stats[event_type]["targets"] += len(ground_truth)
        stats[event_type]["rows"] += 1

    metrics: dict[str, Any] = {}
    overall = 0.0

    for event_type, values in stats.items():
        targets = values["targets"]
        recall = values["hits"] / targets if targets else 0.0
        metrics[event_type] = {
            "recall@20": recall,
            "hits": values["hits"],
            "targets": targets,
            "rows": values["rows"],
            "weight": TYPE_WEIGHTS[event_type],
        }
        overall += TYPE_WEIGHTS[event_type] * recall

    metrics["overall_recall@20"] = overall
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OTTO local validation predictions.")
    parser.add_argument("--predictions-file", required=True, type=Path)
    parser.add_argument("--labels-file", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--exp-name", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(
        "validation",
        log_dir=LOG_DIR / args.exp_name,
        run_name="validation",
    )

    logger.info("predictions file: %s", args.predictions_file)
    logger.info("labels file: %s", args.labels_file)

    with log_duration(logger, "load predictions"):
        predictions_df = _read_table(args.predictions_file)
        predictions = normalize_predictions(predictions_df)

    with log_duration(logger, "load labels"):
        labels_df = _read_table(args.labels_file)
        labels = normalize_labels(labels_df)

    with log_duration(logger, "compute metrics"):
        metrics = compute_metrics(predictions, labels)

    ensure_dir(args.metrics_output.parent)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    for event_type in TYPE_WEIGHTS:
        event_metrics = metrics[event_type]
        logger.info(
            "%s recall@20=%.6f hits=%s targets=%s",
            event_type,
            event_metrics["recall@20"],
            event_metrics["hits"],
            event_metrics["targets"],
        )

    logger.info("overall recall@20=%.6f", metrics["overall_recall@20"])
    logger.info("metrics written to %s", args.metrics_output)


if __name__ == "__main__":
    main()
