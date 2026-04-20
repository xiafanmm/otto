from __future__ import annotations

from pathlib import Path

import pandas as pd

from code.utils import LOG_DIR, VALIDATION_DIR, log_duration, setup_logger


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "This example needs parquet support. Install pyarrow or fastparquet first."
        ) from exc


def main() -> None:
    logger = setup_logger("example", log_dir=LOG_DIR / "example", run_name="demo")
    test_dir = VALIDATION_DIR / "test_parquet"
    label_path = VALIDATION_DIR / "test_labels.parquet"
    first_shard = sorted(test_dir.glob("*.parquet"))[0]

    logger.info("validation dir: %s", VALIDATION_DIR)
    logger.info("first shard: %s", first_shard)
    logger.info("label file: %s", label_path)

    with log_duration(logger, "load first validation shard"):
        shard_df = read_parquet(first_shard)
    logger.info("shard shape: %s", shard_df.shape)
    logger.info("shard columns: %s", list(shard_df.columns))
    logger.info("shard preview: %s", shard_df.head(3).to_dict(orient="records"))

    with log_duration(logger, "load validation labels"):
        labels_df = read_parquet(label_path)
    logger.info("labels shape: %s", labels_df.shape)
    logger.info("labels columns: %s", list(labels_df.columns))
    logger.info("labels preview: %s", labels_df.head(3).to_dict(orient="records"))


if __name__ == "__main__":
    main()
