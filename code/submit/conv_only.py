from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm


EVENT_TYPES = ("clicks", "carts", "orders")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "raw" / "parquet_chunks" / "test_parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "submissions" / "conv_only"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conv-only OTTO submissions from count_info.pkl files."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["base", "v12", "v13", "v15", "v16", "v17", "v18", "v19"],
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for precomputed session-level parquet cache.",
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild cached session parquet files even if they already exist.",
    )
    return parser.parse_args()


def get_count_info_path(processed_root: Path, version: str) -> Path:
    return (
        processed_root
        / f"convis_{version}_r1"
        / "recall"
        / "convisitation"
        / "count_info.pkl"
    )


def load_count_info(path: Path) -> list[Any]:
    with path.open("rb") as fp:
        return pickle.load(fp)


def predict_labels(
    recent_unique: list[int], count_info_list: list[Any], topk: int
) -> list[int]:
    if not recent_unique:
        return []

    labels: list[int] = []
    label_set: set[int] = set()
    seed_aid = recent_unique[0]

    if 0 <= seed_aid < len(count_info_list):
        seed_info = count_info_list[seed_aid]
        if isinstance(seed_info, list) and seed_info:
            raw_candidates = seed_info[0]
            if isinstance(raw_candidates, list):
                for candidate in raw_candidates:
                    candidate_int = int(candidate)
                    if candidate_int in label_set:
                        continue
                    labels.append(candidate_int)
                    label_set.add(candidate_int)
                    if len(labels) >= topk:
                        return labels

    for aid in recent_unique:
        if aid in label_set:
            continue
        labels.append(aid)
        label_set.add(aid)
        if len(labels) >= topk:
            break
    return labels


def get_cache_path(cache_dir: Path, parquet_file: Path, topk: int) -> Path:
    return cache_dir / f"{parquet_file.stem}.top{topk}.parquet"


def prepare_session_cache_file(
    parquet_file: Path, cache_path: Path, topk: int
) -> None:
    frame = (
        pl.scan_parquet(parquet_file)
        .select(["session", "aid", "ts"])
        .group_by("session", maintain_order=True)
        .agg(
            pl.col("aid")
            .sort_by("ts")
            .reverse()
            .unique(maintain_order=True)
            .head(topk)
            .alias("recent_aids")
        )
        .collect()
    )
    frame.write_parquet(cache_path)


def prepare_session_cache(
    parquet_files: list[Path], cache_dir: Path, topk: int, refresh_cache: bool
) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_files: list[Path] = []
    for parquet_file in tqdm(parquet_files, desc="prepare cache"):
        cache_path = get_cache_path(cache_dir, parquet_file, topk)
        if refresh_cache or not cache_path.exists():
            prepare_session_cache_file(parquet_file, cache_path, topk)
        cache_files.append(cache_path)
    return cache_files


def write_version_submission(
    *,
    version: str,
    count_info_list: list[Any],
    cache_files: list[Path],
    output_file: Path,
    topk: int,
) -> None:
    with output_file.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["session_type", "labels"])

        for cache_file in tqdm(cache_files, desc=f"submit {version}"):
            frame = pl.read_parquet(cache_file, columns=["session", "recent_aids"])

            for session, recent_aids in frame.iter_rows():
                labels = predict_labels(list(recent_aids), count_info_list, topk)
                label_text = " ".join(str(aid) for aid in labels)
                for event_type in EVENT_TYPES:
                    writer.writerow([f"{session}_{event_type}", label_text])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(args.test_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.test_dir}")

    cache_dir = args.cache_dir or (args.output_dir / ".session_cache")
    cache_files = prepare_session_cache(
        parquet_files=parquet_files,
        cache_dir=cache_dir,
        topk=args.topk,
        refresh_cache=args.refresh_cache,
    )

    for version in args.versions:
        count_info_path = get_count_info_path(args.processed_root, version)
        if not count_info_path.exists():
            raise FileNotFoundError(f"Missing count_info.pkl for version={version}: {count_info_path}")

        count_info_list = load_count_info(count_info_path)
        output_file = args.output_dir / f"submission_{version}.csv"
        write_version_submission(
            version=version,
            count_info_list=count_info_list,
            cache_files=cache_files,
            output_file=output_file,
            topk=args.topk,
        )
        print(f"Generated: {output_file}")


if __name__ == "__main__":
    main()
