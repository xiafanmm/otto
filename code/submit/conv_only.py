from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm


EVENT_TYPES = ("clicks", "carts", "orders")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conv-only OTTO submissions from count_info.pkl files."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(
            "/Users/xiafan/Desktop/otto_self/data/raw/parquet_chunks/test_parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/xiafan/Desktop/otto_self/data/submissions/conv_only"
        ),
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["base", "v12", "v13", "v15", "v16", "v17", "v18", "v19"],
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/Users/xiafan/Desktop/otto_self/data/processed"),
    )
    parser.add_argument("--topk", type=int, default=20)
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


def get_recent_unique_aids(aids: list[int]) -> list[int]:
    seen: set[int] = set()
    recent_unique: list[int] = []
    for aid in reversed(aids):
        if aid in seen:
            continue
        seen.add(aid)
        recent_unique.append(aid)
    return recent_unique


def predict_labels(
    aids: list[int], count_info_list: list[Any], topk: int
) -> list[int]:
    recent_unique = get_recent_unique_aids(aids)
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


def write_version_submission(
    *,
    version: str,
    count_info_list: list[Any],
    parquet_files: list[Path],
    output_file: Path,
    topk: int,
) -> None:
    with output_file.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["session_type", "labels"])

        for parquet_file in tqdm(parquet_files, desc=f"submit {version}"):
            frame = (
                pl.read_parquet(parquet_file, columns=["session", "aid", "ts"])
                .sort(["session", "ts"])
                .group_by("session", maintain_order=True)
                .agg(pl.col("aid"))
            )

            for session, aids in frame.iter_rows():
                labels = predict_labels(list(aids), count_info_list, topk)
                label_text = " ".join(str(aid) for aid in labels)
                for event_type in EVENT_TYPES:
                    writer.writerow([f"{session}_{event_type}", label_text])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(args.test_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.test_dir}")

    for version in args.versions:
        count_info_path = get_count_info_path(args.processed_root, version)
        if not count_info_path.exists():
            raise FileNotFoundError(f"Missing count_info.pkl for version={version}: {count_info_path}")

        count_info_list = load_count_info(count_info_path)
        output_file = args.output_dir / f"submission_{version}.csv"
        write_version_submission(
            version=version,
            count_info_list=count_info_list,
            parquet_files=parquet_files,
            output_file=output_file,
            topk=args.topk,
        )
        print(f"Generated: {output_file}")


if __name__ == "__main__":
    main()
