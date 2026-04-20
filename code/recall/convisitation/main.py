from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from code.utils import LOG_DIR, ensure_dir, log_duration, setup_logger

# OTTO 原始 type 列可能是整数编码，也可能已经是字符串；这里统一映射到名字。
TYPE_TO_NAME = {
    0: "clicks",
    1: "carts",
    2: "orders",
    "clicks": "clicks",
    "carts": "carts",
    "orders": "orders",
}

# 常用权重版本预设。run.sh 里会把最终权重显式传进来，这里主要保留版本语义。
WEIGHT_PRESETS = {
    "base": {"clicks": 1, "carts": 3, "orders": 6},
    "default": {"clicks": 1, "carts": 3, "orders": 6},
    "v13": {"clicks": 0, "carts": 3, "orders": 6},
    "v14": {"clicks": 1, "carts": 9, "orders": 1},
    "v15": {"clicks": 1, "carts": 0, "orders": 0},
    "v17": {"clicks": 1, "carts": 9, "orders": 6},
    "v18": {"clicks": 1, "carts": 15, "orders": 20},
    "v19": {"clicks": 1, "carts": 9, "orders": 1},
}

DEFAULT_MAX_WORKERS = 4

WORK_LOOKBACK = 2
WORK_RECENT_THRESHOLD: int | None = None
WORK_WEIGHTS: dict[str, int] = {}
WORK_TMP_DIR = ""


def parse_args() -> argparse.Namespace:
    # 这个入口由 run.sh 驱动，因此参数尽量和 run.sh 保持一一对应。
    parser = argparse.ArgumentParser(description="Build covisitation matrix for OTTO.")
    parser.add_argument("--mode", choices=["train", "submit"], required=True)
    parser.add_argument("--exp-name", default="baseline")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-version", default="base")
    parser.add_argument("--click-weight", type=int, required=True)
    parser.add_argument("--cart-weight", type=int, required=True)
    parser.add_argument("--order-weight", type=int, required=True)
    parser.add_argument("--n-lookback", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def get_n_lookback(weight_version: str, override: int | None) -> int:
    # 允许命令行显式覆盖，否则沿用旧实验版本的默认窗口。
    if override is not None:
        return override
    if weight_version == "v16":
        return 5
    if weight_version == "v17":
        return 20
    return 2


def get_topk(weight_version: str, override: int | None) -> int:
    # v19/v20 保留更多候选，其余版本默认 top100。
    if override is not None:
        return override
    if weight_version in {"v19", "v20"}:
        return 300
    return 100


def should_use_recent_window(weight_version: str) -> bool:
    # v12 只保留最近 14 天行为。
    return weight_version == "v12"


def get_recent_threshold(parquet_files: list[Path]) -> int | None:
    # 纯 polars 路线：用 lazy scan 直接算全局 max(ts)，避免重复写 parquet backend 分支。
    max_ts_df = (
        pl.scan_parquet([str(path) for path in parquet_files])
        .select(pl.col("ts").max().alias("max_ts"))
        .collect()
    )
    if max_ts_df.is_empty():
        return None
    max_ts = max_ts_df.item(0, "max_ts")
    return max_ts - 60 * 60 * 24 * 1000 * 14


def get_n_workers(override: int | None) -> int:
    # 进程数优先级：命令行 > 环境变量 > 默认值。
    if override is not None:
        return max(1, override)
    cpu_count = os.cpu_count() or 1
    env_workers = os.getenv("COVISIT_N_WORKERS")
    if env_workers:
        return max(1, int(env_workers))
    return max(1, min(cpu_count, DEFAULT_MAX_WORKERS))


def discover_parquet_files(input_dir: Path) -> list[Path]:
    # 当前项目既支持传 raw/validation 这样的父目录，也支持直接传 parquet 目录。
    if input_dir.is_file():
        return [input_dir]

    train_dir = input_dir / "train_parquet"
    test_dir = input_dir / "test_parquet"

    parquet_files: list[Path] = []
    if train_dir.exists():
        parquet_files.extend(sorted(train_dir.glob("*.parquet")))
    if test_dir.exists():
        parquet_files.extend(sorted(test_dir.glob("*.parquet")))
    if not parquet_files:
        parquet_files.extend(sorted(input_dir.glob("*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")
    return parquet_files


def read_parquet_file(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    # 统一走 polars 读 parquet，但避免调用 to_pandas()，因为那会额外要求 pyarrow。
    frame = pl.read_parquet(path, columns=columns)
    return pd.DataFrame(frame.to_dict(as_series=False))


def normalize_type_column(series: pd.Series) -> pd.Series:
    # type 映射失败时直接报错，避免静默把未知类型变成 NaN。
    mapped = series.map(TYPE_TO_NAME)
    if mapped.isna().any():
        unknown_types = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unknown interaction types: {unknown_types}")
    return mapped


def prepare_frame(
    df: pd.DataFrame,
    *,
    recent_threshold: int | None,
    weights: dict[str, int],
) -> pd.DataFrame:
    # 这里做轻量预处理：按需截最近 14 天、把 type 映射成权重，并按 session/ts 排序。
    if recent_threshold is not None:
        df = df[df["ts"] > recent_threshold]

    if df.empty:
        return df

    df = df.copy()
    df["type_name"] = normalize_type_column(df["type"])
    df["weight"] = df["type_name"].map(weights).astype("int16")
    return (
        df[["session", "ts", "aid", "weight"]]
        .sort_values(["session", "ts"], kind="stable")
        .reset_index(drop=True)
    )


def count_covisitation(df: pd.DataFrame, n_lookback: int) -> pd.DataFrame:
    # 对一个 parquet shard 内的数据计算共现：
    # 当前商品的得分 += 它之前 n 步内商品的共现分数，分数取当前交互行为的权重。
    if len(df) < 2:
        return pd.DataFrame(columns=["aid_key", "aid_future", "score"])

    sessions = df["session"].to_numpy()
    aids = df["aid"].to_numpy()
    weights = df["weight"].to_numpy(dtype=np.int16)

    pair_frames: list[pd.DataFrame] = []
    for lag in range(1, n_lookback + 1):
        # 通过 numpy 切片构造 “lag 步前商品 -> 当前商品” 的配对，
        # 并过滤掉跨 session 的配对。
        current_sessions = sessions[lag:]
        previous_sessions = sessions[:-lag]
        mask = current_sessions == previous_sessions
        if not mask.any():
            continue

        scores = weights[lag:][mask]
        nonzero_mask = scores != 0
        if not nonzero_mask.any():
            continue

        pair_frames.append(
            pd.DataFrame(
                {
                    "aid_key": aids[:-lag][mask][nonzero_mask],
                    "aid_future": aids[lag:][mask][nonzero_mask],
                    "score": scores[nonzero_mask],
                }
            )
        )

    if not pair_frames:
        return pd.DataFrame(columns=["aid_key", "aid_future", "score"])

    pairs = pd.concat(pair_frames, ignore_index=True)
    agg = (
        pairs.groupby(["aid_key", "aid_future"], as_index=False, sort=False)
        .agg(score=("score", "sum"))
    )
    agg["aid_key"] = agg["aid_key"].astype("int64")
    agg["aid_future"] = agg["aid_future"].astype("int64")
    agg["score"] = agg["score"].astype("int32")
    return agg


def init_worker(
    n_lookback: int,
    recent_threshold: int | None,
    weights: dict[str, int],
    tmp_dir: str,
) -> None:
    # 多进程 worker 初始化共享只读配置，避免每个任务都重复传大对象。
    global WORK_LOOKBACK, WORK_RECENT_THRESHOLD, WORK_WEIGHTS, WORK_TMP_DIR
    WORK_LOOKBACK = n_lookback
    WORK_RECENT_THRESHOLD = recent_threshold
    WORK_WEIGHTS = weights
    WORK_TMP_DIR = tmp_dir


def process_shard(parquet_path_str: str) -> str | None:
    # 每个进程只处理一个 parquet shard，先局部聚合，再把中间结果落盘。
    parquet_path = Path(parquet_path_str)
    shard_df = read_parquet_file(parquet_path, columns=["session", "ts", "aid", "type"])
    shard_df = prepare_frame(
        shard_df,
        recent_threshold=WORK_RECENT_THRESHOLD,
        weights=WORK_WEIGHTS,
    )
    if len(shard_df) < 2:
        return None

    agg_df = count_covisitation(shard_df, WORK_LOOKBACK)
    if agg_df.empty:
        return None

    shard_output = Path(WORK_TMP_DIR) / f"{parquet_path.stem}.pkl"
    agg_df.to_pickle(shard_output)
    return str(shard_output)


def merge_partial_counts(partial_files: list[Path]) -> dict[tuple[int, int], int]:
    # 把各个 shard 的局部共现分数合并成全局 pair -> score 字典。
    merged: dict[tuple[int, int], int] = defaultdict(int)
    for partial_file in tqdm(partial_files, desc="merge partial counts"):
        partial_df = pd.read_pickle(partial_file)
        for aid_key, aid_future, score in partial_df.itertuples(index=False):
            merged[(int(aid_key), int(aid_future))] += int(score)
    return merged


def build_count_info_list(
    merged_scores: dict[tuple[int, int], int],
    topk: int,
) -> list[Any]:
    # 最终输出格式沿用原始比赛代码风格：
    # list[aid_key] = [候选 aid 列表, 对应 score 列表]
    by_aid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    max_aid = -1

    for (aid_key, aid_future), score in merged_scores.items():
        by_aid[aid_key].append((aid_future, score))
        if aid_key > max_aid:
            max_aid = aid_key

    if max_aid < 0:
        return []

    count_info_list: list[Any] = [-1] * (max_aid + 1)
    for aid_key, candidates in tqdm(by_aid.items(), desc="build topk"):
        candidates.sort(key=lambda item: item[1], reverse=True)
        top_candidates = candidates[:topk]
        count_info_list[aid_key] = [
            [aid_future for aid_future, _ in top_candidates],
            [score for _, score in top_candidates],
        ]
    return count_info_list


def write_metadata(
    output_dir: Path,
    *,
    mode: str,
    exp_name: str,
    input_dir: Path,
    weight_version: str,
    weights: dict[str, int],
    n_lookback: int,
    topk: int,
    n_workers: int,
    n_shards: int,
    recent_threshold: int | None,
    output_file: Path,
) -> None:
    # 额外保存一份配置快照，后面查实验结果时不用再反推参数。
    metadata = {
        "mode": mode,
        "exp_name": exp_name,
        "input_dir": str(input_dir),
        "weight_version": weight_version,
        "weights": weights,
        "n_lookback": n_lookback,
        "topk": topk,
        "n_workers": n_workers,
        "n_shards": n_shards,
        "recent_threshold": recent_threshold,
        "output_file": str(output_file),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    # 日志按实验名隔离，方便和其它召回/ranker 日志一起管理。
    logger = setup_logger(
        "convisitation",
        log_dir=LOG_DIR / args.exp_name,
        run_name=f"convisitation_{args.mode}",
    )

    preset_name = args.weight_version if args.weight_version in WEIGHT_PRESETS else "custom"
    logger.info("mode=%s exp=%s", args.mode, args.exp_name)
    logger.info("input_dir=%s output_dir=%s", args.input_dir, args.output_dir)

    weights = {
        "clicks": args.click_weight,
        "carts": args.cart_weight,
        "orders": args.order_weight,
    }
    logger.info("weight_version=%s resolved_weights=%s", preset_name, weights)

    n_lookback = get_n_lookback(args.weight_version, args.n_lookback)
    topk = get_topk(args.weight_version, args.topk)
    n_workers = get_n_workers(args.n_workers)
    logger.info("n_lookback=%s topk=%s n_workers=%s", n_lookback, topk, n_workers)

    ensure_dir(args.output_dir)
    tmp_dir = args.output_dir / "tmp_partials"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    ensure_dir(tmp_dir)

    # 先找出这次要处理的 parquet shard 列表。
    with log_duration(logger, "discover parquet files"):
        parquet_files = discover_parquet_files(args.input_dir)
    logger.info("found %s parquet shards", len(parquet_files))

    recent_threshold = None
    if should_use_recent_window(args.weight_version):
        # v12 需要先算“最近 14 天”的时间边界。
        with log_duration(logger, "compute recent threshold"):
            recent_threshold = get_recent_threshold(parquet_files)
        logger.info("recent threshold=%s", recent_threshold)

    logger.info("building partial covisitation counts...")
    partial_files: list[Path] = []
    init_args = (n_lookback, recent_threshold, weights, str(tmp_dir))

    try:
        if args.max_shards is not None:
            parquet_files = parquet_files[: max(1, args.max_shards)]
            logger.info("debug mode: only processing first %s parquet shard(s)", len(parquet_files))

        if n_workers == 1:
            # 单进程模式下直接串行处理，便于本地调试。
            init_worker(*init_args)
            for parquet_file in tqdm(parquet_files, desc="process shards"):
                partial = process_shard(str(parquet_file))
                if partial:
                    partial_files.append(Path(partial))
        else:
            # 多进程模式按 shard 并行，最后只合并局部统计结果。
            with get_context("fork").Pool(
                processes=n_workers,
                initializer=init_worker,
                initargs=init_args,
            ) as pool:
                iterator = pool.imap_unordered(
                    process_shard, [str(path) for path in parquet_files]
                )
                for partial in tqdm(iterator, total=len(parquet_files), desc="process shards"):
                    if partial:
                        partial_files.append(Path(partial))

        if not partial_files:
            raise ValueError("No covisitation pairs were generated. Check input data and weights.")

        with log_duration(logger, "merge partial covisitation counts"):
            merged_scores = merge_partial_counts(partial_files)
        logger.info("merged pair count=%s", len(merged_scores))

        with log_duration(logger, "build topk candidate list"):
            count_info_list = build_count_info_list(merged_scores, topk)

        # 主输出仍然保存成 pickle，和原来比赛代码保持兼容。
        output_file = args.output_dir / "count_info.pkl"
        with output_file.open("wb") as fp:
            pickle.dump(count_info_list, fp)
        logger.info("saved count info to %s", output_file)

        write_metadata(
            args.output_dir,
            mode=args.mode,
            exp_name=args.exp_name,
            input_dir=args.input_dir,
            weight_version=args.weight_version,
            weights=weights,
            n_lookback=n_lookback,
            topk=topk,
            n_workers=n_workers,
            n_shards=len(parquet_files),
            recent_threshold=recent_threshold,
            output_file=output_file,
        )
        logger.info("saved metadata to %s", args.output_dir / "metadata.json")
    finally:
        # 中间临时文件只用于合并阶段，结束后清理掉。
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
