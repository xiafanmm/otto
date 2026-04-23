import math
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = Path(__file__).stem
sys.path.insert(0, str(ROOT / "codes"))

from utils.logger import setup_logger  # noqa: E402

TRAIN_DIR = ROOT / "data/raw/validation/train_parquet"
TEST_DIR = ROOT / "data/raw/validation/test_parquet"
OUT_DIR = ROOT / "data/recall/nn"

TYPE_NAMES = ("clicks", "carts", "orders")
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 24 * 3600
MILLISECONDS_PER_SECOND = 1000
METRIC_SMOOTHING_COEF = 0.99
RUN_TEST_TRAIN_FILES = 2
RUN_TEST_INFER_FILES = 1
RUN_TEST_MAX_SESSIONS = 2048


@dataclass
class Args:
    mode: str
    exp_name: str
    n_length: int
    n_label: int
    days: int
    emb_dim: int
    time_emb_dim: int
    batch_size: int
    n_epoch: int
    lr: float
    temperature: float
    n_neg_coef: int
    neg_topk: int
    topk: int
    click: int
    cart: int
    order: int
    run_test: bool


@dataclass
class RuntimeConfig:
    args: Args
    n_aid: int
    pad_aid: int
    h_min: int
    h_max: int
    n_h: int
    pad_h: int
    ts_scale: int
    ts_per_hour: int
    ts_per_day: int
    type_weights: dict[str, int]
    label_types: tuple[tuple[str, int], ...]
    max_label_weight: int
    x_cols: list[str]
    y_cols: list[str]


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Build embedding recall for OTTO.")
    parser.add_argument("--mode", choices=["train", "submit"], required=True)
    parser.add_argument("--exp-name", default="baseline")
    parser.add_argument("--n-length", type=int, default=10)
    parser.add_argument("--n-label", type=int, default=10)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--time-emb-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--n-neg-coef", type=int, default=30)
    parser.add_argument("--neg-topk", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--click", type=int, default=1)
    parser.add_argument("--cart", type=int, default=3)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="smoke mode: limit train/infer shards and sessions, and stop training after 1000 steps",
    )

    namespace = parser.parse_args()
    args = Args(**vars(namespace))
    if min(
        args.n_length,
        args.n_label,
        args.days,
        args.emb_dim,
        args.time_emb_dim,
        args.batch_size,
        args.n_epoch,
        args.n_neg_coef,
        args.neg_topk,
        args.topk,
        args.click,
        args.cart,
        args.order,
    ) <= 0:
        raise ValueError("all integer hyperparameters must be positive")
    if args.lr <= 0 or args.temperature <= 0:
        raise ValueError("--lr and --temperature must be positive")
    return args


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def metric_smoothing(
    metrics: dict[str, float | torch.Tensor],
    smooth: dict[str, float | None],
) -> tuple[str, dict[str, float | None]]:
    parts: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            current = float(value.detach().item())
        else:
            current = float(value)
        previous = smooth[key]
        if previous is None:
            smooth[key] = current
        else:
            smooth[key] = previous * METRIC_SMOOTHING_COEF + current * (
                1.0 - METRIC_SMOOTHING_COEF
            )
        parts.append(f"{key}={smooth[key]:.4f}")
    return ", ".join(parts), smooth


def event_scan(paths: list[Path]) -> pl.LazyFrame:
    return pl.scan_parquet([str(path) for path in paths]).select(
        pl.col("session").cast(pl.Int32),
        pl.col("aid").cast(pl.Int32),
        pl.col("ts").cast(pl.Int64),
        pl.col("type").cast(pl.Utf8),
    )


def scan_runtime_config(args: Args, logger) -> RuntimeConfig:
    train_files = sorted(TRAIN_DIR.glob("*.parquet"))
    test_files = sorted(TEST_DIR.glob("*.parquet"))
    all_files = train_files + test_files
    if not train_files or not test_files:
        raise FileNotFoundError("train/test parquet files are required")

    stats = (
        # h_min_sec = 如果 ts 是秒，最小小时编号
        # h_max_sec = 如果 ts 是秒，最大小时编号

        # h_min_ms = 如果 ts 是毫秒，最小小时编号
        # h_max_ms = 如果 ts 是毫秒，最大小时编号
        event_scan(all_files)
        .select(
            pl.col("aid").max().alias("aid_max"),
            pl.col("ts").min().alias("ts_min"),
            pl.col("ts").max().alias("ts_max"),
            (pl.col("ts") // SECONDS_PER_HOUR).min().alias("h_min_sec"),
            (pl.col("ts") // SECONDS_PER_HOUR).max().alias("h_max_sec"),
            (pl.col("ts") // (SECONDS_PER_HOUR * MILLISECONDS_PER_SECOND)).min().alias("h_min_ms"),
            (pl.col("ts") // (SECONDS_PER_HOUR * MILLISECONDS_PER_SECOND)).max().alias("h_max_ms"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )

    aid_max = int(stats["aid_max"])
    ts_min = int(stats["ts_min"])
    ts_max = int(stats["ts_max"])
    ts_scale = MILLISECONDS_PER_SECOND if ts_max >= 10**12 else 1
    ts_per_hour = SECONDS_PER_HOUR * ts_scale
    ts_per_day = SECONDS_PER_DAY * ts_scale
    if ts_scale == MILLISECONDS_PER_SECOND:
        h_min = int(stats["h_min_ms"])
        h_max = int(stats["h_max_ms"])
        ts_unit = "milliseconds"
    else:
        h_min = int(stats["h_min_sec"])
        h_max = int(stats["h_max_sec"])
        ts_unit = "seconds"
    logger.info(
        "timestamp unit detected: %s, ts_range=[%d, %d], hour bucket h = ts // %d",
        ts_unit,
        ts_min,
        ts_max,
        ts_per_hour,
    )

    n_aid = aid_max + 3
    pad_aid = n_aid - 1
    pad_h = h_max - h_min + 3
    n_h = pad_h + 1

    logger.info(
        "dynamic stats: aid_max=%d n_aid=%d h_min=%d h_max=%d n_h=%d",
        aid_max,
        n_aid,
        h_min,
        h_max,
        n_h,
    )

    x_cols = [f"aid_pre{i}" for i in range(args.n_length)]
    x_cols += [f"h_pre{i}" for i in range(args.n_length)]
    x_cols += [f"weight_pre{i}" for i in range(args.n_length)]
    y_cols = [f"aid_label{i}" for i in range(args.n_label)]
    y_cols += [f"weight_label{i}" for i in range(args.n_label)]

    type_weights = {"clicks": args.click, "carts": args.cart, "orders": args.order}
    label_types = tuple((name, type_weights[name]) for name in TYPE_NAMES)
    max_label_weight = max(type_weights.values())

    return RuntimeConfig(
        args=args,
        n_aid=n_aid,
        pad_aid=pad_aid,
        h_min=h_min,
        h_max=h_max,
        n_h=n_h,
        pad_h=pad_h,
        ts_scale=ts_scale,
        ts_per_hour=ts_per_hour,
        ts_per_day=ts_per_day,
        type_weights=type_weights,
        label_types=label_types,
        max_label_weight=max_label_weight,
        x_cols=x_cols,
        y_cols=y_cols,
    )


def get_train_files(args: Args) -> list[Path]:
    files = sorted(TRAIN_DIR.glob("*.parquet"))
    if args.mode == "submit":
        files += sorted(TEST_DIR.glob("*.parquet"))
    if args.run_test:
        files = files[:RUN_TEST_TRAIN_FILES]
    return files


def get_inference_files(args: Args) -> list[Path]:
    files = sorted(TEST_DIR.glob("*.parquet"))
    if args.run_test:
        files = files[:RUN_TEST_INFER_FILES]
    return files


def scan_train_cutoff_ts(files: list[Path], cfg: RuntimeConfig, logger) -> int:
    ts_max = (
        event_scan(files)
        .select(pl.col("ts").max().alias("ts_max"))
        .collect(engine="streaming")
        .item()
    )
    cutoff_ts = int(ts_max) - cfg.args.days * cfg.ts_per_day
    logger.info("training cutoff ts >= %d (last %d days)", cutoff_ts, cfg.args.days)
    return cutoff_ts


def load_frame(path: Path, cutoff_ts: int | None = None) -> pl.DataFrame:
    lf = event_scan([path])
    if cutoff_ts is not None:
        lf = lf.filter(pl.col("ts") >= cutoff_ts)
    return lf.collect(engine="streaming")


def empty_train_tensor(cfg: RuntimeConfig) -> torch.Tensor:
    width = len(cfg.x_cols) + len(cfg.y_cols)
    return torch.empty((0, width), dtype=torch.int32)


def empty_infer_tensor(cfg: RuntimeConfig) -> tuple[np.ndarray, torch.Tensor]:
    return np.empty((0,), dtype=np.int32), torch.empty(
        (0, len(cfg.x_cols)), dtype=torch.int32
    )


def preprocess_df(
    df: pl.DataFrame,
    cfg: RuntimeConfig,
    *,
    with_label: bool,
) -> torch.Tensor | tuple[np.ndarray, torch.Tensor]:
    if df.height == 0:
        return empty_train_tensor(cfg) if with_label else empty_infer_tensor(cfg)

    df = (
        df.sort(["session", "ts"])
        .with_columns(
            pl.col("type")
            .replace_strict(cfg.type_weights, default=0, return_dtype=pl.Int32)
            .alias("weight"),
            ((pl.col("ts") // cfg.ts_per_hour) - cfg.h_min)
            .clip(0, cfg.pad_h)
            .cast(pl.Int32)
            .alias("h"),
        )
    )

    history_exprs: list[pl.Expr] = []
    for i in range(cfg.args.n_length):
        history_exprs.extend(
            [
                pl.col("aid")
                .shift(i)
                .over("session")
                .fill_null(cfg.pad_aid)
                .cast(pl.Int32)
                .alias(f"aid_pre{i}"),
                pl.col("h")
                .shift(i)
                .over("session")
                .fill_null(cfg.pad_h)
                .cast(pl.Int32)
                .alias(f"h_pre{i}"),
                pl.col("weight")
                .shift(i)
                .over("session")
                .fill_null(0)
                .cast(pl.Int32)
                .alias(f"weight_pre{i}"),
            ]
        )
    df = df.with_columns(history_exprs)

    if with_label:
        label_exprs: list[pl.Expr] = []
        for i in range(cfg.args.n_label):
            label_exprs.extend(
                [
                    pl.col("aid")
                    .shift(-(i + 1))
                    .over("session")
                    .fill_null(cfg.pad_aid)
                    .cast(pl.Int32)
                    .alias(f"aid_label{i}"),
                    pl.col("weight")
                    .shift(-(i + 1))
                    .over("session")
                    .fill_null(0)
                    .cast(pl.Int32)
                    .alias(f"weight_label{i}"),
                ]
            )
        df = df.with_columns(label_exprs)

        arr = np.asarray(
            df.filter(pl.col("aid_label0") != cfg.pad_aid)
            .select(cfg.x_cols + cfg.y_cols)
            .to_numpy(),
            dtype=np.int32,
        )
        if arr.size == 0:
            return empty_train_tensor(cfg)
        return torch.from_numpy(arr)

    last_rows = (
        df.group_by("session", maintain_order=True)
        .last()
        .select("session", *cfg.x_cols)
        .sort("session")
    )
    if cfg.args.run_test and last_rows.height > RUN_TEST_MAX_SESSIONS:
        last_rows = last_rows.head(RUN_TEST_MAX_SESSIONS)
    if last_rows.height == 0:
        return empty_infer_tensor(cfg)

    sessions = np.asarray(last_rows["session"].to_numpy(), dtype=np.int32)
    x_array = np.asarray(last_rows.select(cfg.x_cols).to_numpy(), dtype=np.int32)
    return sessions, torch.from_numpy(x_array)


class Encoder(nn.Module):
    def __init__(self, cfg: RuntimeConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.n_aid, cfg.args.emb_dim, padding_idx=cfg.pad_aid)
        self.emb_h = nn.Embedding(
            cfg.n_h,
            cfg.args.time_emb_dim,
            padding_idx=cfg.pad_h,
        )
        self.emb_label_type = nn.Embedding(
            cfg.max_label_weight + 1, cfg.args.emb_dim, padding_idx=0
        )

        inp_dim = (
            cfg.args.emb_dim + 2 + 1 + cfg.args.time_emb_dim
        ) * cfg.args.n_length
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Linear(inp_dim, inp_dim),
            nn.GELU(),
            nn.BatchNorm1d(inp_dim),
            nn.Linear(inp_dim, inp_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(inp_dim // 2),
            nn.Linear(inp_dim // 2, cfg.args.emb_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.emb.weight, -1.0, 1.0)
        nn.init.uniform_(self.emb_h.weight, -1.0, 1.0)
        nn.init.uniform_(self.emb_label_type.weight, -1.0, 1.0)
        with torch.no_grad():
            self.emb.weight[self.cfg.pad_aid].zero_()
            self.emb_h.weight[self.cfg.pad_h].zero_()
            self.emb_label_type.weight[0].zero_()

    def do_emb(self, aids: torch.Tensor) -> torch.Tensor:
        return self.emb(aids)

    def _calc_query_emb(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.reshape(-1, 3, self.cfg.args.n_length).transpose(1, 2)

        aid = x_batch[:, :, 0]
        hour = x_batch[:, :, 1]
        weight = x_batch[:, :, 2].float().unsqueeze(2)

        aid_emb = self.emb(aid)
        hour_float = hour.float()
        phase = (hour_float % 24.0) / 24.0 * (2.0 * math.pi)
        cyc_feat = torch.stack([torch.sin(phase), torch.cos(phase)], dim=2)

        hour_max = self.cfg.pad_h
        hour_emb = self.emb_h(torch.clamp(hour, min=0, max=hour_max))
        hour_emb = hour_emb + self.emb_h(torch.clamp(hour + 1, min=0, max=hour_max))
        hour_emb = hour_emb + self.emb_h(torch.clamp(hour + 2, min=0, max=hour_max))

        x = torch.cat([aid_emb, cyc_feat, weight, hour_emb], dim=2)
        mask = ~aid.eq(self.cfg.pad_aid).unsqueeze(2)
        x = x * mask
        x = x.reshape(len(x), -1)
        return self.mlp(x)

    def calc_query_emb(
        self,
        x_batch: torch.Tensor,
        label_type: torch.Tensor,
    ) -> torch.Tensor:
        return self._calc_query_emb(x_batch) + self.emb_label_type(label_type)

    def forward(self, xy_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_width = len(self.cfg.x_cols)
        aid_label_width = self.cfg.args.n_label

        x_batch = xy_batch[:, :x_width]
        y_batch = xy_batch[:, x_width : x_width + aid_label_width]
        label_type = xy_batch[:, x_width + aid_label_width :]

        batch_size = xy_batch.shape[0]
        neg_count = min(
            self.cfg.n_aid - 1,
            max(self.cfg.args.neg_topk, self.cfg.args.n_neg_coef * batch_size),
        )
        neg_aids = torch.randperm(self.cfg.n_aid - 1, device=xy_batch.device)[:neg_count]

        query_emb = self._calc_query_emb(x_batch)
        query_emb = query_emb.unsqueeze(1) + self.emb_label_type(label_type)

        pos_emb = self.do_emb(y_batch)
        neg_emb = self.do_emb(neg_aids)

        query_emb = F.normalize(query_emb, dim=2, eps=1e-6)
        pos_emb = F.normalize(pos_emb, dim=2, eps=1e-6)
        neg_emb = F.normalize(neg_emb, dim=1, eps=1e-6)

        cossim_pos_all = torch.sum(query_emb * pos_emb, dim=2)
        pos_mask = y_batch.eq(self.cfg.pad_aid)
        cossim_pos_min = cossim_pos_all.masked_fill(pos_mask, 1.0).min(dim=1).values

        label_weight = label_type.float()
        cossim_pos_mean = torch.sum(cossim_pos_all * label_weight, dim=1) / torch.clamp(
            label_weight.sum(dim=1), min=1.0
        )
        cossim_pos = (cossim_pos_min + cossim_pos_mean) / 2.0

        cossim_neg = torch.matmul(query_emb[:, 0], neg_emb.T)
        hard_neg_k = min(self.cfg.args.neg_topk, cossim_neg.shape[1])
        cossim_neg = torch.topk(cossim_neg, hard_neg_k, dim=1).values

        logits = torch.cat([cossim_pos.unsqueeze(1), cossim_neg], dim=1)
        logits = logits / self.cfg.args.temperature
        log_prob = F.log_softmax(logits, dim=1)
        loss = -log_prob[:, 0].mean()
        acc = (log_prob.argmax(dim=1) == 0).float().mean()
        return loss, acc


class MyOptimizer:
    def __init__(self, model: nn.Module, cfg: RuntimeConfig, logger):
        params_main: list[nn.Parameter] = []
        params_emb: list[nn.Parameter] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("emb.weight"):
                params_emb.append(param)
            else:
                params_main.append(param)

        self.opt = torch.optim.AdamW(params_main, lr=cfg.args.lr)
        self.opt_em = torch.optim.AdamW(params_emb, lr=cfg.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.5)
        self.scheduler_em = torch.optim.lr_scheduler.StepLR(
            self.opt_em,
            step_size=1,
            gamma=0.5,
        )
        self.count = 0
        self.logger = logger

    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)
        self.opt_em.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.opt.step()
        self.opt_em.step()

    def step_scheduler(self) -> bool:
        self.scheduler.step()
        self.scheduler_em.step()
        self.count += 1
        self.logger.info("scheduler stepped: count=%d", self.count)
        return self.count > 4


def train_one_epoch(
    model: nn.Module,
    optimizer: MyOptimizer,
    files: list[Path],
    cutoff_ts: int,
    cfg: RuntimeConfig,
    logger,
) -> tuple[nn.Module, bool]:
    model.train()
    device = next(model.parameters()).device
    smooth = {"loss": None, "acc": None}
    smooth_acc_history = [0.0]
    step = 0
    should_stop = False
    loaded_shards = 0
    total_rows = 0

    file_order = np.random.permutation(len(files))
    for file_idx in file_order:
        path = files[int(file_idx)]
        df = load_frame(path, cutoff_ts=cutoff_ts)
        xy = preprocess_df(df, cfg, with_label=True)
        del df

        if xy.shape[0] == 0:
            logger.info("skip empty training shard: %s", path.name)
            del xy
            continue
        loaded_shards += 1
        total_rows += xy.shape[0]
        logger.info("loaded %s rows=%d", path.name, xy.shape[0])

        if xy.shape[0] < 2:
            del xy
            continue

        n_split = max(1, math.ceil(xy.shape[0] / cfg.args.batch_size))
        for indices in np.array_split(np.random.permutation(xy.shape[0]), n_split):
            if len(indices) < 2:
                continue

            xy_batch = xy[indices].to(device=device, non_blocking=True).long()
            optimizer.zero_grad()
            loss, acc = model(xy_batch)
            loss.backward()
            optimizer.step()

            step += 1
            metrics_str, smooth = metric_smoothing({"loss": loss, "acc": acc}, smooth)
            if step % 100 == 0:
                logger.info("step %d: %s", step, metrics_str)

            if cfg.args.run_test and step > 1000:
                should_stop = True
                break

            if step % 2000 == 0 and smooth["acc"] is not None:
                current_acc = float(smooth["acc"])
                if current_acc <= smooth_acc_history[-1]:
                    should_stop = optimizer.step_scheduler()
                    if should_stop:
                        break
                smooth_acc_history.append(current_acc)
                logger.info(
                    "smoothed acc: %.4f -> %.4f",
                    smooth_acc_history[-2],
                    smooth_acc_history[-1],
                )
        if should_stop:
            del xy
            break
        del xy

    logger.info("training shards=%d total_rows=%d", loaded_shards, total_rows)
    if loaded_shards == 0:
        raise ValueError("no training samples after preprocessing")
    return model, should_stop


def item_chunk_topk(
    module: nn.Module,
    query_emb: torch.Tensor,
    cfg: RuntimeConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_size = 200_000 if device.type == "cuda" else 100_000
    topk = min(cfg.args.topk, cfg.n_aid - 1)
    running_scores: torch.Tensor | None = None
    running_ids: torch.Tensor | None = None

    for start in range(0, cfg.n_aid - 1, chunk_size):
        end = min(cfg.n_aid - 1, start + chunk_size)
        aid_ids = torch.arange(start, end, device=device, dtype=torch.long)
        item_emb = F.normalize(module.do_emb(aid_ids).detach(), dim=1, eps=1e-6)
        chunk_scores = torch.matmul(query_emb, item_emb.T)
        chunk_ids = aid_ids.unsqueeze(0).expand(query_emb.shape[0], -1)

        if running_scores is None or running_ids is None:
            running_k = min(topk, chunk_scores.shape[1])
            running_scores, local_idx = torch.topk(chunk_scores, running_k, dim=1)
            running_ids = chunk_ids.gather(1, local_idx)
            continue

        merged_scores = torch.cat([running_scores, chunk_scores], dim=1)
        merged_ids = torch.cat([running_ids, chunk_ids], dim=1)
        running_k = min(topk, merged_scores.shape[1])
        running_scores, local_idx = torch.topk(merged_scores, running_k, dim=1)
        running_ids = merged_ids.gather(1, local_idx)

    if running_scores is None or running_ids is None:
        raise ValueError("no item embeddings available for inference")
    return running_scores, running_ids


def infer_one_target(
    model: nn.Module,
    x_tensor: torch.Tensor,
    label_value: int,
    cfg: RuntimeConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    module = unwrap_model(model)
    query_batch_size = 64 if device.type == "cuda" else 32
    candidates_parts: list[np.ndarray] = []
    scores_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, x_tensor.shape[0], query_batch_size):
            end = min(x_tensor.shape[0], start + query_batch_size)
            x_batch = x_tensor[start:end].to(device=device, non_blocking=True).long()
            label_type = torch.full(
                (x_batch.shape[0],),
                label_value,
                device=device,
                dtype=torch.long,
            )
            query_emb = module.calc_query_emb(x_batch, label_type)
            query_emb = F.normalize(query_emb, dim=1, eps=1e-6)
            top_scores, top_ids = item_chunk_topk(module, query_emb, cfg, device)
            candidates_parts.append(top_ids.cpu().numpy().astype(np.int32, copy=False))
            scores_parts.append(top_scores.cpu().numpy().astype(np.float32, copy=False))

    return np.vstack(candidates_parts), np.vstack(scores_parts)


def inference(
    model: nn.Module,
    files: list[Path],
    cfg: RuntimeConfig,
    logger,
    device: torch.device,
) -> pl.DataFrame:
    logger.info("start inference on %d test shards", len(files))
    model.eval()
    logger.info("streaming item embeddings in chunks during top-k search")

    frames: list[pl.DataFrame] = []
    for path in files:
        df = load_frame(path)
        sessions, x_tensor = preprocess_df(df, cfg, with_label=False)
        if x_tensor.shape[0] == 0:
            logger.info("skip empty inference shard: %s", path.name)
            continue

        logger.info("infer shard %s sessions=%d", path.name, len(sessions))
        for event_type, label_value in cfg.label_types:
            candidates, scores = infer_one_target(
                model,
                x_tensor,
                label_value,
                cfg,
                device,
            )
            frames.append(
                pl.DataFrame(
                    {
                        "type": pl.Series(
                            "type",
                            [event_type] * len(sessions),
                            dtype=pl.Utf8,
                        ),
                        "session": pl.Series("session", sessions, dtype=pl.Int32),
                        "candidates": pl.Series(
                            "candidates",
                            candidates,
                            dtype=pl.List(pl.Int32),
                        ),
                        "scores": pl.Series(
                            "scores",
                            scores,
                            dtype=pl.List(pl.Float32),
                        ),
                    }
                )
            )

    if not frames:
        return pl.DataFrame(
            schema={
                "type": pl.Utf8,
                "session": pl.Int32,
                "candidates": pl.List(pl.Int32),
                "scores": pl.List(pl.Float32),
            }
        )
    return pl.concat(frames)


def save_model(model: nn.Module, out_dir: Path, logger) -> None:
    state_dict = {
        key: value.detach().cpu()
        for key, value in unwrap_model(model).state_dict().items()
    }
    model_path = out_dir / "model.pt"
    torch.save(state_dict, model_path)
    logger.info("saved model state_dict to %s", model_path)


def main():
    args = parse_args()
    logger = setup_logger(args.exp_name, stage="nn", run_name=SCRIPT_NAME)
    logger.info("*" * 150)

    logger.info("args: %s", args)
    logger.info("load_train_data from: %s", TRAIN_DIR)
    logger.info("load_test_data from: %s", TEST_DIR)
    logger.info("=" * 150)

    cfg = scan_runtime_config(args, logger)
    train_files = get_train_files(args)
    infer_files = get_inference_files(args)
    logger.info(
        "train shards=%d infer shards=%d run_test=%s",
        len(train_files),
        len(infer_files),
        args.run_test,
    )

    cutoff_ts = scan_train_cutoff_ts(train_files, cfg, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    logger.info("device=%s", device)

    model = nn.DataParallel(Encoder(cfg)).to(device)
    optimizer = MyOptimizer(model, cfg, logger)

    should_stop = False
    for epoch in range(args.n_epoch):
        logger.info("epoch %d/%d", epoch + 1, args.n_epoch)
        model, should_stop = train_one_epoch(
            model,
            optimizer,
            train_files,
            cutoff_ts,
            cfg,
            logger,
        )
        if should_stop:
            logger.info("early stop triggered")
            break

    out_dir = OUT_DIR / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, out_dir, logger)

    recall_df = inference(model, infer_files, cfg, logger, device)
    out_path = out_dir / f"{args.mode}.parquet"
    recall_df.write_parquet(out_path)
    logger.info("saved recall parquet to %s", out_path)


if __name__ == "__main__":
    main()
