# Env Usage

`experiments/*.env` 是实验配置文件，只写变量赋值。

最小示例：

```bash
EXP_NAME=baseline
ENABLE_CONVISITATION=1
ENABLE_NN=1
ENABLE_FUSION=1
ENABLE_RANKER=1
CONVIS_WEIGHT_VERSION=base
RUN_NOTES="full baseline pipeline"
```

常用命令：

```bash
bash scripts/run_train.sh --config experiments/baseline.env
bash scripts/run_validation.sh --config experiments/baseline.env
bash scripts/run_submit.sh --config experiments/baseline.env --submission-name baseline.csv
```

共现权重：

```bash
CONVIS_WEIGHT_VERSION=v18
```

或直接写：

```bash
CONVIS_CLICK_WEIGHT=1
CONVIS_CART_WEIGHT=15
CONVIS_ORDER_WEIGHT=20
```

注意：

- 不要写成 `A = 1`，要写 `A=1`
- 正式实验尽量改 `EXP_NAME`，避免覆盖默认 `baseline`
