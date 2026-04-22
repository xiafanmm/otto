# OTTO co-visitation 实验清单
# 从项目根目录执行：bash codes/recall/convisitation/run.sh
# 输出：data/recall/covisit/<exp_name>/train.parquet
# 日志：data/logs/<exp_name>/cov_<timestamp>.log

SCRIPT=codes/recall/convisitation/cov.py
COMMON="--mode train --days 14 --topk 100"

# ---- time_decay A/B（权重 1:3:6，lookback=2, hours=24）----
python $SCRIPT $COMMON --exp-name weights_1_3_6_nodecay --click 1 --cart 3 --order 6 --n-lookback 2 --hours 24 --no-time-decay
python $SCRIPT $COMMON --exp-name weights_1_3_6_decay   --click 1 --cart 3 --order 6 --n-lookback 2 --hours 24

# ---- 权重 A/B ----
python $SCRIPT $COMMON --exp-name weights_1_1_1 --click 1 --cart 1 --order 1 --n-lookback 2 --hours 24
python $SCRIPT $COMMON --exp-name weights_1_6_3 --click 1 --cart 6 --order 3 --n-lookback 2 --hours 24

# ---- 消融：单一事件类型 ----
python $SCRIPT $COMMON --exp-name only_click --click 1 --cart 0 --order 0 --n-lookback 2 --hours 24
python $SCRIPT $COMMON --exp-name only_cart  --click 0 --cart 1 --order 0 --n-lookback 2 --hours 24
python $SCRIPT $COMMON --exp-name only_order --click 0 --cart 0 --order 1 --n-lookback 2 --hours 24

# ---- hours 扫描（hours=24 对照组即 weights_1_3_6_decay）----
python $SCRIPT $COMMON --exp-name hours_1 --click 1 --cart 3 --order 6 --n-lookback 2 --hours 1
python $SCRIPT $COMMON --exp-name hours_6 --click 1 --cart 3 --order 6 --n-lookback 2 --hours 6

# ---- n_lookback 扫描（lookback=2 对照组即 weights_1_3_6_decay）----
python $SCRIPT $COMMON --exp-name lookback_10 --click 1 --cart 3 --order 6 --n-lookback 10 --hours 24
python $SCRIPT $COMMON --exp-name lookback_30 --click 1 --cart 3 --order 6 --n-lookback 30 --hours 24

# ---- 最终 submit（选定最优参数后手动打开）----
# python $SCRIPT --mode submit --exp-name final --click 1 --cart 3 --order 6 --n-lookback 30 --hours 24 --days 14 --topk 100
