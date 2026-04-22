python codes/recall/convisitation/cov.py --mode train --exp-name baseline --click 1 --cart 3 --order 6 --n-lookback 2 --hours 24 --days 14 --topk 100


# only click, only cart, only order
python codes/recall/convisitation/cov.py --mode train --exp-name only_click --click 1 --cart 0 --order 0 --n-lookback 2 --hours 24 --days 14 --topk 100
python codes/recall/convisitation/cov.py --mode train --exp-name only_cart --click 0 --cart 1 --order 0 --n-lookback 2 --hours 24 --days 14 --topk 100
python codes/recall/convisitation/cov.py --mode train --exp-name only_order --click 0 --cart 0 --order 1 --n-lookback 2 --hours 24 --days 14 --topk 100



python codes/recall/convisitation/cov.py --mode train --exp-name try --click 1 --cart 6 --order 3 --n-lookback 2 --hours 24 --days 14 --topk 100

python codes/recall/convisitation/cov.py --mode train --exp-name try --click 1 --cart 6 --order 3 --n-lookback 2 --hours 24 --days 14 --topk 100