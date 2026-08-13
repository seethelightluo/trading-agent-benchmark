"""miner_1 2033-05-12 candidate C3: close-to-close return skewness (60d).

Motivation: library has intraday_ret_skew_20 (close/open) and range_skew_20
(range width skew), but not classic close-to-close return skewness. Negative
skew assets (crash-prone) may underperform; positive skew may be rewarded.
Distinct horizon (60d) and return basis.
"""
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel
from miner1_eval_helper import eval_candidate


def panel_fn(prices):
    def f(df, s):
        return df['close'].pct_change().rolling(60, min_periods=40).skew()
    return factor_to_panel(f, prices)


if __name__ == '__main__':
    res = eval_candidate('ret_skew_60', panel_fn)
    print(json.dumps(res, indent=1, default=str))
