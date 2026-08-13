"""miner_1 2033-05-12 candidate C8: momentum term spread 20d - 60d.

Motivation: mom_accel_60_120 is 60-120 acceleration; a shorter-term spread
(20d minus 60d momentum, both skip-5) isolates the recent-trend vs
intermediate-trend gap. Positive spread = acceleration at short end.
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
        c = df['close']
        m20 = c.shift(5) / c.shift(25) - 1.0
        m60 = c.shift(5) / c.shift(65) - 1.0
        return m20 - m60
    return factor_to_panel(f, prices)


if __name__ == '__main__':
    res = eval_candidate('term_mom_20_60', panel_fn)
    print(json.dumps(res, indent=1, default=str))
