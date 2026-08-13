"""miner_1 2033-05-12 candidate C4: volatility term ratio 60d/20d.

Motivation: vol_of_vol20x60 captures dispersion of 20d vol; the ratio
vol(60)/vol(20) instead captures the current vol TREND direction (rising
vol regime when >1). Interpretable, low-turnover vol-timing signal.
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
        r = df['close'].pct_change()
        v20 = r.rolling(20).std()
        v60 = r.rolling(60).std()
        return (v60 / v20).replace([np.inf, -np.inf], np.nan)
    return factor_to_panel(f, prices)


if __name__ == '__main__':
    res = eval_candidate('vol_ratio_60_20', panel_fn)
    print(json.dumps(res, indent=1, default=str))
