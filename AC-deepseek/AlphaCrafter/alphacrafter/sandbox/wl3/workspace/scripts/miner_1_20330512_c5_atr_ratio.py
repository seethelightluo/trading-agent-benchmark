"""miner_1 2033-05-12 candidate C5: ATR ratio 10/50 (true-range vol trend).

Motivation: classic range-based volatility trend using average true range.
ATR(10)/ATR(50) > 1 signals expanding realized range (vol up). Distinct from
return-based vol factors; uses high/low/close structure.
"""
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel
from miner1_eval_helper import eval_candidate


def atr(df, w):
    pc = df['close'].shift(1)
    tr = pd.concat([(df['high'] - df['low']).rename('a'),
                    (df['high'] - pc).abs().rename('b'),
                    (df['low'] - pc).abs().rename('c')], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=int(w * 0.5)).mean()


def panel_fn(prices):
    def f(df, s):
        a10 = atr(df, 10)
        a50 = atr(df, 50)
        return (a10 / a50).replace([np.inf, -np.inf], np.nan)
    return factor_to_panel(f, prices)


if __name__ == '__main__':
    res = eval_candidate('atr_ratio_10_50', panel_fn)
    print(json.dumps(res, indent=1, default=str))
