"""miner_1 2033-05-12 candidate C6: short-horizon reversal 5d.

Motivation: library momentum factors skip 5d (vol_adj_mom_20_60 uses 5..25,
mom_accel uses 5..65). Pure 5d close-to-close reversal (contrarian) tests the
short end: recent 5d winners often mean-revert in cross-asset settings.
Factor = -1 * 5d return (high factor = recent loser).
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
        return -(df['close'] / df['close'].shift(5) - 1.0)
    return factor_to_panel(f, prices)


if __name__ == '__main__':
    res = eval_candidate('mom_rev_5d', panel_fn)
    print(json.dumps(res, indent=1, default=str))
