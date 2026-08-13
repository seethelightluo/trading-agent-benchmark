"""miner_1 2033-05-12 candidate C1: US10Y yield-change beta (60d).

Motivation: library has cn10y_beta_60 (China rates beta) but not the US side.
Assets negatively exposed to US10Y yield rises (duration-sensitive) should
underperform when the US curve reprices. Tests whether US rates beta adds a
distinct cross-asset signal. Expected direction: -1.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel
from miner1_libfuncs import build_refs
from miner1_eval_helper import eval_candidate


def panel_fn(prices):
    refs = build_refs(prices)
    us10y_d = prices['US10Y']['close'].diff()

    def rb(r, m, w=60):
        z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
        b = z['r'].rolling(w, min_periods=30).cov(z['m']) / \
            z['m'].rolling(w, min_periods=30).var().replace(0, np.nan)
        return b.reindex(r.index)

    def f(df, s):
        return rb(df['close'].pct_change(), us10y_d)

    return factor_to_panel(f, prices)


if __name__ == '__main__':
    res = eval_candidate('us10y_beta_60', panel_fn)
    print(json.dumps(res, indent=1, default=str))
