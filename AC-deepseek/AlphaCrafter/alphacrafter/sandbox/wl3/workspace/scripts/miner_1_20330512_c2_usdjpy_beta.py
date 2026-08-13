"""miner_1 2033-05-12 candidate C2: USDJPY beta (60d).

Motivation: USDJPY is a global risk-appetite / carry-trade barometer. Assets
whose returns move with USDJPY (equities, cryptos) vs against (gold, bonds)
may predict relative performance over 10d. Novel macro-beta axis absent from
library (DXY/EURUSD/VIX/CN10Y betas exist, USDJPY does not).
"""
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel
from miner1_libfuncs import build_refs
from miner1_eval_helper import eval_candidate


def panel_fn(prices):
    refs = build_refs(prices)
    jpy = refs['jpy'] if 'jpy' in refs else None
    from miner1_libfuncs import load_index_csv
    jpy = load_index_csv('USDJPY', prices)
    jpy_r = jpy['close'].pct_change()

    def rb(r, m, w=60):
        z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
        b = z['r'].rolling(w, min_periods=30).cov(z['m']) / \
            z['m'].rolling(w, min_periods=30).var().replace(0, np.nan)
        return b.reindex(r.index)

    def f(df, s):
        return rb(df['close'].pct_change(), jpy_r)

    return factor_to_panel(f, prices)


if __name__ == '__main__':
    res = eval_candidate('usdjpy_beta_60', panel_fn)
    print(json.dumps(res, indent=1, default=str))
