"""miner_1 2033-05-12 candidate C7: SPX return correlation (60d).

Motivation: spx_beta_60 measures market exposure; correlation normalizes by
idiosyncratic vol and lies in [-1,1]. High-correlation assets are less
diversifying; low-correlation (defensive/alternative) assets may be rewarded
in this long-only all-weather book. Tests comovement as a distinct signal.
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
    spx_r = refs['spx_r']

    def f(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), spx_r.rename('m')], axis=1).dropna()
        corr = z['r'].rolling(60, min_periods=30).corr(z['m'])
        return corr.reindex(r.index)
    return factor_to_panel(f, prices)


if __name__ == '__main__':
    res = eval_candidate('corr_spx_60', panel_fn)
    print(json.dumps(res, indent=1, default=str))
