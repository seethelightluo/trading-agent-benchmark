"""Validate drawdown-duration factor: log(1 + days since last 120d rolling high).

Motivation: assets deep into a drawdown (relative to their 120d high) tend to
mean-revert or underperform depending on regime; duration captures how "stale"
the high is. Log transform dampens the long tail.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2100)
print('n assets:', len(prices), '| max date:', max(df.index.max() for df in prices.values()))


def dd_duration_120(df, s):
    c = df['close']
    h = c.rolling(120, min_periods=60).max()
    is_high = (c >= h).fillna(False)
    idx_high = np.flatnonzero(is_high.values)
    pos = np.arange(len(c))
    last = np.searchsorted(idx_high, pos) - 1
    dur = np.where(last >= 0, pos - idx_high[np.maximum(last, 0)], np.nan)
    return pd.Series(np.log1p(dur), index=c.index)


metrics, panel = evaluate_candidate('dd_duration_120', dd_duration_120, prices)
if metrics is not None:
    print('n_ic_dates:', metrics['n_ic_dates'], '| coverage:', round(metrics['coverage_asset_days'], 3),
          '| dates_ge8:', round(metrics['coverage_dates_ge8'], 3))
