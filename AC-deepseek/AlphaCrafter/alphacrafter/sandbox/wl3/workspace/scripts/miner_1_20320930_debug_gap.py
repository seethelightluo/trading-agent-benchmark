"""miner_1 2032-09-30: debug gap_freq_60 OOS IC anomaly (why only 60 IC dates)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, forward_returns, rank_ic_series, VAL_END, factor_to_panel

np.seterr(all='ignore')
prices = load_prices(days=3300)

FROZEN = {'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y'}
LIVE = [s for s in WATCHLIST if s not in FROZEN]

def f_gap_freq_60(df, s):
    gap = (df['open'] / df['close'].shift(1) - 1.0).abs()
    return (gap > 0.01).astype(float).rolling(60, min_periods=30).mean()

panel = factor_to_panel(f_gap_freq_60, prices)
print(f"panel range: {panel.index.min().date()} .. {panel.index.max().date()}, shape {panel.shape}")
oos_start = VAL_END + pd.Timedelta(days=1)
oos_p = panel[panel.index >= oos_start]
print(f"oos_p range: {oos_p.index.min().date()} .. {oos_p.index.max().date()}, shape {oos_p.shape}")
print(f"oos_p[LIVE] valid-per-date: min={oos_p[LIVE].notna().sum(axis=1).min()} max={oos_p[LIVE].notna().sum(axis=1).max()}")

fwd = forward_returns(prices, 10)
f = fwd.reindex(oos_p.index)[LIVE]
print(f"fwd[LIVE] valid-per-date: min={f.notna().sum(axis=1).min()} max={f.notna().sum(axis=1).max()}")

ic = rank_ic_series(oos_p[LIVE], f, min_valid=8)
print(f"OOS IC dates: {len(ic)}, range {ic.index.min().date()} .. {ic.index.max().date()}")
print(f"last 5 IC dates: {list(ic.index[-5:].strftime('%Y-%m-%d')) if len(ic) else 'none'}")

# manual scan: find dates where >=8 valid but IC missing
common = oos_p.index.intersection(fwd.index)
cnt_ge8 = 0
cnt_ic = 0
first_ge8 = None
for d in common:
    x = oos_p.loc[d]
    y = fwd.loc[d]
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 8:
        cnt_ge8 += 1
        if first_ge8 is None:
            first_ge8 = d
        v = x[m].rank().corr(y[m].rank())
        if np.isfinite(v):
            cnt_ic += 1
print(f"dates with >=8 valid: {cnt_ge8}, with finite rank corr: {cnt_ic}, first: {first_ge8.date() if first_ge8 is not None else None}")

# check corr of last date
d = common[-1]
x = oos_p.loc[d]; y = fwd.loc[d]
m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
print(f"last common date {d.date()}: valid={int(m.sum())}, factor values:\n{x[m].to_string()}")
print(f"fwd values:\n{y[m].to_string()}")
