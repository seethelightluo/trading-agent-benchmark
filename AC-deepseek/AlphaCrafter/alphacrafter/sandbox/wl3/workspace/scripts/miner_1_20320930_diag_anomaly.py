"""miner_1 2032-09-30: diagnose why cn10y_beta_60 / gap_freq_60 have only 60 OOS IC dates."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, forward_returns, rank_ic_series, VAL_END, factor_to_panel

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3300)
max_date = max(dd.index.max() for dd in prices.values())
print(f"prices: {len(prices)} assets, last {max_date.date()} ({time.time()-t0:.1f}s)")

# 1) CN10Y constancy check
cn = prices['CN10Y']['close']
d = cn.diff().abs()
print(f"\nCN10Y last date with nonzero change: {d[d > 0].index.max()}")
print(f"CN10Y last 30 closes: {cn.iloc[-30:].values}")

# 2) gap_freq panel validity in OOS window
def f_gap_freq_60(df, s):
    gap = (df['open'] / df['close'].shift(1) - 1.0).abs()
    return (gap > 0.01).astype(float).rolling(60, min_periods=30).mean()

gap_panel = factor_to_panel(f_gap_freq_60, prices)
oos_start = VAL_END + pd.Timedelta(days=1)
g = gap_panel[gap_panel.index >= oos_start]
valid_per_date = g.notna().sum(axis=1)
print(f"\ngap_freq_60 OOS panel: {g.shape}, valid-per-date: min={valid_per_date.min()} max={valid_per_date.max()}")
print(f"dates with >=8 valid: {(valid_per_date >= 8).sum()} / {len(g)}")
# where does it break?
brk = valid_per_date[valid_per_date < 8]
if len(brk):
    print(f"first date with <8 valid: {brk.index.min().date()}")

# check open/close availability per asset in recent period
print("\nopen availability last 500 days (NaN count per asset):")
for s in WATCHLIST:
    df = prices[s]
    tail = df.iloc[-500:]
    print(f"  {s:10s} open_nan={tail['open'].isna().sum():5d} close_nan={tail['close'].isna().sum():5d}")

# 3) cn10y beta panel validity in OOS window
cn10y_d = cn.diff()
def f_cn10y_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), cn10y_d.rename('m')], axis=1).dropna()
    if len(z) < 30:
        return pd.Series(np.nan, index=r.index)
    b = z['r'].rolling(60, min_periods=30).cov(z['m']) / z['m'].rolling(60, min_periods=30).var().replace(0, np.nan)
    return b.reindex(r.index)

cn_panel = factor_to_panel(f_cn10y_beta_60, prices)
c = cn_panel[cn_panel.index >= oos_start]
valid_per_date_c = c.notna().sum(axis=1)
print(f"\ncn10y_beta_60 OOS panel: {c.shape}, valid-per-date: min={valid_per_date_c.min()} max={valid_per_date_c.max()}")
print(f"dates with >=8 valid: {(valid_per_date_c >= 8).sum()} / {len(c)}")
brk = valid_per_date_c[valid_per_date_c < 8]
if len(brk):
    print(f"first date with <8 valid: {brk.index.min().date()}")
# rolling var of cn10y_d
v = cn10y_d.rolling(60, min_periods=30).var()
print(f"cn10y_d rolling var: last non-nan date {v.notna().index.max().date() if v.notna().any() else 'none'}, "
      f"dates where var>0 in OOS: {(v[v.index>=oos_start] > 0).sum()} / {(v.index>=oos_start).sum()}")
