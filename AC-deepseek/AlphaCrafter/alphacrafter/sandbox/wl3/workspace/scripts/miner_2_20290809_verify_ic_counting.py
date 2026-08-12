"""miner_2 2029-08-09: verify IC date counting + diagnose cn10y_beta_60 collapse.

Checks:
1. How many trading dates exist in each window (raw, and with >=8 valid
   instruments for a normal factor vs cn10y_beta_60).
2. CN10Y close behaviour in the recent window (missing runs, zero diff runs).
3. cn10y_beta_60 valid-count-per-date timeline in OOS.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index, factor_to_panel, \
    forward_returns, rank_ic_series, VAL_START, VAL_END

np.seterr(all='ignore')
prices = load_prices(days=3200)
max_date = max(dd.index.max() for dd in prices.values())
print(f"prices: {len(prices)} assets, max_date={max_date.date()}", flush=True)

FROZEN = {'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y'}
LIVE = [s for s in WATCHLIST if s not in FROZEN]
print(f"live assets ({len(LIVE)}): {LIVE}", flush=True)

fwd10 = forward_returns(prices, 10)
OOS_START = pd.Timestamp('2026-07-16')
REC_START = max_date - pd.Timedelta(days=365)

# raw date counts per window
for lo, hi, name in [(VAL_START, VAL_END, 'WARM'), (OOS_START, max_date, 'OOS'),
                     (REC_START, max_date, 'RECENT')]:
    n_raw = len(fwd10[(fwd10.index >= lo) & (fwd10.index <= hi)])
    n_live_ge8 = int((fwd10[LIVE].notna().sum(axis=1) >= 8)
                     .loc[lambda s: s.index >= lo].loc[lambda s: s.index <= hi].sum())
    print(f"{name:7s} raw dates={n_raw:5d}  fwd10-live dates with >=8 valid={n_live_ge8}", flush=True)

# ---- CN10Y close diagnostics ----
cn = prices['CN10Y']['close']
print(f"\nCN10Y close: last date={cn.index.max().date()}, n={len(cn)}", flush=True)
recent = cn[cn.index >= REC_START]
print(f"CN10Y recent window: first={recent.index[0].date()} last={recent.index[-1].date()} n={len(recent)}", flush=True)
d = cn.diff()
print(f"CN10Y zero-diff days in OOS: {int((d[d.index >= OOS_START] == 0).sum())} / "
      f"{int((d.index >= OOS_START).sum())} OOS days", flush=True)
print(f"CN10Y zero-diff days in recent: {int((d[d.index >= REC_START] == 0).sum())} / "
      f"{int((d.index >= REC_START).sum())} recent days", flush=True)
print(f"CN10Y close head/tail:\n{cn.head(3).to_string()}\n...\n{cn.tail(5).to_string()}", flush=True)

# ---- cn10y_beta_60 panel: valid count per date in OOS ----
cn10y_d = cn.diff()
def rb(r, m, w, min_obs=0.5):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    if len(z) < 30:
        return pd.Series(np.nan, index=r.index)
    b = z['r'].rolling(w, min_periods=int(w * min_obs)).cov(z['m']) / \
        z['m'].rolling(w, min_periods=int(w * min_obs)).var().replace(0, np.nan)
    return b.reindex(r.index)

def f_cn10y_beta_60(df, s):
    return rb(df['close'].pct_change(), cn10y_d, 60)

panel = factor_to_panel(f_cn10y_beta_60, prices)
sub = panel[(panel.index >= OOS_START) & (panel.index <= max_date)]
valid_per_date = sub[LIVE].notna().sum(axis=1)
print(f"\ncn10y_beta_60 OOS: dates with >=8 valid live = {(valid_per_date >= 8).sum()} / {len(sub)}", flush=True)
print("valid-per-date quantiles:", valid_per_date.quantile([0, .25, .5, .75, 1]).round(1).to_dict(), flush=True)
# last 10 dates with >=8 valid
ge8 = valid_per_date[valid_per_date >= 8]
print(f"last 5 dates with >=8 valid: {[d.date().isoformat() for d in ge8.index[-5:]]}", flush=True)
# distribution by quarter
q = valid_per_date.groupby(valid_per_date.index.to_period('Q')).agg(['count', 'mean'])
print("\nquarterly valid-count (live):")
print(q.tail(14).to_string(), flush=True)

# ---- same check for a normal factor (hilo_vol_ratio_20) for contrast ----
def f_hilo_vol_ratio_20(df, s):
    c = df['close']
    rng = (c.rolling(20).max() - c.rolling(20).min()) / c
    v = c.pct_change().rolling(20).std()
    return (rng / v).replace([np.inf, -np.inf], np.nan)

panel2 = factor_to_panel(f_hilo_vol_ratio_20, prices)
sub2 = panel2[(panel2.index >= OOS_START) & (panel2.index <= max_date)]
vp2 = sub2[LIVE].notna().sum(axis=1)
print(f"\nhilo_vol_ratio_20 OOS: dates with >=8 valid live = {(vp2 >= 8).sum()} / {len(sub2)}", flush=True)
