"""miner_2 2030-09-05: diagnostic for cn10y_beta_60 recent-window n=0 anomaly."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           forward_returns, rank_ic_series, VAL_START, VAL_END)

prices = load_prices(days=3200)
max_date = max(dd.index.max() for dd in prices.values())
print(f"max_date = {max_date.date()}")

cn10y = prices['CN10Y']['close']
print(f"CN10Y close last 10 dates:\n{cn10y.tail(10)}")
cn10y_d = cn10y.diff()
print(f"CN10Y diff non-null last 15:\n{cn10y_d.tail(15)}")
print(f"CN10Y diff non-null count total: {cn10y_d.notna().sum()}")

# replicate rolling beta for one asset vs CN10Y
def rb(r, m, w, min_obs=0.5):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    if len(z) < 30:
        return pd.Series(np.nan, index=r.index)
    b = z['r'].rolling(w, min_periods=int(w * min_obs)).cov(z['m']) / \
        z['m'].rolling(w, min_periods=int(w * min_obs)).var().replace(0, np.nan)
    return b.reindex(r.index)

spx_r = prices['SPX']['close'].pct_change()
b_spx = rb(spx_r, cn10y_d, 60)
print(f"\nSPX vs CN10Y rolling beta: non-null total={b_spx.notna().sum()}, last non-null idx={b_spx.last_valid_index()}")

recent_start = max_date - pd.Timedelta(days=365)
print(f"\nrecent window starts {recent_start.date()}")
print(f"SPX beta non-null in recent window: {b_spx[b_spx.index >= recent_start].notna().sum()}")

# forward returns availability
fwd = forward_returns(prices, 10)
print(f"\nfwd non-null in recent window per asset:")
for s in WATCHLIST:
    print(f"  {s:10s} {int(fwd[s][fwd.index >= recent_start].notna().sum())}")
