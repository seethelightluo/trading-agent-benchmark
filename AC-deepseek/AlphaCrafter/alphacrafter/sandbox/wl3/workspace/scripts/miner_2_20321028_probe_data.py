"""miner_2 2032-10-28: probe data availability (start dates, NaN patterns)."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=5200)
print(f"loaded {len(prices)} assets in {time.time()-t0:.1f}s")
for s, df in prices.items():
    print(f"  {s:10s} rows={len(df):5d} first={df.index.min().date()} last={df.index.max().date()} "
          f"nan_close={(~np.isfinite(df['close'])).sum()} nan_vol={(~np.isfinite(df['volume'])).sum() if 'volume' in df else 'NA'}")
max_date = max(dd.index.max() for dd in prices.values())
print(f"max visible date: {max_date.date()}")

for ix in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    d = load_index(ix, prices=prices)
    if d is not None:
        print(f"  idx {ix:8s} rows={len(d):5d} first={d.index.min().date()} last={d.index.max().date()}")

# volume sample stats for a couple of assets
for s in ['SPX', 'BTC', 'WTI', '000300.SH']:
    df = prices[s]
    v = df['volume']
    print(f"  vol {s}: mean={v.mean():.2e} pct_zero={(v==0).mean():.3f} last5={list(v.tail(5).round(2))}")
