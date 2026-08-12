"""miner_2 2030-08-22: data probe - confirm visible horizon, coverage, index signals."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index

t0 = time.time()
prices = load_prices(days=3200)
max_date = max(dd.index.max() for dd in prices.values())
min_date = min(dd.index.min() for dd in prices.values())
print(f"prices: {len(prices)} assets, range {min_date.date()} .. {max_date.date()} ({time.time()-t0:.1f}s)", flush=True)
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"  {s}: MISSING", flush=True)
    else:
        print(f"  {s}: n={len(df)} last={df.index.max().date()} last_close={df['close'].iloc[-1]:.2f}", flush=True)

for ix in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    d = load_index(ix, prices=prices)
    if d is None:
        print(f"index {ix}: MISSING", flush=True)
    else:
        print(f"index {ix}: n={len(d)} last={d.index.max().date()} last={d['close'].iloc[-1]:.2f}", flush=True)
