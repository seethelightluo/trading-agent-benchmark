"""miner_1 2032-01-22: probe data availability for factor research cycle."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index

t0 = time.time()
prices = load_prices(days=3200)
max_date = max(dd.index.max() for dd in prices.values())
print(f"prices: {len(prices)} assets loaded in {time.time()-t0:.1f}s; max visible date: {max_date.date()}")
print(f"min date across assets: {min(dd.index.min() for dd in prices.values()).date()}")

for s in WATCHLIST:
    df = prices[s]
    print(f"{s:10s} rows={len(df):5d} start={df.index.min().date()} end={df.index.max().date()} "
          f"close_last={df['close'].iloc[-1]:.2f}")

# observation-only signals
for sig in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    ix = load_index(sig, prices=prices)
    if ix is not None:
        print(f"{sig:8s} rows={len(ix):5d} start={ix.index.min().date()} end={ix.index.max().date()} "
              f"close_last={ix['close'].iloc[-1]:.2f}")
    else:
        print(f"{sig:8s} MISSING")

# count dates in windows
import json
from factor_common import VAL_START, VAL_END
oos_start = VAL_END + pd.Timedelta(days=1)
recent_start = max_date - pd.Timedelta(days=365)
print(f"\nVAL window: {VAL_START.date()}..{VAL_END.date()}")
print(f"OOS window: {oos_start.date()}..{max_date.date()}")
print(f"RECENT window: {recent_start.date()}..{max_date.date()}")
