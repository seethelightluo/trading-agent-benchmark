"""miner_1 2032-02-05: data probe — confirm available history horizon."""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index, VAL_START, VAL_END

t0 = time.time()
prices = load_prices(days=3300)
max_date = max(dd.index.max() for dd in prices.values())
min_date = min(dd.index.min() for dd in prices.values())
print(f"prices: {len(prices)} assets, min {min_date.date()} max {max_date.date()} ({time.time()-t0:.1f}s)", flush=True)
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"  {s}: MISSING", flush=True)
    else:
        print(f"  {s}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()} last_close={df['close'].iloc[-1]:.2f}", flush=True)

for sig in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    df = load_index(sig, prices=prices)
    if df is None:
        print(f"  {sig}: MISSING", flush=True)
    else:
        print(f"  {sig}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()}", flush=True)

print(f"VAL window: {VAL_START.date()}..{VAL_END.date()}", flush=True)
