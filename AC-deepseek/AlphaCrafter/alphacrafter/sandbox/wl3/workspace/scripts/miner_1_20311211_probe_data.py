"""miner_1 2031-12-11: probe data availability for revalidation + novel factor mining."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import load_prices, load_index, WATCHLIST, INDEX_SIGNALS

t0 = time.time()
prices = load_prices(days=3200)
max_date = max(dd.index.max() for dd in prices.values())
print(f"prices: {len(prices)} assets, last date {max_date.date()} ({time.time()-t0:.1f}s)", flush=True)

for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"{s}: MISSING", flush=True)
        continue
    print(f"{s}: rows={len(df)} range={df.index.min().date()}..{df.index.max().date()} "
          f"vol_na={(df['volume'].isna().mean() if 'volume' in df else 1.0):.2%} "
          f"open_na={df['open'].isna().mean():.2%}", flush=True)

for sym in INDEX_SIGNALS:
    idx = load_index(sym, days=3200, prices=prices)
    if idx is None:
        print(f"{sym}: MISSING", flush=True)
    else:
        print(f"{sym}: rows={len(idx)} range={idx.index.min().date()}..{idx.index.max().date()}", flush=True)
