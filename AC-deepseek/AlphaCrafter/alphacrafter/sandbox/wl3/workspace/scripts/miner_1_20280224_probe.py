"""miner_1 2028-02-24: data availability probe through current visible date."""
import sys, time
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import load_prices, load_index, WATCHLIST, INDEX_SIGNALS

t0 = time.time()
prices = load_prices(days=4200)
print(f"prices loaded in {time.time()-t0:.1f}s for {len(prices)} symbols")

for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"{s}: MISSING")
        continue
    n = len(df)
    vol_frac = df['volume'].notna().mean()
    print(f"{s}: n={n} {df.index.min().date()}..{df.index.max().date()} vol_frac={vol_frac:.2f}")

print("\nIndex signals:")
for s in INDEX_SIGNALS:
    idx = load_index(s, days=4200, prices=prices)
    if idx is None:
        print(f"{s}: MISSING")
        continue
    print(f"{s}: n={len(idx)} {idx.index.min().date()}..{idx.index.max().date()} last={idx['close'].iloc[-1]:.4f}")

# span of visible data
mx = max(df.index.max() for df in prices.values())
mn = min(df.index.min() for df in prices.values())
print(f"\nvisible span: {mn.date()} .. {mx.date()}  total_days={len(pd.bdate_range(mn, mx))}")
