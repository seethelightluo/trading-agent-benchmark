"""miner_1 2028-01-27: probe data availability & quality through current date.
Checks: price history span, volume availability, index-signal availability, and
a quick look at yield-series behavior (US10Y/CN10Y as tradable)."""
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
    last = df.index.max()
    first = df.index.min()
    has_vol = df['volume'].notna().sum()
    vol_frac = has_vol / n if n else 0
    print(f"{s}: n={n} {first.date()}..{last.date()} vol_frac={vol_frac:.2f} last_close={df['close'].iloc[-1]:.4f}")

print("\nIndex signals:")
for s in INDEX_SIGNALS:
    idx = load_index(s, days=4200, prices=prices)
    if idx is None:
        print(f"{s}: MISSING")
        continue
    print(f"{s}: n={len(idx)} {idx.index.min().date()}..{idx.index.max().date()} last={idx['close'].iloc[-1]:.4f}")

# quick sanity: pct_change on yield series
for s in ['US10Y', 'CN10Y']:
    df = prices[s]
    r = df['close'].pct_change().dropna()
    print(f"{s} ret: mean={r.mean():.6f} std={r.std():.6f} min={r.min():.6f} max={r.max():.6f} n={len(r)}")
