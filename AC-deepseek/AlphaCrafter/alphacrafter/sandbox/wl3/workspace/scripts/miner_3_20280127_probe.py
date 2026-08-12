"""Probe data availability as of current sim date (2028-01-27)."""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, load_index, canonical_grid, WATCHLIST, VAL_START, VAL_END
import pandas as pd

prices = load_prices(days=2600)
print(f"assets loaded: {len(prices)}")
for s in WATCHLIST:
    df = prices.get(s)
    if df is not None:
        print(f"  {s:10s} n={len(df):5d} {df.index.min().date()} .. {df.index.max().date()}")
    else:
        print(f"  {s:10s} MISSING")

grid = canonical_grid(prices)
print(f"\ncanonical grid: {len(grid)} dates, {grid.min().date()} .. {grid.max().date()}")

# check index signals
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    ix = load_index(s, prices=prices)
    if ix is not None:
        print(f"  {s:8s} n={len(ix):5d} {ix.index.min().date()} .. {ix.index.max().date()}")
    else:
        print(f"  {s:8s} MISSING")

# volume availability
import numpy as np
for s in WATCHLIST:
    df = prices[s]
    v = df['volume']
    print(f"  vol {s:10s} valid={v.notna().sum()}/{len(v)} nonzero={(v>0).sum()} last={v.iloc[-1]:.0f}")
