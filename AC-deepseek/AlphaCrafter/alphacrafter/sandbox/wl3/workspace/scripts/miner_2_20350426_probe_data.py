"""miner_2 2035-04-26 data probe: check data coverage, volume availability, and macro signals."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index, canonical_grid

prices = load_prices(days=4000)
grid = canonical_grid(prices)
print(f"assets: {len(prices)}")
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"  {s:10s} MISSING")
        continue
    print(f"  {s:10s} rows={len(df):5d} first={df.index.min().date()} last={df.index.max().date()} "
          f"vol_nan={(df['volume'].isna().mean() if 'volume' in df else 1):.2f} "
          f"close_nan={df['close'].isna().mean():.2f}")
print(f"\ncanonical grid: {grid.min().date()}..{grid.max().date()} n={len(grid)}")

for ix in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    df = load_index(ix, prices=prices)
    if df is None:
        print(f"index {ix}: MISSING")
    else:
        print(f"index {ix}: rows={len(df)} last={df.index.max().date()}")

# check volume availability in recent history for a few names
for s in ['000300.SH', 'SPX', 'BTC', 'WTI', 'US10Y']:
    df = prices[s]
    v = df['volume']
    print(f"vol check {s}: recent20 non-null={int(v.tail(20).notna().sum())}/20, total non-null={int(v.notna().sum())}/{len(v)}")
