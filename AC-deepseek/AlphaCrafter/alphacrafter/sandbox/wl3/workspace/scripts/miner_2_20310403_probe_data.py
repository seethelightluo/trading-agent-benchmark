"""miner_2 probe: data availability through 2031-04-02 (visible), canonical grid check."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import WATCHLIST, load_prices, load_index, canonical_grid

prices = load_prices(days=3200)
print("symbols loaded:", len(prices))
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(s, "MISSING")
        continue
    print(s, "rows:", len(df), "range:", df.index.min().date(), "->", df.index.max().date(),
          "cols:", list(df.columns))

grid = canonical_grid(prices)
print("\ncanonical grid (warm-up 2020-01-01..2026-07-15): n =", len(grid),
      grid.min().date(), "->", grid.max().date())

# macro signals availability
for sym in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    idx = load_index(sym, days=3200, prices=prices)
    if idx is None:
        print(sym, "MISSING")
    else:
        print(sym, "rows:", len(idx), "range:", idx.index.min().date(), "->", idx.index.max().date())
