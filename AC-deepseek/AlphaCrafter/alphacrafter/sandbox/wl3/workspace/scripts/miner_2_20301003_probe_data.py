"""miner_2 probe: data availability at 2030-10-03."""
import time, sys
sys.path.insert(0, 'scripts')
import pandas as pd
import numpy as np
from factor_common import WATCHLIST, load_prices, load_index, VAL_START, VAL_END

t0 = time.time()
prices = load_prices(days=4000)
print(f"load_prices: {time.time()-t0:.1f}s")
print(f"{'symbol':12s} {'n_days':>6s} {'first':>12s} {'last':>12s} {'vol_nan%':>8s} {'close_nan%':>9s}")
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"{s:12s} MISSING")
        continue
    vol_nan = float(df['volume'].isna().mean()) if 'volume' in df else 1.0
    close_nan = float(df['close'].isna().mean())
    print(f"{s:12s} {len(df):6d} {str(df.index.min().date()):>12s} {str(df.index.max().date()):>12s} {vol_nan:8.2f} {close_nan:9.3f}")

for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    df = load_index(s, days=4000, prices=prices)
    if df is None:
        print(f"INDEX {s}: MISSING")
    else:
        print(f"INDEX {s:8s} n={len(df):5d} {str(df.index.min().date())}..{str(df.index.max().date())}")

grid = None
try:
    from factor_common import canonical_grid
    grid = canonical_grid(prices)
    print(f"\ncanonical grid: {len(grid)} dates, {grid.min().date()}..{grid.max().date()}")
    print(f"grid within warm window: {(grid>=VAL_START).sum()} after VAL_START, {(grid<=VAL_END).sum()} <= VAL_END")
except Exception as e:
    print("grid err", e)

last = max(df.index.max() for df in prices.values() if df is not None)
print(f"\nlatest tradable date: {last.date()}")
print(f"days since 2026-07-15: {(last - pd.Timestamp('2026-07-15')).days}")
