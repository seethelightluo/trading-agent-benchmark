"""miner_2 probe (2029-05-03): data coverage through visible horizon 2029-05-02."""
import sys, json, glob, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index, canonical_grid, VAL_START, VAL_END

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=4000)
grid = canonical_grid(prices)
T, N = len(grid), len(WATCHLIST)
maxd = max(d.index.max() for d in prices.values())
print(f"canonical grid (warm-up): {T} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)} | {time.time()-t0:.1f}s", flush=True)
print(f"full data end: {maxd.date()} | days requested 4000", flush=True)
for s, df in prices.items():
    print(f"  {s}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()}", flush=True)

# full visible window (2020-01-01 .. visible horizon)
all_dates = sorted(set().union(*[set(d.index) for d in prices.values()]))
full_grid = pd.DatetimeIndex(all_dates)
print(f"full grid: {len(full_grid)} dates {full_grid.min().date()}..{full_grid.max().date()}", flush=True)

# index data availability
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    df = load_index(s, days=4000, prices=prices)
    if df is not None:
        print(f"  idx {s}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()}", flush=True)
    else:
        print(f"  idx {s}: MISSING", flush=True)

# frozen-asset check (unique closes over trailing 120d)
print("--- frozen check (unique closes in last 120d) ---", flush=True)
for s, df in prices.items():
    tail = df['close'].tail(120)
    u = tail.nunique()
    if u <= 2:
        print(f"  FROZEN {s}: {u} unique", flush=True)
print("---", flush=True)
print(f"done {time.time()-t0:.1f}s", flush=True)
