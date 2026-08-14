"""miner_1 2035-12-06: verify data availability and canonical grid."""
import time, sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, VAL_START, VAL_END, load_prices, canonical_grid

t0 = time.time()
prices = load_prices(days=6000)
print(f"load_prices: {round(time.time()-t0,1)}s; assets={len(prices)}")
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"  {s}: MISSING")
    else:
        print(f"  {s}: {df.index.min().date()}..{df.index.max().date()} n={len(df)}")
grid = canonical_grid(prices)
print(f"canonical grid: {len(grid)} dates {grid.min().date()}..{grid.max().date()}")
print(f"VAL window {VAL_START.date()}..{VAL_END.date()}")
# check how many dates past 2026-07-15 available (recent window for regime checks)
recent = [d for d in grid if d > VAL_END]
print(f"dates > VAL_END in grid: {len(recent)}")
allidx = sorted(set().union(*[set(df.index) for df in prices.values()]))
after = [d for d in allidx if pd.Timestamp(d) > VAL_END]
print(f"all dates after VAL_END: {len(after)}; last={max(after).date() if after else None}")
