"""Probe: how do current API grids compare to persisted artifact grids (2388 dates)?"""
import sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, VAL_START, VAL_END)

for days in (2000, 2500, 4000):
    prices = load_prices(days=days)
    grid = canonical_grid(prices)
    print(f"days={days}: grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}", flush=True)
    # per-asset date ranges
    for s, df in prices.items():
        sub = df.index[(df.index >= VAL_START) & (df.index <= VAL_END)]
        print(f"   {s}: {len(sub)} dates {sub.min().date() if len(sub) else '-'}..{sub.max().date() if len(sub) else '-'}", flush=True)

# artifact grid check for one factor
d = json.load(open('factors/vol_adj_mom_20_60.json'))
print("artifact grid meta:", d['signal_artifact_grid'])
arr = np.load('factors/vol_adj_mom_20_60_signal.npy', allow_pickle=False)
print("artifact shape:", arr.shape)

# Try to align: does the stored grid have any 2020-01-01? Build approximate stored grid from meta.
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print("days=2500 grid:", len(grid), grid.min().date(), grid.max().date())
