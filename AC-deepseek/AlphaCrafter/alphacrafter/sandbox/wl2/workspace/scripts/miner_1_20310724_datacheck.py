"""miner_1 2031-07-24: data availability sanity check at current visible date."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    GRID, ASSETS, load_asset, load_macro, N_GRID,
)

print("grid:", len(GRID), "from", GRID[0], "to", GRID[-1])
print("assets:", ASSETS)
for s in ASSETS:
    df = load_asset(s, days=4000)
    if df is None:
        print(f"{s}: NO DATA")
        continue
    print(f"{s}: {len(df)} rows, first={df.index[0]}, last={df.index[-1]}, "
          f"close_last={df['close'].iloc[-1]:.4f}")
for m in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    s = load_macro(m)
    print(f"{m}: {len(s)} rows, last={s.index[-1]}, last_val={s.iloc[-1]:.4f}")
