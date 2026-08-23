"""miner_2 2028-10-05: probe volume data quality across the 15-asset universe."""
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=2500)
    if df is None or len(df) < 300:
        print(f"{sym}: insufficient"); continue
    df = df.set_index("date").sort_index()
    v = df["volume"]
    total = len(v)
    nz = (v > 0).sum()
    zeros = total - nz
    vv = v[v > 0]
    print(f"{sym:10s} rows={total:5d} zero_vol={zeros:4d} ({zeros/total:.2%}) "
          f"vol_med={vv.median() if len(vv) else 0:.1f} vol_mean={vv.mean() if len(vv) else 0:.1f} "
          f"last5_vol={list(v.tail(5).round(1))}")