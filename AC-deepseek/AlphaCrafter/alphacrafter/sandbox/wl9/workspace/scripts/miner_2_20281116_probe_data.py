"""miner_2 2028-11-16: probe data availability for factor research."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=3000)
    if df is None or len(df) < 100:
        print(f"{sym}: NONE/insufficient")
        continue
    df = df.set_index("date").sort_index()
    print(f"{sym}: rows={len(df)} first={df.index.min().date()} last={df.index.max().date()} "
          f"cols={list(df.columns)}")

for m in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    df = get_index_daily_data(symbol=m, days=3000)
    if df is None or len(df) < 100:
        print(f"{m}: N/A/insufficient")
        continue
    df = df.set_index("date").sort_index()
    print(f"{m}: rows={len(df)} last={df.index.max().date()}")