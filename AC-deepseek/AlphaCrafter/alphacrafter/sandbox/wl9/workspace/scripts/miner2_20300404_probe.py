"""Probe live data availability for miner_2 cycle 2030-04-04 via sim API."""
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is None or len(df) == 0:
        df = get_index_daily_data(symbol=sym, days=4000)
    if df is None or len(df) == 0:
        print(sym, "NO DATA")
        continue
    print(sym, "rows", len(df), "last", str(df["date"].iloc[-1])[:10], "first", str(df["date"].iloc[0])[:10],
          "vol_nz", int((df["volume"].fillna(0) > 0).sum()))

print("---MACRO---")
for sym in MACRO:
    try:
        df = get_index_daily_data(symbol=sym, days=4000)
        if df is None or len(df) == 0:
            print(sym, "NO DATA via API")
        else:
            print(sym, "rows", len(df), "last", str(df["date"].iloc[-1])[:10])
    except Exception as e:
        print(sym, "ERR", e)