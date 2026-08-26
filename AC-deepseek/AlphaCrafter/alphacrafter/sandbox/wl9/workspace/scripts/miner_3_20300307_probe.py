"""Probe live data availability for miner_3 cycle 2030-03-07."""
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=3000)
    if df is None or len(df) == 0:
        df = get_index_daily_data(symbol=sym, days=3000)
    if df is None or len(df) == 0:
        print(sym, "NO DATA")
        continue
    print(sym, "rows", len(df), "last", df["date"].iloc[-1], "first", df["date"].iloc[0], "cols", list(df.columns))

print("---MACRO---")
for sym in MACRO:
    try:
        df = get_index_daily_data(symbol=sym, days=3000)
        if df is None or len(df) == 0:
            print(sym, "NO DATA via API")
        else:
            print(sym, "rows", len(df), "last", df["date"].iloc[-1])
    except Exception as e:
        print(sym, "ERR", e)