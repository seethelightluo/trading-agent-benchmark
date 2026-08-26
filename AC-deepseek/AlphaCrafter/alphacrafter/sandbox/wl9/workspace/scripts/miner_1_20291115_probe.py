"""Probe data availability for miner_1 cycle 2029-11-15."""
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

print("---WATCH (stock api first)---")
for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=800)
    if df is None or len(df) == 0:
        print(sym, "NO DATA via stock")
        continue
    print(sym, "rows", len(df), "last", df["date"].iloc[-1], "cols", list(df.columns))

print("---MACRO (index api / csv)---")
import os
for sym in MACRO:
    p = f"../persistent/index_data/{sym}.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(sym, "csv rows", len(df), "last", df["date"].iloc[-1] if "date" in df.columns else "?")
    else:
        try:
            df = get_index_daily_data(symbol=sym, days=800)
            print(sym, "api rows", None if df is None else len(df))
        except Exception as e:
            print(sym, "ERR", str(e)[:80])