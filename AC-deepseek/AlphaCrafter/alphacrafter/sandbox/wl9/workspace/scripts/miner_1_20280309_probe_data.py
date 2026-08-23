"""Probe data availability for the 15-instrument watchlist and macro index files."""
import pandas as pd
import numpy as np
import os
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

watch = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
macro = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

print("=== TRADABLE WATCHLIST (get_stock_daily_data) ===")
for s in watch:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) == 0:
        print(f"{s}: NO DATA")
        continue
    df = df.reset_index(drop=True)
    print(f"{s}: rows={len(df)} from {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} cols={list(df.columns)}")

print("\n=== INDEX DATA (get_index_daily_data) ===")
for s in macro:
    df = get_index_daily_data(symbol=s, days=4000)
    if df is None or len(df) == 0:
        print(f"{s}: NO DATA")
        continue
    df = df.reset_index(drop=True)
    print(f"{s}: rows={len(df)} from {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

print("\n=== PERSISTENT INDEX FILES ===")
try:
    files = os.listdir("../persistent/index_data")
    print("n files:", len(files))
    for fn in sorted(files):
        print(" ", fn)
except Exception as e:
    print("err:", e)