"""MINER_1 2031-01-13: probe data availability/coverage for updated research window."""
import os
import pandas as pd

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA_DIR = "../persistent/stock_data"
IDX_DIR = "../persistent/index_data"

for a in ASSETS:
    df = pd.read_csv(os.path.join(DATA_DIR, a + ".csv"), parse_dates=["date"])
    print(f"{a:10s} n={len(df):5d} {df['date'].min().date()} -> {df['date'].max().date()}")

print("\n--- macro observation-only ---")
for m in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
    df = pd.read_csv(os.path.join(IDX_DIR, m + ".csv"), parse_dates=["date"])
    print(f"{m:10s} n={len(df):5d} {df['date'].min().date()} -> {df['date'].max().date()}")
