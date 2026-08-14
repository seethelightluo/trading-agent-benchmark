"""miner_3 probe: check data availability and date coverage as of 2035-07-19."""
import os, sys
import pandas as pd
import numpy as np

from alphacrafter.sim.utils import (
    get_stock_daily_data,
    get_index_daily_data,
    get_account_dict,
)

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

acct = get_account_dict()
print("watch_list from account:", acct.get("watch_list"))

for sym in WATCH:
    try:
        df = get_stock_daily_data(symbol=sym, days=4000)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        try:
            df = get_index_daily_data(symbol=sym, days=4000)
        except Exception:
            df = None
    if df is None or len(df) == 0:
        print(f"{sym}: NO DATA")
        continue
    print(f"{sym}: n={len(df)} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()} cols={list(df.columns)}")
    # check volume availability
    if "volume" in df.columns:
        vol = df["volume"].dropna()
        print(f"   volume non-null: {len(vol)}/{len(df)}")

# macro observation data
print("\n--- macro csvs ---")
for f in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    p = f"../persistent/index_data/{f}.csv"
    if os.path.exists(p):
        m = pd.read_csv(p)
        print(f, m.shape, list(m.columns)[:6], m.iloc[-1].to_dict() if len(m) else None)
