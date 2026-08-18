"""Data availability probe for factor mining cycle 2032-08-19."""
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
import pandas as pd
import os

acct = get_account_dict()
wl = acct.get("watch_list", [])
print("watch_list from account:", wl)

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBS = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is None or len(df) == 0:
        print(f"{sym}: NO DATA")
        continue
    print(f"{sym}: n={len(df)} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()} cols={list(df.columns)}")

print("\n--- observation-only index CSVs ---")
for sym in OBS:
    p = f"../persistent/index_data/{sym}.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"{sym}: {p} n={len(df)} first={df.iloc[0,0]} last={df.iloc[-1,0]} cols={list(df.columns)[:6]}")
    else:
        print(f"{sym}: MISSING {p}")