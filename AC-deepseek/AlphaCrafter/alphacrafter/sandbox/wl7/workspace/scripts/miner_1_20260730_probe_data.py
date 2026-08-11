"""miner_1 probe: verify data loading conventions for the 15-instrument universe."""
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

for s in WATCH:
    df = get_stock_daily_data(s, days=4000)
    if df is None:
        print(f"{s}: NONE")
        continue
    print(f"{s}: rows={len(df)} start={df['date'].iloc[0].date()} end={df['date'].iloc[-1].date()} cols={list(df.columns)}")

print("--- macro ---")
for s in MACRO:
    df = get_index_daily_data(s, days=4000)
    if df is None:
        print(f"{s}: NONE")
        continue
    print(f"{s}: rows={len(df)} start={df['date'].iloc[0].date()} end={df['date'].iloc[-1].date()}")
