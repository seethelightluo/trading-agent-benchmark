"""miner_2 cycle 2028-03-28: data availability + window sanity check."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
DAYS = 4000

for s in WATCH:
    df = get_stock_daily_data(s, days=DAYS)
    if df is None or not len(df):
        print(f"{s}: NO DATA")
    else:
        df = df.set_index("date")
        last = df.index[-1]
        print(f"{s}: n={len(df)} last={last.date()} last_close={df['close'].iloc[-1]:.2f}")

print()
for s in MACRO:
    df = get_index_daily_data(s, days=DAYS)
    if df is None or not len(df):
        print(f"{s}: NO DATA")
    else:
        df = df.set_index("date")
        print(f"{s}: n={len(df)} last={df.index[-1].date()} last={df['close'].iloc[-1]:.2f}")
