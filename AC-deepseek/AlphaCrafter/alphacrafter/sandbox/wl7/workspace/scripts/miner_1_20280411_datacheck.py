"""miner_1 quick data check 2028-04-11: confirm panel extent + volume/open availability."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

for s in WATCH:
    df = get_stock_daily_data(s, days=4000)
    if df is None or not len(df):
        print(f"{s:10s} NO DATA")
        continue
    df = df.set_index("date")
    print(f"{s:10s} rows={len(df):5d} start={df.index[0].date()} end={df.index[-1].date()} "
          f"vol_nan={(df['volume'].isna() | (df['volume']==0)).mean():.2f} "
          f"open_nan={df['open'].isna().mean():.2f}")

print("--- macro ---")
for m in MACRO:
    df = get_index_daily_data(m, days=4000)
    if df is None or not len(df):
        print(f"{m:10s} NO DATA")
        continue
    df = df.set_index("date")
    print(f"{m:10s} rows={len(df):5d} start={df.index[0].date()} end={df.index[-1].date()}")
