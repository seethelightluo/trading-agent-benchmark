import os
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

watch = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
obs = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def probe(sym, days=2500):
    df = get_stock_daily_data(symbol=sym, days=days)
    if df is None or len(df) == 0:
        try:
            df = get_index_daily_data(symbol=sym, days=days)
        except Exception:
            return None
    return df

print("=== WATCHLIST SPAN ===")
for s in watch:
    df = probe(s)
    if df is None or len(df)==0:
        print(s, "NO DATA")
        continue
    # use long window
    df2 = get_stock_daily_data(s, 3000) or get_index_daily_data(s, 3000)
    d0 = str(df2['date'].iloc[0])[:10]
    d1 = str(df2['date'].iloc[-1])[:10]
    print(f"{s}: rows={len(df2)} {d0}..{d1}")

print("=== OBSERVATION SPAN ===")
for s in obs:
    df = get_index_daily_data(symbol=s, days=3000)
    if df is None or len(df)==0:
        print(s, "NO DATA via get_index")
        continue
    d0 = str(df['date'].iloc[0])[:10]
    d1 = str(df['date'].iloc[-1])[:10]
    print(f"{s}: rows={len(df)} {d0}..{d1}")