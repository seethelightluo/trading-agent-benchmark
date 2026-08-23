"""miner_1 probe: verify data availability up to current date (2035-02-14) via simulator API."""
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is None:
        print(sym, "NONE")
        continue
    d = pd.to_datetime(df["date"])
    print(f"{sym:10s} rows={len(df):5d} first={d.min().date()} last={d.max().date()}")

for sym in ["VIX","DXY","USDCNY","USDJPY","EURUSD"]:
    try:
        df = get_index_daily_data(symbol=sym, days=4000)
        if df is None:
            print(sym, "NONE")
            continue
        d = pd.to_datetime(df["date"])
        print(f"{sym:10s} rows={len(df):5d} first={d.min().date()} last={d.max().date()}")
    except Exception as e:
        print(sym, "ERR", e)