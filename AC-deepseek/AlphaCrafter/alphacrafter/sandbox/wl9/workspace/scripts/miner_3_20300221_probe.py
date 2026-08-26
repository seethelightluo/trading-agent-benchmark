"""Probe sim data availability for miner_3 cycle 2030-02-21."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCH = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX',
         'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

for sym in WATCH:
    try:
        df = get_stock_daily_data(symbol=sym, days=4000)
        if df is None or len(df)==0:
            df = get_index_daily_data(symbol=sym, days=4000)
        if df is None or len(df)==0:
            print(sym, "NO DATA")
            continue
        print(f"{sym:9s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    except Exception as e:
        print(sym, "ERR", e)

print("---MACRO---")
for m in ['VIX','DXY','USDCNY','USDJPY','EURUSD']:
    try:
        df = get_index_daily_data(symbol=m, days=4000)
        if df is None or len(df)==0:
            print(m, "NO DATA")
        else:
            print(f"{m:8s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    except Exception as e:
        print(m, "ERR", e)