"""Probe data availability through current sim date (2029-02-08)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("=== Tradable assets (stock API) ===")
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=5000)
    if df is not None and len(df):
        print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} cols={list(df.columns)}")
    else:
        print(f"{sym:10s} NO DATA")

print("\n=== Macro (index API) ===")
for sym in ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    df = get_index_daily_data(symbol=sym, days=5000)
    if df is not None and len(df):
        print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} cols={list(df.columns)}")
    else:
        print(f"{sym}: NONE")

print("\n=== Persistent index_data CSVs (tail rows) ===")
import os
p = '../persistent/index_data'
for f in sorted(os.listdir(p)):
    if f.endswith('.csv'):
        try:
            csv = pd.read_csv(os.path.join(p, f))
            print(f, 'rows=', len(csv), 'cols=', list(csv.columns), 'last=', csv.iloc[-1].tolist()[:4])
        except Exception as e:
            print(f, 'ERR', e)