"""miner_2 probe: data availability through 2029-03-08 (visible through 2029-03-07)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("=== Tradable assets (stock API) ===")
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is not None and len(df):
        print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} volNaN={df['volume'].isna().sum() if 'volume' in df.columns else 'NA'}")
    else:
        print(f"{sym:10s} NO DATA")

print("\n=== Macro (index API) ===")
for sym in ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    df = get_index_daily_data(symbol=sym, days=4000)
    if df is not None and len(df):
        print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    else:
        print(f"{sym}: NONE")