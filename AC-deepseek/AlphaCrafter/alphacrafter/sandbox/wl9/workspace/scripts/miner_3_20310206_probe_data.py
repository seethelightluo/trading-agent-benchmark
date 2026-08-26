"""Probe persistent data coverage for miner_3 cycle 2031-02-06."""
import pandas as pd
from pathlib import Path

STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("--- ASSETS ---")
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    if not f.exists():
        f = INDEX_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    print(f"{a:9s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} cols={list(df.columns)}")

print("--- MACRO (observation-only) ---")
for m in ['VIX','DXY','USDCNY','USDJPY','EURUSD']:
    f = INDEX_DIR / f'{m}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    print(f"{m:8s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")