"""miner_1 cycle 2031-06-12: probe persistent data coverage (no lookahead)."""
import pandas as pd
from pathlib import Path

STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF = '2031-06-12'

print("--- ASSETS ---")
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    if not f.exists():
        f = INDEX_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    tail = df[df['date'] <= CUTOFF]
    print(f"{a:9s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} | visible<={tail['date'].max().date()} rows={len(tail)} cols={list(df.columns)}")

print("--- MACRO (observation-only) ---")
for m in ['VIX','DXY','USDCNY','USDJPY','EURUSD']:
    f = INDEX_DIR / f'{m}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    tail = df[df['date'] <= CUTOFF]
    print(f"{m:9s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} | visible<={tail['date'].max().date()} rows={len(tail)}")
