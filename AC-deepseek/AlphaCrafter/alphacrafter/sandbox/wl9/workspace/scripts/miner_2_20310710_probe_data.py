"""miner_2 probe: data availability up to current sim date 2031-07-10."""
import pandas as pd
from pathlib import Path

STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
VISIBLE_END = pd.Timestamp('2031-07-10')

print("=== stock_data (capped at 2031-07-10) ===")
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    if not f.exists():
        f = INDEX_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date'])
    dfv = df[df['date'] <= VISIBLE_END]
    print(f"{a}: {len(dfv)} rows , last {dfv['date'].max():%Y-%m-%d}, cols={list(df.columns)}")

print("\n=== index_data (macro obs-only) ===")
for m in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    f = INDEX_DIR / f'{m}.csv'
    if f.exists():
        df = pd.read_csv(f, parse_dates=['date'])
        dfv = df[df['date'] <= VISIBLE_END]
        print(f"{m}: {len(dfv)} rows, {dfv['date'].min():%Y-%m-%d} -> {dfv['date'].max():%Y-%m-%d}")
    else:
        print(f"{m}: MISSING")