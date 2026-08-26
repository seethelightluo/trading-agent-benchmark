"""Probe data availability up to current sim date 2031-02-20."""
import pandas as pd
from pathlib import Path

STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("=== stock_data ===")
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date'])
    print(f"{a}: {len(df)} rows, {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}, cols={list(df.columns)}")

print("\n=== index_data (macro obs-only) ===")
for m in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    f = INDEX_DIR / f'{m}.csv'
    if f.exists():
        df = pd.read_csv(f, parse_dates=['date'])
        print(f"{m}: {len(df)} rows, {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}, cols={list(df.columns)}")
    else:
        print(f"{m}: MISSING")
