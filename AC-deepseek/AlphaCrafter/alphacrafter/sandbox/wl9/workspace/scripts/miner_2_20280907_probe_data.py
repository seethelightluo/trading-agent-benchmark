"""miner_2 2028-09-07: probe data availability for factor research."""
import pandas as pd
import numpy as np
from pathlib import Path

STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    if not f.exists():
        print(f'{a}: MISSING'); continue
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    last = df['date'].iloc[-1]
    print(f'{a}: rows={len(df)} last={last.date()} cols={list(df.columns)}')

for m in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    f = INDEX_DIR / f'{m}.csv'
    if not f.exists():
        print(f'{m}: MISSING'); continue
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    print(f'{m}: rows={len(df)} last={df["date"].iloc[-1].date()} cols={list(df.columns)}')