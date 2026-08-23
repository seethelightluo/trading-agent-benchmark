"""miner_1 2028-10-19: probe data availability; cutoff = last completed trading day before 2028-10-19."""
import pandas as pd
import numpy as np
from pathlib import Path

CUTOFF = pd.Timestamp('2028-10-18')
TRADABLE = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E',
            'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

for a in TRADABLE:
    f = Path(f'../persistent/stock_data/{a}.csv')
    if not f.exists():
        print(f'{a}: MISSING'); continue
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= CUTOFF]
    print(f'{a}: rows={len(df)} last={df["date"].iloc[-1].date()} cols={list(df.columns)}')

closes = {}
for a in TRADABLE:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    closes[a] = df['close']

panel = pd.DataFrame(closes)
print('\nClose panel shape:', panel.shape)
print('Panel date range:', panel.index.min().date(), '->', panel.index.max().date())
print('Assets with >= 1500 obs:', (panel.notna().sum() >= 1500).sum(), 'of', len(TRADABLE))
valid = panel.notna().sum(axis=1)
print('Dates with >=8 assets:', (valid >= 8).sum(), '/', len(panel))

for m in MACRO:
    f = Path(f'../persistent/index_data/{m}.csv')
    if not f.exists():
        print(f'{m}: MISSING'); continue
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= CUTOFF]
    print(f'{m}: rows={len(df)} last={df["date"].iloc[-1].date()}')