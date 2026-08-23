"""Probe: data availability and panel alignment for factor mining at 2028-06-15.
Truncate at visible_through = 2028-06-14 to avoid lookahead.
"""
import pandas as pd
import numpy as np

CUTOFF = pd.Timestamp('2028-06-14')
TRADABLE = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E',
            'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

closes = {}
rets = {}
vols = {}
for s in TRADABLE:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    closes[s] = df['close']
    rets[s] = df['close'].pct_change()
    vols[s] = df['volume'] if 'volume' in df.columns else None

macro = {}
for s in MACRO:
    df = pd.read_csv(f'../persistent/index_data/{s}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    macro[s] = df['close']

panel = pd.DataFrame(closes)
print('Close panel shape:', panel.shape)
print('Panel date range:', panel.index.min(), '->', panel.index.max())
print('Assets with >= 1000 obs:', (panel.notna().sum() >= 1000).sum(), 'of', len(TRADABLE))
print('NaN per asset:')
print(panel.isna().sum().to_string())

# volume availability
print('\nVolume availability (nonzero rows per asset):')
for s in TRADABLE:
    if vols[s] is not None:
        nz = (vols[s] > 0).sum()
        print(f'  {s}: {nz} nonzero / {len(vols[s])} rows')
    else:
        print(f'  {s}: NO volume column')

print('\nMacro series last dates:', {s: str(macro[s].index.max().date()) for s in MACRO})
print('Macro obs counts:', {s: len(macro[s]) for s in MACRO})

# alignment check: how many dates have >=8 assets
valid = panel.notna().sum(axis=1)
print('\nDates with >=8 assets:', (valid >= 8).sum(), '/', len(panel))
print('Dates with 15 assets:', (valid == 15).sum())