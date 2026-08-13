"""miner_3: check data availability as of current date 2033-03-07 (visible through last completed trading day)."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
         'COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO = ['VIX','DXY','USDCNY','USDJPY','EURUSD']

print('=== TRADABLE ===')
lasts = {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4500)
    if df is None:
        print(f'{s}: NO DATA'); continue
    df['date'] = pd.to_datetime(df['date'])
    lasts[s] = df['date'].max()
    print(f'{s}: rows={len(df)} last={df["date"].max().date()} first={df["date"].min().date()} cols={list(df.columns)}')

print('=== MACRO ===')
for s in MACRO:
    df = get_index_daily_data(symbol=s, days=4500)
    if df is None:
        print(f'{s}: NO DATA'); continue
    df['date'] = pd.to_datetime(df['date'])
    print(f'{s}: rows={len(df)} last={df["date"].max().date()}')

print('=== min/max last date across tradable ===')
print('min last:', min(lasts.values()).date(), 'max last:', max(lasts.values()).date())
