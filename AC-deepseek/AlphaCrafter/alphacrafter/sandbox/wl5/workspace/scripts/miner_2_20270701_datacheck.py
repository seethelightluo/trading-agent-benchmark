# -*- coding: utf-8 -*-
"""miner_2 2027-07-01: data availability check for 15-asset universe + macro."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

for s in WATCHLIST:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) == 0:
        print(f"{s:10s} NO DATA")
        continue
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    last = df.index.max()
    n = len(df)
    print(f"{s:10s} n={n:5d} first={df.index.min().date()} last={last.date()}")

print("--- macro observation-only ---")
for s in MACRO:
    df = get_index_daily_data(symbol=s, days=4000)
    if df is None or len(df) == 0:
        print(f"{s:10s} NO DATA")
        continue
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    print(f"{s:10s} n={len(df):5d} first={df.index.min().date()} last={df.index.max().date()}")
