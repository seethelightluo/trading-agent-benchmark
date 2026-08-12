# -*- coding: utf-8 -*-
"""miner_3 2027-06-17: data quality check for the 15-asset universe."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_stock_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

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
    n_vol0 = int((df['volume'] == 0).sum()) if 'volume' in df else -1
    n_hl_na = int(df[['high', 'low']].isna().sum().sum())
    n_o_na = int(df['open'].isna().sum()) if 'open' in df else -1
    flat = 'FLAT!' if df['close'].iloc[-60:].nunique() <= 2 else ''
    print(f"{s:10s} n={n:5d} last={last.date()} vol0={n_vol0:5d} hl_na={n_hl_na:5d} o_na={n_o_na:5d} {flat}")
