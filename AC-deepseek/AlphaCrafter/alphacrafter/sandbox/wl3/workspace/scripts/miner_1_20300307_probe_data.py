"""Probe data availability through current visible date (2030-03-06)."""
import sys
sys.path.insert(0, 'scripts')
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
for s in WATCHLIST:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None:
        print(f"{s}: None")
        continue
    print(f"{s}: n={len(df)} range={df['date'].min()}..{df['date'].max()}")
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    df = get_index_daily_data(symbol=s, days=4000)
    if df is None:
        print(f"{s}(idx): None")
        continue
    print(f"{s}(idx): n={len(df)} range={df['date'].min()}..{df['date'].max()}")
