"""Compute asset-level block returns 2029-02-02 -> 2029-02-16 for memory log."""
import sys
sys.path.insert(0, ".")
from alphacrafter.sim.utils import get_stock_daily_data
import pandas as pd

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D0, D1 = '2029-02-02', '2029-02-16'
for a in assets:
    df = get_stock_daily_data(symbol=a, days=170)
    df['date'] = pd.to_datetime(df['date'])
    sub = df[(df['date']>=pd.Timestamp(D0)) & (df['date']<=pd.Timestamp(D1))]
    if len(sub) >= 2:
        c0 = float(sub['close'].iloc[0]); c1 = float(sub['close'].iloc[-1])
        print(f'{a}: {(c1/c0-1)*100:+.2f}%')
    else:
        print(f'{a}: insufficient data {len(sub)}')
