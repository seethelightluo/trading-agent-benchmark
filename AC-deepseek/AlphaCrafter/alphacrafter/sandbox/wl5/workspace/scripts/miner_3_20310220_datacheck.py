# -*- coding: utf-8 -*-
"""miner_3 2031-02-20 datacheck: verify data visibility through 2031-02-19."""
import sys
import pandas as pd
sys.path.insert(0, 'scripts')
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
VISIBLE = pd.Timestamp('2031-02-19')

print('=== WATCH (stock_data) ===')
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) == 0:
        print(f'{s:10s} NO DATA')
        continue
    last = pd.Timestamp(df['date'].iloc[-1]).normalize()
    n = len(df)
    print(f'{s:10s} rows={n:5d} last={last.date()} visible_ok={last <= VISIBLE}')

print('=== MACRO (index_data) ===')
for s in MACRO:
    try:
        m = pd.read_csv(f'../persistent/index_data/{s}.csv', parse_dates=['date'])
        last = pd.Timestamp(m['date'].iloc[-1]).normalize()
        print(f'{s:10s} rows={len(m):5d} last={last.date()} visible_ok={last <= VISIBLE}')
    except Exception as e:
        print(f'{s:10s} ERR {e}')
