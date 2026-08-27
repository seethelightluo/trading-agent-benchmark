from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

VISIBLE = pd.Timestamp('2035-05-09')
for sym in ['SPX', '000300.SH', 'XAU', 'BTC', 'US10Y', 'COPPER', 'CN10Y', '000688.SH', 'NDX', 'SOX', 'HSI', 'N225', 'SX5E', 'WTI', 'ETH']:
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is None: print(sym, 'None-stock'); continue
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VISIBLE]
    print(sym, 'stock rows', len(df), 'range', df['date'].min().date(), '..', df['date'].max().date())
for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    df = get_index_daily_data(symbol=m, days=4000)
    if df is None: print(m, 'None-index'); continue
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VISIBLE]
    print(m, 'index rows', len(df), 'range', df['date'].min().date(), '..', df['date'].max().date())