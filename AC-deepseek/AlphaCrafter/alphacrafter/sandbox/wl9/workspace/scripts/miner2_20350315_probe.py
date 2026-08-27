from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

VISIBLE = pd.Timestamp('2035-03-14')
for sym in ['SPX', '000300.SH', 'XAU', 'BTC', 'US10Y']:
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is None:
        print(sym, 'None'); continue
    df['date'] = pd.to_datetime(df['date'])
    print(sym, 'rows', len(df), 'range', df['date'].min().date(), '..', df['date'].max().date())
for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    df = get_index_daily_data(symbol=m, days=4000)
    if df is None:
        print(m, 'None'); continue
    df['date'] = pd.to_datetime(df['date'])
    print(m, 'rows', len(df), 'range', df['date'].min().date(), '..', df['date'].max().date())