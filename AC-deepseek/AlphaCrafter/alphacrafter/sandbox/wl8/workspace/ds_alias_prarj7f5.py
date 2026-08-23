from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
for sym in ['SPX','BTC','000300.SH','US10Y']:
    df = get_stock_daily_data(sym, 2500)
    print(sym, 'len=', len(df) if df is not None else None, 'start=', str(df.iloc[0]['date'])[:10] if df is not None else '-', 'end=', str(df.iloc[-1]['date'])[:10] if df is not None else '-')
for sym in ['VIX','DXY','USDCNY','USDJPY','EURUSD']:
    df = get_index_daily_data(sym, 2500)
    print(sym, 'len=', len(df) if df is not None else None, 'end=', str(df.iloc[-1]['date'])[:10] if df is not None else '-')