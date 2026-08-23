from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
W = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for sym in W + ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    df = get_stock_daily_data(symbol=sym, days=5000)
    if df is None:
        df = get_index_daily_data(symbol=sym, days=5000)
    print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")