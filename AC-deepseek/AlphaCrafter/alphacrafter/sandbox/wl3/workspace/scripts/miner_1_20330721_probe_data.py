"""Probe data availability for factor mining at current date 2033-07-21."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=4200)
    if df is None:
        print(f"{sym:10s} None")
        continue
    print(f"{sym:10s} rows={len(df):5d} last={df['date'].iloc[-1].date()} first={df['date'].iloc[0].date()} cols={list(df.columns)}")

print("--- observation-only ---")
for sym in OBS:
    df = get_index_daily_data(symbol=sym, days=4200)
    if df is None:
        print(f"{sym:10s} None")
        continue
    print(f"{sym:10s} rows={len(df):5d} last={df['date'].iloc[-1].date()} first={df['date'].iloc[0].date()} cols={list(df.columns)}")
