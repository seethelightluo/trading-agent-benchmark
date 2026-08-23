"""miner_2 probe: recent regime summary through 2029-06-27 (visible)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("=== last close levels & 1m/3m/6m returns (through 2029-06-27) ===")
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=300)
    if df is None or len(df) < 130:
        print(f"{sym:10s} insufficient"); continue
    c = df['close']
    last = c.iloc[-1]
    r1m = last/c.iloc[-22]-1
    r3m = last/c.iloc[-66]-1
    r6m = last/c.iloc[-130]-1
    print(f"{sym:10s} close={last:12.4f} r1m={r1m:8.2%} r3m={r3m:8.2%} r6m={r6m:8.2%}")

print("\n=== macro levels ===")
for sym in ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    df = get_index_daily_data(symbol=sym, days=300)
    c = df.set_index('date')['close']
    print(f"{sym:10s} close={c.iloc[-1]:10.4f} 20d_roc={c.iloc[-1]/c.iloc[-21]-1:8.2%} 60d_roc={c.iloc[-1]/c.iloc[-61]-1:8.2%}")