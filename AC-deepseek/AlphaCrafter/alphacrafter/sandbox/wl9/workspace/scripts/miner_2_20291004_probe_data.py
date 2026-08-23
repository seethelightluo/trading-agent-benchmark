"""miner_2 probe: recent regime summary through 2029-10-03 (visible)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("=== last close levels & returns (through visible date) ===")
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=400)
    if df is None or len(df) < 130:
        print(f"{sym:10s} insufficient"); continue
    c = df['close']
    last = c.iloc[-1]
    r1m = last/c.iloc[-22]-1
    r3m = last/c.iloc[-66]-1
    r6m = last/c.iloc[-130]-1
    vol = c.pct_change().tail(60).std()
    print(f"{sym:10s} close={last:12.4f} r1m={r1m:8.2%} r3m={r3m:8.2%} r6m={r6m:8.2%} v60={vol:8.2%} last_date={df['date'].iloc[-1]}")

print("\n=== macro levels ===")
for sym in ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    try:
        df = get_index_daily_data(symbol=sym, days=400)
    except Exception as e:
        print(sym, 'err', e); continue
    if df is None or len(df) < 130:
        print(sym, 'insufficient'); continue
    c = df.set_index('date')['close']
    print(f"{sym:10s} close={c.iloc[-1]:10.4f} 20d_roc={c.iloc[-1]/c.iloc[-21]-1:8.2%} 60d_roc={c.iloc[-1]/c.iloc[-61]-1:8.2%}")

print("\n=== volume availability check ===")
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=300)
    if df is None: print(f"{sym:10s} NO DATA"); continue
    v = df['volume']
    nun = v.nunique()
    print(f"{sym:10s} rows={len(df)} vol_nunique={nun} vol_last={v.iloc[-1]:.4g}")