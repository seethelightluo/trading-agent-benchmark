"""Probe data availability for factor research as of 2030-04-08."""
import sys
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

print("=== TRADABLE WATCHLIST ===")
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None:
        df = get_index_daily_data(symbol=s, days=4000)
    if df is None:
        print(f"{s:10s} NO DATA")
        continue
    print(f"{s:10s} rows={len(df):5d} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()} cols={list(df.columns)}")

print("=== MACRO (OBS ONLY) ===")
for s in MACRO:
    try:
        df = get_index_daily_data(symbol=s, days=4000)
        if df is None:
            df = get_stock_daily_data(symbol=s, days=4000)
        if df is None:
            print(f"{s:10s} NO DATA via API")
            continue
        print(f"{s:10s} rows={len(df):5d} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()}")
    except Exception as e:
        print(f"{s:10s} ERR {e}")

print("=== index_data dir ===")
import os
p = os.path.join('..', 'persistent', 'index_data')
if os.path.isdir(p):
    for f in sorted(os.listdir(p))[:40]:
        print(f)
else:
    print("no ../persistent/index_data dir", os.path.abspath('..'))
