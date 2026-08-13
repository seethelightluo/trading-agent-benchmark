"""Probe data availability for all watchlist instruments and macro signals up to current date (fixed paths)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

print("=== WATCHLIST (tradable, from stock_data) ===")
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) == 0:
        print(f"{s:12s} NO DATA")
    else:
        print(f"{s:12s} rows={len(df):5d} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()} cols={list(df.columns)[:12]}")

print("\n=== MACRO (observation-only, from index_data) ===")
for s in MACRO:
    df = get_index_daily_data(symbol=s, days=4000)
    if df is None or len(df) == 0:
        df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) == 0:
        print(f"{s:12s} NO DATA")
    else:
        print(f"{s:12s} rows={len(df):5d} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()} cols={list(df.columns)[:12]}")

print("\n=== VOLUME AVAILABILITY (last 250 rows) ===")
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=250)
    if df is not None and len(df) > 0:
        v = df['volume']
        nz = (v > 0).sum()
        print(f"{s:12s} volume rows={len(v)} nonzero={nz}")
