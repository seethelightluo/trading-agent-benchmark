"""Probe data availability: tradable via stock_data, macro via index_data."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

for sym in TRADABLE:
    df = get_stock_daily_data(symbol=sym, days=2000)
    if df is None or len(df) == 0:
        print(f"{sym:10s} NO DATA")
        continue
    print(f"{sym:10s} rows={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")

for sym in MACRO:
    df = get_index_daily_data(symbol=sym, days=2000)
    if df is None or len(df) == 0:
        print(f"{sym:10s} NO DATA")
        continue
    print(f"{sym:10s} rows={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
