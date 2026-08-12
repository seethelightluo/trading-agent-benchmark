"""miner1 2030-04-22 probe: confirm data window via sim API and check market state."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
            "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]

for s in TRADABLE:
    df = get_stock_daily_data(symbol=s, days=3000)
    if df is None:
        print(s, "NONE")
    else:
        print(s, len(df), str(df['date'].iloc[0])[:10], "->", str(df['date'].iloc[-1])[:10])

for s in MACRO:
    df = get_index_daily_data(symbol=s, days=3000)
    if df is None:
        print(s, "NONE")
    else:
        print(s, len(df), str(df['date'].iloc[0])[:10], "->", str(df['date'].iloc[-1])[:10])
