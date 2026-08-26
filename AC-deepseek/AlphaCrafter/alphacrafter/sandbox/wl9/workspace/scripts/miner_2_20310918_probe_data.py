"""miner_2 probe: data availability at 2031-09-18 cycle."""
import sys, os
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

panels = {}
for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=2600)
    if df is None or len(df) == 0:
        print(sym, "NO DATA")
        continue
    df = df.set_index('date')
    has_vol = 'volume' in df.columns and df['volume'].notna().mean() > 0.5
    print(f"{sym}: rows={len(df)} last={df.index[-1].date()} has_volume={has_vol} vol_frac={df['volume'].notna().mean() if has_vol else 0:.3f}")
    panels[sym] = df

print("--- macro ---")
for s in MACRO:
    try:
        dfp = pd.read_csv(f'../persistent/index_data/{s}.csv')
        dfp['date'] = pd.to_datetime(dfp['date'])
        print(f"{s}: rows={len(dfp)} last={dfp['date'].iloc[-1].date()}")
    except Exception as e:
        print(s, "ERR", e)