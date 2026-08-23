"""Data check for all 15 assets on current date 2032-04-29."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
import numpy as np

watchlist = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
macro = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

print("=" * 100)
print("DATA CHECK: current simulation date contexts")
print("=" * 100)

print("\n--- TRADABLE ASSETS ---")
print(f"{'Symbol':15s} {'Ndays':6s} {'LastDate':14s} {'Close':12s} {'5d%':8s} {'21d%':8s} {'63d%':8s}")
print("-" * 75)
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=300)
    if df is None or len(df) < 20:
        df = get_index_daily_data(symbol=sym, days=300)
    if df is None:
        print(f"{sym:15s} NO DATA")
        continue
    
    df = df.sort_values('date')
    nd = len(df)
    ld = str(df.date.iloc[-1])[:10]
    cl = df.close.iloc[-1]
    
    close_series = df['close'].values
    
    def pct_chg(offset):
        if nd > offset and offset > 0:
            try:
                return (close_series[-1] / close_series[-(offset+1)] - 1) * 100
            except:
                return None
        return None
    
    p5 = pct_chg(5)
    p21 = pct_chg(21)
    p63 = pct_chg(63)
    
    p5s = f"{p5:7.2f}%" if p5 is not None else "  None "
    p21s = f"{p21:7.2f}%" if p21 is not None else "  None "
    p63s = f"{p63:7.2f}%" if p63 is not None else "  None "
    
    print(f"{sym:15s} {nd:6d} {ld:14s} {cl:12.2f} {p5s:8s} {p21s:8s} {p63s:8s}")

print("\n--- MACRO OBSERVATION SIGNALS ---")
for sym in macro:
    df = get_index_daily_data(symbol=sym, days=300)
    if df is None:
        df = get_stock_daily_data(symbol=sym, days=300)
    if df is not None:
        df = df.sort_values('date')
        nd = len(df)
        ld = str(df.date.iloc[-1])[:10]
        cl = df.close.iloc[-1]
        close_arr = df['close'].values
        p5 = (close_arr[-1] / close_arr[-6] - 1) * 100 if nd >= 6 else None
        p63 = (close_arr[-1] / close_arr[-64] - 1) * 100 if nd >= 64 else None
        p5s = f"{p5:7.2f}%" if p5 is not None else "  None"
        p63s = f"{p63:7.2f}%" if p63 is not None else "  None"
        print(f"{sym:15s} {nd:6d} {ld:14s} {cl:12.2f} 5d:{p5s:8s} 63d:{p63s}")
    else:
        print(f"{sym:15s} NO DATA")

import json, os
print("\n--- FACTOR LIBRARY STATUS ---")
bak_files = [f for f in os.listdir('factors') if f.endswith('.bak')]
active_files = [f for f in os.listdir('factors') if f.endswith('.json') and not f.endswith('.bak') and f != 'factor_ensemble.json']
print(f"Active (non-bak) factor JSONs: {active_files}")
print(f"Bak files count: {len(bak_files)}")

print("\nDone.")