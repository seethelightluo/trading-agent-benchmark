# -*- coding: utf-8 -*-
"""miner_3 2026-12-03: quick data availability check through prev completed day."""
import sys
sys.path.insert(0, 'scripts')
import miner3_lib as L
import pandas as pd

C, V, H, Lw, O = L.load_close_panel(4000)
print("Close panel shape:", C.shape)
print("Date range:", C.index.min().date(), "->", C.index.max().date())
print("\nPer-symbol rows (close notna):")
print(C.notna().sum().to_string())
print("\nLast 5 rows tail:")
print(C.tail(3).to_string())
print("\nVolume coverage (fraction notna):")
print((V.notna().mean()).round(3).to_string())
print("\nVolume last date non-null count:", V.notna().iloc[-1].sum())
print("\nMacro files:")
import os
for f in ['DXY', 'VIX', 'USDJPY', 'EURUSD', 'USDCNY']:
    p = f'../persistent/index_data/{f}.csv'
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=['date'])
        print(f, df['date'].min(), '->', df['date'].max(), 'rows', len(df))
