"""Probe data availability and recent regime for miner3 research cycle 2030-07-25."""
import sys, os
sys.path.insert(0, 'scripts')
import pandas as pd
import numpy as np
from factor_validation_lib import load_panel, load_macro, TRADABLE

panel = load_panel(max_date='2030-07-24')
print("panel shape:", panel.shape)
print("date range:", panel.index.min().date(), "->", panel.index.max().date())
print("\nlast 5 rows (close):")
print(panel.tail(5).round(2).to_string())

# Recent returns snapshot
ret = panel.pct_change()
print("\nRecent 60d return by asset:")
print((panel.iloc[-1] / panel.iloc[-61] - 1).round(4).sort_values(ascending=False).to_string())

print("\nRecent 20d return by asset:")
print((panel.iloc[-1] / panel.iloc[-21] - 1).round(4).sort_values(ascending=False).to_string())

for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = load_macro(m, max_date='2030-07-24')
    print(f"{m}: last={s.iloc[-1]:.2f} 20d_chg={ (s.iloc[-1]/s.iloc[-21]-1)*100:.1f}% 60d_chg={(s.iloc[-1]/s.iloc[-61]-1)*100:.1f}%")
