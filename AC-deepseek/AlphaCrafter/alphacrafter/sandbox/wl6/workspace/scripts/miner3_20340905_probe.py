"""Probe data availability and recent regime for miner3 research cycle 2034-09-05."""
import sys, os
sys.path.insert(0, 'scripts')
import pandas as pd
from factor_validation_lib import load_panel, load_macro

panel = load_panel()
print("full panel shape:", panel.shape)
print("date range:", panel.index.min().date(), "->", panel.index.max().date())

# use last valid common date
vis = panel.index.max()
panel = panel[panel.index <= vis]
print("effective VIS:", vis.date())

ret = panel.pct_change()
print("\nRecent 60d return by asset:")
print((panel.iloc[-1] / panel.iloc[-61] - 1).round(4).sort_values(ascending=False).to_string())
print("\nRecent 20d return by asset:")
print((panel.iloc[-1] / panel.iloc[-21] - 1).round(4).sort_values(ascending=False).to_string())

for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = load_macro(m)
    print(f"{m}: last={s.iloc[-1]:.2f} 20d_chg={(s.iloc[-1]/s.iloc[-21]-1)*100:.1f}% 60d_chg={(s.iloc[-1]/s.iloc[-61]-1)*100:.1f}%")