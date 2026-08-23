"""miner3 probe 2035-01-03. Check data availability and recent regime."""
import sys, os
sys.path.insert(0, 'scripts')
import pandas as pd
from factor_validation_lib import load_panel, load_macro, TRADABLE

panel = load_panel(max_date="2035-01-02")
print("full panel shape:", panel.shape)
print("date range:", panel.index.min().date(), "->", panel.index.max().date())

ret = panel.pct_change()
print("\nRecent 60d return by asset:")
print((panel.iloc[-1] / panel.iloc[-61] - 1).round(4).sort_values(ascending=False).to_string())
print("\nRecent 20d return by asset:")
print((panel.iloc[-1] / panel.iloc[-21] - 1).round(4).sort_values(ascending=False).to_string())

for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = load_macro(m, max_date="2035-01-02")
    l20 = s.iloc[-1]/s.iloc[-21]-1 if len(s)>21 else float('nan')
    l60 = s.iloc[-1]/s.iloc[-61]-1 if len(s)>61 else float('nan')
    print(f"{m}: last={s.iloc[-1]:.2f} 20d_chg={l20*100:.1f}% 60d_chg={l60*100:.1f}%")