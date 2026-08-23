"""miner_1 probe 2035-05-23: verify data availability + regime snapshot.
VIS = 2035-05-22 (previous completed trading day).
"""
import sys, os
sys.path.insert(0, 'scripts')
import pandas as pd
import numpy as np
from factor_validation_lib import load_panel, load_macro, TRADABLE

VIS = "2035-05-22"
panel = load_panel(max_date=VIS)
print("panel shape:", panel.shape, "range:", panel.index.min().date(), "->", panel.index.max().date(), flush=True)

ret = panel.pct_change()
print("\nRecent 60d return by asset:")
print((panel.iloc[-1] / panel.iloc[-61] - 1).round(4).sort_values(ascending=False).to_string(), flush=True)
print("\nRecent 20d return by asset:")
print((panel.iloc[-1] / panel.iloc[-21] - 1).round(4).sort_values(ascending=False).to_string(), flush=True)
print("\nRecent 10d return by asset:")
print((panel.iloc[-1] / panel.iloc[-11] - 1).round(4).sort_values(ascending=False).to_string(), flush=True)

for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = load_macro(m, max_date=VIS)
    l10 = s.iloc[-1]/s.iloc[-11]-1 if len(s)>11 else float('nan')
    l20 = s.iloc[-1]/s.iloc[-21]-1 if len(s)>21 else float('nan')
    l60 = s.iloc[-1]/s.iloc[-61]-1 if len(s)>61 else float('nan')
    print(f"{m}: last={s.iloc[-1]:.2f} 10d={l10*100:.1f}% 20d={l20*100:.1f}% 60d={l60*100:.1f}%", flush=True)

# Frozen assets check
frozen = [c for c in panel.columns if np.allclose(panel[c].tail(30).pct_change().fillna(0).mean(), 0, atol=1e-12)]
print("\nFrozen (zero mean ret last 30d):", frozen, flush=True)

spx_ret = ret['SPX'].dropna()
print("\nSPX 20d realized vol (ann):", round(float(spx_ret.tail(20).std()*np.sqrt(252))*100,1), "%", flush=True)
disp = ret.rolling(20).std().mean(axis=1)
print("Cross-sectional dispersion (mean 20d vol):", round(float(disp.iloc[-1]*np.sqrt(252))*100,1), "%", flush=True)