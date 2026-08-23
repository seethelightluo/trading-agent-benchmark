"""miner_1 probe 2035-05-09: data availability + volume/macro check + recent regime snapshot.
VIS = 2035-05-08 (previous completed trading day) per daily-decision rule.
"""
import sys, os
sys.path.insert(0, 'scripts')
import pandas as pd
import numpy as np
from factor_validation_lib import load_panel, load_macro, TRADABLE

VIS = "2035-05-08"
panel = load_panel(max_date=VIS)
print("panel shape:", panel.shape, "range:", panel.index.min().date(), "->", panel.index.max().date())

# volume availability
vol = {}
for sym in TRADABLE:
    try:
        from alphacrafter.sim.utils import get_stock_daily_data
        df = get_stock_daily_data(symbol=sym, days=4000)
        if df is not None and "volume" in df.columns:
            v = df.set_index(pd.to_datetime(df["date"]))["volume"].astype(float)
            v = v[v.index <= pd.Timestamp(VIS)]
            vol[sym] = float(v.notna().sum())
    except Exception as e:
        vol[sym] = f"ERR {e}"
print("\nvolume non-null counts:", vol)

ret = panel.pct_change()
print("\nRecent 60d return by asset:")
print((panel.iloc[-1] / panel.iloc[-61] - 1).round(4).sort_values(ascending=False).to_string())
print("\nRecent 20d return by asset:")
print((panel.iloc[-1] / panel.iloc[-21] - 1).round(4).sort_values(ascending=False).to_string())

for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = load_macro(m, max_date=VIS)
    l10 = s.iloc[-1]/s.iloc[-11]-1 if len(s)>11 else float('nan')
    l20 = s.iloc[-1]/s.iloc[-21]-1 if len(s)>21 else float('nan')
    l60 = s.iloc[-1]/s.iloc[-61]-1 if len(s)>61 else float('nan')
    print(f"{m}: last={s.iloc[-1]:.2f} 10d={l10*100:.1f}% 20d={l20*100:.1f}% 60d={l60*100:.1f}%")

print("\nSPX 20d realized vol (ann):", round(float(ret['SPX'].tail(20).std()*np.sqrt(252))*100, 1), "%")
disp = ret.rolling(20).std().mean(axis=1)
print("Cross-sectional dispersion (mean 20d vol):", round(float(disp.iloc[-1]*np.sqrt(252))*100, 1), "%")