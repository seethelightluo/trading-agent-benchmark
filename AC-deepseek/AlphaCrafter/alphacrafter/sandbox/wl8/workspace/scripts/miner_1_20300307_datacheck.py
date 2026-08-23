"""miner_1 2030-03-07: data state check before factor screening (asof = visible_through)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_1_common import load_panel, load_macro_panel

px, vol = load_panel(start="2020-01-01")
print("panel shape:", px.shape, "range:", px.index[0].date(), "->", px.index[-1].date())

rets = px.pct_change()
std60 = rets.rolling(60).std()
print("\nfrozen at asof (std60~0):", {s: bool(v) for s, v in (std60.iloc[-1] < 1e-10).items()})

freeze_start = {}
for s in px.columns:
    live = std60[s][std60[s] > 1e-10]
    if len(live) == 0 or live.index[-1] < px.index[-1] - pd.Timedelta(days=60):
        freeze_start[s] = str(live.index[-1].date()) if len(live) else "start"
    else:
        freeze_start[s] = None
print("freeze_start:", freeze_start)

for m in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    s = load_macro_panel(m, start="2029-06-01")
    print(m, "last:", s.index[-1].date(), "value:", round(float(s.iloc[-1]), 4), "n:", len(s))

r20 = (px.iloc[-1] / px.iloc[-21] - 1).sort_values()
print("\n20d returns through", px.index[-1].date(), ":")
print(r20.round(4).to_string())
r60 = (px.iloc[-1] / px.iloc[-61] - 1).sort_values()
print("\n60d returns through", px.index[-1].date(), ":")
print(r60.round(4).to_string())
