"""miner_1 2028-02-21 (ASOF 2030-02-20): quick data check before factor screen."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_1_common import load_panel, load_macro_panel

px, vol = load_panel(start="2020-01-01")
print("panel shape:", px.shape, "range:", px.index[0].date(), "->", px.index[-1].date())

# freeze detection: assets with zero price movement over trailing 60d
rets = px.pct_change()
std60 = rets.rolling(60).std()
frozen_now = (std60.iloc[-1] < 1e-10).astype(int)
print("\nfrozen at asof (std60~0):", {s: bool(v) for s, v in frozen_now.items()})

# find freeze start date per asset
freeze_start = {}
for s in px.columns:
    tail_std = std60[s]
    # last date where std was > 1e-10
    live = tail_std[tail_std > 1e-10]
    if len(live) == 0 or live.index[-1] < px.index[-1] - pd.Timedelta(days=60):
        freeze_start[s] = str(live.index[-1].date()) if len(live) else "start"
    else:
        freeze_start[s] = None
print("\nfreeze_start:", freeze_start)

# macro recency
for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = load_macro_panel(m, start="2030-01-01")
    print(m, "last:", s.index[-1].date(), "value:", round(float(s.iloc[-1]), 4), "n:", len(s))

# recent market state summary (20d returns through asof)
r20 = (px.iloc[-1] / px.iloc[-21] - 1).sort_values()
print("\n20d returns through", px.index[-1].date(), ":")
print(r20.round(4).to_string())
