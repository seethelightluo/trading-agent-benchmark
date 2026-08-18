# -*- coding: utf-8 -*-
"""miner_1: 2031-09-18 cycle data/regime check (visible through 2031-09-17)."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20310918_common import price_panel, macro_panel, WATCH, VISIBLE_THROUGH

close = price_panel("close")
print(f"panel shape {close.shape}, {close.index.min().date()} .. {close.index.max().date()}")

# recent regime: 10/20/60/120d returns
ret = close.pct_change()
print("\n=== recent returns by asset (10/20/60/120d, sorted by 60d) ===")
snap = pd.DataFrame({
    "r10": close.iloc[-1] / close.iloc[-11] - 1,
    "r20": close.iloc[-1] / close.iloc[-21] - 1,
    "r60": close.iloc[-1] / close.iloc[-61] - 1,
    "r120": close.iloc[-1] / close.iloc[-121] - 1,
}).sort_values("r60")
print(snap.round(4).to_string())

# macro snapshot
print("\n=== macro snapshot ===")
for s in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    m = macro_panel(s)
    print(f"{s:8s} last={m.iloc[-1]:.2f} 10d={m.iloc[-1]/m.iloc[-11]-1:+.2%} 60d={m.iloc[-1]/m.iloc[-61]-1:+.2%}")

# cross-sectional dispersion
disp10 = ret.tail(300).std(axis=1)
disp60 = ret.tail(300).mean(axis=1)
print("\n10d x-section disp: mean %.4f last %.4f" % (disp10.mean(), disp10.iloc[-1]))

# 10d forward return alignment check (last date usable for IC)
fwd10 = close.shift(-10) / close - 1.0
print("last 10 IC-usable date:", fwd10.dropna(how="all").index[-1].date())
