"""miner_3 2030-05-02: datacheck of current regime (visible-through 2030-05-01).
Print recent returns by asset, macro levels, coverage, and frozen-asset status."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices, load_macro, load_visible_through

ASOF = load_visible_through()
print("visible_through:", ASOF)
px = load_prices(ASOF)
macro = load_macro(ASOF)

print("\n=== last close per asset ===")
print(px.iloc[-1].round(2))

print("\n=== returns (1m / 3m / 6m) ===")
for s in WATCH:
    v = px[s].dropna()
    if len(v) < 130:
        print(f"{s:10s} n={len(v):4d} insufficient history")
        continue
    r1 = v.iloc[-1]/v.iloc[-22 if len(v)>22 else 0]-1
    r3 = v.iloc[-1]/v.iloc[-66 if len(v)>66 else 0]-1
    r6 = v.iloc[-1]/v.iloc[-132 if len(v)>132 else 0]-1
    print(f"{s:10s} 1m={r1*100:7.2f}% 3m={r3*100:7.2f}% 6m={r6*100:7.2f}%  last_frozen={int(v.iloc[-1]==v.iloc[-5])}")

print("\n=== macro last levels ===")
for m in macro.columns:
    v = macro[m].dropna()
    if len(v):
        print(f"{m:8s} {v.iloc[-1]:.3f}  1m={ (v.iloc[-1]/v.iloc[-22]-1)*100:7.2f}%")

# coverage/width check: assets with flat last 60d
print("\n=== assets flat last 60d (frozen) ===")
for s in WATCH:
    v = px[s].dropna().tail(60)
    if len(v) and (v.std() < 1e-12):
        print("  frozen:", s)