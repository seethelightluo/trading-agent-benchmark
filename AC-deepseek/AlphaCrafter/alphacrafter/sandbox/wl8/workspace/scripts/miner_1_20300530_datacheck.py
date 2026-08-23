"""miner_1 2030-05-30: datacheck - verify data availability/activity through visible_through 2030-05-29."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, load_visible_through

ASOF = load_visible_through()
print("visible_through:", ASOF)
px = load_prices(ASOF)
print("price panel shape:", px.shape, "range:", px.index.min().date(), "->", px.index.max().date())
mac = load_macro(ASOF)
print("macro cols:", list(mac.columns), mac.shape)
print("macro last:", mac.tail(2).index.max().date() if len(mac) else None)

# activity: non-flat closes in last 30 trading days
last30 = px.tail(30)
flat = (last30.diff().abs() < 1e-12).all()
for s in WATCH:
    ser = px[s].dropna()
    n = len(ser)
    last = ser.iloc[-1]
    r1 = ser.iloc[-1]/ser.iloc[-21]-1 if n >= 21 else np.nan
    r10 = ser.iloc[-1]/ser.iloc[-11]-1 if n >= 11 else np.nan
    r3 = ser.iloc[-1]/ser.iloc[-4]-1 if n >= 4 else np.nan
    print(f"{s:10s} n={n:5d} last={str(last.date()) if hasattr(last,'date') else last} flat60={bool(flat[s])} r3={r3:+.2%} r10={r10:+.2%} r20={r20:+.2%}")

# macro last values
for c in mac.columns:
    s = mac[c].dropna()
    if len(s):
        print(f"MACRO {c}: last={s.iloc[-1]:.4f} d10={s.iloc[-1]/s.iloc[-11]-1 if len(s)>10 else np.nan:+.2%}")