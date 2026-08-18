"""miner_1 2027-07-01 datacheck: live cross-section, recent regime stats."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices, load_macro

ASOF = '2027-06-30'
px = load_prices(ASOF)
macro = load_macro(ASOF)

print("=== Recent closes & liveness ===")
last = px.index[-1]
for s in WATCH:
    v = px[s].dropna()
    if len(v) == 0:
        print(f"{s:10s} NO DATA"); continue
    last_date = v.index[-1]
    n_stale = (last - last_date).days
    ret10 = v.iloc[-1] / v.iloc[-11] - 1 if len(v) > 11 else np.nan
    ret21 = v.iloc[-1] / v.iloc[-22] - 1 if len(v) > 22 else np.nan
    vol20 = v.pct_change().tail(20).std() * np.sqrt(252) if len(v) > 21 else np.nan
    flag = 'FROZEN' if n_stale > 5 else 'live'
    print(f"{s:10s} last={str(last_date.date()):12s} stale_d={n_stale:4d} ret10={ret10:+.3f} ret21={ret21:+.3f} vol20_ann={vol20:.2f} [{flag}]")

print("\n=== Macro last values ===")
for c in macro.columns:
    v = macro[c].dropna()
    if len(v):
        r10 = v.iloc[-1] / v.iloc[-11] - 1 if len(v) > 11 else np.nan
        print(f"{c:8s} last={v.iloc[-1]:.4f} ret10={r10:+.4f}")

print("\n=== Cross-sectional dispersion of 10d returns (live only) ===")
live = [s for s in WATCH if (px.index[-1] - px[s].dropna().index[-1]).days <= 5]
r10 = (px[live].iloc[-1] / px[live].iloc[-11] - 1).dropna()
print("n_live:", len(live), "assets:", live)
print("dispersion (std of r10):", round(r10.std(), 4))
print(r10.sort_values().round(4).to_string())
