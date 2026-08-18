"""miner_2 datacheck 2027-07-01 (ASOF visible_through 2027-06-30).
Check which of the 15 watchlist assets are live vs frozen, and basic regime stats.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro

ASOF = '2027-06-30'
px = load_prices(ASOF)
macro = load_macro(ASOF)

print("=== ASSET LIVE CHECK (last 5 close dates through ASOF) ===")
frozen = []
for s in WATCH:
    col = px[s].dropna()
    last = col.index[-1]
    n_flat = 0
    recent = col.tail(20)
    if len(recent) > 2:
        n_flat = int((recent.diff().abs() < 1e-12).sum())
    status = 'LIVE' if last == pd.Timestamp(ASOF) else f'FROZEN since {last.date()}'
    if status != 'LIVE':
        frozen.append(s)
    print(f"{s:10s} rows={len(col):5d} last={str(last.date()):12s} flat20={n_flat:2d} {status}")

print(f"\nFROZEN: {frozen} ({len(frozen)}/15)")

print("\n=== 21d returns asof ASOF ===")
r21 = (px / px.shift(21) - 1).iloc[-1]
print(r21.round(4).to_string())

print("\n=== Macro last values ===")
print(macro.tail(1).round(4).to_string())

print("\n=== Macro 21d changes ===")
m21 = (macro / macro.shift(21) - 1).iloc[-1]
print(m21.round(4).to_string())

print("\n=== Data span ===")
print(f"px rows: {len(px)}, start {px.index[0].date()}, end {px.index[-1].date()}")
print(f"macro rows: {len(macro)}, start {macro.index[0].date()}, end {macro.index[-1].date()}")
