"""miner_1 datacheck asof 2027-06-02 (visible through)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices, load_macro

ASOF = '2027-06-02'
px = load_prices(ASOF)
macro = load_macro(ASOF)
print("px shape:", px.shape, "range:", px.index.min().date(), "->", px.index.max().date())
print("macro shape:", macro.shape, "range:", macro.index.min().date(), "->", macro.index.max().date())

# last 15 rows check for frozen feeds
tail = px.tail(15)
for s in WATCH:
    last = px[s].dropna()
    if len(last) < 5:
        print(f"{s}: only {len(last)} values, last={last.index[-1].date() if len(last) else 'NA'}")
        continue
    recent = last.tail(20)
    nuniq = recent.nunique()
    last_date = last.index[-1]
    stale = (pd.Timestamp(ASOF) - last_date).days
    print(f"{s}: n={len(last)} last_date={last_date.date()} stale_days={stale} uniq_last20={nuniq} last_close={last.iloc[-1]:.4f}")

# last 5-day returns of live assets
r = px.pct_change().tail(5)
print("\nlast 5d pct change (live rows):")
print(r.round(4).tail(5).to_string())

# VIX and other macro last values
print("\nmacro tail:")
print(macro.tail(3).round(4).to_string())
