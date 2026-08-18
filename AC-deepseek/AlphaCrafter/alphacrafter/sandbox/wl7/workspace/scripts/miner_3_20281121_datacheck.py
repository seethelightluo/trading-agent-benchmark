"""Data quality check through visible_through date for miner_3 (2028-11-21 cycle)."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import ASSETS, MACRO, load_close, load_macro

END = "2028-11-20"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()

print(f"Calendar days: {len(close)}  assets: {len(close.columns)}")
print(f"Close range: {close.index.min().date()} .. {close.index.max().date()}")

# Frozen feeds: last 120d with ~0 return
for a in ASSETS:
    r = ret[a].dropna()
    last120 = r.tail(120)
    n_zero = float((last120.abs() < 1e-12).mean())
    last_val = close[a].iloc[-1]
    print(f"{a:10s} last={last_val:12.4f}  zero-return share(120d)={n_zero:6.3f}  n_obs={len(r):5d}")

print("\nMacro last values:")
for m in MACRO:
    print(f"{m:8s} last={macro[m].iloc[-1]:12.4f}  20d chg={macro[m].iloc[-1]/macro[m].iloc[-21]-1:+.4f}  60d chg={macro[m].iloc[-1]/macro[m].iloc[-61]-1:+.4f}")

# Recent regime summary (last 60d)
print("\nAsset 20d/60d returns (last rows):")
for a in ASSETS:
    r20 = close[a].iloc[-1] / close[a].iloc[-21] - 1
    r60 = close[a].iloc[-1] / close[a].iloc[-61] - 1
    print(f"{a:10s} r20={r20:+.4f}  r60={r60:+.4f}")
