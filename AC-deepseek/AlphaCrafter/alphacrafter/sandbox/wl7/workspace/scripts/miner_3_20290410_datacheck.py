"""miner_3 datacheck through 2029-04-09 (current cycle, current date 2029-04-10)."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import ASSETS, MACRO, load_close, load_macro

END = "2029-04-09"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()

print(f"Calendar days: {len(close)}  assets: {len(close.columns)}")
print(f"Close range: {close.index.min().date()} .. {close.index.max().date()}")

for a in ASSETS:
    r = ret[a].dropna()
    last120 = r.tail(120)
    n_zero = float((last120.abs() < 1e-12).mean())
    print(f"{a:10s} last={close[a].iloc[-1]:12.4f}  zero-share(120d)={n_zero:6.3f}  n_obs={len(r):5d}")

print("\nMacro last values:")
for m in MACRO:
    v = macro[m].iloc[-1]
    r20 = macro[m].iloc[-1] / macro[m].iloc[-21] - 1
    r60 = macro[m].iloc[-1] / macro[m].iloc[-61] - 1
    print(f"{m:8s} last={v:12.4f}  20d={r20:+.4f}  60d={r60:+.4f}")

print("\nAsset 20d/60d returns (last rows):")
for a in ASSETS:
    r20 = close[a].iloc[-1] / close[a].iloc[-21] - 1
    r60 = close[a].iloc[-1] / close[a].iloc[-61] - 1
    print(f"{a:10s} r20={r20:+.4f}  r60={r60:+.4f}")

cs_std = ret.tail(60).std(axis=1)
print(f"\nCS daily dispersion last 60d: mean={cs_std.mean():.4f} last={cs_std.iloc[-1]:.4f}")

# regime via VIX
vix = macro["VIX"]
print(f"VIX last={vix.iloc[-1]:.2f}  r20={vix.iloc[-1]/vix.iloc[-21]-1:+.3f}  r60={vix.iloc[-1]/vix.iloc[-61]-1:+.3f}")
