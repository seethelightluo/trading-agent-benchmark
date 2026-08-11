# -*- coding: utf-8 -*-
"""miner_1 2027-04-22 cycle: data availability + quirk check."""
import sys
sys.path.insert(0, 'scripts')
import miner3_lib as L
import numpy as np
import pandas as pd

C, V, H, Lo, O = L.load_close_panel(4000)
print(f"Panel dates: {C.index.min().date()} -> {C.index.max().date()}, {len(C)} rows, {C.shape[1]} assets")
print("\nClose tail:")
print(C.tail(3).round(2).to_string())
print("\nVolume availability (non-null count / total):")
for s in V.columns:
    v = V[s]
    print(f"  {s}: {v.notna().sum()}/{len(v)}  last={v.dropna().iloc[-1] if v.notna().any() else 'NA'}")
print("\nVolume zero-fraction per asset:")
for s in V.columns:
    v = V[s].dropna()
    if len(v):
        print(f"  {s}: {(v == 0).mean():.3f} zero frac")
print("\nHigh-Low sanity (any H<L or H<C etc.):")
bad = (H < Lo).sum().sum()
print("  H<L count:", bad)
print("\nRecent 5d returns (from close):")
R = C.pct_change()
print(R.tail(5).round(4).to_string())
