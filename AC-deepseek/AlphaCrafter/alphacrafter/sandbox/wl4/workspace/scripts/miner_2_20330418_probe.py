"""miner_2 probe 2033-04-18 - check data availability and current library state."""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, TRADABLE, MACRO

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s")
print("\nPer-asset last date and rows:")
for a in TRADABLE:
    if a in panels:
        df = panels[a]
        print(f"  {a:10s} rows={len(df):5d} last={df.index.max().date()}")
    else:
        print(f"  {a:10s} MISSING")
print("\nMacro signals:")
for m in MACRO:
    if m in panels:
        df = panels[m]
        print(f"  {m:10s} rows={len(df):5d} last={df.index.max().date()}")
    else:
        print(f"  {m:10s} MISSING")
print("\nLast 5 closes (cross-section):")
print(closes.tail(5).round(4))
