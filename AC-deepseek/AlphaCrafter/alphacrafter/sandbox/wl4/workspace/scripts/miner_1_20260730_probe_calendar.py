"""Inspect per-asset calendar structure of the 15 tradable instruments."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE

panels = load_panels()
closes = close_panel(panels)
print("union index: n =", len(closes), closes.index[0].date(), "->", closes.index[-1].date())
print("\nper-asset valid close stats:")
for a in TRADABLE:
    s = closes[a]
    v = s.dropna()
    if len(v) == 0:
        print(f"{a:12s} NO DATA")
        continue
    gaps = s.notna()
    # longest consecutive run of valid days in union index
    run = cur = 0
    for x in gaps:
        cur = cur + 1 if x else 0
        run = max(run, cur)
    # weekly cadence estimate: fraction of valid days that fall on Mon-Fri
    wd = v.index.dayofweek
    print(f"{a:12s} n_valid={len(v):5d} start={v.index[0].date()} end={v.index[-1].date()} "
          f"max_consec={run:4d} weekday_frac={ (wd<5).mean():.2f}")
# also check macro panels
for s in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    if s in panels:
        v = panels[s]["close"].dropna()
        print(f"{s:12s} n_valid={len(v):5d} start={v.index[0].date()} end={v.index[-1].date()}")
